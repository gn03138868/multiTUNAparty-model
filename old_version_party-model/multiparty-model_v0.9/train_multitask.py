"""
多任務 TransUNet 訓練腳本 - 優化版本
主要優化：
1. 按任務分組批量處理（而非逐樣本處理）
2. 混合精度訓練 (AMP) - 2-3x 加速
3. 向量化損失計算
4. 優化的數據加載
5. 梯度累積支援
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU 記憶體管理工具
# ============================================================================

def clear_gpu_memory():
    """徹底清理 GPU 記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        import gc
        gc.collect()

def get_gpu_memory_info():
    """獲取 GPU 記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    return None

def reset_peak_memory():
    """重置峰值記憶體統計"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# ============================================================================
# 優化的損失函數（向量化計算，不用 for 迴圈）
# ============================================================================

class OptimizedBoundaryAwareLoss(nn.Module):
    """
    優化的邊界感知損失函數
    使用向量化操作，避免逐樣本計算
    """
    def __init__(self, boundary_weights=None, foreground_weights=None, 
                 deep_supervision_weights=None, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
        # 預設權重
        self.boundary_weights = boundary_weights or {0: 2.0, 1: 3.0, 2: 5.0}
        self.foreground_weights = foreground_weights or {0: 1.0, 1: 1.5, 2: 3.0}
        self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.2, 0.1]
        
        # 轉換為 tensor 以便向量化（最多支援 10 個任務）
        max_tasks = 10
        self.register_buffer('boundary_weight_tensor', 
            torch.tensor([self.boundary_weights.get(i, 2.0) for i in range(max_tasks)]))
        self.register_buffer('foreground_weight_tensor',
            torch.tensor([self.foreground_weights.get(i, 1.0) for i in range(max_tasks)]))
    
    def forward(self, outputs, targets, task_ids, boundary_targets=None):
        """
        向量化的損失計算
        """
        # 處理輸出格式
        if isinstance(outputs, dict):
            seg_out = outputs.get('refined', outputs.get('seg'))
            boundary_out = outputs.get('boundary')
            deep_segs = outputs.get('deep_seg', [])
            deep_boundaries = outputs.get('deep_boundary', [])
        else:
            seg_out = outputs
            boundary_out = None
            deep_segs = []
            deep_boundaries = []
        
        # 生成邊界目標（如果沒有提供）
        if boundary_targets is None:
            boundary_targets = self._generate_boundary_targets(targets)
        
        total_loss = 0.0
        loss_dict = {}
        
        # 獲取任務權重（向量化）
        task_weights = self.foreground_weight_tensor[task_ids]  # [B]
        boundary_task_weights = self.boundary_weight_tensor[task_ids]  # [B]
        
        # 1. 主分割損失（向量化）
        seg_loss = self._vectorized_seg_loss(seg_out, targets, task_weights)
        total_loss += seg_loss
        loss_dict['seg_loss'] = seg_loss.item()
        
        # 2. 邊界損失（如果有）
        if boundary_out is not None:
            boundary_loss = self._vectorized_boundary_loss(
                boundary_out, boundary_targets, boundary_task_weights)
            total_loss += boundary_loss
            loss_dict['boundary_loss'] = boundary_loss.item()
        
        # 3. 深度監督損失
        if len(deep_segs) > 0:
            deep_loss = 0.0
            for i, (ds, db) in enumerate(zip(deep_segs, deep_boundaries)):
                weight = self.deep_supervision_weights[i] if i < len(self.deep_supervision_weights) else 0.1
                deep_loss += weight * self._vectorized_seg_loss(ds, targets, task_weights)
                if db is not None:
                    deep_loss += weight * 0.5 * self._vectorized_boundary_loss(
                        db, boundary_targets, boundary_task_weights)
            total_loss += deep_loss
            loss_dict['deep_loss'] = deep_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _vectorized_seg_loss(self, pred, target, task_weights):
        """向量化的分割損失"""
        B = pred.shape[0]
        
        # Dice Loss（批量計算）
        pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
        pred_flat = pred_sigmoid.view(B, -1)
        target_flat = target.view(B, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1 - dice)  # [B]
        
        # BCE Loss（批量計算）
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        bce_loss = bce_loss.view(B, -1).mean(dim=1)  # [B]
        
        # 前景加權 BCE
        fg_mask = (target > 0.5).float()
        bg_mask = 1 - fg_mask
        fg_count = fg_mask.view(B, -1).sum(dim=1)
        bg_count = bg_mask.view(B, -1).sum(dim=1)
        
        # 動態權重
        fg_weight = torch.clamp(bg_count / (fg_count + 1e-7), min=1.0, max=10.0)
        
        weighted_bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weight_map = fg_mask * fg_weight.view(B, 1, 1, 1) + bg_mask
        weighted_bce = (weighted_bce * weight_map).view(B, -1).mean(dim=1)  # [B]
        
        # 組合損失，乘以任務權重
        sample_loss = (dice_loss + bce_loss + weighted_bce) * task_weights
        
        return sample_loss.mean()
    
    def _vectorized_boundary_loss(self, pred, target, task_weights):
        """向量化的邊界損失"""
        B = pred.shape[0]
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        bce_loss = bce_loss.view(B, -1).mean(dim=1)  # [B]
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
        pred_flat = pred_sigmoid.view(B, -1)
        target_flat = target.view(B, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1 - dice)  # [B]
        
        sample_loss = (bce_loss + dice_loss) * task_weights
        
        return sample_loss.mean()
    
    def _generate_boundary_targets(self, targets):
        """生成邊界目標"""
        kernel_size = 3
        targets_float = targets.float()
        dilated = F.max_pool2d(targets_float, kernel_size, stride=1, padding=kernel_size//2)
        eroded = -F.max_pool2d(-targets_float, kernel_size, stride=1, padding=kernel_size//2)
        boundary = dilated - eroded
        return boundary


# ============================================================================
# 任務平衡採樣器
# ============================================================================

class TaskBalancedSampler:
    """
    任務平衡採樣器
    確保每個 batch 中包含不同任務的樣本
    """
    def __init__(self, dataset, batch_size=4, drop_last=False):
        """
        Args:
            dataset: MultiTaskSegmentationDataset
            batch_size: batch 大小
            drop_last: 是否丟棄最後不完整的 batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 動態偵測任務數量
        max_task_id = 0
        
        # 先掃描一遍找出所有任務
        for idx in range(len(dataset.patches)):
            _, _, task_id = dataset.patches[idx]
            if task_id > max_task_id:
                max_task_id = task_id
        
        self.num_tasks = max_task_id + 1
        
        # 按任務分組索引
        self.task_indices = {i: [] for i in range(self.num_tasks)}
        for idx, (_, _, task_id) in enumerate(dataset.patches):
            self.task_indices[task_id].append(idx)
        
        # 計算每個任務的樣本數
        self.task_counts = {
            task_id: len(indices) 
            for task_id, indices in self.task_indices.items()
        }
        
        print(f"TaskBalancedSampler: {self.num_tasks} tasks, counts: {self.task_counts}")
    
    def __iter__(self):
        """產生平衡的 batch 索引"""
        # 複製索引列表（避免修改原始列表）
        task_indices_copy = {
            task_id: list(indices) 
            for task_id, indices in self.task_indices.items()
        }
        
        # 隨機打亂每個任務的索引
        for task_id in task_indices_copy:
            np.random.shuffle(task_indices_copy[task_id])
        
        # 當前各任務的指針
        task_pointers = {task_id: 0 for task_id in range(self.num_tasks)}
        
        batch_indices = []
        
        while True:
            # 檢查是否所有任務都已用完
            all_exhausted = all(
                task_pointers[task_id] >= len(task_indices_copy[task_id])
                for task_id in range(self.num_tasks)
            )
            if all_exhausted:
                break
            
            # 嘗試從每個任務中取樣
            for task_id in range(self.num_tasks):
                if task_pointers[task_id] < len(task_indices_copy[task_id]):
                    idx = task_indices_copy[task_id][task_pointers[task_id]]
                    batch_indices.append(idx)
                    task_pointers[task_id] += 1
                    
                    # 當 batch 滿了，返回
                    if len(batch_indices) == self.batch_size:
                        yield batch_indices
                        batch_indices = []
        
        # 處理剩餘的樣本
        if len(batch_indices) > 0 and not self.drop_last:
            yield batch_indices
    
    def __len__(self):
        """計算總 batch 數"""
        total_samples = sum(self.task_counts.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


# ============================================================================
# 優化的訓練器
# ============================================================================

class OptimizedMultiTaskTrainer:
    """
    優化的多任務訓練器
    
    主要優化：
    1. 按任務分組批量處理
    2. 混合精度訓練 (AMP)
    3. 梯度累積
    4. 優化的數據加載
    """
    
    def __init__(self, config_path='configs/default.yaml'):
        # 載入配置
        self.config = self._load_config(config_path)
        
        # 設置設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            # 啟用 cudnn benchmark 以加速
            torch.backends.cudnn.benchmark = True
        
        # 任務配置
        self.num_tasks = self.config.get('num_tasks', 3)
        self.task_names = self.config.get('task_names', {0: 'Cell', 1: 'Blood', 2: 'Root'})
        if isinstance(self.task_names, list):
            self.task_names = {i: name for i, name in enumerate(self.task_names)}
        
        # 混合精度訓練
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度累積
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Number of tasks: {self.num_tasks}")
        print(f"  Mixed Precision (AMP): {self.use_amp}")
        print(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"{'='*60}")
        
        # 創建輸出目錄
        self.output_dir = Path('outputs')
        self.model_dir = self.output_dir / 'models'
        self.pred_dir = self.output_dir / 'predictions'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        
        # 訓練歷史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_iou': [], 'val_dice': [],
            'task_metrics': {i: [] for i in range(self.num_tasks)},
            'loss_components': {'train': [], 'val': []}
        }
        
        # 初始化
        self._init_model()
        self._init_data()
        self._init_optimizer()
    
    def _load_config(self, config_path):
        """載入配置"""
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}, using defaults")
            return self._default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _default_config(self):
        """預設配置"""
        return {
            'batch_size': 4,
            'epochs': 200,
            'lr': 1e-5,
            'patch_size': 400,
            'num_decoder_conv_layers': 80,
            'data_path': 'data/',
            'task_structure': 'subfolder',
            'num_tasks': 3,
            'use_amp': True,  # 混合精度訓練
            'gradient_accumulation_steps': 2,  # 梯度累積，減少記憶體使用
            'num_workers': 4,  # DataLoader workers
            'gpu_cache_clear_freq': 20,  # 每多少個 batch 清理一次 GPU（較小值=更頻繁清理）
            'boundary_weights': {0: 2.0, 1: 3.0, 2: 5.0},
            'foreground_weights': {0: 1.0, 1: 1.5, 2: 3.0},
            'use_deep_supervision': True,
            'gpu_cache_clear_freq': 50,  # 每多少個 batch 清理一次 GPU 記憶體
        }
    
    def _init_model(self):
        """初始化模型"""
        print("\nInitializing model...")
        
        # 嘗試導入模型
        try:
            from model_multitask_boundaryversion import MultiTaskTransUNet
            print("  ✓ Using boundary-aware model")
        except ImportError:
            try:
                from model_multitask import MultiTaskTransUNet
                print("  ✓ Using original model")
            except ImportError:
                raise ImportError("Cannot find model_multitask.py or model_multitask_boundaryversion.py")
        
        self.model = MultiTaskTransUNet(
            img_size=self.config['patch_size'],
            patch_size=16,
            in_channels=3,
            num_classes=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            num_decoder_layers=self.config['num_decoder_conv_layers'],
            num_tasks=self.num_tasks,
            task_embed_dim=256,
            use_deep_supervision=self.config.get('use_deep_supervision', True)
        ).to(self.device)
        
        # 參數統計
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # 載入預訓練權重
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """載入預訓練權重"""
        pretrained_path = self.config.get('pretrained_model_path')
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading pretrained weights from: {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # 部分載入
                model_dict = self.model.state_dict()
                matched = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(matched)
                self.model.load_state_dict(model_dict)
                print(f"  ✓ Loaded {len(matched)}/{len(model_dict)} layers")
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
    
    def _init_data(self):
        """初始化數據載入器"""
        print("\nInitializing datasets...")
        
        from dataset_multitask import MultiTaskSegmentationDataset
        
        # 訓練集
        self.train_dataset = MultiTaskSegmentationDataset(
            data_root=self.config['data_path'],
            mode='train',
            patch_size=self.config['patch_size'],
            task_structure=self.config.get('task_structure', 'subfolder')
        )
        
        # 驗證集
        self.val_dataset = MultiTaskSegmentationDataset(
            data_root=self.config['data_path'],
            mode='val',
            patch_size=self.config['patch_size'],
            task_structure=self.config.get('task_structure', 'subfolder')
        )
        
        # 任務平衡採樣器
        train_sampler = TaskBalancedSampler(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            drop_last=True
        )
        
        num_workers = self.config.get('num_workers', 4)
        
        # 數據載入器（優化配置）
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,  # 保持 workers 存活
            prefetch_factor=2 if num_workers > 0 else None  # 預取
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
    
    def _init_optimizer(self):
        """初始化優化器"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=0.01
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        
        # 損失函數
        self.criterion = OptimizedBoundaryAwareLoss(
            boundary_weights=self.config.get('boundary_weights'),
            foreground_weights=self.config.get('foreground_weights'),
            deep_supervision_weights=self.config.get('deep_supervision_weights', [0.4, 0.2, 0.1])
        ).to(self.device)
        
        print("  ✓ Optimizer and loss function initialized")
    
    def _batch_forward_by_task(self, images, task_ids):
        """
        按任務分組批量前向傳播
        這是關鍵優化：避免逐樣本處理
        """
        batch_size = images.shape[0]
        unique_tasks = torch.unique(task_ids)
        
        # 如果只有一個任務，直接批量處理
        if len(unique_tasks) == 1:
            return self.model(images, task_id=unique_tasks[0].item())
        
        # 多個任務：按任務分組處理，然後重新排列
        outputs_dict = {}
        task_indices = {}
        
        # 分組
        for task in unique_tasks:
            mask = (task_ids == task)
            task_indices[task.item()] = mask
            task_images = images[mask]
            
            if task_images.shape[0] > 0:
                task_output = self.model(task_images, task_id=task.item())
                outputs_dict[task.item()] = task_output
        
        # 重新組合輸出（保持原始順序）
        first_output = list(outputs_dict.values())[0]
        
        if isinstance(first_output, dict):
            # 字典輸出
            combined = {}
            for key in first_output.keys():
                if isinstance(first_output[key], torch.Tensor):
                    # 創建空 tensor
                    shape = list(first_output[key].shape)
                    shape[0] = batch_size
                    combined_tensor = torch.zeros(shape, device=images.device, dtype=first_output[key].dtype)
                    
                    # 填充各任務的結果
                    for task, indices in task_indices.items():
                        if task in outputs_dict:
                            combined_tensor[indices] = outputs_dict[task][key]
                    
                    combined[key] = combined_tensor
                elif isinstance(first_output[key], list):
                    # 深度監督列表
                    combined_list = []
                    for i in range(len(first_output[key])):
                        shape = list(first_output[key][i].shape)
                        shape[0] = batch_size
                        combined_tensor = torch.zeros(shape, device=images.device, dtype=first_output[key][i].dtype)
                        
                        for task, indices in task_indices.items():
                            if task in outputs_dict:
                                combined_tensor[indices] = outputs_dict[task][key][i]
                        
                        combined_list.append(combined_tensor)
                    combined[key] = combined_list
            
            return combined
        else:
            # Tensor 輸出
            shape = list(first_output.shape)
            shape[0] = batch_size
            combined = torch.zeros(shape, device=images.device, dtype=first_output.dtype)
            
            for task, indices in task_indices.items():
                if task in outputs_dict:
                    combined[indices] = outputs_dict[task]
            
            return combined
    
    def train_epoch(self, epoch):
        """訓練一個 epoch（優化版，加入 OOM 處理）"""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        # Epoch 開始前清理記憶體
        clear_gpu_memory()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images, masks, task_ids = batch[:3]
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                task_ids = task_ids.to(self.device, non_blocking=True)
                
                # 混合精度前向傳播
                with autocast(enabled=self.use_amp):
                    # 按任務分組批量處理
                    outputs = self._batch_forward_by_task(images, task_ids)
                    
                    # 計算損失
                    loss, loss_dict = self.criterion(outputs, masks, task_ids)
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向傳播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度累積
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                
                # 記錄損失（在清理前提取數值）
                loss_value = loss.item() * self.gradient_accumulation_steps
                total_loss += loss_value
                num_batches += 1
                
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                
                # 明確刪除不需要的變量，釋放記憶體
                del outputs, loss, loss_dict
                del images, masks, task_ids
                
                # 更新進度條
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # 定期清理 GPU 記憶體（更頻繁）
                cache_clear_freq = self.config.get('gpu_cache_clear_freq', 30)
                if (batch_idx + 1) % cache_clear_freq == 0:
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n⚠ OOM at batch {batch_idx}, clearing memory and skipping...")
                    # 清理所有可能的變量
                    if 'outputs' in locals(): del outputs
                    if 'loss' in locals(): del loss
                    if 'images' in locals(): del images
                    if 'masks' in locals(): del masks
                    clear_gpu_memory()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e
        
        # Epoch 結束時徹底清理
        clear_gpu_memory()
        
        # 學習率更新
        self.scheduler.step()
        
        # 計算平均損失
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_components = {k: v / num_batches for k, v in loss_components.items()}
        else:
            avg_loss = 0.0
            avg_components = {}
        
        return avg_loss, avg_components
    
    @torch.no_grad()
    def validate(self, epoch):
        """驗證（優化版，加入更好的記憶體管理）"""
        # 驗證前徹底清理 GPU 記憶體
        clear_gpu_memory()
        
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        all_preds = []
        all_targets = []
        all_task_ids = []
        
        # 收集每個任務的樣本用於視覺化
        task_samples = {i: None for i in range(self.num_tasks)}
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating')):
            images, masks, task_ids = batch[:3]
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            task_ids = task_ids.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                outputs = self._batch_forward_by_task(images, task_ids)
                loss, loss_dict = self.criterion(outputs, masks, task_ids)
            
            total_loss += loss.item()
            
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            # 收集預測結果
            if isinstance(outputs, dict):
                seg_out = outputs.get('refined', outputs.get('seg'))
            else:
                seg_out = outputs
            
            # 立即轉移到 CPU 並刪除 GPU 上的變量
            pred_cpu = torch.sigmoid(seg_out).cpu()
            masks_cpu = masks.cpu()
            task_ids_cpu = task_ids.cpu()
            
            all_preds.append(pred_cpu)
            all_targets.append(masks_cpu)
            all_task_ids.append(task_ids_cpu)
            
            # 收集每個任務的第一個樣本用於視覺化
            batch_size = images.shape[0]
            for i in range(batch_size):
                task_id = task_ids_cpu[i].item()
                if task_id < self.num_tasks and task_samples.get(task_id) is None:
                    task_samples[task_id] = {
                        'image': images[i:i+1].cpu(),
                        'mask': masks_cpu[i:i+1],
                        'pred': seg_out[i:i+1].cpu(),
                        'task_id': task_id
                    }
            
            # 明確刪除 GPU 變量
            del outputs, loss, seg_out, images, masks, task_ids
            
            # 定期清理 GPU 記憶體
            cache_clear_freq = self.config.get('gpu_cache_clear_freq', 30)
            if (batch_idx + 1) % cache_clear_freq == 0:
                clear_gpu_memory()
        
        # 驗證結束徹底清理
        clear_gpu_memory()
        
        # 計算指標
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_task_ids = torch.cat(all_task_ids, dim=0)
        
        metrics = self._compute_metrics(all_preds, all_targets, all_task_ids)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        
        # 列印驗證結果
        print(f"\nValidation Loss: {avg_loss:.4f}")
        print(f"Overall IoU: {metrics['overall']['iou']:.4f}, Dice: {metrics['overall']['dice']:.4f}")
        for task_id in range(self.num_tasks):
            task_name = self.task_names.get(task_id, f'Task_{task_id}')
            if metrics[task_id]['count'] > 0:
                print(f"  {task_name}: IoU={metrics[task_id]['iou']:.4f}, Dice={metrics[task_id]['dice']:.4f}")
        
        # 保存視覺化樣本（每 10 個 epoch 或最後一個 epoch）
        if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
            self._save_validation_samples(task_samples, epoch)
        
        return avg_loss, metrics, avg_components, task_samples
    
    def _compute_metrics(self, preds, targets, task_ids, threshold=0.5):
        """計算分割指標"""
        preds_binary = (preds > threshold).float()
        
        metrics = {
            'overall': {'iou': 0, 'dice': 0, 'count': 0}
        }
        for i in range(self.num_tasks):
            metrics[i] = {'iou': 0, 'dice': 0, 'count': 0}
        
        for task_id in range(self.num_tasks):
            mask = (task_ids == task_id)
            if mask.sum() == 0:
                continue
            
            task_preds = preds_binary[mask]
            task_targets = targets[mask]
            
            # IoU
            intersection = (task_preds * task_targets).sum()
            union = task_preds.sum() + task_targets.sum() - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            
            # Dice
            dice = (2 * intersection + 1e-7) / (task_preds.sum() + task_targets.sum() + 1e-7)
            
            metrics[task_id]['iou'] = iou.item()
            metrics[task_id]['dice'] = dice.item()
            metrics[task_id]['count'] = mask.sum().item()
        
        # Overall
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum() - intersection
        metrics['overall']['iou'] = ((intersection + 1e-7) / (union + 1e-7)).item()
        metrics['overall']['dice'] = ((2 * intersection + 1e-7) / (preds_binary.sum() + targets.sum() + 1e-7)).item()
        metrics['overall']['count'] = len(preds)
        
        return metrics
    
    def _save_validation_samples(self, task_samples, epoch):
        """保存驗證樣本的視覺化"""
        try:
            nrows = self.num_tasks
            ncols = 3
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
            
            # 如果只有一個任務，確保 axes 是 2D
            if nrows == 1:
                axes = axes.reshape(1, -1)
            
            for task_id in range(self.num_tasks):
                task_name = self.task_names.get(task_id, f'Task_{task_id}')
                sample = task_samples.get(task_id)
                
                if sample is None:
                    # 沒有這個任務的樣本
                    for col in range(3):
                        axes[task_id, col].text(
                            0.5, 0.5, 
                            f'⚠ No {task_name} samples\nin validation set', 
                            ha='center', va='center', fontsize=14, color='red',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        )
                        axes[task_id, col].set_xlim(0, 1)
                        axes[task_id, col].set_ylim(0, 1)
                        axes[task_id, col].axis('off')
                    
                    axes[task_id, 0].set_title(f'{task_name} - Image', fontsize=12, fontweight='bold')
                    axes[task_id, 1].set_title(f'{task_name} - Ground Truth', fontsize=12, fontweight='bold')
                    axes[task_id, 2].set_title(f'{task_name} - Prediction', fontsize=12, fontweight='bold')
                    continue
                
                # 原始影像
                img = sample['image'][0].permute(1, 2, 0).numpy()
                # 確保影像值在 [0, 1] 範圍內
                img = np.clip(img, 0, 1)
                axes[task_id, 0].imshow(img)
                axes[task_id, 0].set_title(f'{task_name} - Image', fontsize=12, fontweight='bold')
                axes[task_id, 0].axis('off')
                
                # 真實標籤
                mask = sample['mask'][0, 0].numpy()
                axes[task_id, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
                axes[task_id, 1].set_title(f'{task_name} - Ground Truth', fontsize=12, fontweight='bold')
                axes[task_id, 1].axis('off')
                
                # 預測結果（顯示概率熱圖）
                pred = torch.sigmoid(sample['pred'][0, 0]).numpy()
                
                # 使用 jet colormap 讓預測結果更明顯
                im = axes[task_id, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
                axes[task_id, 2].set_title(
                    f'{task_name} - Prediction\n(min: {pred.min():.3f}, max: {pred.max():.3f}, mean: {pred.mean():.3f})', 
                    fontsize=10, fontweight='bold'
                )
                axes[task_id, 2].axis('off')
                
                # 添加 colorbar
                cbar = plt.colorbar(im, ax=axes[task_id, 2], fraction=0.046, pad=0.04)
                cbar.set_label('Probability', rotation=270, labelpad=15)
            
            plt.suptitle(f'Validation Results - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # 確保輸出目錄存在
            self.pred_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = self.pred_dir / f'val_epoch{epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved validation visualization to {save_path}")
            
        except Exception as e:
            print(f"⚠ Error saving validation samples: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
    
    def plot_history(self):
        """繪製訓練歷史"""
        try:
            # 檢查是否有數據可以繪製
            if len(self.history['train_loss']) == 0:
                print("⚠ No training history to plot yet")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 損失曲線
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # IoU 曲線
            if len(self.history['val_iou']) > 0:
                axes[0, 1].plot(self.history['val_iou'], label='Overall IoU', linewidth=2, color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('IoU')
            axes[0, 1].set_title('Validation IoU', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Dice 曲線
            if len(self.history['val_dice']) > 0:
                axes[0, 2].plot(self.history['val_dice'], label='Overall Dice', linewidth=2, color='orange')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Dice')
            axes[0, 2].set_title('Validation Dice', fontsize=14, fontweight='bold')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 各任務 IoU
            for task_id in range(self.num_tasks):
                task_name = self.task_names.get(task_id, f'Task_{task_id}')
                if task_id in self.history['task_metrics'] and len(self.history['task_metrics'][task_id]) > 0:
                    task_ious = [m.get('iou', 0) for m in self.history['task_metrics'][task_id]]
                    axes[1, 0].plot(task_ious, label=f'{task_name} IoU', linewidth=1.5)
            
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU')
            axes[1, 0].set_title('Task-specific IoU', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 損失分量
            if 'loss_components' in self.history and len(self.history['loss_components']['train']) > 0:
                train_components = self.history['loss_components']['train']
                if len(train_components) > 0 and isinstance(train_components[0], dict):
                    component_keys = list(train_components[0].keys())
                    
                    for key in component_keys:
                        if key != 'total_loss':
                            train_values = [comp.get(key, 0) for comp in train_components]
                            if len(train_values) > 0 and any(v != 0 for v in train_values):
                                axes[1, 1].plot(train_values, label=f'{key}', alpha=0.7, linewidth=1)
                    
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Loss')
                    axes[1, 1].set_title('Training Loss Components', fontsize=14, fontweight='bold')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            # 訓練資訊摘要
            axes[1, 2].axis('off')
            info_text = f"Model: Boundary-Aware MultiTaskTransUNet\n"
            info_text += f"Tasks: {self.num_tasks}\n"
            info_text += f"AMP: {self.use_amp}\n"
            info_text += f"Epochs completed: {len(self.history['train_loss'])}\n"
            if len(self.history['val_iou']) > 0:
                info_text += f"Best IoU: {max(self.history['val_iou']):.4f}\n"
                info_text += f"Final Train Loss: {self.history['train_loss'][-1]:.4f}\n"
                info_text += f"Final Val Loss: {self.history['val_loss'][-1]:.4f}"
            
            axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, 
                           verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle('Training History - Boundary-Aware MultiTaskTransUNet', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # 確保輸出目錄存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = self.output_dir / 'training_history.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved training history plot to {save_path}")
            
        except Exception as e:
            print(f"⚠ Error plotting training history: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # 確保關閉所有圖形
    
    def _save_history_json(self):
        """保存訓練歷史為 JSON"""
        history_json_path = self.output_dir / 'training_history.json'
        try:
            history_to_save = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_iou': [float(x) for x in self.history['val_iou']],
                'val_dice': [float(x) for x in self.history['val_dice']],
                'task_metrics': {},
                'num_tasks': self.num_tasks,
                'task_names': self.task_names,
                'config': {k: v for k, v in self.config.items() if not callable(v)}
            }
            
            # 保存各任務的指標
            for task_id, metrics_list in self.history['task_metrics'].items():
                history_to_save['task_metrics'][str(task_id)] = [
                    {k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()}
                    for m in metrics_list
                ]
            
            with open(history_json_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Training history saved to: {history_json_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to save training history as JSON: {e}")
    
    def train(self):
        """主訓練循環"""
        print(f"\n{'='*60}")
        print(f"Starting Optimized Multi-Task Training")
        print(f"  AMP: {self.use_amp}")
        print(f"  Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"  GPU Cache Clear Freq: {self.config.get('gpu_cache_clear_freq', 30)} batches")
        print(f"{'='*60}\n")
        
        best_val_iou = 0.0
        
        for epoch in range(self.config['epochs']):
            # 每個 epoch 開始前徹底清理記憶體
            clear_gpu_memory()
            reset_peak_memory()
            
            # 顯示 GPU 記憶體狀態（每 10 個 epoch）
            if epoch % 10 == 0:
                mem_info = get_gpu_memory_info()
                if mem_info:
                    print(f"\n[GPU Memory] Allocated: {mem_info['allocated']:.2f}GB, "
                          f"Reserved: {mem_info['reserved']:.2f}GB")
            
            # 訓練
            train_loss, train_components = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['loss_components']['train'].append(train_components)
            
            # 訓練後清理
            clear_gpu_memory()
            
            # 驗證（返回 4 個值：loss, metrics, components, task_samples）
            val_loss, val_metrics, val_components, _ = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_metrics['overall']['iou'])
            self.history['val_dice'].append(val_metrics['overall']['dice'])
            self.history['loss_components']['val'].append(val_components)
            
            # 各任務指標
            for task_id in range(self.num_tasks):
                self.history['task_metrics'][task_id].append(val_metrics.get(task_id, {'iou': 0, 'dice': 0}))
            
            # 保存最佳模型
            if val_metrics['overall']['iou'] > best_val_iou:
                best_val_iou = val_metrics['overall']['iou']
                torch.save(self.model.state_dict(), self.model_dir / 'best_model.pth')
                print(f"✓ Best model saved (IoU: {best_val_iou:.4f})")
            
            # 定期保存檢查點
            if (epoch + 1) % 20 == 0:
                # 保存前清理記憶體
                clear_gpu_memory()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': self.history,
                    'num_tasks': self.num_tasks,
                    'task_names': self.task_names,
                    'config': self.config
                }
                torch.save(checkpoint, self.model_dir / f'checkpoint_epoch{epoch+1:03d}.pth')
                print(f"✓ Checkpoint saved at epoch {epoch+1}")
            
            # 定期繪製訓練歷史（第一個 epoch 也畫，方便確認程式運行）
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.plot_history()
            
            # 每個 epoch 結束後徹底清理記憶體
            clear_gpu_memory()
        
        # 保存最終模型
        torch.save(self.model.state_dict(), self.model_dir / 'final_model.pth')
        
        # 最終繪製訓練歷史
        self.plot_history()
        
        # 保存訓練歷史為 JSON
        print(f"\n{'='*60}")
        print("Saving training history...")
        self._save_history_json()
        print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Best Validation IoU: {best_val_iou:.4f}")
        print(f"{'='*60}\n")


# ============================================================================
# 主程式
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Task TransUNet (Optimized)')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    trainer = OptimizedMultiTaskTrainer(config_path=args.config)
    trainer.train()


if __name__ == '__main__':
    main()
