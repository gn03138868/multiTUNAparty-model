"""
多任務 TransUNet 訓練腳本
支援 2、3 或更多任務的靈活訓練
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# 假設這些模組都在同一目錄
try:
    from model_multitask import MultiTaskTransUNet
    from dataset_multitask import MultiTaskSegmentationDataset
    from losses_multitask import MultiTaskLoss, compute_metrics, print_metrics, TaskBalancedSampler
except ImportError:
    print("Please ensure model_multitask.py, dataset_multitask.py, and losses_multitask.py are in the same directory")
    raise


class MultiTaskTrainer:
    """
    多任務訓練器
    支援靈活的任務數量（2、3 或更多）
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
        
        # 從配置獲取任務數量和名稱
        self.num_tasks = self.config.get('num_tasks', 3)
        self.task_names = self.config.get('task_names', None)
        
        # 如果沒有指定任務名稱，使用預設
        if self.task_names is None:
            if self.num_tasks == 3:
                self.task_names = {0: 'Cell', 1: 'Blood', 2: 'Root'}
            elif self.num_tasks == 2:
                self.task_names = {0: 'Task_0', 1: 'Task_1'}
            else:
                self.task_names = {i: f'Task_{i}' for i in range(self.num_tasks)}
        
        # 如果 task_names 是列表，轉換為字典
        if isinstance(self.task_names, list):
            self.task_names = {i: name for i, name in enumerate(self.task_names)}
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Number of tasks: {self.num_tasks}")
        print(f"  Task names: {self.task_names}")
        print(f"{'='*60}")
        
        # 創建輸出目錄
        self.output_dir = Path('outputs')
        self.model_dir = self.output_dir / 'models'
        self.pred_dir = self.output_dir / 'predictions'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        
        # 先初始化訓練歷史（在 _init_data 之前，因為 _init_data 可能會更新 num_tasks）
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'task_metrics': {i: [] for i in range(self.num_tasks)}
        }
        
        # 初始化模型、數據和優化器
        self._init_model()
        self._init_data()
        self._init_optimizer()
    
    def _load_config(self, config_path):
        """載入配置檔案"""
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Using default configuration...")
            return self._default_config()
        
        # 明確指定 UTF-8 編碼以支援中文註釋（Windows 兼容性）
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _default_config(self):
        """預設配置"""
        return {
            'batch_size': 4,
            'epochs': 200,
            'lr': 1e-5,
            'patch_size': 400,
            'num_decoder_conv_layers': 80,
            'data_path': 'data/',
            'val_split': 0.2,
            'task_structure': 'subfolder',  # or 'filename'
            # 任務數量（可設為 2, 3 或更多）
            'num_tasks': 3,
            'task_names': {0: 'Cell', 1: 'Blood', 2: 'Root'},
            # 任務特定的損失權重
            'base_weights': {0: 1.0, 1: 1.0, 2: 1.0},
            'boundary_weights': {0: 2.0, 1: 3.0, 2: 20.0},
            'foreground_weights': {0: 1.0, 1: 1.5, 2: 15.0}
        }
    
    def _extend_weights(self, weights, num_tasks):
        """
        擴展權重字典以覆蓋所有任務
        如果配置的權重數量少於任務數量，使用平均值填充
        """
        if weights is None:
            return {i: 1.0 for i in range(num_tasks)}
        
        # 將字符串鍵轉換為整數
        weights = {int(k): v for k, v in weights.items()}
        
        # 計算預設權重（使用現有權重的平均值）
        default_weight = 1.0
        if len(weights) > 0:
            default_weight = sum(weights.values()) / len(weights)
        
        # 擴展到所有任務
        return {i: weights.get(i, default_weight) for i in range(num_tasks)}
    
    def _init_model(self):
        """初始化模型"""
        print("\nInitializing model...")
        
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
            num_tasks=self.num_tasks,  # 使用動態任務數量
            task_embed_dim=256
        ).to(self.device)
        
        # 計算參數量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 載入預訓練權重（智能部分載入）
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """
        智能載入預訓練權重
        支持三種來源：
        1. 完整的多任務模型權重（直接載入）
        2. 原始 TransUNet 權重（部分載入，只載入 encoder）
        3. ImageNet 預訓練的 ViT 權重（只載入 patch_embed 和 blocks）
        """
        # 檢查配置中的預訓練路徑
        pretrained_path = None
        
        # 優先級 1: 配置文件中指定的路徑
        if 'pretrained_model_path' in self.config and self.config['pretrained_model_path']:
            pretrained_path = Path(self.config['pretrained_model_path'])
        
        # 優先級 2: 預設路徑
        if pretrained_path is None or not pretrained_path.exists():
            pretrained_path = Path(self.config['data_path']) / 'pretrained model' / 'pretrained_model.pth'
        
        if not pretrained_path.exists():
            print("No pretrained weights found. Training from scratch.")
            return
        
        print(f"\n{'='*60}")
        print(f"Loading pretrained weights from: {pretrained_path}")
        print(f"{'='*60}")
        
        try:
            # 載入預訓練權重
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)
            
            # 如果是完整的 checkpoint（包含 optimizer 等）
            if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']
                print("Detected checkpoint format (extracting model_state_dict)")
            
            model_dict = self.model.state_dict()
            
            # 統計載入情況
            total_params = len(model_dict)
            matched_params = 0
            shape_mismatch = 0
            
            # 嘗試完整載入（如果是相同架構）
            try:
                self.model.load_state_dict(pretrained_dict, strict=True)
                print("✓ Full model loaded successfully (same architecture)")
                return
            except:
                print("⚠ Full loading failed. Attempting partial loading...")
            
            # 部分載入：只載入形狀匹配的層
            matched_dict = {}
            
            print("\nMatching layers:")
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        matched_dict[k] = v
                        matched_params += 1
                        # 只顯示重要層
                        if 'weight' in k and len(v.shape) >= 2:
                            print(f"  ✓ {k:50s} {str(v.shape):20s}")
                    else:
                        shape_mismatch += 1
                        if 'weight' in k:
                            print(f"  ✗ {k:50s} shape mismatch: {v.shape} vs {model_dict[k].shape}")
            
            # 更新模型權重
            model_dict.update(matched_dict)
            self.model.load_state_dict(model_dict)
            
            # 統計報告
            print(f"\n{'='*60}")
            print(f"Pretrained Loading Summary:")
            print(f"{'='*60}")
            print(f"  Total parameters in current model: {total_params}")
            print(f"  Matched and loaded:                 {matched_params} ({matched_params/total_params*100:.1f}%)")
            print(f"  Shape mismatch (skipped):           {shape_mismatch}")
            print(f"  Not in pretrained (initialized):    {total_params - matched_params}")
            print(f"{'='*60}")
            
            if matched_params > 0:
                print(f"✓ Partial loading successful!")
                print(f"  Loaded: encoder layers (ViT blocks)")
                print(f"  Initialized from scratch: task_embedding, aspp, decoder")
            else:
                print("⚠ No matching parameters found. Training from scratch.")
            
        except Exception as e:
            print(f"✗ Error loading pretrained weights: {e}")
            print("Training from scratch.")
            import traceback
            traceback.print_exc()
    
    def _init_data(self):
        """初始化數據載入器"""
        print("\nInitializing datasets...")
        
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
        
        # 自動偵測實際任務數量（跨整個資料集抽樣）
        detected_tasks = set()
        dataset_len = len(self.train_dataset)
        
        # 從整個資料集均勻抽樣，而不是只看前 100 個
        sample_indices = np.linspace(0, dataset_len - 1, min(500, dataset_len), dtype=int)
        
        for i in sample_indices:
            try:
                sample = self.train_dataset[i]
                if len(sample) >= 3:
                    task_id = sample[2]
                    if isinstance(task_id, torch.Tensor):
                        task_id = task_id.item()
                    elif isinstance(task_id, np.ndarray):
                        task_id = int(task_id)
                    detected_tasks.add(int(task_id))
            except Exception as e:
                pass
        
        if len(detected_tasks) > 0:
            actual_num_tasks = max(detected_tasks) + 1
            print(f"\n  Detected tasks in dataset: {sorted(detected_tasks)}")
            
            # 只有當偵測到的任務數量與配置不同時才更新
            if actual_num_tasks != self.num_tasks:
                print(f"⚠ Detected {actual_num_tasks} tasks in dataset, config specified {self.num_tasks}")
                # 使用較大的值，確保不會漏掉任務
                self.num_tasks = max(actual_num_tasks, self.num_tasks)
                print(f"  Using: {self.num_tasks} tasks")
                # 更新 history
                self.history['task_metrics'] = {i: [] for i in range(self.num_tasks)}
                # 更新 task_names
                for i in range(self.num_tasks):
                    if i not in self.task_names:
                        self.task_names[i] = f'Task_{i}'
        
        # 使用任務平衡採樣器
        print("\nUsing task-balanced sampling...")
        train_sampler = TaskBalancedSampler(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            drop_last=True
        )
        
        # 數據載入器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"\nTrain batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def _init_optimizer(self):
        """初始化優化器和學習率調度器"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=0.01
        )
        
        # 使用 Cosine Annealing 學習率調度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # 第一次重啟的週期
            T_mult=2,  # 每次重啟後週期的倍數
            eta_min=1e-7
        )
        
        # 擴展權重以覆蓋所有任務
        base_weights = self._extend_weights(self.config.get('base_weights'), self.num_tasks)
        boundary_weights = self._extend_weights(self.config.get('boundary_weights'), self.num_tasks)
        foreground_weights = self._extend_weights(self.config.get('foreground_weights'), self.num_tasks)
        
        # 初始化損失函數
        self.criterion = MultiTaskLoss(
            base_weights=base_weights,
            boundary_weights=boundary_weights,
            foreground_weights=foreground_weights
        )
    
    def _unpack_batch(self, batch):
        """
        靈活解包 batch 數據
        支援 Dataset 返回 3 個值 (images, masks, task_ids) 
        或 4 個值 (images, masks, task_ids, image_names) 的情況
        """
        if len(batch) == 3:
            images, masks, task_ids = batch
            return images, masks, task_ids
        elif len(batch) == 4:
            images, masks, task_ids, _ = batch  # 忽略 image_names
            return images, masks, task_ids
        elif len(batch) >= 5:
            # 處理更多返回值的情況
            images, masks, task_ids = batch[0], batch[1], batch[2]
            return images, masks, task_ids
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
    
    def train_epoch(self, epoch):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            # 靈活解包 batch
            images, masks, task_ids = self._unpack_batch(batch)
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            task_ids = task_ids.to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            
            # 批量處理：為每個樣本使用其對應的 task_id
            batch_size = images.shape[0]
            outputs = []
            
            # 如果 batch 中所有樣本的 task_id 相同，可以批量處理
            unique_tasks = torch.unique(task_ids)
            
            if len(unique_tasks) == 1:
                # 所有樣本是同一任務，批量處理
                outputs = self.model(images, task_id=unique_tasks[0].item())
            else:
                # 不同任務，分別處理（保持向後兼容）
                for i in range(batch_size):
                    out = self.model(images[i:i+1], task_id=task_ids[i].item())
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0)
            
            # 計算損失
            loss = self.criterion(outputs, masks, task_ids)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 定期清理 GPU 記憶體（每 100 個 batch）
            if batch_idx % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        
        # 更新學習率
        self.scheduler.step()
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """驗證"""
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        # 收集每個任務的樣本用於視覺化（動態任務數量）
        task_samples = {i: None for i in range(self.num_tasks)}
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # 靈活解包 batch
            images, masks, task_ids = self._unpack_batch(batch)
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            task_ids = task_ids.to(self.device)
            
            # 前向傳播
            batch_size = images.shape[0]
            unique_tasks = torch.unique(task_ids)
            
            if len(unique_tasks) == 1:
                # 所有樣本是同一任務，批量處理
                outputs = self.model(images, task_id=unique_tasks[0].item())
            else:
                # 不同任務，分別處理
                outputs = []
                for i in range(batch_size):
                    out = self.model(images[i:i+1], task_id=task_ids[i].item())
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0)
            
            # 計算損失
            loss = self.criterion(outputs, masks, task_ids)
            total_loss += loss.item()
            
            # 計算指標
            metrics = compute_metrics(outputs, masks, task_ids)
            all_metrics.append(metrics)
            
            # 收集每個任務的第一個樣本用於視覺化
            for i in range(batch_size):
                task_id = task_ids[i].item()
                if task_id < self.num_tasks and task_samples.get(task_id) is None:
                    task_samples[task_id] = {
                        'image': images[i:i+1].cpu(),
                        'mask': masks[i:i+1].cpu(),
                        'pred': outputs[i:i+1].cpu(),
                        'task_id': task_id
                    }
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 聚合所有指標
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # 列印指標
        print(f"\nValidation Loss: {avg_loss:.4f}")
        print_metrics(aggregated_metrics)
        
        # 保存驗證樣本（從收集的樣本中）
        if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
            self._save_collected_samples(task_samples, epoch)
        
        return avg_loss, aggregated_metrics
    
    def _aggregate_metrics(self, all_metrics):
        """聚合多個 batch 的指標（動態任務數量）"""
        aggregated = {
            'overall': {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0}
        }
        # 動態添加各任務
        for i in range(self.num_tasks):
            aggregated[i] = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0}
        
        for metrics in all_metrics:
            for key in aggregated:
                if key in metrics and metrics[key]['count'] > 0:
                    for metric in ['iou', 'dice', 'precision', 'recall']:
                        aggregated[key][metric] += metrics[key][metric] * metrics[key]['count']
                    aggregated[key]['count'] += metrics[key]['count']
        
        # 計算平均值
        for key in aggregated:
            if aggregated[key]['count'] > 0:
                for metric in ['iou', 'dice', 'precision', 'recall']:
                    aggregated[key][metric] /= aggregated[key]['count']
        
        return aggregated
    
    def _save_collected_samples(self, task_samples, epoch):
        """從收集的樣本中保存視覺化（動態任務數量）"""
        # 根據任務數量動態設置子圖
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
                # 沒有這個任務的樣本（資料集中確實沒有）
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
        plt.savefig(self.pred_dir / f'val_epoch{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved validation visualization to {self.pred_dir / f'val_epoch{epoch:03d}.png'}")
    
    def plot_history(self):
        """繪製訓練歷史（動態任務數量）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失曲線
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU 曲線
        axes[0, 1].plot(self.history['val_iou'], label='Overall IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice 曲線
        axes[1, 0].plot(self.history['val_dice'], label='Overall Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].set_title('Validation Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 各任務 IoU（動態任務數量）
        for task_id in range(self.num_tasks):
            task_name = self.task_names.get(task_id, f'Task_{task_id}')
            if len(self.history['task_metrics'][task_id]) > 0:
                task_ious = [m['iou'] for m in self.history['task_metrics'][task_id]]
                axes[1, 1].plot(task_ious, label=f'{task_name} IoU')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_title('Task-specific IoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """主訓練循環"""
        print(f"\n{'='*60}")
        print(f"Starting Multi-Task Training ({self.num_tasks} tasks)")
        print(f"{'='*60}\n")
        
        best_val_iou = 0.0
        
        for epoch in range(self.config['epochs']):
            # 訓練
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # 驗證
            val_loss, val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_metrics['overall']['iou'])
            self.history['val_dice'].append(val_metrics['overall']['dice'])
            
            # 記錄各任務指標（動態任務數量）
            for task_id in range(self.num_tasks):
                if task_id in val_metrics:
                    self.history['task_metrics'][task_id].append(val_metrics[task_id])
                else:
                    # 如果沒有這個任務的樣本，添加佔位符
                    self.history['task_metrics'][task_id].append({
                        'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0
                    })
            
            # 保存最佳模型
            if val_metrics['overall']['iou'] > best_val_iou:
                best_val_iou = val_metrics['overall']['iou']
                torch.save(
                    self.model.state_dict(),
                    self.model_dir / 'best_model.pth'
                )
                print(f"✓ Best model saved (IoU: {best_val_iou:.4f})")
            
            # 定期保存檢查點
            if (epoch + 1) % 20 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'history': self.history,
                        'num_tasks': self.num_tasks,
                        'task_names': self.task_names
                    },
                    self.model_dir / f'checkpoint_epoch{epoch+1:03d}.pth'
                )
            
            # 繪製訓練歷史
            if (epoch + 1) % 10 == 0:
                self.plot_history()
            
            # 每個 epoch 結束後清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 保存最終模型
        torch.save(
            self.model.state_dict(),
            self.model_dir / 'final_model.pth'
        )
        
        # 最終繪圖
        self.plot_history()
        
        # 保存訓練歷史為 JSON
        print(f"\n{'='*60}")
        print("Saving training history...")
        self._save_history_json()
        print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Best Validation IoU: {best_val_iou:.4f}")
        print(f"{'='*60}\n")
    
    def _save_history_json(self):
        """保存訓練歷史為 JSON"""
        history_json_path = self.output_dir / 'training_history.json'
        try:
            # 將 numpy 數組轉換為列表以便 JSON 序列化
            history_to_save = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_iou': [float(x) for x in self.history['val_iou']],
                'val_dice': [float(x) for x in self.history['val_dice']],
                'task_metrics': {},
                'num_tasks': self.num_tasks,
                'task_names': self.task_names
            }
            
            # 保存各任務的指標
            for task_id, metrics_list in self.history['task_metrics'].items():
                history_to_save['task_metrics'][str(task_id)] = [
                    {k: float(v) for k, v in m.items()}
                    for m in metrics_list
                ]
            
            with open(history_json_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Training history saved to: {history_json_path}")
            print(f"  - Train Loss: {len(history_to_save['train_loss'])} epochs")
            print(f"  - Val IoU: {len(history_to_save['val_iou'])} epochs")
            print(f"  - Task Metrics: {self.num_tasks} tasks")
        except Exception as e:
            print(f"⚠ Warning: Failed to save training history as JSON: {e}")


# ============================================================================
# 主程式
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Task TransUNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # 創建訓練器
    trainer = MultiTaskTrainer(config_path=args.config)
    
    # 開始訓練
    trainer.train()


if __name__ == '__main__':
    main()
