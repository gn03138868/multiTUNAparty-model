"""
多任務損失函數
針對不同任務使用不同的權重策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class MultiTaskLoss(nn.Module):
    """
    多任務損失函數
    
    針對不同任務的特性設計不同的損失權重：
    - 植物細胞 (task 0): 平衡的權重
    - 血球 (task 1): 略微增加邊界權重
    - 根系 (task 2): 大幅增加邊界權重和前景權重
    """
    def __init__(
        self,
        base_weights=None,
        boundary_weights=None,
        foreground_weights=None,
        smooth=1e-5
    ):
        """
        Args:
            base_weights: 基礎損失權重 {task_id: weight}
            boundary_weights: 邊界損失權重 {task_id: weight}
            foreground_weights: 前景損失權重 {task_id: weight}
        """
        super().__init__()
        
        # 預設權重配置
        self.base_weights = base_weights or {
            0: 1.0,  # 細胞
            1: 1.0,  # 血球
            2: 1.0   # 根系
        }
        
        self.boundary_weights = boundary_weights or {
            0: 2.0,   # 細胞：中等邊界權重
            1: 3.0,   # 血球：較高邊界權重（圓形邊界很重要）
            2: 5.0    # 根系：最高邊界權重（線性結構全是邊界）
        }
        
        self.foreground_weights = foreground_weights or {
            0: 1.0,   # 細胞：平衡
            1: 1.5,   # 血球：略微增加前景權重
            2: 3.0    # 根系：大幅增加前景權重（前景比例極小）
        }
        
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, task_ids):
        """
        Args:
            predictions: 模型輸出 [B, 1, H, W]
            targets: 真實標籤 [B, 1, H, W]
            task_ids: 任務ID [B]
        """
        batch_size = predictions.shape[0]
        total_loss = 0.0
        
        # 分別計算每個樣本的損失
        for i in range(batch_size):
            pred = predictions[i:i+1]
            target = targets[i:i+1]
            task_id = task_ids[i].item()
            
            # 基礎 Dice + BCE 損失
            base_loss = self._compute_base_loss(pred, target)
            
            # 邊界損失
            boundary_loss = self._compute_boundary_loss(pred, target)
            
            # 前景加權損失（針對前景比例極小的情況，如根系）
            foreground_loss = self._compute_foreground_loss(pred, target)
            
            # 組合損失，使用任務特定的權重
            sample_loss = (
                self.base_weights[task_id] * base_loss +
                self.boundary_weights[task_id] * boundary_loss +
                self.foreground_weights[task_id] * foreground_loss
            )
            
            total_loss += sample_loss
        
        return total_loss / batch_size
    
    def _compute_base_loss(self, pred, target):
        """基礎 Dice + BCE 損失"""
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # BCE Loss
        bce_loss = self.bce(pred, target).mean()
        
        return dice_loss + bce_loss
    
    def _compute_boundary_loss(self, pred, target):
        """
        邊界損失：使用距離變換強調邊界區域
        """
        pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
        target_binary = (target > 0.5).float()
        
        # 計算距離變換（在 CPU 上）
        target_np = target_binary[0, 0].detach().cpu().numpy().astype(np.uint8)
        
        # 計算背景的距離變換
        dist_map = cv2.distanceTransform(1 - target_np, cv2.DIST_L2, 5)
        dist_tensor = torch.tensor(dist_map, dtype=pred.dtype, device=pred.device)
        
        # 限制最大距離，避免過大的權重
        dist_tensor = torch.clamp(dist_tensor, max=10.0)
        
        # 使用距離作為權重計算損失
        loss = torch.mean(dist_tensor * torch.abs(pred_sigmoid[0, 0] - target_binary[0, 0]))
        
        return loss
    
    def _compute_foreground_loss(self, pred, target):
        """
        前景加權損失：針對前景比例極小的情況（如根系）
        給予前景像素更高的權重
        """
        pred_sigmoid = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
        
        # 計算前景和背景的權重
        fg_mask = (target > 0.5).float()
        bg_mask = (target <= 0.5).float()
        
        # 前景像素數量
        fg_count = fg_mask.sum()
        bg_count = bg_mask.sum()
        
        if fg_count == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 動態計算權重：前景越少，權重越高
        fg_weight = bg_count / (fg_count + 1e-7)
        fg_weight = torch.clamp(fg_weight, min=1.0, max=10.0)
        
        # 計算加權的 BCE 損失
        fg_loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        fg_loss = fg_loss * (fg_mask * fg_weight + bg_mask)
        
        return fg_loss.mean()


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
        
        # 按任務分組索引
        self.task_indices = {0: [], 1: [], 2: []}
        for idx, (_, _, task_id) in enumerate(dataset.patches):
            self.task_indices[task_id].append(idx)
        
        # 計算每個任務的樣本數
        self.task_counts = {
            task_id: len(indices) 
            for task_id, indices in self.task_indices.items()
        }
        
        print(f"Task counts: {self.task_counts}")
    
    def __iter__(self):
        """產生平衡的 batch 索引"""
        # 隨機打亂每個任務的索引
        for task_id in self.task_indices:
            np.random.shuffle(self.task_indices[task_id])
        
        # 當前各任務的指針
        task_pointers = {0: 0, 1: 0, 2: 0}
        
        batch_indices = []
        
        while True:
            # 檢查是否所有任務都已用完
            if all(
                task_pointers[task_id] >= len(self.task_indices[task_id])
                for task_id in [0, 1, 2]
            ):
                break
            
            # 嘗試從每個任務中取樣
            for task_id in [0, 1, 2]:
                if task_pointers[task_id] < len(self.task_indices[task_id]):
                    idx = self.task_indices[task_id][task_pointers[task_id]]
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
# 輔助函數
# ============================================================================

def compute_metrics(predictions, targets, task_ids, threshold=0.5):
    """
    計算分割指標
    
    Returns:
        dict: 包含整體和各任務的 IoU, Dice, Precision, Recall
    """
    pred_binary = (torch.sigmoid(predictions) > threshold).float()
    target_binary = (targets > 0.5).float()
    
    metrics = {
        'overall': {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0},
        0: {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0},
        1: {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0},
        2: {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'count': 0}
    }
    
    batch_size = predictions.shape[0]
    
    for i in range(batch_size):
        pred = pred_binary[i].view(-1)
        target = target_binary[i].view(-1)
        task_id = task_ids[i].item()
        
        # 計算交集和並集
        intersection = (pred * target).sum().item()
        union = (pred + target).clamp(0, 1).sum().item()
        
        # 計算 TP, FP, FN
        tp = intersection
        fp = (pred * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        
        # IoU
        iou = intersection / (union + 1e-7)
        
        # Dice
        dice = (2 * intersection) / (pred.sum().item() + target.sum().item() + 1e-7)
        
        # Precision and Recall
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        # 更新整體指標
        metrics['overall']['iou'] += iou
        metrics['overall']['dice'] += dice
        metrics['overall']['precision'] += precision
        metrics['overall']['recall'] += recall
        metrics['overall']['count'] += 1
        
        # 更新任務特定指標
        metrics[task_id]['iou'] += iou
        metrics[task_id]['dice'] += dice
        metrics[task_id]['precision'] += precision
        metrics[task_id]['recall'] += recall
        metrics[task_id]['count'] += 1
    
    # 計算平均值
    for key in metrics:
        if metrics[key]['count'] > 0:
            for metric in ['iou', 'dice', 'precision', 'recall']:
                metrics[key][metric] /= metrics[key]['count']
    
    return metrics


def print_metrics(metrics, epoch=None):
    """列印指標"""
    if epoch is not None:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Metrics")
        print(f"{'='*60}")
    
    task_names = {0: 'Cell', 1: 'Blood', 2: 'Root'}
    
    # 整體指標
    print(f"\nOverall:")
    print(f"  IoU:       {metrics['overall']['iou']:.4f}")
    print(f"  Dice:      {metrics['overall']['dice']:.4f}")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall:    {metrics['overall']['recall']:.4f}")
    
    # 各任務指標
    for task_id in [0, 1, 2]:
        if metrics[task_id]['count'] > 0:
            print(f"\n{task_names[task_id]}:")
            print(f"  IoU:       {metrics[task_id]['iou']:.4f}")
            print(f"  Dice:      {metrics[task_id]['dice']:.4f}")
            print(f"  Precision: {metrics[task_id]['precision']:.4f}")
            print(f"  Recall:    {metrics[task_id]['recall']:.4f}")


# ============================================================================
# 測試程式碼
# ============================================================================

if __name__ == '__main__':
    print("Testing MultiTaskLoss...")
    
    # 創建測試數據
    batch_size = 4
    predictions = torch.randn(batch_size, 1, 256, 256)
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    task_ids = torch.tensor([0, 1, 2, 0])  # 混合任務
    
    # 測試損失函數
    loss_fn = MultiTaskLoss()
    loss = loss_fn(predictions, targets, task_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # 測試指標計算
    metrics = compute_metrics(predictions, targets, task_ids)
    print_metrics(metrics)
    
    print("\n✓ All tests passed!")
