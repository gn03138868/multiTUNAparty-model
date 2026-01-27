# ==== src/losses.py ====
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def boundary_loss(pred, target):
    """
    利用距离变换计算边界损失：
      - 将 target 二值化
      - 计算 (1 - target) 的距离变换
      - 以距离作为权重计算 L1 损失
    """
    # 使用 sigmoid 并 clamp 防止极端数值
    pred = torch.sigmoid(pred).clamp(min=1e-7, max=1-1e-7)
    target = (target > 0.5).float()
    
    loss = 0.0
    B = target.shape[0]
    for i in range(B):
        target_np = target[i, 0].detach().cpu().numpy().astype(np.uint8)
        # 计算距离变换，限制最大值以防止过大权重
        dist_map = cv2.distanceTransform(1 - target_np, cv2.DIST_L2, 5)
        dist_tensor = torch.tensor(dist_map, dtype=pred.dtype, device=pred.device)
        dist_tensor = torch.clamp(dist_tensor, max=10.0)
        loss += torch.mean(dist_tensor * torch.abs(pred[i, 0] - target[i, 0]))
    return loss / max(B, 1)

class DiceBCELoss(nn.Module):
    """
    混合 Dice Loss 与 BCE Loss。
    """
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 使用 sigmoid 并 clamp 以确保数值稳定
        probs = torch.sigmoid(inputs).clamp(min=1e-7, max=1-1e-7)
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        if union < self.smooth:
            return self.bce(inputs, targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        bce_loss = self.bce(inputs, targets)
        return 0.5 * dice_loss + 0.5 * bce_loss

class FocalLoss(nn.Module):
    """
    Focal Loss 用于二元分割，加入 clamp 防止极端数值。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=-88, max=88)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss).clamp(min=1e-7, max=1-1e-7)
        gamma_factor = torch.clamp((1 - pt) ** self.gamma, max=100.0)
        focal_loss = self.alpha * gamma_factor * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DynamicDiceBCELoss(nn.Module):
    """
    动态调整 Dice+BCE 损失中 BCE 的比例：
      - 初始时 Dice 与 BCE 各占 0.5
      - 从 stable_epoch 开始，BCE 的比重逐步上升到 max_bce_weight，
        而 Dice 的比重相应下降，使总权重为 1.0
    """
    def __init__(self, smooth=1e-5, initial_bce_weight=0.5, max_bce_weight=1.0, stable_epoch=50, schedule_epochs=50):
        super(DynamicDiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.initial_bce_weight = initial_bce_weight
        self.max_bce_weight = max_bce_weight
        self.stable_epoch = stable_epoch
        self.schedule_epochs = schedule_epochs

    def forward(self, inputs, targets, epoch):
        probs = torch.sigmoid(inputs).clamp(min=1e-7, max=1-1e-7)
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        if union < self.smooth:
            dice_loss = self.bce(inputs, targets)
        else:
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
        bce_loss = self.bce(inputs, targets)
        
        if epoch < self.stable_epoch:
            bce_weight = self.initial_bce_weight
        else:
            factor = min(1.0, (epoch - self.stable_epoch) / self.schedule_epochs)
            bce_weight = self.initial_bce_weight + (self.max_bce_weight - self.initial_bce_weight) * factor
        dice_weight = 1.0 - bce_weight
        
        return bce_weight * bce_loss + dice_weight * dice_loss

class CombinedLoss(nn.Module):
    """
    综合损失函数：
      - 初期仅使用 DynamicDiceBCELoss（Dice+BCE 部分），
      - forward() 需要传入当前 epoch
      - 返回一个元组 (总损失, dice_bce_loss, focal_loss, tversky_loss, boundary_loss)
      这里其他损失项暂时返回 0.
    """
    def __init__(self, **kwargs):
        super(CombinedLoss, self).__init__()
        self.dynamic_dice_bce = DynamicDiceBCELoss(**kwargs)
    
    def forward(self, inputs, targets, epoch):
        loss = self.dynamic_dice_bce(inputs, targets, epoch)
        # 此版本只使用 DynamicDiceBCELoss，其他部分返回 0.
        return loss, loss, torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device), torch.tensor(0.0, device=inputs.device)

