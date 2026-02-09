"""
多任務 TransUNet - 邊界感知增強版本
針對植物細胞、血球、根系三種影像的邊界清晰分割

主要改進：
1. 雙頭輸出：分割頭 + 邊界檢測頭
2. 邊界感知精煉模組 (Boundary-Aware Refinement)
3. 梯度增強邊界檢測 (Gradient-based Edge Detection)
4. 深度監督 (Deep Supervision)
5. 多尺度邊界融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 注意力模塊
# ============================================================================

class ChannelAttention(nn.Module):
    """通道注意力 - 幫助模型關注重要的特徵通道"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)  # 確保至少有8個通道
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """空間注意力 - 幫助模型關注重要的空間位置（特別是邊界）"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    """CBAM - 結合通道和空間注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================================
# 邊界檢測模塊
# ============================================================================

class GradientEdgeDetector(nn.Module):
    """
    基於梯度的邊界檢測
    使用Sobel算子計算梯度，增強邊界特徵
    """
    def __init__(self, in_channels):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        
        self.in_channels = in_channels
    
    def forward(self, x):
        # 計算梯度
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        
        # 計算梯度幅度
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return gradient_magnitude


class BoundaryDetectionHead(nn.Module):
    """
    邊界檢測頭
    專門用於檢測和增強邊界特徵
    """
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        
        # 梯度檢測
        self.gradient_detector = GradientEdgeDetector(in_channels)
        
        # 邊界特徵提取
        self.boundary_conv = nn.Sequential(
            # 小感受野，捕捉細節
            nn.Conv2d(in_channels * 2, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 卷積融合
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            
            # 輸出邊界概率
            nn.Conv2d(mid_channels, 1, 1)
        )
        
        # 多尺度邊界檢測（小kernel捕捉精細邊界）
        self.edge_conv_3x3 = nn.Conv2d(in_channels, mid_channels // 2, 3, padding=1, bias=False)
        self.edge_conv_1x1 = nn.Conv2d(in_channels, mid_channels // 2, 1, bias=False)
        
        self.edge_fusion = nn.Sequential(
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1)
        )
    
    def forward(self, x):
        # 梯度特徵
        gradient_feat = self.gradient_detector(x)
        
        # 結合原始特徵和梯度特徵
        combined = torch.cat([x, gradient_feat], dim=1)
        boundary_out = self.boundary_conv(combined)
        
        # 多尺度邊界
        edge_3x3 = self.edge_conv_3x3(x)
        edge_1x1 = self.edge_conv_1x1(x)
        multi_scale_edge = torch.cat([edge_3x3, edge_1x1], dim=1)
        multi_scale_out = self.edge_fusion(multi_scale_edge)
        
        # 融合兩種邊界檢測結果
        boundary = (boundary_out + multi_scale_out) / 2
        
        return boundary


class BoundaryAwareRefinement(nn.Module):
    """
    邊界感知精煉模組
    利用邊界資訊來精煉分割結果
    """
    def __init__(self, seg_channels, boundary_channels=1, out_channels=1):
        super().__init__()
        
        # 邊界特徵擴展
        self.boundary_expand = nn.Sequential(
            nn.Conv2d(boundary_channels, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        
        # 分割特徵處理
        self.seg_conv = nn.Sequential(
            nn.Conv2d(seg_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        
        # 邊界引導注意力
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )
    
    def forward(self, seg_feat, boundary_pred):
        # 處理邊界預測
        boundary_feat = self.boundary_expand(torch.sigmoid(boundary_pred))
        
        # 處理分割特徵
        seg_feat = self.seg_conv(seg_feat)
        
        # 邊界引導注意力
        boundary_attn = self.boundary_attention(boundary_feat)
        
        # 使用邊界注意力增強分割特徵（邊界區域得到更多關注）
        enhanced_seg = seg_feat * (1 + boundary_attn)
        
        # 融合
        combined = torch.cat([enhanced_seg, boundary_feat], dim=1)
        refined = self.fusion(combined)
        
        return refined


# ============================================================================
# 多尺度特徵融合模塊
# ============================================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling - 多尺度特徵提取
    使用 GroupNorm 以支援 batch_size=1 的情況
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 確保 out_channels 可被 groups 整除
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        
        # 不同膨脹率的卷積 - 捕捉不同尺度的特徵
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合所有分支
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.fusion(out)
        
        return out


# ============================================================================
# 任務嵌入模塊
# ============================================================================

class TaskEmbedding(nn.Module):
    """
    任務條件化 - 讓模型知道當前處理的是哪種影像
    task_id: 0=植物細胞, 1=血球, 2=根系
    """
    def __init__(self, num_tasks=3, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, task_id, batch_size):
        if isinstance(task_id, int):
            task_id = torch.tensor([task_id] * batch_size, dtype=torch.long)
        
        task_emb = self.embedding(task_id.to(self.embedding.weight.device))
        task_emb = self.norm(task_emb)
        
        return task_emb


# ============================================================================
# 改進的 Decoder 塊（加入邊界感知）
# ============================================================================

class BoundaryAwareDecoderBlock(nn.Module):
    """
    邊界感知解碼塊
    在解碼過程中保持邊界清晰度
    """
    def __init__(self, in_channels, out_channels, task_embed_dim=256):
        super().__init__()
        
        # 任務條件化的投影層
        self.task_proj = nn.Linear(task_embed_dim, in_channels)
        
        groups = min(32, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        
        in_groups = min(32, in_channels)
        while in_channels % in_groups != 0:
            in_groups -= 1
        
        # 主要卷積層（使用 GroupNorm）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 邊界增強卷積（小kernel保持邊界細節）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力機制
        self.attention = CBAM(out_channels)
        
        # 殘差連接的投影層
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, task_emb=None):
        residual = x
        
        # 如果有任務嵌入，添加到特徵中
        if task_emb is not None:
            task_weight = self.task_proj(task_emb)
            task_weight = task_weight.unsqueeze(-1).unsqueeze(-1)
            x = x + task_weight
        
        # 主要卷積
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 邊界增強
        edge_feat = self.edge_conv(out)
        out = out + edge_feat
        
        # 注意力
        out = self.attention(out)
        
        # 殘差連接
        out = out + self.residual_proj(residual)
        
        return out


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 主模型：邊界感知多任務 TransUNet
# ============================================================================

class MultiTaskTransUNet(nn.Module):
    """
    邊界感知多任務 TransUNet
    
    主要特性：
    1. 雙頭輸出：分割頭 + 邊界檢測頭
    2. 邊界感知精煉模組
    3. 任務條件化編碼器和解碼器
    4. 多尺度特徵融合 (ASPP)
    5. 深度監督（可選）
    """
    def __init__(
        self,
        img_size=400,
        patch_size=16,
        in_channels=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_decoder_layers=80,
        num_tasks=3,
        task_embed_dim=256,
        use_deep_supervision=True
    ):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.img_size = img_size
        
        # 任務嵌入
        self.task_embedding = TaskEmbedding(num_tasks, task_embed_dim)
        
        # ViT Encoder
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 多尺度特徵融合
        self.aspp = ASPP(embed_dim, 256)
        
        # Decoder - 使用邊界感知解碼塊
        decoder_channels = [256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = 256
        for i, out_ch in enumerate(decoder_channels):
            layers = []
            num_blocks = num_decoder_layers // len(decoder_channels)
            for j in range(num_blocks):
                if j == 0:
                    layers.append(BoundaryAwareDecoderBlock(in_ch, out_ch, task_embed_dim))
                else:
                    layers.append(BoundaryAwareDecoderBlock(out_ch, out_ch, task_embed_dim))
            self.decoder_blocks.append(nn.ModuleList(layers))
            in_ch = out_ch
        
        # 上採樣層
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # ============== 分割頭 ==============
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )
        
        # ============== 邊界檢測頭 ==============
        self.boundary_head = BoundaryDetectionHead(32, mid_channels=32)
        
        # ============== 邊界感知精煉模組 ==============
        self.boundary_refinement = BoundaryAwareRefinement(
            seg_channels=num_classes, 
            boundary_channels=1, 
            out_channels=num_classes
        )
        
        # ============== 深度監督（可選）==============
        if use_deep_supervision:
            # 從不同解碼器層輸出
            self.deep_seg_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, 1) for ch in [256, 128, 64]
            ])
            self.deep_boundary_heads = nn.ModuleList([
                nn.Conv2d(ch, 1, 1) for ch in [256, 128, 64]
            ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, task_id=0):
        """
        Args:
            x: 輸入影像 [B, 3, H, W]
            task_id: 任務ID (0=植物細胞, 1=血球, 2=根系)
        
        Returns:
            訓練時: dict with 'seg', 'boundary', 'refined', 'deep_seg', 'deep_boundary'
            推論時: 精煉後的分割結果
        """
        B, C, H, W = x.shape
        
        # 獲取任務嵌入
        task_emb = self.task_embedding(task_id, B)
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoding
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial
        grid_size = int(math.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, -1, grid_size, grid_size)
        
        # 多尺度特徵融合
        x = self.aspp(x)
        
        # Decoder with task conditioning
        deep_features = []  # 用於深度監督
        
        for i, stage_blocks in enumerate(self.decoder_blocks):
            for block in stage_blocks:
                x = block(x, task_emb)
            
            # 保存深度監督用的特徵（上採樣前）
            if self.use_deep_supervision and i < 3:
                deep_features.append(x)
            
            x = self.upsample(x)
        
        # ============== 分割輸出 ==============
        seg_out = self.seg_head(x)
        
        # ============== 邊界輸出 ==============
        boundary_out = self.boundary_head(x)
        
        # ============== 邊界感知精煉 ==============
        refined_out = self.boundary_refinement(seg_out, boundary_out)
        
        # 最終輸出 = 原始分割 + 精煉結果（殘差學習）
        final_out = seg_out + refined_out
        
        # 調整到原始大小
        if final_out.shape[2:] != (H, W):
            final_out = F.interpolate(final_out, size=(H, W), mode='bilinear', align_corners=False)
            seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)
            boundary_out = F.interpolate(boundary_out, size=(H, W), mode='bilinear', align_corners=False)
        
        if self.training:
            output = {
                'seg': seg_out,
                'boundary': boundary_out,
                'refined': final_out,
            }
            
            # 深度監督輸出
            if self.use_deep_supervision:
                deep_segs = []
                deep_boundaries = []
                for i, feat in enumerate(deep_features):
                    deep_seg = self.deep_seg_heads[i](feat)
                    deep_boundary = self.deep_boundary_heads[i](feat)
                    deep_seg = F.interpolate(deep_seg, size=(H, W), mode='bilinear', align_corners=False)
                    deep_boundary = F.interpolate(deep_boundary, size=(H, W), mode='bilinear', align_corners=False)
                    deep_segs.append(deep_seg)
                    deep_boundaries.append(deep_boundary)
                
                output['deep_seg'] = deep_segs
                output['deep_boundary'] = deep_boundaries
            
            return output
        else:
            # 推論時返回精煉後的結果
            return final_out


# ============================================================================
# 邊界感知損失函數
# ============================================================================

class BoundaryAwareLoss(nn.Module):
    """
    邊界感知損失函數
    結合分割損失和邊界損失
    """
    def __init__(
        self,
        boundary_weights=None,
        foreground_weights=None,
        deep_supervision_weights=None,
        smooth=1e-5
    ):
        super().__init__()
        
        self.boundary_weights = boundary_weights or {
            0: 2.0,   # 細胞
            1: 3.0,   # 血球
            2: 5.0    # 根系
        }
        
        self.foreground_weights = foreground_weights or {
            0: 1.0,
            1: 1.5,
            2: 3.0
        }
        
        # 深度監督權重（從深到淺遞減）
        self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.2, 0.1]
        
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets, task_ids, boundary_targets=None):
        """
        Args:
            outputs: dict from model containing 'seg', 'boundary', 'refined', etc.
            targets: 真實分割標籤 [B, 1, H, W]
            task_ids: 任務ID [B]
            boundary_targets: 真實邊界標籤（可選，如果沒有會自動生成）
        """
        batch_size = targets.shape[0]
        
        # 如果沒有邊界標籤，自動生成
        if boundary_targets is None:
            boundary_targets = self._generate_boundary_targets(targets)
        
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 主分割損失
        seg_loss = self._compute_seg_loss(outputs['seg'], targets, task_ids)
        total_loss += seg_loss
        loss_dict['seg_loss'] = seg_loss.item()
        
        # 2. 邊界損失
        boundary_loss = self._compute_boundary_loss(outputs['boundary'], boundary_targets, task_ids)
        total_loss += boundary_loss
        loss_dict['boundary_loss'] = boundary_loss.item()
        
        # 3. 精煉後的分割損失
        refined_loss = self._compute_seg_loss(outputs['refined'], targets, task_ids)
        total_loss += refined_loss
        loss_dict['refined_loss'] = refined_loss.item()
        
        # 4. 深度監督損失
        if 'deep_seg' in outputs:
            deep_loss = 0.0
            for i, (deep_seg, deep_boundary) in enumerate(zip(outputs['deep_seg'], outputs['deep_boundary'])):
                weight = self.deep_supervision_weights[i]
                deep_loss += weight * self._compute_seg_loss(deep_seg, targets, task_ids)
                deep_loss += weight * 0.5 * self._compute_boundary_loss(deep_boundary, boundary_targets, task_ids)
            total_loss += deep_loss
            loss_dict['deep_loss'] = deep_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _generate_boundary_targets(self, targets):
        """從分割標籤自動生成邊界標籤"""
        # 使用形態學運算生成邊界
        kernel_size = 3
        
        # Dilation - Erosion = Boundary
        targets_float = targets.float()
        
        # 膨脹
        dilated = F.max_pool2d(targets_float, kernel_size, stride=1, padding=kernel_size//2)
        
        # 腐蝕（使用負值再取max再取負）
        eroded = -F.max_pool2d(-targets_float, kernel_size, stride=1, padding=kernel_size//2)
        
        # 邊界 = 膨脹 - 腐蝕
        boundary = dilated - eroded
        
        return boundary
    
    def _compute_seg_loss(self, pred, target, task_ids):
        """計算分割損失"""
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            task_id = task_ids[i].item()
            p = pred[i:i+1]
            t = target[i:i+1]
            
            # Dice Loss
            pred_sigmoid = torch.sigmoid(p).clamp(min=1e-7, max=1-1e-7)
            pred_flat = pred_sigmoid.view(-1)
            target_flat = t.view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            # BCE Loss
            bce_loss = self.bce(p, t).mean()
            
            # 前景加權
            fg_mask = (t > 0.5).float()
            bg_mask = (t <= 0.5).float()
            fg_count = fg_mask.sum()
            bg_count = bg_mask.sum()
            
            if fg_count > 0:
                fg_weight = torch.clamp(bg_count / (fg_count + 1e-7), min=1.0, max=10.0)
                weighted_bce = self.bce(p, t)
                weighted_bce = weighted_bce * (fg_mask * fg_weight + bg_mask)
                weighted_bce = weighted_bce.mean()
            else:
                weighted_bce = bce_loss
            
            sample_loss = dice_loss + bce_loss + self.foreground_weights[task_id] * weighted_bce
            total_loss += sample_loss
        
        return total_loss / batch_size
    
    def _compute_boundary_loss(self, pred, target, task_ids):
        """計算邊界損失"""
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            task_id = task_ids[i].item()
            p = pred[i:i+1]
            t = target[i:i+1]
            
            # BCE Loss for boundary
            bce_loss = self.bce(p, t).mean()
            
            # Dice Loss for boundary
            pred_sigmoid = torch.sigmoid(p).clamp(min=1e-7, max=1-1e-7)
            pred_flat = pred_sigmoid.view(-1)
            target_flat = t.view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            sample_loss = self.boundary_weights[task_id] * (bce_loss + dice_loss)
            total_loss += sample_loss
        
        return total_loss / batch_size


# ============================================================================
# 測試和實用函數
# ============================================================================

def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """測試模型"""
    print("Testing Boundary-Aware MultiTaskTransUNet...")
    print("="*60)
    
    model = MultiTaskTransUNet(
        img_size=400,
        patch_size=16,
        num_decoder_layers=80,
        num_tasks=3,
        use_deep_supervision=True
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # 測試訓練模式
    model.train()
    x = torch.randn(2, 3, 400, 400)
    
    for task_id, task_name in enumerate(['Plant Cell', 'Blood Cell', 'Root']):
        print(f"\nTesting {task_name} (task_id={task_id})...")
        outputs = model(x, task_id=task_id)
        
        print(f"  Seg output shape: {outputs['seg'].shape}")
        print(f"  Boundary output shape: {outputs['boundary'].shape}")
        print(f"  Refined output shape: {outputs['refined'].shape}")
        
        if 'deep_seg' in outputs:
            print(f"  Deep supervision outputs: {len(outputs['deep_seg'])} levels")
        
        assert outputs['refined'].shape == (2, 1, 400, 400), f"Output shape mismatch"
    
    # 測試推論模式
    model.eval()
    with torch.no_grad():
        y = model(x, task_id=0)
        print(f"\nInference output shape: {y.shape}")
        assert y.shape == (2, 1, 400, 400)
    
    # 測試損失函數
    print("\n" + "="*60)
    print("Testing BoundaryAwareLoss...")
    
    criterion = BoundaryAwareLoss()
    
    model.train()
    outputs = model(x, task_id=0)
    targets = torch.randint(0, 2, (2, 1, 400, 400)).float()
    task_ids = torch.tensor([0, 0])
    
    loss, loss_dict = criterion(outputs, targets, task_ids)
    print(f"Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")

if __name__ == '__main__':
    test_model()
