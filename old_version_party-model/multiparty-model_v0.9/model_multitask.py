"""
多任務 TransUNet - 針對植物細胞、血球、根系三種影像的改進版本
主要改進：
1. 任務條件化 (Task Conditioning)
2. 多尺度特徵融合 (ASPP)
3. 通道和空間注意力機制
4. 計算成本增加不到 15%
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
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """空間注意力 - 幫助模型關注重要的空間位置"""
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
# 多尺度特徵融合模塊
# ============================================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling - 多尺度特徵提取
    對於細胞用小膨脹率，對於根系用大膨脹率
    使用 GroupNorm 以支援 batch_size=1 的情況
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 不同膨脹率的卷積 - 捕捉不同尺度的特徵
        # 使用 GroupNorm 替代 BatchNorm（不受 batch size 限制）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # 32 groups
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合所有分支
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
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
        # task_id: 單一整數或 tensor
        if isinstance(task_id, int):
            task_id = torch.tensor([task_id] * batch_size, dtype=torch.long)
        
        task_emb = self.embedding(task_id.to(self.embedding.weight.device))
        task_emb = self.norm(task_emb)
        
        return task_emb  # [B, embed_dim]


# ============================================================================
# 改進的 Decoder 塊
# ============================================================================

class ImprovedDecoderBlock(nn.Module):
    """
    改進的解碼塊：
    1. 整合任務嵌入
    2. 添加注意力機制
    3. 殘差連接
    使用 GroupNorm 以支援 batch_size=1
    """
    def __init__(self, in_channels, out_channels, task_embed_dim=256):
        super().__init__()
        
        # 任務條件化的投影層
        self.task_proj = nn.Linear(task_embed_dim, in_channels)
        
        # 主要卷積層（使用 GroupNorm）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),  # 確保 groups <= channels
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
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
            task_weight = self.task_proj(task_emb)  # [B, in_channels]
            task_weight = task_weight.unsqueeze(-1).unsqueeze(-1)  # [B, in_channels, 1, 1]
            x = x + task_weight
        
        # 主要卷積
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 注意力
        out = self.attention(out)
        
        # 殘差連接
        out = out + self.residual_proj(residual)
        
        return out


# ============================================================================
# 主模型：多任務 TransUNet
# ============================================================================

class MultiTaskTransUNet(nn.Module):
    """
    多任務 TransUNet
    
    主要特性：
    1. 任務條件化編碼器和解碼器
    2. 多尺度特徵融合 (ASPP)
    3. 注意力機制
    4. 保持原始 TransUNet 的 ViT encoder 結構
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
        task_embed_dim=256
    ):
        super().__init__()
        
        # 任務嵌入
        self.task_embedding = TaskEmbedding(num_tasks, task_embed_dim)
        
        # ViT Encoder (保持原始 TransUNet 結構)
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
        
        # Decoder - 使用改進的解碼塊
        decoder_channels = [256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = 256
        for i, out_ch in enumerate(decoder_channels):
            layers = []
            # 每個階段使用多個改進的解碼塊
            num_blocks = num_decoder_layers // len(decoder_channels)
            for j in range(num_blocks):
                if j == 0:
                    layers.append(ImprovedDecoderBlock(in_ch, out_ch, task_embed_dim))
                else:
                    layers.append(ImprovedDecoderBlock(out_ch, out_ch, task_embed_dim))
            self.decoder_blocks.append(nn.ModuleList(layers))
            in_ch = out_ch
        
        # 上採樣層
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 最終輸出層
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化其他權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        """
        B, C, H, W = x.shape
        
        # 獲取任務嵌入
        task_emb = self.task_embedding(task_id, B)  # [B, task_embed_dim]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
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
        for stage_blocks in self.decoder_blocks:
            for block in stage_blocks:
                x = block(x, task_emb)
            x = self.upsample(x)
        
        # 最終輸出
        x = self.final_conv(x)
        
        # 調整到原始大小
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


# ============================================================================
# Transformer Block (保持原始結構)
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
        # Multi-head attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================================
# 測試和實用函數
# ============================================================================

def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    """測試模型"""
    print("Testing MultiTaskTransUNet...")
    
    model = MultiTaskTransUNet(
        img_size=400,
        patch_size=16,
        num_decoder_layers=80,
        num_tasks=3
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # 測試不同任務
    x = torch.randn(2, 3, 400, 400)
    
    for task_id, task_name in enumerate(['Plant Cell', 'Blood Cell', 'Root']):
        print(f"\nTesting {task_name} (task_id={task_id})...")
        y = model(x, task_id=task_id)
        print(f"Output shape: {y.shape}")
        assert y.shape == (2, 1, 400, 400), f"Output shape mismatch: {y.shape}"
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_model()
