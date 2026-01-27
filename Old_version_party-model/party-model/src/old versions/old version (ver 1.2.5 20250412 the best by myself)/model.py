# ==== src/model.py ====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# DropPath (Stochastic Depth) 實作
#########################################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 為每個 sample 生成隨機遮罩
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

#########################################
# Patch Embedding
#########################################

class PatchEmbed(nn.Module):
    """
    將圖像分割成 patch 並以卷積映射到 embed_dim
    """
    def __init__(self, img_size=400, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

#########################################
# MLP block
#########################################

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#########################################
# Multi-head Self-Attention
#########################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, N, dim]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, num_heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        x = x.transpose(1,2).reshape(B, N, C)  # [B, N, dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#########################################
# Transformer Block
#########################################

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, dropout=dropout)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#########################################
# Full Vision Transformer
#########################################

class VisionTransformer(nn.Module):
    def __init__(self, img_size=400, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0, drop_path_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token與位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 隨機深度衰減率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, interpolate_pos_encoding=False):
        # x: [B, 3, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        if interpolate_pos_encoding:
            N = x.shape[1]
            if N != self.pos_embed.shape[1]:
                # 插值位置嵌入
                cls_pos_embed = self.pos_embed[:, 0:1, :]
                patch_pos_embed = self.pos_embed[:, 1:, :]
                dim = x.shape[-1]
                num_patches = int(math.sqrt(N-1))
                orig_size = int(math.sqrt(self.pos_embed.shape[1]-1))
                patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
                patch_pos_embed = F.interpolate(patch_pos_embed, size=(num_patches, num_patches), mode='bicubic', align_corners=False)
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches*num_patches, dim)
                x = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
            else:
                x = x + self.pos_embed
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states

#########################################
# TransUNet: 結合完整 ViT 與 UNet-style Decoder
#########################################

class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 建立完整的 ViT 模型
        self.vit = VisionTransformer(img_size=400, patch_size=16, in_chans=3,
                                     embed_dim=768, depth=12, num_heads=12,
                                     mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1)
        # 選擇一層作為 skip connection (例如第 6 層，index 5)
        self.skip_layer_index = 5
        
        # Decoder: 輸入通道 = 768 * 2 = 1536 (skip + final token特徵)
        self.decoder = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 1, kernel_size=1)
        )
    
    def forward(self, x):
        B = x.shape[0]
        # 得到 ViT 最終輸出與各層隱藏狀態
        vit_out, hidden_states = self.vit(x, interpolate_pos_encoding=True)  # vit_out: [B, 1+N, 768]
        # 去除 class token 得到 token 序列
        final_tokens = vit_out[:, 1:, :]  # [B, N, 768]
        # 取 skip connection 的 token (從第 skip_layer_index 層)
        skip_tokens = hidden_states[self.skip_layer_index][:, 1:, :]  # [B, N, 768]
        
        # 假設 N 可以排列成正方形：H = W = sqrt(N)
        N = final_tokens.shape[1]
        H = W = int(math.sqrt(N))
        final_feat = final_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)
        skip_feat = skip_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)
        
        # Concat skip + final feature maps
        feats = torch.cat([skip_feat, final_feat], dim=1)  # [B, 1536, H, W]
        out = self.decoder(feats)
        return out

