# ==== src/model.py ====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# DropPath (Stochastic Depth) 
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
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 為每個 sample 生成遮罩
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
        q, k, v = qkv[0], qkv[1], qkv[2]   # each: [B, num_heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)  # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, dim]
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

        # Class token 與位置嵌入
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
    def __init__(self, num_decoder_conv_layers=5):
        """
        num_decoder_conv_layers:
         decoder 中要堆疊的卷積層數量（不含上採樣與最後 1x1 卷積）。
         此參數可以從 configs/default.yaml 調整，例如設為 30。
        """
        super().__init__()
        # 建立完整的 ViT 模型 (輸入尺寸固定為 400)
        self.vit = VisionTransformer(img_size=400, patch_size=16, in_chans=3,
                                     embed_dim=768, depth=12, num_heads=12,
                                     mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1)
        # 選擇一層作為 skip connection (例如第 6 層, index 5)
        self.skip_layer_index = 5
        
        # 建立深層 decoder，輸入通道數固定為 768*2 = 1536 (skip + final token)
        # 修改 decoder 結構使得輸出解析度能從 25x25 上採樣至 400x400，
        # 需要 4 次上採樣，故將 decoder 分為 5 個 block
        self.decoder = self.build_deep_decoder(num_layers=num_decoder_conv_layers)
    
    def build_deep_decoder(self, num_layers):
        """
        動態構造 decoder:
         - 將 decoder 分成 5 個 block，每個 block 中均勻分配卷積層數。
         - 每個 block 完成後（除最後一個 block 外）加入上採樣層 (scale_factor=2)。
         - 最後加入一個 1x1 卷積輸出分割結果。
        """
        num_blocks = 5  # 固定 block 數，這樣上採樣次數 = 4, 輸出解析度 25*(2^4)=400
        convs_per_block = num_layers // num_blocks
        remainder = num_layers % num_blocks  # 前 remainder 個 block 多一層
        
        layers = []
        in_channels = 1536  # 輸入通道數
        # 為每個 block 設計不同的輸出通道，參考原有結構： [512, 256, 128, 64, 32]
        channel_scheme = [512, 256, 128, 64, 32]
        for i in range(num_blocks):
            out_channels = channel_scheme[i]
            # 決定本 block 卷積層數
            layers_in_block = convs_per_block + (1 if i < remainder else 0)
            for j in range(layers_in_block):
                if j == 0:
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                else:
                    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            # 除了最後一個 block，每個 block後加入上採樣
            if i < num_blocks - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            in_channels = out_channels
        # 最後一層: 1x1 卷積輸出單通道結果
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        B = x.shape[0]
        # 透過 ViT 得到最終 token 與各層隱藏狀態
        vit_out, hidden_states = self.vit(x, interpolate_pos_encoding=True)  # vit_out: [B, 1+N, 768]
        # 移除 class token 得到 token 序列
        final_tokens = vit_out[:, 1:, :]  # [B, N, 768]
        # 取得 skip connection token (來自第 skip_layer_index 層)
        skip_tokens = hidden_states[self.skip_layer_index][:, 1:, :]  # [B, N, 768]
        
        # 假設 token 數量 N 可排列成正方形，轉換成 2D feature maps
        N = final_tokens.shape[1]
        H = W = int(math.sqrt(N))
        final_feat = final_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)
        skip_feat = skip_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)
        
        # 串接 skip 與 encoder 最終特徵: [B, 1536, H, W]，H 與 W 應為 25
        feats = torch.cat([skip_feat, final_feat], dim=1)
        out = self.decoder(feats)
        return out

# 測試用
if __name__ == '__main__':
    # 可在 configs/default.yaml 中設定 num_decoder_conv_layers (如 30)
    model = TransUNet(num_decoder_conv_layers=30)
    dummy_input = torch.randn(1, 3, 400, 400)
    output = model(dummy_input)
    print("模型輸出 shape:", output.shape)

