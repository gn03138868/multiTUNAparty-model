# ==== src/model.py ====
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 優化後ViT配置
        config = ViTConfig(
            image_size=400,
            patch_size=16,
            num_channels=3,
            hidden_size=1024,      # 特徵維度
            num_hidden_layers=12,   # 層數20-24適合醫療用高精度的圖像
            num_attention_heads=16,  # 注意力頭數
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.vit = ViTModel(config)
        
        # 改進的解碼器結構
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        vit_output = self.vit(x)
        features = vit_output.last_hidden_state  # shape: [batch, 1+num_patches, hidden_size]
        
        # 排除 class token 並重塑為2D特徵圖
        batch_size, seq_len, hidden_dim = features.shape
        # 假設patches數量為正方形，排除第一個class token
        h = w = int((seq_len - 1) ** 0.5)
        features = features[:, 1:].permute(0, 2, 1).view(batch_size, hidden_dim, h, w)
        
        return self.decoder(features)
