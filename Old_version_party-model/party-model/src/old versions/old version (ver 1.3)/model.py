# ==== src/model.py ====
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class TransUNet(nn.Module):
    def __init__(self, input_size=224, model_patch_size=16):
        """
        :param input_size: 輸入圖像尺寸（例如 224，對應 dataset 的 patch_size）
        :param model_patch_size: ViT 分割小 patch 的尺寸（例如 16）
        """
        super().__init__()
        
        # 更新 ViT 配置：增加 transformer 隱藏層數（num_hidden_layers 由 6 增加到 12）
        config = ViTConfig(
            image_size=input_size,
            patch_size=model_patch_size,
            num_channels=3,
            hidden_size=1024,      
            num_hidden_layers=12,      # 層數增加
            num_attention_heads=16,    
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.vit = ViTModel(config)
        
        # 改進的 decoder 結構：每個上採樣階段加入額外卷積層以提取更多細節
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        vit_output = self.vit(x)
        features = vit_output.last_hidden_state  # shape: [batch, 1+num_patches, hidden_size]
        
        # 排除 class token 並重塑為 2D 特徵圖
        batch_size, seq_len, hidden_dim = features.shape
        # 假設 patch 數量為正方形，排除第一個 class token
        h = w = int((seq_len - 1) ** 0.5)
        features = features[:, 1:].permute(0, 2, 1).view(batch_size, hidden_dim, h, w)
        
        return self.decoder(features)

