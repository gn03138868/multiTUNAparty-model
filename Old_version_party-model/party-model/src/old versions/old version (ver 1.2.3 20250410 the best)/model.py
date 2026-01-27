# ==== src/model.py ====
import torch
import torch.nn as nn
from transformers import ViTModel

class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 使用預訓練的 ViT，設定 output_hidden_states=True
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_hidden_states=True)
        # 更新配置（改變 image_size）
        self.vit.config.image_size = 400  # 設定目標輸入尺寸
        hidden_size = self.vit.config.hidden_size  # 預設 768
        
        # Decoder：我們使用 UNet-style skip connection (使用第4層的 hidden_states) 與最後層輸出做 concat
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size * 2, 512, 3, padding=1),
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
        # 傳入 interpolate_pos_encoding=True，讓預訓練模型在位置嵌入上進行插值，
        # 這樣就能接受 400×400 的輸入（原本 pretrained ViT 是 224×224）
        vit_output = self.vit(x, interpolate_pos_encoding=True)
        all_hidden = vit_output.hidden_states  # tuple of [B, 1+N, hidden_size]
        # 取最後一層 (去掉 class token)
        last_hidden = vit_output.last_hidden_state[:, 1:, :]  # [B, N, hidden_size]
        # 取中間一層作為 skip connection (例如第 4 層, 請根據實際需求調整)
        skip_hidden = all_hidden[4][:, 1:, :]  # [B, N, hidden_size]
        
        B, N, C = last_hidden.shape
        H = W = int(N ** 0.5)  # 假設 patch token 可以排列成正方形
        last_features = last_hidden.permute(0, 2, 1).view(B, C, H, W)
        skip_features = skip_hidden.permute(0, 2, 1).view(B, C, H, W)
        
        # Concatenate skip connection 和最後一層特徵
        features = torch.cat([skip_features, last_features], dim=1)  # [B, 2*C, H, W]
        out = self.decoder(features)
        return out

