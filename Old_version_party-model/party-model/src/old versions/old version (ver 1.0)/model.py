import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义 ViT 配置，支持 400x400 输入
        config = ViTConfig(
            image_size=400,  # 修改为 400
            patch_size=16,   # 保持 patch 大小不变
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=False,
            num_labels=1000,
            torch_dtype="float32",
            transform_act_fn="gelu",
        )
        self.vit = ViTModel(config)
        
        # 解码器部分保持不变
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # ViT 输出处理
        outputs = self.vit(x)
        hidden_states = outputs.last_hidden_state  # (B, 197, 768)
        
        # 重塑为 (B, 768, 14, 14)
        cls_token = hidden_states[:, 0]
        patch_tokens = hidden_states[:, 1:]
        b, n, c = patch_tokens.shape
        h = w = int(n**0.5)
        features = patch_tokens.permute(0, 2, 1).view(b, c, h, w)
        
        # 解码器
        return self.decoder(features)