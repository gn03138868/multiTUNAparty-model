# ==== src/model_with_timm.py ====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# 從原有模型匯入 TransUNet 結構（保留 decoder 與 skip connection）
from model import TransUNet

class TimMEncoderWrapper(nn.Module):
    def __init__(self, timm_model):
        super().__init__()
        self.timm_model = timm_model
        self.patch_embed = timm_model.patch_embed
        self.cls_token = timm_model.cls_token
        self.pos_embed = timm_model.pos_embed
        self.pos_drop = timm_model.pos_drop
        self.blocks = timm_model.blocks
        self.norm = timm_model.norm

    def interpolate_pos_encoding(self, x, w, h):
        n_patches = x.shape[1] - 1  # 排除 class token
        N = self.pos_embed.shape[1] - 1
        if n_patches == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1, :]
        patch_pos_embed = self.pos_embed[:, 1:, :]
        dim = x.shape[-1]
        orig_size = int(math.sqrt(N))

        patch_size = self.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        new_grid_h = h // patch_size[0]
        new_grid_w = w // patch_size[1]
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_grid_h, new_grid_w),
                                        mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states

class TransUNetWithTimm(TransUNet):
    """
    利用 timm 預訓練 ViT（例如 vit_base_patch16_224_in21k）作 encoder，
    保持原有 TransUNet decoder 與 skip connection 結構，
    同時支援不同尺寸的輸入/輸出。
    """
    def __init__(self, img_size=400):
        super().__init__()
        # 建立 timm 預訓練模型，並更新模型內部尺寸參數以支援非 224 尺寸輸入
        timm_vit = create_model('vit_base_patch16_224_in21k', pretrained=True)
        # 將預設的 img_size 參數更新為指定尺寸（如 400）
        timm_vit.patch_embed.img_size = (img_size, img_size)
        if 'img_size' in timm_vit.default_cfg:
            timm_vit.default_cfg['img_size'] = img_size
        self.vit = TimMEncoderWrapper(timm_vit)

    def forward(self, x):
        B = x.shape[0]
        vit_out, hidden_states = self.vit(x)
        final_tokens = vit_out[:, 1:, :]
        skip_tokens = hidden_states[self.skip_layer_index][:, 1:, :]

        N = final_tokens.shape[1]
        H = W = int(math.sqrt(N))
        final_feat = final_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)
        skip_feat = skip_tokens.transpose(1, 2).contiguous().view(B, 768, H, W)

        feats = torch.cat([skip_feat, final_feat], dim=1)
        out = self.decoder(feats)
        return out

if __name__ == '__main__':
    model = TransUNetWithTimm(img_size=400)
    dummy_input = torch.randn(1, 3, 400, 400)
    output = model(dummy_input)
    print("模型輸出 shape:", output.shape)
