# ==== src/train.py ====
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from dataset import SegmentationDataset
from model import TransUNet
from tqdm import tqdm

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        prob = torch.sigmoid(logits)
        intersection = (prob * targets).sum(dim=(2,3))
        union = prob.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        
        return bce_loss + dice_loss.mean()

def main(config_path='configs/default.yaml'):
    # 处理Windows路径问题
    config_path = config_path.replace('\\', '/')
    
    try:
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # ==== 参数类型验证 ====
        required_config = {
            'data_path': str,
            'batch_size': int,
            'epochs': int,
            'lr': float,
            'patch_size': int,
            'val_split': float
        }
        
        # 检查必需参数是否存在
        missing_keys = [k for k in required_config if k not in config]
        if missing_keys:
            raise ValueError(f"配置文件缺少必需参数: {missing_keys}")
            
        # 强制类型转换
        for key, dtype in required_config.items():
            try:
                config[key] = dtype(config[key])
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"参数 '{key}' 类型错误，应为 {dtype.__name__}，实际类型为 {type(config[key]).__name__}"
                ) from e
                
        # 参数合理性验证
        if config['val_split'] <= 0 or config['val_split'] >= 1:
            raise ValueError("val_split 必须在0和1之间")
        if config['lr'] <= 0:
            raise ValueError("学习率必须大于0")
            
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        return

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 数据加载
        full_dataset = SegmentationDataset(
            config['data_path'],
            mode='train',
            patch_size=config['patch_size']
        )
        
        # 划分验证集
        val_size = int(len(full_dataset) * config['val_split'])
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 确保可重复性
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            pin_memory_device=str(device) if device.type == 'cuda' else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    try:
        # 模型初始化
        model = TransUNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=1e-4
        )
        criterion = DiceBCELoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min', 
            patience=3,
            verbose=True
        )
        scaler = GradScaler()
        
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        try:
            # 训练阶段
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(
                train_loader, 
                desc=f'Train Epoch {epoch+1}/{config["epochs"]}',
                bar_format='{l_bar}{bar:20}{r_bar}'
            )
            
            for images, masks in progress_bar:
                images, masks = images.to(device), masks.to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # 验证阶段
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs("outputs/models", exist_ok=True)
                torch.save(model.state_dict(), f"outputs/models/best_model.pth")
                print(f"保存新最佳模型，验证损失: {avg_val_loss:.4f}")
            
            print(f"Epoch {epoch+1} 统计: "
                  f"训练损失 = {train_loss/len(train_loader):.4f}, "
                  f"验证损失 = {avg_val_loss:.4f}")
            
        except KeyboardInterrupt:
            print("训练被用户中断")
            return
        except Exception as e:
            print(f"Epoch {epoch+1} 训练出错: {str(e)}")
            return

if __name__ == '__main__':
    main()