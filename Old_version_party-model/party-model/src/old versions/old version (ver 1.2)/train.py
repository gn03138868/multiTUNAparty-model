# ==== src/train.py ====
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import TransUNet
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def main(config_path='configs/default.yaml'):
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 數據加載
    train_dataset = SegmentationDataset(config['data_path'], mode='train', patch_size=config['patch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型初始化
    model = TransUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # 訓練迴圈
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 每10個epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/sievenet_epoch{epoch+1}.pth")
        
        print(f'Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}')

if __name__ == '__main__':
    main()
