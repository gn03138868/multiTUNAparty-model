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
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_dataset = SegmentationDataset(config['data_path'], mode='train')
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
    
    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for patches in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            for images, masks in patches:
                images = images.to(device)
                masks = masks.to(device)
                
                # 混合精度训练
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
        
        # 保存模型
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 
                      f"outputs/models/sievenet_epoch{epoch+1}.pth")
        
        print(f'Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}')

if __name__ == '__main__':
    main()