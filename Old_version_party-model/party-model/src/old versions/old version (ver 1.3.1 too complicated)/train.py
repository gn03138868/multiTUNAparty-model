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
from utils import calculate_metrics  # 用於驗證階段評估

def main(config_path='configs/default.yaml'):
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 數據加載：訓練與驗證資料集（假設驗證資料放在 data/val 下）
    train_dataset = SegmentationDataset(config['data_path'], mode='train', patch_size=config['patch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dir = os.path.join(config['data_path'], 'val')
    if os.path.exists(val_dir):
        val_dataset = SegmentationDataset(config['data_path'], mode='val', patch_size=config['patch_size'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
        print("未發現驗證資料夾，跳過驗證階段。")
    
    # 模型初始化：使用 dataset 的 patch_size 作為 input_size，並從配置讀取 model_patch_size
    model = TransUNet(input_size=config['patch_size'], model_patch_size=config['model_patch_size']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # 若啟用學習率調度器，使用 ReduceLROnPlateau（監控驗證 loss）
    scheduler = None
    if config.get('lr_scheduler', False) and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('lr_patience', 10),
            min_lr=config.get('min_lr', 1.0e-7),
            verbose=True
        )
    
    best_val_loss = float('inf')
    os.makedirs("outputs/models", exist_ok=True)
    
    # 取得驗證預測閾值設定
    val_threshold = config.get('val_threshold', 0.3)
    
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
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}')
        
        # 驗證階段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
            
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc='Validation', leave=False):
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # 轉換預測與真實 mask 為 numpy 格式進行指標計算
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > val_threshold).astype('uint8') * 255
                    true_masks = (masks.cpu().numpy() * 255).astype('uint8')
                    
                    for t, p in zip(true_masks, preds):
                        metrics = calculate_metrics(t[0], p[0])
                        for key in all_metrics:
                            all_metrics[key].append(metrics[key])
            
            avg_val_loss = val_loss / len(val_loader)
            avg_metrics = {key: sum(vals)/len(vals) for key, vals in all_metrics.items()}
            print(f'Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | Metrics: {avg_metrics}')
            
            if scheduler:
                scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"outputs/models/sievenet_best.pth")
                print("發現新的最佳驗證 loss，儲存模型。")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"outputs/models/sievenet_epoch{epoch+1}.pth")

if __name__ == '__main__':
    main()
