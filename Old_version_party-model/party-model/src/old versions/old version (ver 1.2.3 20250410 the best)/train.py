# ==== src/train.py ====
import os
import random
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import TransUNet
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils import visualize  # 可視化工具
from predict import SievePredictor  # 預測模組

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def main(config_path='configs/default.yaml'):
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 數據加載：訓練資料
    train_dataset = SegmentationDataset("data", mode='train', patch_size=config['patch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 數據加載：驗證資料
    val_dataset = SegmentationDataset("data", mode='val', patch_size=config['patch_size'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型初始化（使用改進後模型）
    model = TransUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_model_state = None
    
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
        
        print(f'Epoch {epoch+1} Training Loss: {epoch_loss / len(train_loader):.4f}')
        
        # 每100個 epoch 執行驗證並輸出可視化結果，可調整
        if (epoch + 1) % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}')
            
            # 從驗證集隨機選取一張完整圖片進行可視化
            val_img_dir = os.path.join("data", "val", "images")
            val_mask_dir = os.path.join("data", "val", "masks")
            val_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))]
            if len(val_files) > 0:
                img_file = random.choice(val_files)
                img_path = os.path.join(val_img_dir, img_file)
                mask_path = os.path.join(val_mask_dir, os.path.splitext(img_file)[0] + '.png')
                orig_img = cv2.imread(img_path)
                true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if orig_img is None or true_mask is None:
                    print("驗證圖片或真實掩碼讀取失敗，跳過可視化")
                else:
                    # 使用目前模型以 patch 融合方式進行完整圖預測
                    predictor = SievePredictor(model=model, patch_size=config['patch_size'],
                                                threshold=0.4, pred_batch_size=16)
                    pred_mask = predictor.predict_single_return(img_path)
                    
                    # 將原圖、正解、預測結果並排可視化並保存
                    os.makedirs("outputs/predictions", exist_ok=True)
                    save_path = os.path.join("outputs/predictions", f"val_epoch{epoch+1}.png")
                    visualize(orig_img, true_mask, pred_mask, save_path)
                    print(f"驗證結果圖已保存至: {save_path}")
            
            # 如果驗證 loss 改善則更新模型狀態，否則回復最佳點
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                os.makedirs("outputs/models", exist_ok=True)
                torch.save(best_model_state, f"outputs/models/best_model_epoch{epoch+1}.pth")
                print("更新最佳模型。")
            else:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("驗證結果較差，回復到上次最佳模型狀態。")
            
            # 每10個 epoch 保存一次當前模型
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/sievenet_epoch{epoch+1}.pth")

if __name__ == '__main__':
    main()
