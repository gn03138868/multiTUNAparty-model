# ==== src/train.py ====
import matplotlib
matplotlib.use('Agg')

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
from utils import visualize
from predict import SievePredictor
from losses import CombinedLoss
import matplotlib.pyplot as plt

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def update_loss_curve(loss_history, save_path="outputs/loss_curve.png"):
    epochs = range(1, len(loss_history['dice_bce']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history['dice_bce'], label="Dice+BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (Dice+BCE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve updated: {save_path}")

def main(config_path='configs/default.yaml'):
    # 載入配置檔
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 建立資料載入器
    train_dataset = SegmentationDataset("data", mode='train', patch_size=config['patch_size'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataset = SegmentationDataset("data", mode='val', patch_size=config['patch_size'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = TransUNet().to(device)
    
    # 載入預訓練模型（若有）
    pretrained_path = os.path.join("data", "pretrained model", "pretrained_model.pth")
    if os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"成功載入預訓練模型: {pretrained_path}")
        except Exception as e:
            print(f"載入預訓練模型失敗: {e}\n將從頭開始訓練。")
    else:
        print(f"在 {pretrained_path} 找不到預訓練模型，將從頭開始訓練。")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 使用動態 Dice+BCE 損失
    criterion = CombinedLoss(initial_bce_weight=0.5, max_bce_weight=1.0,
                              stable_epoch=50, schedule_epochs=50)
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_model_state = None
    revert_count = 0  # 記錄連續回滾次數
    max_revert = 500  # 超過此次數則停止訓練
    
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)  # 儲存損失折線圖
    
    loss_history = {"dice_bce": []}
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            try:
                with autocast():
                    outputs = model(images)
                    loss, dice_bce_val, _, _, _ = criterion(outputs, masks, epoch)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf detected at epoch {epoch+1}. Skipping batch.")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                valid_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if valid_batches > 0:
            avg_train_loss = epoch_loss / valid_batches
            print(f'Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}')
        else:
            print(f'Epoch {epoch+1} No valid training batches.')
            continue
        
        # 更新loss history (僅記錄 Dice+BCE 損失)
        loss_history["dice_bce"].append(dice_bce_val.item())
        update_loss_curve(loss_history, save_path="outputs/loss_curve.png")
        
        # 決定是否進行驗證：前100個 epoch，每 5 次驗證；100次以後，每個 epoch驗證
        do_validation = False
        if epoch + 1 < 100:
            if (epoch + 1) % 5 == 0:
                do_validation = True
        else:
            do_validation = True
        
        if do_validation:
            model.eval()
            val_loss = 0.0
            valid_val_batches = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    try:
                        with autocast():
                            outputs = model(images)
                            loss_val, _, _, _, _ = criterion(outputs, masks, epoch)
                        if not (torch.isnan(loss_val).any() or torch.isinf(loss_val).any()):
                            val_loss += loss_val.item()
                            valid_val_batches += 1
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            if valid_val_batches > 0:
                avg_val_loss = val_loss / valid_val_batches
                print(f'Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}')
                scheduler.step(avg_val_loss)
                
                # 可視化驗證結果（選擇隨機一張驗證圖）
                try:
                    val_img_dir = os.path.join("data", "val", "images")
                    val_mask_dir = os.path.join("data", "val", "masks")
                    val_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))]
                    if len(val_files) > 0:
                        img_file = random.choice(val_files)
                        img_path = os.path.join(val_img_dir, img_file)
                        mask_path = os.path.join(val_mask_dir, os.path.splitext(img_file)[0] + '.png')
                        orig_img = cv2.imread(img_path)
                        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if orig_img is not None and true_mask is not None:
                            predictor = SievePredictor(model=model, patch_size=config['patch_size'],
                                                        threshold=0.4, pred_batch_size=16)
                            pred_mask = predictor.predict_single_return(img_path)
                            save_path = os.path.join("outputs/predictions", f"val_epoch{epoch+1}.png")
                            visualize(orig_img, true_mask, pred_mask, save_path)
                            print(f"Validation visualization saved to: {save_path}")
                except Exception as e:
                    print(f"Error during visualization: {e}")
                
                # 若驗證損失改善則更新最佳模型，否則回滾
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, f"outputs/models/best_model_epoch{epoch+1}.pth")
                    print("更新最佳模型.")
                    revert_count = 0  # 重置回滾計數
                else:
                    revert_count += 1
                    print(f"Validation 沒有改善。Revert count: {revert_count}/{max_revert}")
                    if revert_count >= max_revert:
                        print("Early stopping：連續回滾次數已達上限。")
                        break
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print("回滾至最佳模型狀態。")
            else:
                print(f'Epoch {epoch+1} No valid validation batches.')
            
            # 每次驗證後都儲存一次當前模型（供日後檢查）
            torch.save(model.state_dict(), f"outputs/models/sievenet_epoch{epoch+1}.pth")
    
if __name__ == '__main__':
    main()

