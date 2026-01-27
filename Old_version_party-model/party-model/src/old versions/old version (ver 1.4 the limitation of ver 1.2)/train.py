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
from utils import calculate_metrics  # 引入計算評估指標的函數

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def count_parameters(model):
    """返回模型總參數量與可訓練參數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_model_size(model):
    """
    估計模型大小 (MB)，假設每個參數 4 字節（float32）
    """
    total_params, _ = count_parameters(model)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024**2)
    return size_mb

def main(config_path='configs/default.yaml'):
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 數據加載：分別載入 train 與 val 數據夾中的資料
    train_dataset = SegmentationDataset(config['data_path'], mode='train', patch_size=config['patch_size'])
    val_dataset = SegmentationDataset(config['data_path'], mode='val', patch_size=config['patch_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型初始化
    model = TransUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # 計算並打印模型量化指標
    total_params, trainable_params = count_parameters(model)
    model_size = estimate_model_size(model)
    print(f"模型總參數: {total_params}")
    print(f"可訓練參數: {trainable_params}")
    print(f"估計模型大小: {model_size:.2f} MB")
    
    best_val_loss = float('inf')
    
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
        
        # 驗證階段，並計算額外評估指標
        model.eval()
        val_loss = 0.0
        metrics_list = []  # 存放每個 batch 的指標結果
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # 將 logits 經 sigmoid 激活並二值化 (以 0.3 為閾值)
                preds = (torch.sigmoid(outputs) > 0.3).float()
                
                # 將 tensor 轉為 numpy array，並轉換到 0-255 範圍，方便計算指標
                preds_np = (preds.cpu().numpy() * 255).astype('uint8')
                masks_np = (masks.cpu().numpy() * 255).astype('uint8')
                
                # 對每個 sample 計算指標，然後存到列表中
                for i in range(preds_np.shape[0]):
                    metrics = calculate_metrics(masks_np[i, 0, :, :], preds_np[i, 0, :, :])
                    metrics_list.append(metrics)
        
        # 計算平均指標
        avg_precision = sum(m['precision'] for m in metrics_list) / len(metrics_list)
        avg_recall = sum(m['recall'] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m['f1'] for m in metrics_list) / len(metrics_list)
        avg_iou = sum(m['iou'] for m in metrics_list) / len(metrics_list)
        
        train_loss_avg = epoch_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        print(f'Epoch {epoch+1} Training Loss: {train_loss_avg:.4f} Validation Loss: {val_loss_avg:.4f} Precision: {avg_precision:.4f} Recall: {avg_recall:.4f} F1 Score: {avg_f1:.4f} IoU: {avg_iou:.4f}')
        
        # 檢查是否取得更好的驗證結果，保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/best_model.pth")
            print(f"Epoch {epoch+1}: 検証損失が改善されたため、最良のモデルを保存します。Validation loss has improved, saving the best model. 驗證損失改善，保存最佳模型。")
        
        # 每10個epoch保存一次當前模型
        if (epoch + 1) % 10 == 0:
            os.makedirs("outputs/models", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/models/sievenet_epoch{epoch+1}.pth")

if __name__ == '__main__':
    main()
