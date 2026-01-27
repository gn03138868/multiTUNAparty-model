# ==== src/utils.py ====
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def visualize(image, true_mask, pred_mask, save_path=None):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def calculate_metrics(true_mask, pred_mask):
    """計算評估指標，並設定 zero_division 參數避免警告"""
    y_true = true_mask.flatten() > 127
    y_pred = pred_mask.flatten() > 127

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    union = np.sum(y_true | y_pred)
    iou = np.sum(y_true & y_pred) / union if union != 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }