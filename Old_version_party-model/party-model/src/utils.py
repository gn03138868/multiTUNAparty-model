# ==== src/utils.py ====
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def visualize(image, true_mask, pred_mask, save_path=None):
    """
    直接將原圖、正解與預測結果以原始像素值直列堆疊，並以 PNG 格式保存
    備註：輸入圖像 image 由 cv2.imread 取得，為 BGR 格式；
         true_mask 與 pred_mask 為單通道灰階圖，先轉為 BGR 再堆疊
    """
    # 若 true_mask 與 pred_mask 為單通道，轉換為 3 通道方便顯示
    if len(true_mask.shape) == 2:
        true_mask_color = cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR)
    else:
        true_mask_color = true_mask.copy()
    if len(pred_mask.shape) == 2:
        pred_mask_color = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    else:
        pred_mask_color = pred_mask.copy()
    
    # 將三張圖片（原圖、正解、預測）直列（垂直）堆疊
    combined = np.vstack((image, true_mask_color, pred_mask_color))
    
    # 保存時使用 cv2.imwrite，直接輸出原始像素數值的 PNG 檔
    if save_path:
        cv2.imwrite(save_path, combined)

def calculate_metrics(true_mask, pred_mask):
    y_true = true_mask.flatten() > 127
    y_pred = pred_mask.flatten() > 127
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'iou': np.sum(y_true & y_pred) / np.sum(y_true | y_pred)
    }

