# ==== src/predict.py ====
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from model import TransUNet

class SievePredictor:
    def __init__(self, model_path=None, patch_size=400, threshold=0.4, pred_batch_size=16, model=None):
        """
        初始化預測器
        :param model_path: 模型權重路徑 (若不傳入 model，則使用此參數)
        :param patch_size: 輸入 patch 尺寸
        :param threshold: 二值化閾值 (0-1)
        :param pred_batch_size: 預測時每個批次的 patch 數量
        :param model: 可選，直接傳入模型 (例如訓練驗證時使用當前模型)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.threshold = threshold
        self.pred_batch_size = pred_batch_size
        
        if model is not None:
            self.model = model
        else:
            self.model = TransUNet().to(self.device)
            if model_path is None:
                raise ValueError("必須提供 model_path 或 model 參數")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"模型加載失敗: {str(e)}")
        self.model.eval()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = 100

    def predict_single_return(self, image_path):
        """
        預測單張圖像，返回預測後的二值 mask
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)
        pad = self.patch_size // 2
        
        image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_h, padded_w = image_padded.shape[:2]
        stride = self.patch_size // 2
        
        patch_coords = []
        patches = []
        for y in range(0, padded_h - self.patch_size + 1, stride):
            for x in range(0, padded_w - self.patch_size + 1, stride):
                patch = image_padded[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(self._preprocess(patch))
                patch_coords.append((y - pad, x - pad))
        
        predictions = []
        total = len(patches)
        for i in tqdm(range(0, total, self.pred_batch_size), desc="Batch Predict"):
            batch = torch.cat(patches[i:i+self.pred_batch_size], dim=0).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend([probs[j] for j in range(probs.shape[0])])
        
        for i, (y_orig, x_orig) in enumerate(patch_coords):
            y1 = max(y_orig, 0)
            x1 = max(x_orig, 0)
            y2 = min(y_orig + self.patch_size, h)
            x2 = min(x_orig + self.patch_size, w)
            
            patch_pred = predictions[i][0, :, :]
            if y_orig < 0 or x_orig < 0 or (y_orig + self.patch_size) > h or (x_orig + self.patch_size) > w:
                valid_patch = cv2.resize(patch_pred, (x2 - x1, y2 - y1))
            else:
                valid_patch = patch_pred
            
            full_mask[y1:y2, x1:x2] += valid_patch
            count_mask[y1:y2, x1:x2] += 1
        
        count_mask[count_mask == 0] = 1
        avg_mask = full_mask / count_mask
        
        bin_mask = (avg_mask > self.threshold).astype(np.uint8) * 255
        bin_mask = self._final_postprocess(bin_mask)
        return bin_mask

    def predict_single(self, image_path, output_path):
        bin_mask = self.predict_single_return(image_path)
        cv2.imwrite(output_path, bin_mask)
        print(f"預測結果已保存至: {output_path}")

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _final_postprocess(self, mask):
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        return mask

    def predict_batch(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
        if not file_list:
            print(f"目錄 {input_dir} 中沒有支持的圖像文件")
            return
        for filename in file_list:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"pred_{filename}")
            try:
                print(f"\n正在處理: {filename}")
                self.predict_single(input_path, output_path)
            except Exception as e:
                print(f"處理 {filename} 失敗: {str(e)}")

if __name__ == '__main__':
    try:
        predictor = SievePredictor(
            model_path='outputs/models/best_model.pth',  # 調整成你最佳模型的路徑
            patch_size=400,
            threshold=0.4,
            pred_batch_size=16
        )
        predictor.predict_batch(
            input_dir='data/test/images',
            output_dir='outputs/predictions'
        )
    except Exception as e:
        print(f"運行時錯誤: {str(e)}")

