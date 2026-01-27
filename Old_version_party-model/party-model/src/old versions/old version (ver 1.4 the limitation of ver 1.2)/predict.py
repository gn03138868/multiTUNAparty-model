# ==== src/predict.py ====
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from model import TransUNet

class SievePredictor:
    def __init__(self, model_path, patch_size=400, threshold=0.4):
        """
        初始化預測器
        :param model_path: 模型權重路徑
        :param patch_size: 輸入patch尺寸
        :param threshold: 二值化閾值(0-1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.threshold = threshold
        
        # 初始化模型
        self.model = TransUNet().to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"模型加載失敗: {str(e)}")

        # 後處理參數
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = 100  # 最小區域面積閾值

    def predict_single(self, image_path, output_path):
        """
        預測單張圖像
        :param image_path: 輸入圖像路徑
        :param output_path: 輸出掩碼路徑
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
        
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)
        pad = self.patch_size // 2
        
        # 為避免邊界問題進行padding
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
        
        # 將patches組成batch進行推論
        batch_tensor = torch.cat(patches, dim=0).to(self.device)  # shape: [N, C, H, W]
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # 將每個patch的預測融合到全圖（累加再除以次數）
        for i, (y_orig, x_orig) in enumerate(patch_coords):
            # 計算在原圖中的對應區域
            y1 = max(y_orig, 0)
            x1 = max(x_orig, 0)
            y2 = min(y_orig + self.patch_size, h)
            x2 = min(x_orig + self.patch_size, w)
            
            # 取當前patch預測並resize至有效尺寸
            patch_pred = probs[i, 0, :, :]
            if y_orig < 0 or x_orig < 0 or (y_orig + self.patch_size) > h or (x_orig + self.patch_size) > w:
                # 需要resize裁剪區域
                valid_patch = cv2.resize(patch_pred, (x2 - x1, y2 - y1))
            else:
                valid_patch = patch_pred
            
            full_mask[y1:y2, x1:x2] += valid_patch
            count_mask[y1:y2, x1:x2] += 1
        
        # 避免除以零
        count_mask[count_mask == 0] = 1
        avg_mask = full_mask / count_mask
        
        # 二值化處理
        bin_mask = (avg_mask > self.threshold).astype(np.uint8) * 255
        bin_mask = self._final_postprocess(bin_mask)
        
        cv2.imwrite(output_path, bin_mask)
        print(f"預測結果已保存至: {output_path}")

    def _preprocess(self, image):
        """圖像預處理，與訓練一致"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _final_postprocess(self, mask):
        """最終後處理"""
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        return mask

    def predict_batch(self, input_dir, output_dir):
        """
        批量預測
        :param input_dir: 輸入圖像目錄
        :param output_dir: 輸出掩碼目錄
        """
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
            model_path='outputs/models/sievenet_epoch1000.pth',  # 請根據實際情況調整模型路徑
            patch_size=400,
            threshold=0.4
        )
        predictor.predict_batch(
            input_dir='data/test/images',
            output_dir='outputs/predictions'
        )
    except Exception as e:
        print(f"運行時錯誤: {str(e)}")

