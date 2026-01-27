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
        初始化预测器
        :param model_path: 模型权重路径
        :param patch_size: 输入patch尺寸
        :param threshold: 二值化阈值(0-1)
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
            raise RuntimeError(f"模型加载失败: {str(e)}")

        # 后处理参数
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.min_area = 100  # 最小区域面积阈值

    def predict_single(self, image_path, output_path):
        """
        预测单张图像
        :param image_path: 输入图像路径
        :param output_path: 输出掩码路径
        """
        # 读取图像并验证
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 初始化参数
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        pad_size = self.patch_size // 2
        
        # 图像填充处理
        image_padded = cv2.copyMakeBorder(
            image, 
            pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_CONSTANT, value=[0,0,0]
        )
        
        # 计算滑动窗口参数
        max_y = image_padded.shape[0] - self.patch_size
        max_x = image_padded.shape[1] - self.patch_size
        stride = self.patch_size // 2
        total_patches = ((max_y // stride) + 1) * ((max_x // stride) + 1)
        
        # 进度条设置
        progress_bar = tqdm(
            total=total_patches, 
            desc=f"处理 {os.path.basename(image_path)}",
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        
        # 滑动窗口预测
        for y in range(0, max_y + 1, stride):
            for x in range(0, max_x + 1, stride):
                # 提取patch
                patch = image_padded[y:y+self.patch_size, x:x+self.patch_size]
                
                # 预处理并预测
                tensor = self._preprocess(patch).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    prob = torch.sigmoid(logits).cpu().numpy().squeeze()
                
                # 后处理
                mask = self._postprocess(prob)
                
                # 坐标转换
                y_orig = y - pad_size
                x_orig = x - pad_size
                
                # 计算有效区域
                y_start = max(y_orig, 0)
                x_start = max(x_orig, 0)
                y_end = min(y_orig + self.patch_size, h)
                x_end = min(x_orig + self.patch_size, w)
                
                # 调整mask区域
                mask_height = y_end - y_start
                mask_width = x_end - x_start
                if mask_height <=0 or mask_width <=0:
                    continue
                
                valid_mask = cv2.resize(mask, (mask_width, mask_height))
                
                # 融合到完整掩码
                if valid_mask.size > 0:
                    try:
                        # 使用加权平均融合重叠预测
                        full_mask[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                            full_mask[y_start:y_end, x_start:x_end], 0.5,
                            valid_mask, 0.5, 0
                        )
                    except cv2.error as e:
                        print(f"区域融合错误 @ [{y_start}:{y_end}, {x_start}:{x_end}]")
                        print(f"Full mask shape: {full_mask.shape}")
                        print(f"Valid mask shape: {valid_mask.shape}")
                        raise
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        # 最终后处理
        full_mask = self._final_postprocess(full_mask)
        cv2.imwrite(output_path, full_mask)
        print(f"预测结果已保存至: {output_path}")

    def _preprocess(self, image):
        """图像预处理"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

    def _postprocess(self, prob):
        """单个patch后处理"""
        # 二值化
        mask = (prob > self.threshold).astype(np.uint8) * 255
        
        # 形态学操作
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        return mask

    def _final_postprocess(self, mask):
        """最终后处理"""
        # 中值滤波去噪
        mask = cv2.medianBlur(mask, 5)
        
        # 移除小区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        
        return mask

    def predict_batch(self, input_dir, output_dir):
        """
        批量预测
        :param input_dir: 输入图片目录
        :param output_dir: 输出掩码目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件列表
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
        
        if not file_list:
            print(f"目录 {input_dir} 中没有支持的图像文件")
            return
        
        # 批量处理
        for filename in file_list:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"pred_{filename}")
            
            try:
                print(f"\n正在处理: {filename}")
                self.predict_single(input_path, output_path)
            except Exception as e:
                print(f"处理 {filename} 失败: {str(e)}")

if __name__ == '__main__':
    try:
        predictor = SievePredictor(
            model_path='outputs/models/best_model.pth',
            patch_size=400,
            threshold=0.4
        )
        predictor.predict_batch(
            input_dir='data/test/images',
            output_dir='outputs/predictions'
        )
    except Exception as e:
        print(f"运行时错误: {str(e)}")