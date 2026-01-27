import cv2
import torch
import numpy as np
import os
from model import TransUNet

class SievePredictor:
    def __init__(self, model_path, patch_size=400):
        """
        初始化预测器
        :param model_path: 模型权重路径
        :param patch_size: 输入 patch 尺寸（默认 400x400）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransUNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # 修复 map_location
        self.model.eval()
        self.patch_size = patch_size
    
    def preprocess(self, image):
        """
        图像预处理
        :param image: 输入图像（NumPy 数组）
        :return: 预处理后的 Tensor
        """
        # 调整尺寸并归一化
        image = cv2.resize(image, (self.patch_size, self.patch_size))
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    def postprocess(self, logits):
        """
        后处理：将模型输出转换为二值掩码
        :param logits: 模型输出（未经过 Sigmoid）
        :return: 二值掩码图像
        """
        # 将 logits 转换为概率
        prob = torch.sigmoid(logits).cpu().numpy()
        
        # 二值化
        mask = (prob > 0.5).astype(np.uint8) * 255
        return mask

    def predict_patch(self, image):
        """
        预测单个 patch
        :param image: 输入 patch（NumPy 数组）
        :return: 预测的二值掩码 patch
        """
        # 预处理
        input_tensor = self.preprocess(image).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            logits = self.model(input_tensor)
        
        # 后处理
        return self.postprocess(logits.squeeze())

    def predict_large_image(self, image_path, output_path, stride=400):
        """
        预测大图
        :param image_path: 输入大图路径
        :param output_path: 输出掩码路径
        :param stride: 滑动窗口步长（默认 400）
        """
        # 加载大图
        large_image = cv2.imread(image_path)
        if large_image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
    
        # 获取大图尺寸
        h, w = large_image.shape[:2]
    
        # 初始化全图掩码
        full_mask = np.zeros((h, w), dtype=np.uint8)
    
        # 滑动窗口预测
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 提取 patch
                patch = large_image[y:y+self.patch_size, x:x+self.patch_size]
                
               # 获取 patch 实际大小
                h_patch, w_patch = patch.shape[:2]  
                
                # 如果 patch 不足 400x400，填充到 400x400
                if h_patch < self.patch_size or w_patch < self.patch_size:
                    patch = cv2.copyMakeBorder(
                        patch,
                        0, self.patch_size - h_patch,
                        0, self.patch_size - w_patch,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                
                # 预测 patch
                mask_patch = self.predict_patch(patch)
    
                # **修正這裡：只填充原本的 patch 尺寸，避免超出邊界**
                full_mask[y:y+h_patch, x:x+w_patch] = mask_patch[:h_patch, :w_patch]
    
        # 保存结果
        cv2.imwrite(output_path, full_mask)
        print(f"Saved prediction: {output_path}")
    

    def predict_folder(self, input_folder, output_folder, stride=400):
        """
        批量预测文件夹中的所有图片
        :param input_folder: 输入图片文件夹路径
        :param output_folder: 输出掩码文件夹路径
        :param stride: 滑动窗口步长（默认 400）
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 遍历输入文件夹中的所有图片
        for img_name in os.listdir(input_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 输入图片路径
                img_path = os.path.join(input_folder, img_name)
                
                # 输出掩码路径
                output_path = os.path.join(output_folder, f"pred_{img_name}")
                
                # 预测大图
                try:
                    self.predict_large_image(img_path, output_path, stride)
                except Exception as e:
                    print(f"预测失败: {img_path}, 错误: {e}")

if __name__ == '__main__':
    # 初始化预测器
    predictor = SievePredictor('outputs/models/sievenet_epoch100.pth', patch_size=400)
    
    # 批量预测文件夹中的所有图片
    predictor.predict_folder(
        input_folder='data/test/images',  # 输入图片文件夹路径
        output_folder='outputs/predictions'  # 输出掩码文件夹路径
    )