"""
Multi-Task TransUNet - Desktop GUI (Tkinter) - FIXED VERSION
修復大圖片輸出問題：使用滑動窗口方法處理大圖

修改內容：
1. 使用滑動窗口處理大圖，而非直接縮放
2. 輸出保持原始大小
3. 支援任意尺寸的輸入圖片
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import queue
import os
from pathlib import Path
import torch
import cv2


# ============================================================================
# Windows Unicode 路徑相容的圖片讀寫
# cv2.imread / cv2.imwrite 在 Windows 上無法處理含中文或空白的路徑
# ============================================================================

def imread_unicode(filepath):
    """讀取圖片（支援 Unicode 路徑，如中文、空白）"""
    try:
        # 先嘗試標準方式
        img = cv2.imread(str(filepath))
        if img is not None:
            return img
    except Exception:
        pass
    # Fallback: numpy fromfile → imdecode
    try:
        buf = np.fromfile(str(filepath), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(filepath, img, params=None):
    """寫入圖片（支援 Unicode 路徑）"""
    try:
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()
        if not ext:
            ext = '.png'
        encode_params = params or []
        success, buf = cv2.imencode(ext, img, encode_params)
        if success:
            buf.tofile(filepath)
            return True
    except Exception:
        pass
    # Fallback
    try:
        cv2.imwrite(filepath, img)
        return True
    except Exception:
        return False
import numpy as np
import json
import yaml
import math
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import sys

# Import model - 支援兩種版本
MODEL_VERSION = None
try:
    from model_multitask_boundaryversion import MultiTaskTransUNet
    MODEL_AVAILABLE = True
    MODEL_VERSION = "boundary"
    print("✓ Using boundary-aware model version")
except ImportError:
    try:
        from model_multitask import MultiTaskTransUNet
        MODEL_AVAILABLE = True
        MODEL_VERSION = "original"
        print("✓ Using original model version")
    except ImportError:
        MODEL_AVAILABLE = False
        print("Warning: No model file found")

# Global configuration
TASK_MAPPING = {
    'Cell (Plant Cell)': 0,
    'Blood (Blood Cell)': 1,
    'Other (Other System)': 2
}

TASK_COLORS = {
    0: 'Blues',
    1: 'Reds',
    2: 'Greens'
}

# Global variables
loaded_model = None
model_device = None


# ============================================================================
# 滑動窗口推理函數 (核心修復)
# ============================================================================

def sliding_window_inference(model, image, task_id, device, patch_size=400, overlap=0.25):
    """
    使用滑動窗口方法處理大圖片
    
    Args:
        model: 訓練好的模型
        image: 輸入圖片 (H, W, 3) RGB格式
        task_id: 任務ID
        device: 計算設備
        patch_size: patch大小 (預設400)
        overlap: 重疊比例 (預設25%)
    
    Returns:
        prob_map: 概率圖 (H, W)，與原圖大小相同
    """
    H, W = image.shape[:2]
    
    # 如果圖片小於 patch_size，直接處理
    if H <= patch_size and W <= patch_size:
        # Pad to patch_size
        padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        padded[:H, :W] = image
        
        img_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor, task_id=task_id)
            if isinstance(output, dict):
                seg_output = output.get('refined', output.get('seg', output.get('main')))
            else:
                seg_output = output
            prob = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
        
        return prob[:H, :W]
    
    # 計算步長
    stride = int(patch_size * (1 - overlap))
    
    # 計算需要的 patches 數量
    n_h = max(1, int(np.ceil((H - patch_size) / stride)) + 1)
    n_w = max(1, int(np.ceil((W - patch_size) / stride)) + 1)
    
    # 初始化累積數組
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    # 創建權重遮罩 (中心權重高，邊緣權重低)
    weight = create_weight_mask(patch_size)
    
    total_patches = n_h * n_w
    processed = 0
    
    for i in range(n_h):
        for j in range(n_w):
            # 計算 patch 位置
            y_start = min(i * stride, H - patch_size)
            x_start = min(j * stride, W - patch_size)
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # 提取 patch
            patch = image[y_start:y_end, x_start:x_end]
            
            # 預處理
            img_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 推理
            with torch.no_grad():
                output = model(img_tensor, task_id=task_id)
                if isinstance(output, dict):
                    seg_output = output.get('refined', output.get('seg', output.get('main')))
                else:
                    seg_output = output
                prob = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
            
            # 加權累積
            prob_sum[y_start:y_end, x_start:x_end] += prob * weight
            count_map[y_start:y_end, x_start:x_end] += weight
            
            processed += 1
    
    # 計算平均
    prob_map = prob_sum / (count_map + 1e-7)
    
    return prob_map


def create_weight_mask(size):
    """
    創建權重遮罩，中心權重高，邊緣權重低
    使用高斯權重避免拼接邊界
    """
    sigma = size / 4
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    center = size / 2
    weight = np.exp(-((X - center)**2 + (Y - center)**2) / (2 * sigma**2))
    
    # 確保邊緣不是完全為0
    weight = weight + 0.1
    weight = weight / weight.max()
    
    return weight.astype(np.float32)


# ============================================================================
# 形態學分析函數 (Morphological Analysis)
# ============================================================================

def analyze_morphology(binary_mask, pixel_size=1.0, min_area_pixels=10):
    """
    對二值化 mask 進行形態學分析，計算每個物件的面積與直徑
    
    Args:
        binary_mask: 二值化遮罩 (H, W)，值為 0 或 255
        pixel_size: 每像素代表的實際長度（微米/像素），預設 1.0 表示以像素為單位
        min_area_pixels: 最小物件面積（像素），低於此值視為雜訊忽略
    
    Returns:
        dict: {
            'total_area': 總面積,
            'num_objects': 物件數量,
            'objects': [{'id', 'area', 'diameter', 'centroid', 'bbox', 'perimeter', 'circularity'}, ...],
            'unit': 單位字串,
            'labeled_image': 標記圖 (H, W) 每個物件有不同標號
        }
    """
    # 確保是 uint8
    if binary_mask.dtype != np.uint8:
        mask = (binary_mask > 127).astype(np.uint8) * 255
    else:
        mask = binary_mask.copy()
    
    # 連通區域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    unit = "px" if pixel_size == 1.0 else "μm"
    area_unit = "px²" if pixel_size == 1.0 else "μm²"
    
    objects = []
    total_area_pixels = 0
    
    for i in range(1, num_labels):  # 跳過背景 (label=0)
        area_px = stats[i, cv2.CC_STAT_AREA]
        
        # 過濾太小的物件（雜訊）
        if area_px < min_area_pixels:
            continue
        
        # 計算面積（考慮 pixel_size）
        area_real = area_px * (pixel_size ** 2)
        
        # 等效圓直徑: d = sqrt(4 * A / pi)
        diameter_real = math.sqrt(4.0 * area_real / math.pi)
        
        # 取得該物件的 contour 計算周長與圓度
        obj_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        perimeter_px = 0
        circularity = 0
        if contours:
            perimeter_px = cv2.arcLength(contours[0], True)
            perimeter_real = perimeter_px * pixel_size
            if perimeter_real > 0:
                circularity = (4.0 * math.pi * area_real) / (perimeter_real ** 2)
                circularity = min(circularity, 1.0)  # 上限為 1
        else:
            perimeter_real = 0
        
        # Bounding box
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        objects.append({
            'id': len(objects) + 1,
            'label': i,
            'area_px': area_px,
            'area': area_real,
            'diameter': diameter_real,
            'perimeter': perimeter_real,
            'circularity': circularity,
            'centroid': (centroids[i][0], centroids[i][1]),
            'bbox': (x, y, w, h),
        })
        
        total_area_pixels += area_px
    
    total_area_real = total_area_pixels * (pixel_size ** 2)
    
    # 統計摘要
    if objects:
        areas = [o['area'] for o in objects]
        diameters = [o['diameter'] for o in objects]
        summary = {
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'median_area': np.median(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
            'mean_diameter': np.mean(diameters),
            'std_diameter': np.std(diameters),
            'median_diameter': np.median(diameters),
            'min_diameter': np.min(diameters),
            'max_diameter': np.max(diameters),
        }
    else:
        summary = {k: 0 for k in [
            'mean_area', 'std_area', 'median_area', 'min_area', 'max_area',
            'mean_diameter', 'std_diameter', 'median_diameter', 'min_diameter', 'max_diameter'
        ]}
    
    return {
        'total_area': total_area_real,
        'total_area_px': total_area_pixels,
        'num_objects': len(objects),
        'objects': objects,
        'summary': summary,
        'unit': unit,
        'area_unit': area_unit,
        'pixel_size': pixel_size,
        'labeled_image': labels,
    }


def create_analysis_visualization(image, binary_mask, analysis_result, task_name=""):
    """
    建立形態學分析的視覺化圖像
    
    Args:
        image: 原始 RGB 圖 (H, W, 3)
        binary_mask: 二值化遮罩 (H, W)
        analysis_result: analyze_morphology 的回傳結果
        task_name: 任務名稱
    
    Returns:
        fig: matplotlib figure
    """
    objects = analysis_result['objects']
    labels = analysis_result['labeled_image']
    unit = analysis_result['unit']
    area_unit = analysis_result['area_unit']
    summary = analysis_result['summary']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- (0,0) 原圖 + 物件輪廓 + 編號 ---
    overlay = image.copy()
    for obj in objects:
        obj_mask = (labels == obj['label']).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
        cx, cy = int(obj['centroid'][0]), int(obj['centroid'][1])
        cv2.putText(overlay, str(obj['id']), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(f'{task_name} Labeled Objects ({analysis_result["num_objects"]} detected)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # --- (0,1) 面積色彩圖 ---
    area_map = np.zeros(labels.shape, dtype=np.float32)
    for obj in objects:
        area_map[labels == obj['label']] = obj['area']
    
    if objects:
        im = axes[0, 1].imshow(np.ma.masked_where(area_map == 0, area_map),
                                cmap='plasma', interpolation='nearest')
        cbar = plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.set_label(f'Area ({area_unit})', rotation=270, labelpad=15)
    axes[0, 1].set_title(f'Area Heatmap ({area_unit})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # --- (1,0) 面積直方圖 ---
    if objects:
        areas = [o['area'] for o in objects]
        n_bins = min(30, max(5, len(areas) // 3))
        axes[1, 0].hist(areas, bins=n_bins, color='#2196F3', edgecolor='white', alpha=0.85)
        axes[1, 0].axvline(summary['mean_area'], color='red', linestyle='--', linewidth=1.5,
                           label=f'Mean: {summary["mean_area"]:.1f}')
        axes[1, 0].axvline(summary['median_area'], color='orange', linestyle='--', linewidth=1.5,
                           label=f'Median: {summary["median_area"]:.1f}')
        axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xlabel(f'Area ({area_unit})', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Area Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # --- (1,1) 統計摘要表 ---
    axes[1, 1].axis('off')
    
    stats_text = (
        f"{'═' * 42}\n"
        f"  MORPHOLOGICAL ANALYSIS SUMMARY\n"
        f"{'═' * 42}\n\n"
        f"  Task:              {task_name}\n"
        f"  Scale:             1 px = {analysis_result['pixel_size']:.4f} {unit}\n"
        f"  Objects detected:  {analysis_result['num_objects']}\n\n"
        f"{'─' * 42}\n"
        f"  AREA ({area_unit})\n"
        f"{'─' * 42}\n"
        f"  Total:    {analysis_result['total_area']:>14.2f}\n"
        f"  Mean:     {summary['mean_area']:>14.2f}\n"
        f"  Std:      {summary['std_area']:>14.2f}\n"
        f"  Median:   {summary['median_area']:>14.2f}\n"
        f"  Min:      {summary['min_area']:>14.2f}\n"
        f"  Max:      {summary['max_area']:>14.2f}\n\n"
        f"{'─' * 42}\n"
        f"  EQUIV. DIAMETER ({unit})\n"
        f"{'─' * 42}\n"
        f"  Mean:     {summary['mean_diameter']:>14.2f}\n"
        f"  Std:      {summary['std_diameter']:>14.2f}\n"
        f"  Median:   {summary['median_diameter']:>14.2f}\n"
        f"  Min:      {summary['min_diameter']:>14.2f}\n"
        f"  Max:      {summary['max_diameter']:>14.2f}\n"
        f"{'═' * 42}"
    )
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle(f'Morphological Analysis — {task_name}', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def export_analysis_csv(analysis_result, csv_path, image_name="", task_name=""):
    """
    將形態學分析結果匯出為 CSV
    """
    unit = analysis_result['unit']
    area_unit = analysis_result['area_unit']
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 摘要區
        writer.writerow(['=== Summary ==='])
        writer.writerow(['Image', image_name])
        writer.writerow(['Task', task_name])
        writer.writerow(['Scale (unit/px)', analysis_result['pixel_size']])
        writer.writerow(['Unit', unit])
        writer.writerow(['Total Objects', analysis_result['num_objects']])
        writer.writerow([f'Total Area ({area_unit})', f"{analysis_result['total_area']:.4f}"])
        writer.writerow([f'Mean Area ({area_unit})', f"{analysis_result['summary']['mean_area']:.4f}"])
        writer.writerow([f'Mean Diameter ({unit})', f"{analysis_result['summary']['mean_diameter']:.4f}"])
        writer.writerow([])
        
        # 逐物件表
        writer.writerow(['=== Individual Objects ==='])
        writer.writerow([
            'Object_ID',
            f'Area ({area_unit})',
            'Area (px)',
            f'Equiv_Diameter ({unit})',
            f'Perimeter ({unit})',
            'Circularity',
            'Centroid_X (px)',
            'Centroid_Y (px)',
            'BBox_X', 'BBox_Y', 'BBox_W', 'BBox_H'
        ])
        
        for obj in analysis_result['objects']:
            writer.writerow([
                obj['id'],
                f"{obj['area']:.4f}",
                obj['area_px'],
                f"{obj['diameter']:.4f}",
                f"{obj['perimeter']:.4f}",
                f"{obj['circularity']:.4f}",
                f"{obj['centroid'][0]:.1f}",
                f"{obj['centroid'][1]:.1f}",
                obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]
            ])


# ============================================================================
# GUI 類
# ============================================================================

class MultiTaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Task TransUNet - Desktop Edition (Fixed + Morphological Analysis)")
        self.root.geometry("1200x800")
        
        # Create status bar
        self.create_status_bar()
        
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create feature tabs
        self.create_model_tab()
        self.create_predict_tab()
        self.create_analysis_tab()
        self.create_batch_tab()
        self.create_help_tab()
        
        # Analysis state
        self.current_analysis = None
        
        # Update status
        self.update_status("Ready - Please load a model first")
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    # ========================================================================
    # Tab 1: Model Management
    # ========================================================================
    
    def create_model_tab(self):
        """Create model management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📦 Model Management")
        
        # Title
        title = tk.Label(tab, text="Model Loading and Management", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Model selection area
        model_frame = ttk.LabelFrame(tab, text="Select Model", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        # Model path
        path_frame = tk.Frame(model_frame)
        path_frame.pack(fill='x', pady=5)
        
        tk.Label(path_frame, text="Model Path:").pack(side='left', padx=5)
        self.model_path_var = tk.StringVar(value="outputs/models/best_model.pth")
        model_entry = tk.Entry(path_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        tk.Button(
            path_frame, 
            text="Browse...", 
            command=self.browse_model
        ).pack(side='left', padx=5)
        
        # Device selection
        device_frame = tk.Frame(model_frame)
        device_frame.pack(fill='x', pady=5)
        
        tk.Label(device_frame, text="Computing Device:").pack(side='left', padx=5)
        self.device_var = tk.StringVar(
            value="GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        )
        
        ttk.Radiobutton(
            device_frame, 
            text="GPU (CUDA)", 
            variable=self.device_var, 
            value="GPU (CUDA)"
        ).pack(side='left', padx=10)
        
        ttk.Radiobutton(
            device_frame, 
            text="CPU", 
            variable=self.device_var, 
            value="CPU"
        ).pack(side='left', padx=10)
        
        # GPU information
        if torch.cuda.is_available():
            gpu_info = f"✅ GPU Available: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "⚠️ GPU not available, CPU will be used (slower)"
        
        tk.Label(device_frame, text=gpu_info, fg='green' if torch.cuda.is_available() else 'orange').pack(side='left', padx=10)
        
        # Load button
        tk.Button(
            model_frame, 
            text="📥 Load Model", 
            command=self.load_model,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # Model information display
        info_frame = ttk.LabelFrame(tab, text="Model Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame, 
            height=15, 
            wrap=tk.WORD
        )
        self.model_info_text.pack(fill='both', expand=True)
        self.model_info_text.insert('1.0', "No model loaded\n\nPlease select a model file and click 'Load Model'")
    
    def browse_model(self):
        """Browse and select model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """Load model"""
        global loaded_model, model_device
        
        if not MODEL_AVAILABLE:
            messagebox.showerror("Error", "Cannot find model_multitask.py file!")
            return
        
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file does not exist:\n{model_path}")
            return
        
        self.update_status("Loading model...")
        self.model_info_text.delete('1.0', tk.END)
        self.model_info_text.insert('1.0', "Loading model, please wait...\n")
        
        def load_thread():
            global loaded_model, model_device
            try:
                # Set device
                if self.device_var.get() == "GPU (CUDA)" and torch.cuda.is_available():
                    device = torch.device('cuda')
                    device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                else:
                    device = torch.device('cpu')
                    device_info = "CPU"
                
                # Create model
                if MODEL_VERSION == "boundary":
                    model = MultiTaskTransUNet(
                        img_size=400,
                        patch_size=16,
                        num_decoder_layers=80,
                        num_tasks=3,
                        use_deep_supervision=True
                    )
                else:
                    model = MultiTaskTransUNet(
                        img_size=400,
                        patch_size=16,
                        num_decoder_layers=80,
                        num_tasks=3
                    )
                
                # Load weights
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()
                
                loaded_model = model
                model_device = device
                
                # Update info
                param_count = sum(p.numel() for p in model.parameters())
                info = f"""✅ Model loaded successfully!

Model File: {model_path}
Device: {device_info}
Model Version: {MODEL_VERSION}
Parameters: {param_count:,}

🔧 Fixed Version Features:
• Sliding window inference for large images
• Output size matches input size
• Supports images of any size
"""
                self.root.after(0, lambda: self.model_info_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.model_info_text.insert('1.0', info))
                self.root.after(0, lambda: self.update_status(f"Model loaded - {device_info}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model:\n{str(e)}"))
                self.root.after(0, lambda: self.update_status("Model loading failed"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 2: Single Prediction
    # ========================================================================
    
    def create_predict_tab(self):
        """Create single prediction tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Single Prediction")
        
        # Title
        title = tk.Label(tab, text="Single Image Prediction", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Settings
        settings_frame = ttk.LabelFrame(tab, text="Prediction Settings", padding=10)
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        # Image selection
        img_frame = tk.Frame(settings_frame)
        img_frame.pack(fill='x', pady=5)
        
        tk.Label(img_frame, text="Image:").pack(side='left', padx=5)
        self.predict_image_var = tk.StringVar()
        img_entry = tk.Entry(img_frame, textvariable=self.predict_image_var, width=50)
        img_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        tk.Button(img_frame, text="Browse...", command=self.browse_predict_image).pack(side='left', padx=5)
        
        # Task selection
        task_frame = tk.Frame(settings_frame)
        task_frame.pack(fill='x', pady=5)
        
        tk.Label(task_frame, text="Task:").pack(side='left', padx=5)
        self.predict_task_var = tk.StringVar(value="Cell (Plant Cell)")
        ttk.Combobox(
            task_frame, 
            textvariable=self.predict_task_var, 
            values=list(TASK_MAPPING.keys()),
            width=25
        ).pack(side='left', padx=5)
        
        # Threshold
        tk.Label(task_frame, text="Threshold:").pack(side='left', padx=20)
        self.threshold_var = tk.DoubleVar(value=0.5)
        tk.Scale(
            task_frame, 
            from_=0.0, to=1.0, 
            resolution=0.05,
            orient='horizontal', 
            variable=self.threshold_var,
            length=200
        ).pack(side='left', padx=5)
        
        # Overlap setting (new)
        tk.Label(task_frame, text="Overlap:").pack(side='left', padx=10)
        self.overlap_var = tk.DoubleVar(value=0.25)
        tk.Scale(
            task_frame, 
            from_=0.0, to=0.5, 
            resolution=0.05,
            orient='horizontal', 
            variable=self.overlap_var,
            length=100
        ).pack(side='left', padx=5)
        
        # Buttons
        btn_frame = tk.Frame(settings_frame)
        btn_frame.pack(fill='x', pady=10)
        
        tk.Button(
            btn_frame, 
            text="🚀 Start Prediction", 
            command=self.run_prediction,
            bg='#2196F3',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(side='left', padx=5)
        
        tk.Button(
            btn_frame, 
            text="💾 Save Result", 
            command=self.save_prediction,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10),
            padx=10,
            pady=5
        ).pack(side='left', padx=5)
        
        # Result display
        result_frame = ttk.LabelFrame(tab, text="Prediction Result", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.result_canvas = tk.Canvas(result_frame, bg='white', height=400)
        self.result_canvas.pack(fill='both', expand=True)
        
        # Image size label
        self.size_label = tk.Label(result_frame, text="")
        self.size_label.pack()
    
    def browse_predict_image(self):
        """Browse and select image"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if filename:
            self.predict_image_var.set(filename)
    
    def run_prediction(self):
        """Run single prediction with sliding window"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        image_path = self.predict_image_var.get()
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file does not exist:\n{image_path}")
            return
        
        self.update_status("Running prediction with sliding window...")
        
        def predict_thread():
            try:
                # Load image (keep original size)
                image = imread_unicode(image_path)
                if image is None:
                    raise ValueError(f"Cannot read image (check path for special characters):\n{image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                H, W = image.shape[:2]
                
                # Get parameters
                task_id = TASK_MAPPING[self.predict_task_var.get()]
                threshold = self.threshold_var.get()
                overlap = self.overlap_var.get()
                
                # Sliding window inference (FIXED!)
                prob_map = sliding_window_inference(
                    loaded_model, image, task_id, model_device,
                    patch_size=400, overlap=overlap
                )
                
                # Apply threshold
                binary_mask = (prob_map > threshold).astype(np.uint8) * 255
                
                # Store results (original size!)
                self.current_prediction = {
                    'original': image,
                    'prob_map': prob_map,
                    'binary': binary_mask,
                    'threshold': threshold,
                    'size': (H, W)
                }
                
                # Update display
                self.root.after(0, self.display_prediction)
                self.root.after(0, lambda: self.update_status(f"Prediction complete - Output size: {W}x{H}"))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Prediction failed"))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def display_prediction(self):
        """Display prediction results"""
        if not hasattr(self, 'current_prediction'):
            return
        
        pred = self.current_prediction
        H, W = pred['size']
        
        # Update size label
        self.size_label.config(text=f"Output size: {W} x {H} (same as input)")
        
        # Create combined visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(pred['original'])
        axes[0].set_title(f'Original Image ({W}x{H})')
        axes[0].axis('off')
        
        axes[1].imshow(pred['prob_map'], cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        axes[2].imshow(pred['binary'], cmap='gray')
        axes[2].set_title(f"Binary Mask (Threshold: {pred['threshold']:.2f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = Path('outputs/temp_prediction.png')
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Display
        img = Image.open(temp_path)
        img.thumbnail((1000, 500))
        photo = ImageTk.PhotoImage(img)
        
        self.result_canvas.delete('all')
        self.result_canvas.create_image(500, 250, image=photo)
        self.result_canvas.image = photo
    
    def save_prediction(self):
        """Save prediction result at original resolution"""
        if not hasattr(self, 'current_prediction'):
            messagebox.showerror("Error", "No prediction result to save!")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return
        
        pred = self.current_prediction
        base_name = Path(self.predict_image_var.get()).stem
        
        try:
            # Save original (at original size)
            imwrite_unicode(
                os.path.join(save_dir, f'{base_name}_original.png'),
                cv2.cvtColor(pred['original'], cv2.COLOR_RGB2BGR)
            )
            
            # Save heatmap (at original size)
            plt.figure(figsize=(pred['size'][1]/100, pred['size'][0]/100), dpi=100)
            plt.imshow(pred['prob_map'], cmap='jet', vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(save_dir, f'{base_name}_heatmap.png'), 
                       dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save binary mask (at original size)
            imwrite_unicode(
                os.path.join(save_dir, f'{base_name}_binary.png'),
                pred['binary']
            )
            
            # Save overlay (at original size)
            overlay = pred['original'].copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[pred['binary'] > 0] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            imwrite_unicode(
                os.path.join(save_dir, f'{base_name}_overlay.png'),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
            
            messagebox.showinfo("Success", 
                f"Results saved to:\n{save_dir}\n\n"
                f"Files:\n"
                f"• {base_name}_original.png\n"
                f"• {base_name}_heatmap.png\n"
                f"• {base_name}_binary.png\n"
                f"• {base_name}_overlay.png\n\n"
                f"Size: {pred['size'][1]} x {pred['size'][0]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    # ========================================================================
    # Tab 3: Morphological Analysis
    # ========================================================================
    
    def create_analysis_tab(self):
        """Create morphological analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Morphological Analysis")
        
        # Title
        title = tk.Label(tab, text="Morphological Analysis — Area & Diameter", font=('Arial', 16, 'bold'))
        title.pack(pady=5)
        
        # ---- Top control panel ----
        ctrl_frame = ttk.LabelFrame(tab, text="Analysis Settings", padding=8)
        ctrl_frame.pack(fill='x', padx=10, pady=3)
        
        # Row 1: Image & task
        row1 = tk.Frame(ctrl_frame)
        row1.pack(fill='x', pady=3)
        
        tk.Label(row1, text="Image:").pack(side='left', padx=5)
        self.analysis_image_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.analysis_image_var, width=45).pack(side='left', padx=3, fill='x', expand=True)
        tk.Button(row1, text="Browse...", command=self.browse_analysis_image).pack(side='left', padx=3)
        
        tk.Label(row1, text="Task:").pack(side='left', padx=(15, 3))
        self.analysis_task_var = tk.StringVar(value="Cell (Plant Cell)")
        ttk.Combobox(row1, textvariable=self.analysis_task_var,
                     values=list(TASK_MAPPING.keys()), width=20).pack(side='left', padx=3)
        
        # Row 2: Scale calibration & thresholds
        row2 = tk.Frame(ctrl_frame)
        row2.pack(fill='x', pady=3)
        
        tk.Label(row2, text="Pixel Size:").pack(side='left', padx=5)
        self.pixel_size_var = tk.DoubleVar(value=1.0)
        tk.Entry(row2, textvariable=self.pixel_size_var, width=8).pack(side='left', padx=3)
        tk.Label(row2, text="μm/px  (1.0 = pixel units)").pack(side='left', padx=3)
        
        tk.Label(row2, text="Threshold:").pack(side='left', padx=(20, 3))
        self.analysis_threshold_var = tk.DoubleVar(value=0.5)
        tk.Scale(row2, from_=0.0, to=1.0, resolution=0.05, orient='horizontal',
                 variable=self.analysis_threshold_var, length=120).pack(side='left', padx=3)
        
        tk.Label(row2, text="Min Area (px):").pack(side='left', padx=(15, 3))
        self.min_area_var = tk.IntVar(value=10)
        tk.Entry(row2, textvariable=self.min_area_var, width=6).pack(side='left', padx=3)
        
        tk.Label(row2, text="Overlap:").pack(side='left', padx=(15, 3))
        self.analysis_overlap_var = tk.DoubleVar(value=0.25)
        tk.Scale(row2, from_=0.0, to=0.5, resolution=0.05, orient='horizontal',
                 variable=self.analysis_overlap_var, length=80).pack(side='left', padx=3)
        
        # Row 3: Buttons
        row3 = tk.Frame(ctrl_frame)
        row3.pack(fill='x', pady=5)
        
        tk.Button(
            row3, text="🔬 Run Analysis", command=self.run_analysis,
            bg='#9C27B0', fg='white', font=('Arial', 11, 'bold'), padx=15, pady=6
        ).pack(side='left', padx=5)
        
        tk.Button(
            row3, text="📋 Use Current Prediction", command=self.analysis_from_prediction,
            bg='#607D8B', fg='white', font=('Arial', 10), padx=10, pady=4
        ).pack(side='left', padx=5)
        
        tk.Button(
            row3, text="💾 Export CSV", command=self.export_analysis,
            bg='#FF9800', fg='white', font=('Arial', 10), padx=10, pady=4
        ).pack(side='left', padx=5)
        
        tk.Button(
            row3, text="💾 Save Figure", command=self.save_analysis_figure,
            bg='#4CAF50', fg='white', font=('Arial', 10), padx=10, pady=4
        ).pack(side='left', padx=5)
        
        # ---- Main content: left = visualisation, right = table ----
        content = tk.PanedWindow(tab, orient=tk.HORIZONTAL, sashwidth=6)
        content.pack(fill='both', expand=True, padx=10, pady=3)
        
        # Left panel — Visualisation
        left_frame = ttk.LabelFrame(content, text="Visualisation", padding=5)
        content.add(left_frame, width=600)
        
        self.analysis_canvas = tk.Canvas(left_frame, bg='#f0f0f0')
        self.analysis_canvas.pack(fill='both', expand=True)
        
        # Right panel — Results table + summary
        right_frame = ttk.Frame(content)
        content.add(right_frame, width=500)
        
        # Summary text
        summary_frame = ttk.LabelFrame(right_frame, text="Summary Statistics", padding=5)
        summary_frame.pack(fill='x', padx=3, pady=3)
        
        self.analysis_summary_text = scrolledtext.ScrolledText(
            summary_frame, height=10, wrap=tk.WORD, font=('Consolas', 9))
        self.analysis_summary_text.pack(fill='x')
        self.analysis_summary_text.insert('1.0', "Run analysis to see results...")
        
        # Object table
        table_frame = ttk.LabelFrame(right_frame, text="Individual Objects", padding=5)
        table_frame.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Treeview
        columns = ('ID', 'Area', 'Diameter', 'Perimeter', 'Circularity', 'Cx', 'Cy')
        self.analysis_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        col_widths = {'ID': 35, 'Area': 80, 'Diameter': 75, 'Perimeter': 75,
                      'Circularity': 70, 'Cx': 55, 'Cy': 55}
        for col in columns:
            self.analysis_tree.heading(col, text=col)
            self.analysis_tree.column(col, width=col_widths.get(col, 60), anchor='center')
        
        tree_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.analysis_tree.yview)
        self.analysis_tree.configure(yscrollcommand=tree_scroll.set)
        self.analysis_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
    
    def browse_analysis_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if filename:
            self.analysis_image_var.set(filename)
    
    def analysis_from_prediction(self):
        """Use the current prediction result from Single Prediction tab"""
        if not hasattr(self, 'current_prediction') or self.current_prediction is None:
            messagebox.showinfo("Info", "No prediction available yet.\n"
                                "Please run a prediction first in the 'Single Prediction' tab, "
                                "or load an image directly here.")
            return
        
        pred = self.current_prediction
        threshold = self.analysis_threshold_var.get()
        pixel_size = self.pixel_size_var.get()
        min_area = self.min_area_var.get()
        
        binary_mask = (pred['prob_map'] > threshold).astype(np.uint8) * 255
        
        task_name = self.predict_task_var.get().split(' ')[0]
        image_name = Path(self.predict_image_var.get()).name if self.predict_image_var.get() else "current"
        
        self.update_status("Running morphological analysis on current prediction...")
        
        # Run analysis
        result = analyze_morphology(binary_mask, pixel_size=pixel_size, min_area_pixels=min_area)
        self.current_analysis = {
            'result': result,
            'image': pred['original'],
            'binary_mask': binary_mask,
            'task_name': task_name,
            'image_name': image_name,
        }
        
        self._display_analysis()
        self.update_status(f"Analysis complete — {result['num_objects']} objects detected")
    
    def run_analysis(self):
        """Run full pipeline: inference + analysis on the selected image"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        image_path = self.analysis_image_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file.")
            return
        
        self.update_status("Running inference + morphological analysis...")
        
        def analysis_thread():
            try:
                image = imread_unicode(image_path)
                if image is None:
                    raise ValueError(f"Cannot read image:\n{image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                task_id = TASK_MAPPING[self.analysis_task_var.get()]
                threshold = self.analysis_threshold_var.get()
                overlap = self.analysis_overlap_var.get()
                pixel_size = self.pixel_size_var.get()
                min_area = self.min_area_var.get()
                
                # Inference
                prob_map = sliding_window_inference(
                    loaded_model, image, task_id, model_device,
                    patch_size=400, overlap=overlap
                )
                binary_mask = (prob_map > threshold).astype(np.uint8) * 255
                
                # Analysis
                result = analyze_morphology(binary_mask, pixel_size=pixel_size, min_area_pixels=min_area)
                
                task_name = self.analysis_task_var.get().split(' ')[0]
                image_name = Path(image_path).name
                
                self.current_analysis = {
                    'result': result,
                    'image': image,
                    'binary_mask': binary_mask,
                    'task_name': task_name,
                    'image_name': image_name,
                }
                
                # Also store as current_prediction for cross-tab usage
                self.current_prediction = {
                    'original': image,
                    'prob_map': prob_map,
                    'binary': binary_mask,
                    'threshold': threshold,
                    'size': image.shape[:2]
                }
                
                self.root.after(0, self._display_analysis)
                self.root.after(0, lambda: self.update_status(
                    f"Analysis complete — {result['num_objects']} objects detected"))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))
                self.root.after(0, lambda: self.update_status("Analysis failed"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def _display_analysis(self):
        """Display analysis results in the tab"""
        if self.current_analysis is None:
            return
        
        result = self.current_analysis['result']
        image = self.current_analysis['image']
        binary_mask = self.current_analysis['binary_mask']
        task_name = self.current_analysis['task_name']
        unit = result['unit']
        area_unit = result['area_unit']
        summary = result['summary']
        
        # --- Update summary text ---
        self.analysis_summary_text.delete('1.0', tk.END)
        txt = (
            f"Image: {self.current_analysis['image_name']}\n"
            f"Task: {task_name}    Scale: 1 px = {result['pixel_size']:.4f} {unit}\n"
            f"{'─' * 45}\n"
            f"Total objects:     {result['num_objects']}\n"
            f"Total area:        {result['total_area']:.2f} {area_unit}\n"
            f"{'─' * 45}\n"
            f"Area ({area_unit}):\n"
            f"  Mean ± Std:  {summary['mean_area']:.2f} ± {summary['std_area']:.2f}\n"
            f"  Median:      {summary['median_area']:.2f}\n"
            f"  Range:       [{summary['min_area']:.2f}, {summary['max_area']:.2f}]\n"
            f"{'─' * 45}\n"
            f"Equiv. Diameter ({unit}):\n"
            f"  Mean ± Std:  {summary['mean_diameter']:.2f} ± {summary['std_diameter']:.2f}\n"
            f"  Median:      {summary['median_diameter']:.2f}\n"
            f"  Range:       [{summary['min_diameter']:.2f}, {summary['max_diameter']:.2f}]\n"
        )
        self.analysis_summary_text.insert('1.0', txt)
        
        # --- Update table ---
        for row in self.analysis_tree.get_children():
            self.analysis_tree.delete(row)
        
        for obj in result['objects']:
            self.analysis_tree.insert('', tk.END, values=(
                obj['id'],
                f"{obj['area']:.2f}",
                f"{obj['diameter']:.2f}",
                f"{obj['perimeter']:.2f}",
                f"{obj['circularity']:.3f}",
                f"{obj['centroid'][0]:.0f}",
                f"{obj['centroid'][1]:.0f}",
            ))
        
        # --- Update visualisation ---
        try:
            fig = create_analysis_visualization(image, binary_mask, result, task_name)
            temp_path = Path('outputs/temp_analysis.png')
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(temp_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            
            img = Image.open(temp_path)
            # Fit to canvas
            canvas_w = self.analysis_canvas.winfo_width() or 600
            canvas_h = self.analysis_canvas.winfo_height() or 500
            img.thumbnail((max(canvas_w, 500), max(canvas_h, 400)))
            photo = ImageTk.PhotoImage(img)
            
            self.analysis_canvas.delete('all')
            self.analysis_canvas.create_image(
                canvas_w // 2, canvas_h // 2, image=photo, anchor='center')
            self.analysis_canvas.image = photo
        except Exception as e:
            print(f"Visualisation error: {e}")
    
    def export_analysis(self):
        """Export analysis results to CSV"""
        if self.current_analysis is None:
            messagebox.showinfo("Info", "No analysis results to export. Run analysis first.")
            return
        
        csv_path = filedialog.asksaveasfilename(
            title="Save Analysis CSV",
            defaultextension='.csv',
            initialfile=f"{Path(self.current_analysis['image_name']).stem}_analysis.csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_path:
            return
        
        try:
            export_analysis_csv(
                self.current_analysis['result'], csv_path,
                image_name=self.current_analysis['image_name'],
                task_name=self.current_analysis['task_name']
            )
            messagebox.showinfo("Success", f"CSV exported to:\n{csv_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def save_analysis_figure(self):
        """Save the analysis visualisation figure"""
        if self.current_analysis is None:
            messagebox.showinfo("Info", "No analysis results. Run analysis first.")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save Analysis Figure",
            defaultextension='.png',
            initialfile=f"{Path(self.current_analysis['image_name']).stem}_analysis.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("All Files", "*.*")]
        )
        if not save_path:
            return
        
        try:
            fig = create_analysis_visualization(
                self.current_analysis['image'],
                self.current_analysis['binary_mask'],
                self.current_analysis['result'],
                self.current_analysis['task_name']
            )
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            messagebox.showinfo("Success", f"Figure saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    # ========================================================================
    # Tab 4: Batch Processing
    # ========================================================================
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 Batch Processing")
        
        # Title
        title = tk.Label(tab, text="Batch Image Processing (Full Resolution)", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Settings area
        settings_frame = ttk.LabelFrame(tab, text="Batch Settings", padding=10)
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        # Image selection
        tk.Label(settings_frame, text="Selected Images:").pack(anchor='w')
        
        btn_frame = tk.Frame(settings_frame)
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="📁 Select Images", command=self.select_batch_images).pack(side='left', padx=5)
        tk.Button(btn_frame, text="📂 Select Folder", command=self.select_batch_folder).pack(side='left', padx=5)
        tk.Button(btn_frame, text="🗑️ Clear Selection", command=self.clear_batch_selection).pack(side='left', padx=5)
        
        # File list
        self.batch_files_text = scrolledtext.ScrolledText(settings_frame, height=5)
        self.batch_files_text.pack(fill='x', pady=5)
        
        # Task and threshold
        param_frame = tk.Frame(settings_frame)
        param_frame.pack(fill='x', pady=5)
        
        tk.Label(param_frame, text="Task:").pack(side='left', padx=5)
        self.batch_task_var = tk.StringVar(value="Cell (Plant Cell)")
        ttk.Combobox(param_frame, textvariable=self.batch_task_var, values=list(TASK_MAPPING.keys()), width=20).pack(side='left', padx=5)
        
        tk.Label(param_frame, text="Threshold:").pack(side='left', padx=5)
        self.batch_threshold_var = tk.DoubleVar(value=0.5)
        tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.05, orient='horizontal', 
                 variable=self.batch_threshold_var, length=100).pack(side='left', padx=5)
        
        tk.Label(param_frame, text="Overlap:").pack(side='left', padx=5)
        self.batch_overlap_var = tk.DoubleVar(value=0.25)
        tk.Scale(param_frame, from_=0.0, to=0.5, resolution=0.05, orient='horizontal', 
                 variable=self.batch_overlap_var, length=100).pack(side='left', padx=5)
        
        # Morphological analysis params for batch
        morph_frame = tk.Frame(settings_frame)
        morph_frame.pack(fill='x', pady=3)
        
        tk.Label(morph_frame, text="Pixel Size (μm/px):").pack(side='left', padx=5)
        self.batch_pixel_size_var = tk.DoubleVar(value=1.0)
        tk.Entry(morph_frame, textvariable=self.batch_pixel_size_var, width=8).pack(side='left', padx=3)
        
        tk.Label(morph_frame, text="Min Area (px):").pack(side='left', padx=(15, 3))
        self.batch_min_area_var = tk.IntVar(value=10)
        tk.Entry(morph_frame, textvariable=self.batch_min_area_var, width=6).pack(side='left', padx=3)
        
        self.batch_export_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(morph_frame, text="Export morphological analysis (CSV per image + summary)",
                        variable=self.batch_export_analysis_var).pack(side='left', padx=15)
        
        # Run button
        tk.Button(
            settings_frame,
            text="🚀 Start Batch Processing (Full Resolution)",
            command=self.run_batch_prediction,
            bg='#FF9800',
            fg='white',
            font=('Arial', 12, 'bold'),
            pady=10
        ).pack(pady=10)
        
        # Progress
        progress_frame = ttk.LabelFrame(tab, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.batch_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.batch_progress.pack(fill='x', pady=5)
        
        self.batch_status_label = tk.Label(progress_frame, text="Ready")
        self.batch_status_label.pack()
        
        # Results
        results_frame = ttk.LabelFrame(tab, text="Processing Log", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.batch_results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.batch_results_text.pack(fill='both', expand=True)
        
        # Store batch files
        self.batch_files = []
    
    def select_batch_images(self):
        """Select multiple images"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if files:
            self.batch_files.extend(files)
            self.update_batch_file_list()
    
    def select_batch_folder(self):
        """Select folder containing images"""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            folder_path = Path(folder)
            image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.jpeg')) + \
                         list(folder_path.glob('*.png')) + list(folder_path.glob('*.bmp')) + \
                         list(folder_path.glob('*.JPG')) + list(folder_path.glob('*.PNG'))
            self.batch_files.extend([str(f) for f in image_files])
            self.update_batch_file_list()
    
    def clear_batch_selection(self):
        """Clear batch file selection"""
        self.batch_files = []
        self.update_batch_file_list()
    
    def update_batch_file_list(self):
        """Update batch file list display"""
        self.batch_files_text.delete('1.0', tk.END)
        self.batch_files_text.insert('1.0', f"Selected {len(self.batch_files)} images:\n")
        for f in self.batch_files[:10]:
            self.batch_files_text.insert(tk.END, f"  • {Path(f).name}\n")
        if len(self.batch_files) > 10:
            self.batch_files_text.insert(tk.END, f"  ... and {len(self.batch_files) - 10} more\n")
    
    def run_batch_prediction(self):
        """Run batch prediction with sliding window (FIXED!)"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        if not self.batch_files:
            messagebox.showerror("Error", "No images selected!")
            return
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_name = self.batch_task_var.get().split(' ')[0]
        output_dir = Path(f'outputs/batch_predictions/{timestamp}_{task_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_results_text.delete('1.0', tk.END)
        self.batch_results_text.insert('1.0', f"Output directory: {output_dir}\n")
        self.batch_results_text.insert(tk.END, f"🔧 Using sliding window inference (full resolution output)\n\n")
        
        def batch_thread():
            task_id = TASK_MAPPING[self.batch_task_var.get()]
            task_name = self.batch_task_var.get().split(' ')[0]
            threshold = self.batch_threshold_var.get()
            overlap = self.batch_overlap_var.get()
            pixel_size = self.batch_pixel_size_var.get()
            min_area = self.batch_min_area_var.get()
            do_analysis = self.batch_export_analysis_var.get()
            total = len(self.batch_files)
            
            # Collect batch summary for combined CSV
            batch_summary_rows = []
            
            for i, image_path in enumerate(self.batch_files):
                try:
                    # Update progress
                    progress = (i + 1) / total * 100
                    self.root.after(0, lambda p=progress: self.batch_progress.configure(value=p))
                    self.root.after(0, lambda n=i+1, t=total: self.batch_status_label.configure(
                        text=f"Processing {n}/{t}..."))
                    
                    # Load image (keep original size!)
                    image = imread_unicode(image_path)
                    if image is None:
                        raise ValueError(f"Cannot read image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    H, W = image.shape[:2]
                    
                    # Sliding window inference (FIXED!)
                    prob_map = sliding_window_inference(
                        loaded_model, image, task_id, model_device,
                        patch_size=400, overlap=overlap
                    )
                    
                    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
                    
                    # Save results (at original size!)
                    base_name = Path(image_path).stem
                    
                    # Original
                    imwrite_unicode(str(output_dir / f'{base_name}_original.png'), 
                               cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Heatmap (using matplotlib for colormap)
                    plt.figure(figsize=(W/100, H/100), dpi=100)
                    plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.savefig(str(output_dir / f'{base_name}_heatmap.png'), 
                               dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    # Binary mask
                    imwrite_unicode(str(output_dir / f'{base_name}_binary.png'), binary_mask)
                    
                    # Overlay
                    overlay = image.copy()
                    mask_colored = np.zeros_like(overlay)
                    mask_colored[binary_mask > 0] = [0, 255, 0]
                    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                    imwrite_unicode(str(output_dir / f'{base_name}_overlay.png'), 
                               cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
                    # Morphological analysis
                    analysis_info = ""
                    if do_analysis:
                        result = analyze_morphology(binary_mask, pixel_size=pixel_size,
                                                    min_area_pixels=min_area)
                        # Per-image CSV
                        export_analysis_csv(
                            result, str(output_dir / f'{base_name}_analysis.csv'),
                            image_name=Path(image_path).name, task_name=task_name
                        )
                        
                        # Collect for batch summary
                        u = result['unit']
                        au = result['area_unit']
                        batch_summary_rows.append({
                            'image': Path(image_path).name,
                            'width': W, 'height': H,
                            'num_objects': result['num_objects'],
                            'total_area': result['total_area'],
                            'mean_area': result['summary']['mean_area'],
                            'std_area': result['summary']['std_area'],
                            'mean_diameter': result['summary']['mean_diameter'],
                            'std_diameter': result['summary']['std_diameter'],
                        })
                        
                        analysis_info = (f"  → {result['num_objects']} objects, "
                                         f"total area={result['total_area']:.1f} {au}, "
                                         f"mean ø={result['summary']['mean_diameter']:.1f} {u}")
                    
                    # Update log
                    self.root.after(0, lambda p=image_path, w=W, h=H, info=analysis_info:
                        self.batch_results_text.insert(
                            tk.END, f"✓ {Path(p).name} ({w}x{h})\n{info}\n" if info
                            else f"✓ {Path(p).name} ({w}x{h})\n"))
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.root.after(0, lambda p=image_path, err=e: self.batch_results_text.insert(
                        tk.END, f"✗ {Path(p).name}: {err}\n"))
            
            # Write batch summary CSV
            if do_analysis and batch_summary_rows:
                try:
                    unit = "px" if pixel_size == 1.0 else "μm"
                    area_unit = "px²" if pixel_size == 1.0 else "μm²"
                    summary_path = output_dir / 'batch_summary.csv'
                    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'Image', 'Width', 'Height',
                            'Num_Objects',
                            f'Total_Area ({area_unit})',
                            f'Mean_Area ({area_unit})',
                            f'Std_Area ({area_unit})',
                            f'Mean_Diameter ({unit})',
                            f'Std_Diameter ({unit})',
                        ])
                        for row in batch_summary_rows:
                            writer.writerow([
                                row['image'], row['width'], row['height'],
                                row['num_objects'],
                                f"{row['total_area']:.4f}",
                                f"{row['mean_area']:.4f}",
                                f"{row['std_area']:.4f}",
                                f"{row['mean_diameter']:.4f}",
                                f"{row['std_diameter']:.4f}",
                            ])
                except Exception as e:
                    print(f"Warning: failed to write batch summary CSV: {e}")
            
            # Complete
            analysis_note = "\n📊 Morphological analysis CSVs included!" if do_analysis else ""
            self.root.after(0, lambda: self.batch_status_label.configure(text="Complete!"))
            self.root.after(0, lambda: self.batch_results_text.insert(
                tk.END, f"\n✅ Batch processing complete!\n"
                f"Results saved to: {output_dir}\n"
                f"All outputs are at ORIGINAL RESOLUTION!"
                f"{analysis_note}"))
            self.root.after(0, lambda: messagebox.showinfo("Complete", 
                f"Batch processing complete!\n\n"
                f"Results saved to:\n{output_dir}\n\n"
                f"All outputs are at ORIGINAL RESOLUTION!"
                f"{analysis_note}"))
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 5: Help
    # ========================================================================
    
    def create_help_tab(self):
        """Create help tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📖 Help")
        
        help_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=('Arial', 10))
        help_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_content = """
Multi-Task TransUNet - FIXED VERSION + Morphological Analysis

═══════════════════════════════════════════════════════════

📊 MORPHOLOGICAL ANALYSIS (NEW)
═══════════════════════════════════════════════════════════

  The "Morphological Analysis" tab computes:

  • Total area — sum of all detected objects
  • Individual area — area of each connected component
  • Equiv. diameter — diameter of a circle with same area
      d = √(4 × Area / π)
  • Perimeter — contour length of each object
  • Circularity — 4π × Area / Perimeter²
      (1.0 = perfect circle)

  Scale calibration:
  • Set "Pixel Size" (μm/px) to convert pixels → μm.
  • If unknown, leave at 1.0 (results in pixel units).
  • Example: if 1 pixel = 0.5 μm, enter 0.5.

  Two ways to run:
  1. "Run Analysis" — select an image, run inference
      + analysis in one step.
  2. "Use Current Prediction" — reuse the mask from
      the Single Prediction tab (faster, no re-inference).

  Export:
  • "Export CSV" — per-object measurements to .csv
  • "Save Figure" — 4-panel visualisation to PNG/PDF

═══════════════════════════════════════════════════════════

📁 BATCH PROCESSING
═══════════════════════════════════════════════════════════

  When "Export morphological analysis" is checked:
  • Each image gets a *_analysis.csv
  • A batch_summary.csv collects per-image statistics

═══════════════════════════════════════════════════════════

🔧 Overlap (Sliding Window)
═══════════════════════════════════════════════════════════

   • 0.0  = no overlap (fastest, may show tile edges)
   • 0.25 = 25% overlap (good balance)
   • 0.5  = 50% overlap (best quality, slowest)

═══════════════════════════════════════════════════════════

If you have any further questions, please feel free to contact Shitephen.
email: gn03138868@gmail.com; shitephenwang@ntu.edu.tw

═══════════════════════════════════════════════════════════
"""
        
        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')


def main():
    """Main programme"""
    root = tk.Tk()
    app = MultiTaskGUI(root)
    
    print("\n" + "="*60)
    print("🚀 Multi-Task TransUNet - FIXED VERSION + Morphological Analysis")
    print("="*60)
    print("\n✅ Features：")
    print("   • 使用滑動窗口處理大圖")
    print("   • 輸出大小 = 輸入大小")
    print("   • 支援任意解析度")
    print("   • 📊 形態學分析（面積、直徑、周長、圓度）")
    print("   • 📋 CSV 匯出（逐物件 + 批次摘要）")
    print("\n" + "="*60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
    
    main()
