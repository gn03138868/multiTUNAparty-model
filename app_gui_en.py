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
import numpy as np
import json
import yaml
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
# GUI 類
# ============================================================================

class MultiTaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Task TransUNet - Desktop Edition (Fixed)")
        self.root.geometry("1200x800")
        
        # Create status bar
        self.create_status_bar()
        
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create feature tabs
        self.create_model_tab()
        self.create_predict_tab()
        self.create_batch_tab()
        self.create_help_tab()
        
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
                image = cv2.imread(image_path)
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
            cv2.imwrite(
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
            cv2.imwrite(
                os.path.join(save_dir, f'{base_name}_binary.png'),
                pred['binary']
            )
            
            # Save overlay (at original size)
            overlay = pred['original'].copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[pred['binary'] > 0] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(
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
    # Tab 3: Batch Processing
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
            threshold = self.batch_threshold_var.get()
            overlap = self.batch_overlap_var.get()
            total = len(self.batch_files)
            
            for i, image_path in enumerate(self.batch_files):
                try:
                    # Update progress
                    progress = (i + 1) / total * 100
                    self.root.after(0, lambda p=progress: self.batch_progress.configure(value=p))
                    self.root.after(0, lambda n=i+1, t=total: self.batch_status_label.configure(
                        text=f"Processing {n}/{t}..."))
                    
                    # Load image (keep original size!)
                    image = cv2.imread(image_path)
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
                    cv2.imwrite(str(output_dir / f'{base_name}_original.png'), 
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
                    cv2.imwrite(str(output_dir / f'{base_name}_binary.png'), binary_mask)
                    
                    # Overlay
                    overlay = image.copy()
                    mask_colored = np.zeros_like(overlay)
                    mask_colored[binary_mask > 0] = [0, 255, 0]
                    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                    cv2.imwrite(str(output_dir / f'{base_name}_overlay.png'), 
                               cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
                    # Update log
                    self.root.after(0, lambda p=image_path, w=W, h=H: self.batch_results_text.insert(
                        tk.END, f"✓ {Path(p).name} ({w}x{h})\n"))
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.root.after(0, lambda p=image_path, err=e: self.batch_results_text.insert(
                        tk.END, f"✗ {Path(p).name}: {err}\n"))
            
            # Complete
            self.root.after(0, lambda: self.batch_status_label.configure(text="Complete!"))
            self.root.after(0, lambda: self.batch_results_text.insert(
                tk.END, f"\n✅ Batch processing complete!\n"
                f"Results saved to: {output_dir}\n"
                f"All outputs are at ORIGINAL RESOLUTION!"))
            self.root.after(0, lambda: messagebox.showinfo("Complete", 
                f"Batch processing complete!\n\n"
                f"Results saved to:\n{output_dir}\n\n"
                f"All outputs are at ORIGINAL RESOLUTION!"))
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 4: Help
    # ========================================================================
    
    def create_help_tab(self):
        """Create help tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📖 Help")
        
        help_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=('Arial', 10))
        help_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_content = """
Multi-Task TransUNet - FIXED VERSION

═══════════════════════════════════════════════════════════

Overlap：
   • 0.0 = on any overlap,
   • 0.25 = 25% overlap
   • 0.5 = 50% overlap

═══════════════════════════════════════════════════════════

If you have any further questions, please feel free to contect Shitephen.
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
    print("🚀 Multi-Task TransUNet - FIXED VERSION")
    print("="*60)
    print("\n✅ 修復內容：")
    print("   • 使用滑動窗口處理大圖")
    print("   • 輸出大小 = 輸入大小")
    print("   • 支援任意解析度")
    print("\n" + "="*60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
    
    main()
