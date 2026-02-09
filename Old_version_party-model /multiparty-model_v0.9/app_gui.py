"""
多任務 TransUNet - Tkinter 桌面 GUI 版本

不需要 Gradio，使用 Python 標準庫 tkinter
適用於所有環境，包括 RTX 5080

功能：
1. 模型管理
2. 單張預測
3. 批量預測
4. 訓練監控
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

# 導入模型
try:
    from model_multitask import MultiTaskTransUNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: model_multitask.py not found")

# 全局配置
TASK_MAPPING = {
    'Cell (植物細胞)': 0,
    'Blood (血球)': 1,
    'Root (根系)': 2
}

TASK_COLORS = {
    0: 'Blues',
    1: 'Reds',
    2: 'Greens'
}

# 全局變量
loaded_model = None
model_device = None
training_process = None
training_status = {
    'is_training': False,
    'message': '尚未開始訓練'
}


class MultiTaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("多任務 TransUNet - 桌面版")
        self.root.geometry("1200x800")
        
        # 設置圖標（如果有）
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # 創建狀態列
        self.create_status_bar()
        
        # 創建主要的標籤頁
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 創建各個功能頁面
        self.create_model_tab()
        # self.create_training_tab()  # 訓練功能已移至 Gradio 版本
        self.create_predict_tab()
        self.create_batch_tab()
        self.create_monitor_tab()
        self.create_help_tab()
        
        # 更新狀態
        self.update_status("就緒 - 請先載入模型")
    
    def create_status_bar(self):
        """創建狀態列"""
        self.status_bar = tk.Label(
            self.root, 
            text="就緒", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """更新狀態列"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    # ========================================================================
    # Tab 1: 模型管理
    # ========================================================================
    
    def create_model_tab(self):
        """創建模型管理頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📦 模型管理")
        
        # 標題
        title = tk.Label(tab, text="模型載入與管理", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # 模型選擇區域
        model_frame = ttk.LabelFrame(tab, text="選擇模型", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        # 模型路徑
        path_frame = tk.Frame(model_frame)
        path_frame.pack(fill='x', pady=5)
        
        tk.Label(path_frame, text="模型路徑:").pack(side='left', padx=5)
        self.model_path_var = tk.StringVar(value="outputs/models/best_model.pth")
        model_entry = tk.Entry(path_frame, textvariable=self.model_path_var, width=50)
        model_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        tk.Button(
            path_frame, 
            text="瀏覽...", 
            command=self.browse_model
        ).pack(side='left', padx=5)
        
        # 設備選擇
        device_frame = tk.Frame(model_frame)
        device_frame.pack(fill='x', pady=5)
        
        tk.Label(device_frame, text="計算設備:").pack(side='left', padx=5)
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
        
        # GPU 資訊
        if torch.cuda.is_available():
            gpu_info = f"✅ GPU 可用: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "⚠️ GPU 不可用，將使用 CPU（速度較慢）"
        
        tk.Label(device_frame, text=gpu_info, fg='green' if torch.cuda.is_available() else 'orange').pack(side='left', padx=10)
        
        # 載入按鈕
        tk.Button(
            model_frame, 
            text="📥 載入模型", 
            command=self.load_model,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # 模型資訊顯示
        info_frame = ttk.LabelFrame(tab, text="模型資訊", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame, 
            height=15, 
            wrap=tk.WORD
        )
        self.model_info_text.pack(fill='both', expand=True)
        self.model_info_text.insert('1.0', "尚未載入模型\n\n請選擇模型檔案並點擊「載入模型」按鈕")
    
    def browse_model(self):
        """瀏覽選擇模型檔案"""
        filename = filedialog.askopenfilename(
            title="選擇模型檔案",
            filetypes=[("PyTorch 模型", "*.pth"), ("所有檔案", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def load_model(self):
        """載入模型"""
        global loaded_model, model_device
        
        if not MODEL_AVAILABLE:
            messagebox.showerror("錯誤", "找不到 model_multitask.py 檔案！")
            return
        
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("錯誤", f"模型檔案不存在:\n{model_path}")
            return
        
        self.update_status("正在載入模型...")
        self.model_info_text.delete('1.0', tk.END)
        self.model_info_text.insert('1.0', "正在載入模型，請稍候...\n")
        
        # 在背景線程載入
        def load_thread():
            global loaded_model, model_device  # 重要：在內部函數也要聲明 global
            try:
                # 設置設備
                if self.device_var.get() == "GPU (CUDA)" and torch.cuda.is_available():
                    device = torch.device('cuda')
                    device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                else:
                    device = torch.device('cpu')
                    device_info = "CPU"
                
                # 創建模型
                model = MultiTaskTransUNet(
                    img_size=400,
                    patch_size=16,
                    num_decoder_layers=80,
                    num_tasks=3
                )
                
                # 載入權重
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                
                # 更新全局變量
                loaded_model = model
                model_device = device
                
                # 計算參數量
                total_params = sum(p.numel() for p in model.parameters())
                
                # 更新 UI
                info = f"""
✅ 模型載入成功！

📊 模型資訊：
  • 設備: {device_info}
  • 參數量: {total_params:,}
  • 模型路徑: {model_path}
  • 載入時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 支援任務：
  • Cell (植物細胞)
  • Blood (血球)
  • Root (根系)

✓ 現在可以開始預測了！
"""
                
                self.root.after(0, lambda: self.model_info_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.model_info_text.insert('1.0', info))
                self.root.after(0, lambda: self.update_status("模型已載入 - 就緒"))
                self.root.after(0, lambda: messagebox.showinfo("成功", "模型載入成功！"))
                
            except Exception as e:
                error_msg = f"❌ 載入失敗: {str(e)}"
                self.root.after(0, lambda: self.model_info_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.model_info_text.insert('1.0', error_msg))
                self.root.after(0, lambda: self.update_status("載入失敗"))
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 2: 訓練模型
    # ========================================================================
    
    def create_training_tab(self):
        """創建訓練模型頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🚀 訓練模型")
        
        # 標題和說明
        title = tk.Label(tab, text="訓練新模型或繼續訓練", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        info_label = tk.Label(
            tab, 
            text="💡 訓練過程的輸出會顯示在 CMD 視窗，請保持 CMD 視窗開啟",
            fg='blue'
        )
        info_label.pack(pady=5)
        
        # 主要內容區
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 左側：參數設定
        left_frame = ttk.LabelFrame(main_frame, text="訓練參數", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # 基本參數
        basic_frame = ttk.LabelFrame(left_frame, text="基本參數", padding=10)
        basic_frame.pack(fill='x', pady=5)
        
        # Batch Size
        tk.Label(basic_frame, text="Batch Size:").grid(row=0, column=0, sticky='w', pady=2)
        self.batch_size_var = tk.IntVar(value=2)
        batch_spinbox = tk.Spinbox(basic_frame, from_=1, to=16, textvariable=self.batch_size_var, width=10)
        batch_spinbox.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Epochs
        tk.Label(basic_frame, text="訓練輪數 (Epochs):").grid(row=1, column=0, sticky='w', pady=2)
        self.epochs_var = tk.IntVar(value=200)
        epochs_spinbox = tk.Spinbox(basic_frame, from_=1, to=500, textvariable=self.epochs_var, width=10)
        epochs_spinbox.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Learning Rate
        tk.Label(basic_frame, text="學習率 (Learning Rate):").grid(row=2, column=0, sticky='w', pady=2)
        self.lr_var = tk.StringVar(value="1e-5")
        lr_entry = tk.Entry(basic_frame, textvariable=self.lr_var, width=15)
        lr_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Patch Size
        tk.Label(basic_frame, text="Patch 大小:").grid(row=3, column=0, sticky='w', pady=2)
        self.patch_size_var = tk.IntVar(value=400)
        patch_spinbox = tk.Spinbox(basic_frame, from_=128, to=512, increment=32, textvariable=self.patch_size_var, width=10)
        patch_spinbox.grid(row=3, column=1, sticky='w', padx=5, pady=2)
        
        # Decoder Layers
        tk.Label(basic_frame, text="Decoder 層數:").grid(row=4, column=0, sticky='w', pady=2)
        self.num_layers_var = tk.IntVar(value=80)
        layers_spinbox = tk.Spinbox(basic_frame, from_=20, to=120, increment=10, textvariable=self.num_layers_var, width=10)
        layers_spinbox.grid(row=4, column=1, sticky='w', padx=5, pady=2)
        
        # Data Path
        tk.Label(basic_frame, text="資料路徑:").grid(row=5, column=0, sticky='w', pady=2)
        self.data_path_var = tk.StringVar(value="data/")
        data_entry = tk.Entry(basic_frame, textvariable=self.data_path_var, width=30)
        data_entry.grid(row=5, column=1, sticky='w', padx=5, pady=2)
        
        # 資料檢查按鈕
        tk.Button(
            basic_frame,
            text="🔍 檢查資料結構",
            command=self.check_data_structure,
            bg='#2196F3',
            fg='white',
            font=('Arial', 9, 'bold')
        ).grid(row=6, column=0, columnspan=2, pady=10)
        
        # 進階設定
        advanced_frame = ttk.LabelFrame(left_frame, text="進階設定", padding=10)
        advanced_frame.pack(fill='x', pady=5)
        
        # 使用預訓練模型
        self.use_pretrained_var = tk.BooleanVar(value=False)
        pretrained_check = tk.Checkbutton(
            advanced_frame,
            text="使用預訓練模型",
            variable=self.use_pretrained_var
        )
        pretrained_check.pack(anchor='w', pady=5)
        
        # 預訓練模型路徑
        pretrained_frame = tk.Frame(advanced_frame)
        pretrained_frame.pack(fill='x', pady=5)
        
        tk.Label(pretrained_frame, text="預訓練模型:").pack(side='left', padx=5)
        self.pretrained_path_var = tk.StringVar(value="outputs/models/checkpoint_epoch060.pth")
        pretrained_entry = tk.Entry(pretrained_frame, textvariable=self.pretrained_path_var, width=25)
        pretrained_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        tk.Button(
            pretrained_frame,
            text="瀏覽...",
            command=self.browse_pretrained_model
        ).pack(side='left', padx=5)
        
        # 右側：控制和狀態
        right_frame = ttk.LabelFrame(main_frame, text="訓練控制", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # 控制按鈕
        button_frame = tk.Frame(right_frame)
        button_frame.pack(fill='x', pady=5)
        
        tk.Button(
            button_frame,
            text="🚀 開始訓練",
            command=self.start_training,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="⏹️ 停止訓練",
            command=self.stop_training,
            bg='#f44336',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="🔄 刷新進度",
            command=self.refresh_training_progress,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(side='left', padx=5)
        
        # 訓練進度
        progress_frame = ttk.LabelFrame(right_frame, text="訓練進度", padding=10)
        progress_frame.pack(fill='x', pady=5)
        
        self.training_progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.training_progress_bar.pack(fill='x', pady=5)
        
        self.training_progress_label = tk.Label(progress_frame, text="尚未開始訓練")
        self.training_progress_label.pack(pady=5)
        
        # 訓練訊息
        msg_frame = ttk.LabelFrame(right_frame, text="訓練訊息", padding=10)
        msg_frame.pack(fill='both', expand=True, pady=5)
        
        self.training_msg_text = scrolledtext.ScrolledText(msg_frame, wrap=tk.WORD, height=15)
        self.training_msg_text.pack(fill='both', expand=True)
        self.training_msg_text.insert('1.0', "尚未開始訓練\n\n請設定訓練參數後點擊「開始訓練」")
        
        # 資料檢查結果（底部）
        data_frame = ttk.LabelFrame(tab, text="資料檢查結果", padding=10)
        data_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_check_text = scrolledtext.ScrolledText(data_frame, wrap=tk.WORD, height=8)
        self.data_check_text.pack(fill='both', expand=True)
        self.data_check_text.insert('1.0', "點擊「檢查資料結構」查看資料集資訊")
        
        self.training_process = None
    
    def browse_pretrained_model(self):
        """瀏覽選擇預訓練模型"""
        filename = filedialog.askopenfilename(
            title="選擇預訓練模型",
            filetypes=[("PyTorch 模型", "*.pth"), ("所有檔案", "*.*")]
        )
        if filename:
            self.pretrained_path_var.set(filename)
    
    def check_data_structure(self):
        """檢查資料結構"""
        data_path = Path(self.data_path_var.get())
        
        self.data_check_text.delete('1.0', tk.END)
        self.data_check_text.insert('1.0', "正在檢查資料結構...\n\n")
        self.update_status("檢查資料中...")
        
        def check_thread():
            try:
                result = "📁 資料結構檢查\n\n"
                
                if not data_path.exists():
                    result += f"❌ 資料路徑不存在: {data_path}\n"
                    self.root.after(0, lambda: self.data_check_text.delete('1.0', tk.END))
                    self.root.after(0, lambda: self.data_check_text.insert('1.0', result))
                    self.root.after(0, lambda: self.update_status("資料路徑不存在"))
                    return
                
                # 檢查訓練集
                train_path = data_path / 'train'
                val_path = data_path / 'val'
                
                for split_name, split_path in [('訓練集', train_path), ('驗證集', val_path)]:
                    result += f"\n{'='*50}\n{split_name}: {split_path}\n{'='*50}\n"
                    
                    if not split_path.exists():
                        result += f"❌ {split_name}目錄不存在\n"
                        continue
                    
                    for task in ['cell', 'blood', 'root']:
                        task_path = split_path / task
                        if task_path.exists():
                            images_path = task_path / 'images'
                            masks_path = task_path / 'masks'
                            
                            num_images = len(list(images_path.glob('*'))) if images_path.exists() else 0
                            num_masks = len(list(masks_path.glob('*'))) if masks_path.exists() else 0
                            
                            if num_images > 0 and num_masks > 0:
                                result += f"  ✅ {task:10s}: {num_images:3d} 影像, {num_masks:3d} masks\n"
                            elif num_images > 0:
                                result += f"  ⚠️  {task:10s}: {num_images:3d} 影像, {num_masks:3d} masks (不匹配！)\n"
                            else:
                                result += f"  ❌ {task:10s}: 無資料\n"
                        else:
                            result += f"  ❌ {task:10s}: 目錄不存在\n"
                
                result += "\n" + "="*50 + "\n"
                result += "✅ 資料檢查完成！\n"
                
                self.root.after(0, lambda: self.data_check_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.data_check_text.insert('1.0', result))
                self.root.after(0, lambda: self.update_status("資料檢查完成"))
                
            except Exception as e:
                error_msg = f"❌ 檢查失敗: {str(e)}"
                self.root.after(0, lambda: self.data_check_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.data_check_text.insert('1.0', error_msg))
                self.root.after(0, lambda: self.update_status("檢查失敗"))
        
        threading.Thread(target=check_thread, daemon=True).start()
    
    def start_training(self):
        """開始訓練"""
        global training_process, training_status
        
        if training_status['is_training']:
            messagebox.showwarning("警告", "訓練已在進行中！")
            return
        
        self.update_status("準備開始訓練...")
        self.training_msg_text.delete('1.0', tk.END)
        self.training_msg_text.insert('1.0', "正在準備訓練...\n\n")
        
        def train_thread():
            try:
                # 創建配置文件（使用與 Gradio 相同的格式）
                config = {
                    'batch_size': int(self.batch_size_var.get()),
                    'epochs': int(self.epochs_var.get()),
                    'lr': float(self.lr_var.get()),
                    'patch_size': int(self.patch_size_var.get()),
                    'num_decoder_conv_layers': int(self.num_layers_var.get()),  # 注意：是 conv_layers
                    'data_path': self.data_path_var.get(),
                    'task_structure': 'subfolder',
                    'boundary_weights': {0: 2.0, 1: 3.0, 2: 5.0},
                    'foreground_weights': {0: 1.0, 1: 1.5, 2: 3.0}
                }
                
                # 如果使用預訓練模型
                if self.use_pretrained_var.get():
                    pretrained_path = self.pretrained_path_var.get()
                    if os.path.exists(pretrained_path):
                        config['pretrained_model_path'] = pretrained_path  # 注意參數名
                    else:
                        error_msg = f"❌ 預訓練模型不存在: {pretrained_path}"
                        self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
                        return
                
                # 保存配置文件
                config_path = Path('config_gui_training.yaml')
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                # 準備訓練命令（使用配置文件）
                cmd = [
                    sys.executable,
                    'train_multitask.py',
                    '--config', str(config_path)
                ]
                
                # 更新狀態
                training_status['is_training'] = True
                training_status['current_epoch'] = 0
                training_status['total_epochs'] = self.epochs_var.get()
                training_status['message'] = '正在啟動訓練...'
                
                msg = f"""
🚀 開始訓練！

📊 訓練配置：
  • Batch Size: {self.batch_size_var.get()}
  • Epochs: {self.epochs_var.get()}
  • Learning Rate: {self.lr_var.get()}
  • Patch Size: {self.patch_size_var.get()}
  • Decoder Layers: {self.num_layers_var.get()}
  • Data Path: {self.data_path_var.get()}
  • 預訓練模型: {'是' if self.use_pretrained_var.get() else '否'}
  • 配置文件: {config_path}

💡 訓練過程的詳細輸出會顯示在 CMD 視窗
   請保持 CMD 視窗開啟以查看進度

🔄 點擊「刷新進度」查看當前訓練狀態
"""
                
                self.root.after(0, lambda: self.training_msg_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.training_msg_text.insert('1.0', msg))
                self.root.after(0, lambda: self.update_status("訓練進行中..."))
                
                # 啟動訓練程序
                # 設置環境變量以支持 UTF-8 編碼（解決 Windows cp950 問題）
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env=env  # 使用 UTF-8 環境
                )
                
                print("\n" + "="*60)
                print("🚀 訓練已啟動！")
                print("="*60)
                print(f"命令: {' '.join(cmd)}")
                print(f"配置文件: {config_path}")
                print("\n配置內容:")
                print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
                print("\n訓練輸出：\n")
                
                # 讀取輸出
                for line in self.training_process.stdout:
                    print(line, end='')
                
                self.training_process.wait()
                
                # 訓練結束
                training_status['is_training'] = False
                
                if self.training_process.returncode == 0:
                    training_status['message'] = '✅ 訓練完成！'
                    final_msg = "\n\n✅ 訓練成功完成！\n\n模型已保存到 outputs/models/ 目錄"
                else:
                    training_status['message'] = '❌ 訓練失敗'
                    final_msg = f"\n\n❌ 訓練失敗（返回碼: {self.training_process.returncode}）\n請查看 CMD 視窗了解錯誤詳情"
                
                self.root.after(0, lambda: self.training_msg_text.insert(tk.END, final_msg))
                self.root.after(0, lambda: self.update_status("訓練結束"))
                self.root.after(0, lambda: self.training_progress_bar.config(value=100))
                
            except Exception as e:
                training_status['is_training'] = False
                training_status['message'] = f'❌ 錯誤: {str(e)}'
                error_msg = f"\n\n❌ 訓練錯誤: {str(e)}"
                self.root.after(0, lambda: self.training_msg_text.insert(tk.END, error_msg))
                self.root.after(0, lambda: self.update_status("訓練失敗"))
                self.root.after(0, lambda: messagebox.showerror("錯誤", f"訓練失敗: {e}"))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def stop_training(self):
        """停止訓練"""
        global training_status
        
        if not training_status['is_training']:
            messagebox.showinfo("提示", "目前沒有正在進行的訓練")
            return
        
        if self.training_process and self.training_process.poll() is None:
            response = messagebox.askyesno("確認", "確定要停止訓練嗎？\n進度將會丟失。")
            if response:
                self.training_process.terminate()
                training_status['is_training'] = False
                training_status['message'] = '⏹️ 訓練已停止'
                
                self.training_msg_text.insert(tk.END, "\n\n⏹️ 訓練已被用戶停止")
                self.update_status("訓練已停止")
    
    def refresh_training_progress(self):
        """刷新訓練進度"""
        global training_status
        
        if training_status['is_training']:
            current = training_status.get('current_epoch', 0)
            total = training_status.get('total_epochs', 1)
            progress = (current / total * 100) if total > 0 else 0
            
            self.training_progress_bar.config(value=progress)
            self.training_progress_label.config(
                text=f"進度: Epoch {current}/{total} ({progress:.1f}%)"
            )
        else:
            self.training_progress_label.config(text=training_status['message'])
    
    # ========================================================================
    # Tab 3: 單張預測
    # ========================================================================
    
    def create_predict_tab(self):
        """創建單張預測頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 單張預測")
        
        # 左側：控制面板
        left_frame = ttk.Frame(tab)
        left_frame.pack(side='left', fill='both', padx=5, pady=5)
        
        # 上傳影像
        upload_frame = ttk.LabelFrame(left_frame, text="上傳影像", padding=10)
        upload_frame.pack(fill='x', pady=5)
        
        tk.Button(
            upload_frame,
            text="📁 選擇影像檔案",
            command=self.select_predict_image,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(pady=5)
        
        self.predict_image_label = tk.Label(upload_frame, text="未選擇檔案", fg='gray')
        self.predict_image_label.pack(pady=5)
        
        # 預覽影像
        self.preview_label = tk.Label(upload_frame, text="影像預覽", bg='lightgray', width=40, height=15)
        self.preview_label.pack(pady=5)
        
        # 參數設定
        param_frame = ttk.LabelFrame(left_frame, text="預測參數", padding=10)
        param_frame.pack(fill='x', pady=5)
        
        # 任務選擇
        tk.Label(param_frame, text="任務類型:").pack(anchor='w', pady=2)
        self.task_var = tk.StringVar(value='Cell (植物細胞)')
        for task in TASK_MAPPING.keys():
            ttk.Radiobutton(
                param_frame,
                text=task,
                variable=self.task_var,
                value=task
            ).pack(anchor='w', padx=20)
        
        # 閾值
        tk.Label(param_frame, text="分割閾值:").pack(anchor='w', pady=(10, 2))
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(
            param_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient='horizontal',
            variable=self.threshold_var,
            length=200
        )
        threshold_scale.pack(fill='x', padx=10)
        
        # 預測按鈕
        tk.Button(
            left_frame,
            text="🔍 開始預測",
            command=self.predict_single,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # 右側：結果顯示
        right_frame = ttk.Frame(tab)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # 結果顯示區
        result_notebook = ttk.Notebook(right_frame)
        result_notebook.pack(fill='both', expand=True)
        
        # 熱圖
        heatmap_frame = ttk.Frame(result_notebook)
        result_notebook.add(heatmap_frame, text="機率熱圖")
        self.heatmap_label = tk.Label(heatmap_frame, text="預測結果會顯示在這裡", bg='lightgray')
        self.heatmap_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 二值化
        binary_frame = ttk.Frame(result_notebook)
        result_notebook.add(binary_frame, text="二值化結果")
        self.binary_label = tk.Label(binary_frame, text="預測結果會顯示在這裡", bg='lightgray')
        self.binary_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 疊加圖
        overlay_frame = ttk.Frame(result_notebook)
        result_notebook.add(overlay_frame, text="疊加圖")
        self.overlay_label = tk.Label(overlay_frame, text="預測結果會顯示在這裡", bg='lightgray')
        self.overlay_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 統計資訊
        stats_frame = ttk.Frame(result_notebook)
        result_notebook.add(stats_frame, text="統計資訊")
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD)
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.predict_image_path = None
    
    def select_predict_image(self):
        """選擇要預測的影像"""
        filename = filedialog.askopenfilename(
            title="選擇影像檔案",
            filetypes=[
                ("影像檔案", "*.jpg *.jpeg *.png *.bmp"),
                ("所有檔案", "*.*")
            ]
        )
        if filename:
            self.predict_image_path = filename
            self.predict_image_label.config(text=os.path.basename(filename), fg='black')
            
            # 顯示預覽
            try:
                img = Image.open(filename)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=photo, text='')
                self.preview_label.image = photo
            except Exception as e:
                messagebox.showerror("錯誤", f"無法載入影像: {e}")
    
    def predict_single(self):
        """執行單張預測"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("錯誤", "請先載入模型！")
            return
        
        if self.predict_image_path is None:
            messagebox.showerror("錯誤", "請先選擇影像！")
            return
        
        self.update_status("正在預測...")
        
        def predict_thread():
            global loaded_model, model_device  # 確保訪問全局變量
            try:
                # 讀取影像
                image = Image.open(self.predict_image_path)
                image_rgb = np.array(image)
                
                # 確保是 RGB
                if len(image_rgb.shape) == 2:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                elif image_rgb.shape[2] == 4:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
                
                h, w = image_rgb.shape[:2]
                task_id = TASK_MAPPING[self.task_var.get()]
                threshold = self.threshold_var.get()
                
                # 預測
                result = self.predict_image_full(loaded_model, image_rgb, task_id, model_device)
                
                # 二值化
                pred_binary = (result > threshold).astype(np.uint8) * 255
                
                # 創建熱圖
                heatmap_img = self.create_heatmap_image(result, TASK_COLORS[task_id])
                
                # 創建疊加圖
                overlay_img = self.create_overlay_image(image_rgb, result, threshold)
                
                # 統計
                foreground_ratio = (result > threshold).sum() / result.size * 100
                stats = f"""
📊 預測結果：

🎯 任務: {self.task_var.get()}
📏 影像大小: {w} x {h}
🎚️ 閾值: {threshold:.2f}

📈 統計：
  • 最小值: {result.min():.3f}
  • 最大值: {result.max():.3f}
  • 平均值: {result.mean():.3f}
  • 前景比例: {foreground_ratio:.2f}%
  • 前景像素: {int((result > threshold).sum())} / {result.size}

✅ 預測完成！
"""
                
                # 更新 UI
                self.root.after(0, lambda: self.display_predict_results(
                    heatmap_img, pred_binary, overlay_img, stats
                ))
                self.root.after(0, lambda: self.update_status("預測完成"))
                
            except Exception as e:
                error_msg = f"預測失敗: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
                self.root.after(0, lambda: self.update_status("預測失敗"))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def predict_image_full(self, model, image, task_id, device, patch_size=400):
        """完整影像預測"""
        h, w = image.shape[:2]
        result = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        stride = patch_size // 2
        
        with torch.no_grad():
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    y_start = max(0, y_end - patch_size)
                    x_start = max(0, x_end - patch_size)
                    
                    patch = image[y_start:y_end, x_start:x_end]
                    
                    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                        patch_padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                        patch_padded[:patch.shape[0], :patch.shape[1]] = patch
                        patch = patch_padded
                    
                    patch_tensor = torch.from_numpy(
                        patch.astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    pred = model(patch_tensor, task_id=task_id)
                    pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
                    
                    actual_h = min(patch.shape[0], y_end - y_start)
                    actual_w = min(patch.shape[1], x_end - x_start)
                    result[y_start:y_end, x_start:x_end] += pred[:actual_h, :actual_w]
                    count[y_start:y_end, x_start:x_end] += 1
        
        result = result / (count + 1e-7)
        return result
    
    def create_heatmap_image(self, prob_map, colormap='jet'):
        """創建熱圖"""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
        im = ax.imshow(prob_map, cmap=colormap, vmin=0, vmax=1)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout(pad=0)
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(img)
    
    def create_overlay_image(self, image, prob_map, threshold):
        """創建疊加圖"""
        mask = (prob_map > threshold).astype(np.uint8) * 255
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 1] = mask
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        return Image.fromarray(overlay)
    
    def display_predict_results(self, heatmap_img, binary_img, overlay_img, stats):
        """顯示預測結果"""
        # 熱圖
        heatmap_img.thumbnail((600, 600))
        heatmap_photo = ImageTk.PhotoImage(heatmap_img)
        self.heatmap_label.config(image=heatmap_photo, text='')
        self.heatmap_label.image = heatmap_photo
        
        # 二值化
        binary_pil = Image.fromarray(binary_img)
        binary_pil.thumbnail((600, 600))
        binary_photo = ImageTk.PhotoImage(binary_pil)
        self.binary_label.config(image=binary_photo, text='')
        self.binary_label.image = binary_photo
        
        # 疊加
        overlay_img.thumbnail((600, 600))
        overlay_photo = ImageTk.PhotoImage(overlay_img)
        self.overlay_label.config(image=overlay_photo, text='')
        self.overlay_label.image = overlay_photo
        
        # 統計
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats)
    
    # ========================================================================
    # Tab 4: 批量預測
    # ========================================================================
    
    def create_batch_tab(self):
        """創建批量預測頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 批量預測")
        
        # 控制面板
        control_frame = ttk.LabelFrame(tab, text="批量處理", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # 選擇檔案
        tk.Button(
            control_frame,
            text="📁 選擇多個影像",
            command=self.select_batch_images,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(pady=5)
        
        self.batch_files_label = tk.Label(control_frame, text="未選擇檔案", fg='gray')
        self.batch_files_label.pack(pady=5)
        
        # 參數
        param_frame = tk.Frame(control_frame)
        param_frame.pack(fill='x', pady=5)
        
        tk.Label(param_frame, text="任務:").pack(side='left', padx=5)
        self.batch_task_var = tk.StringVar(value='Cell (植物細胞)')
        task_menu = ttk.Combobox(
            param_frame,
            textvariable=self.batch_task_var,
            values=list(TASK_MAPPING.keys()),
            state='readonly',
            width=20
        )
        task_menu.pack(side='left', padx=5)
        
        tk.Label(param_frame, text="閾值:").pack(side='left', padx=5)
        self.batch_threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(
            param_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient='horizontal',
            variable=self.batch_threshold_var,
            length=200
        )
        threshold_scale.pack(side='left', padx=5)
        
        # 處理按鈕
        tk.Button(
            control_frame,
            text="🔍 批量預測",
            command=self.predict_batch,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # 進度條
        self.batch_progress = ttk.Progressbar(
            control_frame,
            mode='determinate',
            length=400
        )
        self.batch_progress.pack(pady=5)
        
        self.batch_progress_label = tk.Label(control_frame, text="")
        self.batch_progress_label.pack()
        
        # 結果顯示
        result_frame = ttk.LabelFrame(tab, text="處理結果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.batch_result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD)
        self.batch_result_text.pack(fill='both', expand=True)
        
        self.batch_files = []
    
    def select_batch_images(self):
        """選擇多個影像"""
        filenames = filedialog.askopenfilenames(
            title="選擇多個影像檔案",
            filetypes=[
                ("影像檔案", "*.jpg *.jpeg *.png *.bmp"),
                ("所有檔案", "*.*")
            ]
        )
        if filenames:
            self.batch_files = list(filenames)
            self.batch_files_label.config(
                text=f"已選擇 {len(self.batch_files)} 個檔案",
                fg='black'
            )
    
    def predict_batch(self):
        """執行批量預測"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("錯誤", "請先載入模型！")
            return
        
        if not self.batch_files:
            messagebox.showerror("錯誤", "請先選擇影像檔案！")
            return
        
        self.update_status("正在批量處理...")
        self.batch_result_text.delete('1.0', tk.END)
        self.batch_result_text.insert('1.0', "開始批量處理...\n\n")
        
        def batch_thread():
            global loaded_model, model_device  # 確保訪問全局變數
            task_id = TASK_MAPPING[self.batch_task_var.get()]
            threshold = self.batch_threshold_var.get()
            total = len(self.batch_files)
            results = []
            
            # 創建輸出目錄
            output_dir = Path('outputs/batch_predictions')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 創建時間戳記目錄
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            task_name = self.batch_task_var.get().split()[0]  # Cell, Blood, Root
            batch_output_dir = output_dir / f"{timestamp}_{task_name}"
            batch_output_dir.mkdir(exist_ok=True)
            
            for idx, filepath in enumerate(self.batch_files):
                try:
                    # 更新進度
                    progress = (idx + 1) / total * 100
                    self.root.after(0, lambda p=progress, i=idx+1: (
                        self.batch_progress.config(value=p),
                        self.batch_progress_label.config(text=f"處理中: {i}/{total}")
                    ))
                    
                    # 讀取和預測
                    image = Image.open(filepath)
                    image_rgb = np.array(image)
                    
                    if len(image_rgb.shape) == 2:
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                    elif image_rgb.shape[2] == 4:
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
                    
                    result = self.predict_image_full(loaded_model, image_rgb, task_id, model_device)
                    foreground_ratio = (result > threshold).sum() / result.size * 100
                    
                    # 生成輸出圖片
                    filename = os.path.basename(filepath)
                    name_without_ext = os.path.splitext(filename)[0]
                    
                    # 1. 保存機率熱圖
                    heatmap_img = self.create_heatmap_image(result, TASK_COLORS[task_id])
                    heatmap_path = batch_output_dir / f"{name_without_ext}_heatmap.png"
                    heatmap_img.save(heatmap_path)
                    
                    # 2. 保存二值化結果
                    pred_binary = (result > threshold).astype(np.uint8) * 255
                    binary_path = batch_output_dir / f"{name_without_ext}_binary.png"
                    cv2.imwrite(str(binary_path), pred_binary)
                    
                    # 3. 保存疊加圖
                    overlay_img = self.create_overlay_image(image_rgb, result, threshold)
                    overlay_path = batch_output_dir / f"{name_without_ext}_overlay.png"
                    overlay_img.save(overlay_path)
                    
                    # 4. 保存原圖（方便對照）
                    original_path = batch_output_dir / f"{name_without_ext}_original.png"
                    Image.fromarray(image_rgb).save(original_path)
                    
                    msg = f"✓ {filename}: {foreground_ratio:.2f}% (已保存)\n"
                    self.root.after(0, lambda m=msg: self.batch_result_text.insert(tk.END, m))
                    
                    results.append((filename, foreground_ratio))
                    
                except Exception as e:
                    filename = os.path.basename(filepath)
                    msg = f"✗ {filename}: 失敗 - {str(e)}\n"
                    self.root.after(0, lambda m=msg: self.batch_result_text.insert(tk.END, m))
            
            # 完成
            avg_ratio = sum(r[1] for r in results) / len(results) if results else 0
            summary = f"""
\n{'='*50}
✅ 批量處理完成！

📊 統計：
  • 成功處理: {len(results)} / {total}
  • 平均前景比例: {avg_ratio:.2f}%
  • 任務: {self.batch_task_var.get()}
  • 閾值: {threshold:.2f}

💾 輸出位置：
  {batch_output_dir}

📁 每張影像生成 4 個文件：
  • *_original.png  - 原始影像
  • *_heatmap.png   - 機率熱圖
  • *_binary.png    - 二值化結果
  • *_overlay.png   - 疊加圖
"""
            self.root.after(0, lambda: self.batch_result_text.insert(tk.END, summary))
            self.root.after(0, lambda: self.update_status("批量處理完成"))
            
            # 顯示完成對話框並詢問是否打開資料夾
            def show_completion():
                response = messagebox.askyesno(
                    "完成", 
                    f"✅ 成功處理 {len(results)}/{total} 個檔案！\n\n"
                    f"圖片已保存到:\n{batch_output_dir}\n\n"
                    f"是否打開輸出資料夾？"
                )
                if response:
                    # 打開輸出資料夾
                    try:
                        if sys.platform == 'win32':
                            os.startfile(batch_output_dir)
                        elif sys.platform == 'darwin':
                            subprocess.run(['open', batch_output_dir])
                        else:
                            subprocess.run(['xdg-open', batch_output_dir])
                    except Exception as e:
                        messagebox.showinfo("提示", f"請手動打開資料夾:\n{batch_output_dir}")
            
            self.root.after(0, show_completion)
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 5: 訓練監控
    # ========================================================================
    
    def create_monitor_tab(self):
        """創建訓練監控頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 訓練監控")
        
        # 按鈕區
        button_frame = tk.Frame(tab)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            button_frame,
            text="📈 載入訓練曲線",
            command=self.load_training_history,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="🖼️ 載入驗證影像",
            command=self.load_validation_images,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(side='left', padx=5)
        
        # 顯示區
        display_notebook = ttk.Notebook(tab)
        display_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 訓練曲線
        curve_frame = ttk.Frame(display_notebook)
        display_notebook.add(curve_frame, text="訓練曲線")
        
        self.curve_label = tk.Label(curve_frame, text="點擊「載入訓練曲線」查看", bg='lightgray')
        self.curve_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 統計資訊
        stats_frame = ttk.Frame(display_notebook)
        display_notebook.add(stats_frame, text="統計資訊")
        
        self.monitor_stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD)
        self.monitor_stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 驗證影像
        val_frame = ttk.Frame(display_notebook)
        display_notebook.add(val_frame, text="驗證影像")
        
        # 創建可滾動的 Canvas
        val_canvas = tk.Canvas(val_frame)
        val_scrollbar = ttk.Scrollbar(val_frame, orient="vertical", command=val_canvas.yview)
        self.val_images_frame = ttk.Frame(val_canvas)
        
        self.val_images_frame.bind(
            "<Configure>",
            lambda e: val_canvas.configure(scrollregion=val_canvas.bbox("all"))
        )
        
        val_canvas.create_window((0, 0), window=self.val_images_frame, anchor="nw")
        val_canvas.configure(yscrollcommand=val_scrollbar.set)
        
        val_canvas.pack(side="left", fill="both", expand=True)
        val_scrollbar.pack(side="right", fill="y")
    
    def load_training_history(self):
        """載入訓練歷史"""
        history_file = Path('outputs/training_history.json')
        
        if not history_file.exists():
            messagebox.showerror("錯誤", f"找不到訓練歷史檔案:\n{history_file}")
            return
        
        self.update_status("載入訓練歷史...")
        
        def load_thread():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # 創建訓練曲線
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
                
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                # Loss
                axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2, marker='o')
                if history.get('val_loss'):
                    axes[0, 0].plot(epochs[:len(history['val_loss'])], history['val_loss'], 
                                   label='Val', linewidth=2, marker='s')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # IoU
                if history.get('val_iou'):
                    axes[0, 1].plot(epochs[:len(history['val_iou'])], history['val_iou'], 
                                   linewidth=2, color='green', marker='o')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('IoU')
                    axes[0, 1].set_title('Validation IoU')
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Dice
                if history.get('val_dice'):
                    axes[1, 0].plot(epochs[:len(history['val_dice'])], history['val_dice'], 
                                   linewidth=2, color='blue', marker='o')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Dice')
                    axes[1, 0].set_title('Validation Dice')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Task IoU
                if history.get('task_metrics'):
                    for task_id, name in [(0, 'Cell'), (1, 'Blood'), (2, 'Root')]:
                        if str(task_id) in history['task_metrics']:
                            metrics = history['task_metrics'][str(task_id)]
                            if metrics:
                                ious = [m.get('iou', 0) for m in metrics if 'iou' in m]
                                if ious:
                                    axes[1, 1].plot(epochs[:len(ious)], ious, label=name, linewidth=2, marker='o')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('IoU')
                    axes[1, 1].set_title('Task-specific IoU')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # 轉換為圖片（修復 matplotlib 兼容性）
                fig.canvas.draw()
                
                # 使用兼容的方法獲取圖片數據
                # 新版本 matplotlib 使用 buffer_rgba()
                try:
                    # 嘗試新方法
                    buf = fig.canvas.buffer_rgba()
                    img = np.asarray(buf)
                    # RGBA 轉 RGB
                    img = img[:, :, :3]
                except AttributeError:
                    # 舊方法作為後備
                    try:
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    except AttributeError:
                        # 最後的後備方案
                        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                        img = img[:, :, 1:]  # 移除 alpha 通道，保留 RGB
                
                plt.close(fig)
                
                curve_img = Image.fromarray(img)
                
                # 統計
                best_epoch = np.argmax(history.get('val_iou', [0]))
                
                # 安全地獲取和格式化統計數據
                best_val_dice = history['val_dice'][best_epoch] if (history.get('val_dice') and len(history['val_dice']) > best_epoch) else None
                best_val_dice_str = f"{best_val_dice:.4f}" if best_val_dice is not None else 'N/A'
                
                final_val_iou = history['val_iou'][-1] if history.get('val_iou') else None
                final_val_iou_str = f"{final_val_iou:.4f}" if final_val_iou is not None else 'N/A'
                
                stats = f"""
📈 訓練歷史統計

🏆 最佳結果（Epoch {best_epoch + 1}）：
  • Val IoU: {history.get('val_iou', [0])[best_epoch]:.4f}
  • Val Dice: {best_val_dice_str}

📊 最終結果：
  • Train Loss: {history['train_loss'][-1]:.4f}
  • Val IoU: {final_val_iou_str}
  
📉 訓練進度：
  • 總 Epochs: {len(history['train_loss'])}
  • Loss 降低: {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%
"""
                
                # 更新 UI
                self.root.after(0, lambda: self.display_training_curve(curve_img, stats))
                self.root.after(0, lambda: self.update_status("訓練歷史已載入"))
                
            except Exception as e:
                error_msg = f"載入失敗: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
                self.root.after(0, lambda: self.update_status("載入失敗"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def display_training_curve(self, curve_img, stats):
        """顯示訓練曲線"""
        curve_img.thumbnail((900, 700))
        photo = ImageTk.PhotoImage(curve_img)
        self.curve_label.config(image=photo, text='')
        self.curve_label.image = photo
        
        self.monitor_stats_text.delete('1.0', tk.END)
        self.monitor_stats_text.insert('1.0', stats)
    
    def load_validation_images(self):
        """載入驗證影像"""
        pred_dir = Path('outputs/predictions')
        
        if not pred_dir.exists():
            messagebox.showerror("錯誤", f"找不到驗證影像目錄:\n{pred_dir}")
            return
        
        val_images = sorted(pred_dir.glob('val_epoch*.png'))
        
        if not val_images:
            messagebox.showerror("錯誤", "沒有找到驗證影像")
            return
        
        # 清空舊的
        for widget in self.val_images_frame.winfo_children():
            widget.destroy()
        
        # 顯示所有驗證影像（而不是只顯示最後 6 張）
        self.update_status(f"正在載入 {len(val_images)} 張驗證影像...")
        
        for img_path in val_images:  # 顯示所有圖片
            try:
                img = Image.open(img_path)
                img.thumbnail((800, 400))
                photo = ImageTk.PhotoImage(img)
                
                # 添加框架以分隔每張圖片
                img_container = tk.Frame(self.val_images_frame, relief=tk.RIDGE, borderwidth=2)
                img_container.pack(pady=10, fill='x')
                
                # 圖片標題
                name_label = tk.Label(
                    img_container, 
                    text=img_path.name,
                    font=('Arial', 10, 'bold'),
                    bg='lightblue'
                )
                name_label.pack(fill='x')
                
                # 圖片
                label = tk.Label(img_container, image=photo)
                label.image = photo
                label.pack(pady=5)
                
            except Exception as e:
                print(f"載入失敗: {img_path} - {e}")
        
        self.update_status(f"✅ 已載入 {len(val_images)} 張驗證影像")
    
    # ========================================================================
    # Tab 6: 使用說明
    # ========================================================================
    
    def create_help_tab(self):
        """創建使用說明頁面"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📖 使用說明")
        
        help_text = scrolledtext.ScrolledText(tab, wrap=tk.WORD, font=('Arial', 10))
        help_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_content = """
🔬 多任務 TransUNet - 預測專用版

版本: Tkinter GUI v2.0 (預測專用)
適用: Windows / Linux / Mac
GPU: 支援 RTX 5080 及所有 NVIDIA GPU

═══════════════════════════════════════════════════════════

⚠️ 重要提示

本版本專注於【模型載入與預測】功能
訓練功能請使用：python app_train.py (Gradio 訓練介面)

═══════════════════════════════════════════════════════════

📖 快速開始

1️⃣ 載入模型
   • 前往「模型管理」頁面
   • 選擇模型檔案（如 outputs/models/best_model.pth）
   • 選擇計算設備（GPU 推薦）
   • 點擊「載入模型」

2️⃣ 單張預測
   • 前往「單張預測」頁面
   • 選擇影像檔案
   • 選擇任務類型（Cell/Blood/Root）
   • 調整閾值
   • 點擊「開始預測」

3️⃣ 批量處理
   • 前往「批量預測」頁面
   • 選擇多個影像
   • 設定參數
   • 點擊「批量預測」
   • 結果自動保存到 outputs/batch_predictions/

4️⃣ 查看訓練結果
   • 前往「訓練監控」頁面
   • 載入訓練曲線
   • 查看驗證影像

═══════════════════════════════════════════════════════════

🎯 任務說明

🌿 Cell (植物細胞)
   • 適用：植物細胞壁影像
   • 特點：多邊形結構
   • 建議閾值：0.5-0.7

🩸 Blood (血球)
   • 適用：血球細胞影像
   • 特點：圓形結構
   • 建議閾值：0.4-0.6

🌱 Root (根系)
   • 適用：植物根系影像
   • 特點：線性結構
   • 建議閾值：0.3-0.5

═══════════════════════════════════════════════════════════

🚀 訓練模型（使用專用介面）

訓練功能已移至專用介面，請執行：

    python app_train.py

這會啟動一個 Web 介面（Gradio），專門用於訓練：
• 開啟瀏覽器自動顯示訓練介面
• 設定訓練參數更直觀
• 訓練狀態監控更方便
• 避免編碼和相容性問題

準備資料結構：
  data/
  ├── train/
  │   ├── cell/images/ + masks/
  │   ├── blood/images/ + masks/
  │   └── root/images/ + masks/
  └── val/（相同結構）

訓練完成後，回到本介面進行預測！

═══════════════════════════════════════════════════════════

📁 批量預測輸出位置

輸出目錄：
  outputs/batch_predictions/YYYYMMDD_HHMMSS_TaskName/

每張影像生成 4 個文件：
  • *_original.png  - 原始影像
  • *_heatmap.png   - 機率熱圖
  • *_binary.png    - 二值化結果
  • *_overlay.png   - 疊加圖

範例：
  outputs/batch_predictions/20251225_163045_Cell/
  ├── image1_original.png
  ├── image1_heatmap.png
  ├── image1_binary.png
  └── image1_overlay.png

批量預測完成後會彈出對話框，詢問是否打開輸出資料夾。

═══════════════════════════════════════════════════════════

⚙️ 系統需求

最低配置：
  • RAM: 8GB
  • CPU: Intel i5 或同等級
  • 作業系統: Windows 10 / Ubuntu 18.04 / macOS 10.14

推薦配置：
  • RAM: 16GB
  • GPU: NVIDIA RTX 3060 或更高（包括 RTX 5080）
  • VRAM: 8GB
  • Python: 3.8 或更高

═══════════════════════════════════════════════════════════

❓ 常見問題

Q: 預測結果全黑？
A: 降低閾值到 0.3，或檢查模型是否正確載入

Q: GPU 無法使用？
A: 確認已安裝 CUDA 版本的 PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Q: 程式運行緩慢？
A: 使用 GPU 模式，速度可提升 10-50 倍

Q: RTX 5080 支援嗎？
A: 完全支援！只需確保 PyTorch 版本支援你的 CUDA 版本

Q: 如何訓練模型？
A: 使用專用訓練介面：python app_train.py

Q: 批量預測的圖片在哪？
A: outputs/batch_predictions/YYYYMMDD_HHMMSS_TaskName/

═══════════════════════════════════════════════════════════

💡 使用技巧

1. 閾值調整
   • 根系使用較低閾值 (0.3-0.5)
   • 細胞使用中等閾值 (0.5-0.7)
   • 可即時調整觀察效果

2. 批量處理
   • 一次可處理任意數量影像
   • 相同類型影像使用相同設定
   • 結果自動保存，不會覆蓋

3. GPU 使用
   • 第一次預測可能較慢（初始化）
   • 後續預測會更快
   • 關閉其他佔用 GPU 的程式

═══════════════════════════════════════════════════════════

📁 檔案位置

模型：outputs/models/
  • best_model.pth - 最佳模型
  • final_model.pth - 最終模型
  • checkpoint_epoch*.pth - 檢查點

批量預測：outputs/batch_predictions/
  • YYYYMMDD_HHMMSS_TaskName/ - 每次批量處理的結果

訓練結果：outputs/
  • training_history.json - 訓練歷史
  • predictions/ - 驗證影像

═══════════════════════════════════════════════════════════

🔧 故障排除

1. 模組導入錯誤
   確認所有必要檔案都在同一目錄：
   • model_multitask.py
   • dataset_multitask.py
   • losses_multitask.py

2. 模型載入失敗
   • 檢查模型檔案是否存在
   • 確認檔案路徑正確
   • 嘗試重新下載模型

3. GPU 記憶體不足
   • 關閉其他佔用 GPU 的程式
   • 使用 CPU 模式
   • 處理較小的影像

═══════════════════════════════════════════════════════════

🔗 相關工具

• 訓練介面：python app_train.py
• 預測介面：python app_gui.py（當前）

═══════════════════════════════════════════════════════════

📞 支援與回饋

如有問題或建議，歡迎回饋！

版本資訊：
• Tkinter GUI v2.0 - 預測專用
• 訓練功能：使用 app_train.py
• 最後更新：2025-12-25
"""
        
        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')


def main():
    """主程式"""
    root = tk.Tk()
    app = MultiTaskGUI(root)
    
    print("\n" + "="*60)
    print("🚀 多任務 TransUNet - Tkinter 桌面版")
    print("="*60)
    print("\n✅ GUI 已啟動！")
    print("💡 提示：")
    print("   • 不需要 Gradio，使用 Python 標準庫")
    print("   • 支援所有 GPU，包括 RTX 5080")
    print("   • 請在 GUI 視窗中操作")
    print("\n" + "="*60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    # 創建必要目錄
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
    
    main()
