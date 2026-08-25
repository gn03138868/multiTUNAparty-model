"""
Multi-Task TransUNet - Desktop GUI (Tkinter)

A graphical user interface for model inference and prediction.
Compatible with all environments including RTX 5080 GPUs.

Features:
1. Model Management
2. Single Image Prediction
3. Batch Processing
4. Training Monitoring
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

# Import model
try:
    from model_multitask import MultiTaskTransUNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: model_multitask.py not found")

# Global configuration
TASK_MAPPING = {
    'Cell (Plant Cell)': 0,
    'Blood (Blood Cell)': 1,
    'Root (Root System)': 2
}

TASK_COLORS = {
    0: 'Blues',
    1: 'Reds',
    2: 'Greens'
}

# Global variables
loaded_model = None
model_device = None
training_process = None
training_status = {
    'is_training': False,
    'message': 'Training not yet started'
}


class MultiTaskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Task TransUNet - Desktop Edition")
        self.root.geometry("1200x800")
        
        # Set icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Create status bar
        self.create_status_bar()
        
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create feature tabs
        self.create_model_tab()
        self.create_predict_tab()
        self.create_batch_tab()
        self.create_monitor_tab()
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
        
        # Load in background thread
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
                
                model.to(device)
                model.eval()
                
                # Update global variables
                loaded_model = model
                model_device = device
                
                # Calculate parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Update UI
                info = f"""
✅ Model loaded successfully!

📊 Model Information:
  • Device: {device_info}
  • Parameters: {total_params:,}
  • Model Path: {model_path}
  • Load Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 Supported Tasks:
  • Cell (Plant Cell)
  • Blood (Blood Cell)
  • Root (Root System)

✓ Ready for prediction!
"""
                
                self.root.after(0, lambda: self.model_info_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.model_info_text.insert('1.0', info))
                self.root.after(0, lambda: self.update_status("Model loaded - Ready"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model loaded successfully!"))
                
            except Exception as e:
                error_msg = f"❌ Loading failed: {str(e)}"
                self.root.after(0, lambda: self.model_info_text.delete('1.0', tk.END))
                self.root.after(0, lambda: self.model_info_text.insert('1.0', error_msg))
                self.root.after(0, lambda: self.update_status("Loading failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 2: Single Image Prediction
    # ========================================================================
    
    def create_predict_tab(self):
        """Create single prediction tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔬 Single Prediction")
        
        # Title
        title = tk.Label(tab, text="Single Image Prediction", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Main content area
        main_frame = tk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=10)
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Settings", padding=10)
        left_frame.pack(side='left', fill='y', padx=5)
        
        # Image selection
        tk.Label(left_frame, text="Input Image:").pack(anchor='w', pady=5)
        
        img_frame = tk.Frame(left_frame)
        img_frame.pack(fill='x', pady=5)
        
        self.predict_image_var = tk.StringVar()
        tk.Entry(img_frame, textvariable=self.predict_image_var, width=30).pack(side='left')
        tk.Button(img_frame, text="Browse", command=self.browse_predict_image).pack(side='left', padx=5)
        
        # Task selection
        tk.Label(left_frame, text="Task Type:").pack(anchor='w', pady=5)
        self.predict_task_var = tk.StringVar(value="Cell (Plant Cell)")
        task_combo = ttk.Combobox(left_frame, textvariable=self.predict_task_var, values=list(TASK_MAPPING.keys()))
        task_combo.pack(fill='x', pady=5)
        
        # Threshold
        tk.Label(left_frame, text="Threshold:").pack(anchor='w', pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(left_frame, from_=0.0, to=1.0, resolution=0.05, 
                                   orient='horizontal', variable=self.threshold_var)
        threshold_scale.pack(fill='x', pady=5)
        
        # Predict button
        tk.Button(
            left_frame, 
            text="🚀 Start Prediction", 
            command=self.run_prediction,
            bg='#2196F3',
            fg='white',
            font=('Arial', 12, 'bold'),
            pady=10
        ).pack(fill='x', pady=20)
        
        # Save button
        tk.Button(
            left_frame, 
            text="💾 Save Result", 
            command=self.save_prediction,
            pady=5
        ).pack(fill='x', pady=5)
        
        # Right panel - Results display
        right_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # Result images
        self.result_canvas = tk.Canvas(right_frame, bg='white')
        self.result_canvas.pack(fill='both', expand=True)
        
        # Metrics display
        self.metrics_label = tk.Label(right_frame, text="", font=('Arial', 10))
        self.metrics_label.pack(pady=5)
    
    def browse_predict_image(self):
        """Browse and select image for prediction"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if filename:
            self.predict_image_var.set(filename)
    
    def run_prediction(self):
        """Run prediction"""
        global loaded_model, model_device
        
        if loaded_model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        image_path = self.predict_image_var.get()
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file does not exist:\n{image_path}")
            return
        
        self.update_status("Running prediction...")
        
        def predict_thread():
            try:
                # Load image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize
                image_resized = cv2.resize(image, (400, 400))
                
                # Preprocess
                img_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.to(model_device)
                
                # Get task ID
                task_id = TASK_MAPPING[self.predict_task_var.get()]
                
                # Predict
                with torch.no_grad():
                    output = loaded_model(img_tensor, task_id=task_id)
                    prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
                
                # Apply threshold
                threshold = self.threshold_var.get()
                binary_mask = (prob_map > threshold).astype(np.uint8) * 255
                
                # Store results
                self.current_prediction = {
                    'original': image_resized,
                    'prob_map': prob_map,
                    'binary': binary_mask,
                    'threshold': threshold
                }
                
                # Update display
                self.root.after(0, self.display_prediction)
                self.root.after(0, lambda: self.update_status("Prediction complete"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Prediction failed"))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def display_prediction(self):
        """Display prediction results"""
        if not hasattr(self, 'current_prediction'):
            return
        
        pred = self.current_prediction
        
        # Create combined visualisation
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(pred['original'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred['prob_map'], cmap='jet')
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
        img.thumbnail((800, 400))
        photo = ImageTk.PhotoImage(img)
        
        self.result_canvas.delete('all')
        self.result_canvas.create_image(400, 200, image=photo)
        self.result_canvas.image = photo
    
    def save_prediction(self):
        """Save prediction result"""
        if not hasattr(self, 'current_prediction'):
            messagebox.showerror("Error", "No prediction result to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        
        if filename:
            pred = self.current_prediction
            
            # Create combined image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(pred['original'])
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred['prob_map'], cmap='jet')
            axes[1].set_title('Probability Map')
            axes[1].axis('off')
            
            axes[2].imshow(pred['binary'], cmap='gray')
            axes[2].set_title(f"Binary Mask (Threshold: {pred['threshold']:.2f})")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            messagebox.showinfo("Success", f"Result saved to:\n{filename}")
    
    # ========================================================================
    # Tab 3: Batch Processing
    # ========================================================================
    
    def create_batch_tab(self):
        """Create batch processing tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📁 Batch Processing")
        
        # Title
        title = tk.Label(tab, text="Batch Image Processing", font=('Arial', 16, 'bold'))
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
                 variable=self.batch_threshold_var, length=150).pack(side='left', padx=5)
        
        # Run button
        tk.Button(
            settings_frame,
            text="🚀 Start Batch Processing",
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
                         list(folder_path.glob('*.png')) + list(folder_path.glob('*.bmp'))
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
        for f in self.batch_files[:10]:  # Show first 10
            self.batch_files_text.insert(tk.END, f"  • {Path(f).name}\n")
        if len(self.batch_files) > 10:
            self.batch_files_text.insert(tk.END, f"  ... and {len(self.batch_files) - 10} more\n")
    
    def run_batch_prediction(self):
        """Run batch prediction"""
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
        self.batch_results_text.insert('1.0', f"Output directory: {output_dir}\n\n")
        
        def batch_thread():
            task_id = TASK_MAPPING[self.batch_task_var.get()]
            threshold = self.batch_threshold_var.get()
            total = len(self.batch_files)
            
            for i, image_path in enumerate(self.batch_files):
                try:
                    # Update progress
                    progress = (i + 1) / total * 100
                    self.root.after(0, lambda p=progress: self.batch_progress.configure(value=p))
                    self.root.after(0, lambda n=i+1, t=total: self.batch_status_label.configure(
                        text=f"Processing {n}/{t}..."))
                    
                    # Load and process image
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image, (400, 400))
                    
                    # Preprocess
                    img_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(model_device)
                    
                    # Predict
                    with torch.no_grad():
                        output = loaded_model(img_tensor, task_id=task_id)
                        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
                    
                    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
                    
                    # Save results
                    base_name = Path(image_path).stem
                    
                    cv2.imwrite(str(output_dir / f'{base_name}_original.png'), 
                               cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                    
                    plt.imsave(str(output_dir / f'{base_name}_heatmap.png'), prob_map, cmap='jet')
                    cv2.imwrite(str(output_dir / f'{base_name}_binary.png'), binary_mask)
                    
                    # Create overlay
                    overlay = image_resized.copy()
                    mask_colored = np.zeros_like(overlay)
                    mask_colored[binary_mask > 0] = [0, 255, 0]
                    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                    cv2.imwrite(str(output_dir / f'{base_name}_overlay.png'), 
                               cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    
                    # Update log
                    self.root.after(0, lambda p=image_path: self.batch_results_text.insert(
                        tk.END, f"✓ {Path(p).name}\n"))
                    
                except Exception as e:
                    self.root.after(0, lambda p=image_path, err=e: self.batch_results_text.insert(
                        tk.END, f"✗ {Path(p).name}: {err}\n"))
            
            # Complete
            self.root.after(0, lambda: self.batch_status_label.configure(text="Complete!"))
            self.root.after(0, lambda: self.batch_results_text.insert(
                tk.END, f"\n✅ Batch processing complete!\nResults saved to: {output_dir}"))
            self.root.after(0, lambda: messagebox.showinfo("Complete", 
                f"Batch processing complete!\n\nResults saved to:\n{output_dir}\n\nOpen output folder?"))
            
            # Open output folder
            if messagebox.askyesno("Open Folder", "Open output folder?"):
                os.startfile(str(output_dir)) if os.name == 'nt' else subprocess.run(['xdg-open', str(output_dir)])
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    # ========================================================================
    # Tab 4: Training Monitor
    # ========================================================================
    
    def create_monitor_tab(self):
        """Create training monitor tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Training Monitor")
        
        # Title
        title = tk.Label(tab, text="Training Progress Monitoring", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Controls
        ctrl_frame = tk.Frame(tab)
        ctrl_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(ctrl_frame, text="🔄 Refresh Training Curve", 
                 command=self.load_training_curve).pack(side='left', padx=5)
        tk.Button(ctrl_frame, text="📷 Load Validation Images", 
                 command=self.load_validation_images).pack(side='left', padx=5)
        
        # Main content with scrollbar
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Training curve display
        curve_frame = ttk.LabelFrame(self.scrollable_frame, text="Training Curve", padding=10)
        curve_frame.pack(fill='x', pady=5)
        
        self.curve_label = tk.Label(curve_frame, text="Click 'Refresh Training Curve' to load")
        self.curve_label.pack()
        
        # Statistics display
        stats_frame = ttk.LabelFrame(self.scrollable_frame, text="Training Statistics", padding=10)
        stats_frame.pack(fill='x', pady=5)
        
        self.monitor_stats_text = scrolledtext.ScrolledText(stats_frame, height=10, wrap=tk.WORD)
        self.monitor_stats_text.pack(fill='x')
        
        # Validation images
        val_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Images", padding=10)
        val_frame.pack(fill='both', expand=True, pady=5)
        
        self.val_images_frame = ttk.Frame(val_frame)
        self.val_images_frame.pack(fill='both', expand=True)
    
    def load_training_curve(self):
        """Load and display training curve"""
        curve_path = Path('outputs/training_history.png')
        history_path = Path('outputs/training_history.json')
        
        if not curve_path.exists():
            messagebox.showerror("Error", f"Training curve not found:\n{curve_path}")
            return
        
        # Load image
        img = Image.open(curve_path)
        img.thumbnail((800, 400))
        photo = ImageTk.PhotoImage(img)
        
        self.curve_label.config(image=photo, text='')
        self.curve_label.image = photo
        
        # Load statistics
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                epochs = len(history.get('train_loss', []))
                stats = f"""
Training Statistics:
{'='*50}

Total Epochs: {epochs}

Loss:
  Initial Train Loss: {history['train_loss'][0]:.4f}
  Final Train Loss:   {history['train_loss'][-1]:.4f}
  Initial Val Loss:   {history['val_loss'][0]:.4f}
  Final Val Loss:     {history['val_loss'][-1]:.4f}

IoU:
  Initial Val IoU: {history['val_iou'][0]:.4f}
  Final Val IoU:   {history['val_iou'][-1]:.4f}
  Improvement:     {history['val_iou'][-1] - history['val_iou'][0]:.4f}

Dice:
  Initial Val Dice: {history['val_dice'][0]:.4f}
  Final Val Dice:   {history['val_dice'][-1]:.4f}
"""
                self.monitor_stats_text.delete('1.0', tk.END)
                self.monitor_stats_text.insert('1.0', stats)
                
            except Exception as e:
                self.monitor_stats_text.delete('1.0', tk.END)
                self.monitor_stats_text.insert('1.0', f"Error loading statistics: {e}")
        
        self.update_status("Training curve loaded")
    
    def load_validation_images(self):
        """Load validation images"""
        pred_dir = Path('outputs/predictions')
        
        if not pred_dir.exists():
            messagebox.showerror("Error", f"Validation image directory not found:\n{pred_dir}")
            return
        
        val_images = sorted(pred_dir.glob('val_epoch*.png'))
        
        if not val_images:
            messagebox.showerror("Error", "No validation images found")
            return
        
        # Clear old images
        for widget in self.val_images_frame.winfo_children():
            widget.destroy()
        
        self.update_status(f"Loading {len(val_images)} validation images...")
        
        for img_path in val_images:
            try:
                img = Image.open(img_path)
                img.thumbnail((800, 400))
                photo = ImageTk.PhotoImage(img)
                
                img_container = tk.Frame(self.val_images_frame, relief=tk.RIDGE, borderwidth=2)
                img_container.pack(pady=10, fill='x')
                
                name_label = tk.Label(
                    img_container, 
                    text=img_path.name,
                    font=('Arial', 10, 'bold'),
                    bg='lightblue'
                )
                name_label.pack(fill='x')
                
                label = tk.Label(img_container, image=photo)
                label.image = photo
                label.pack(pady=5)
                
            except Exception as e:
                print(f"Failed to load: {img_path} - {e}")
        
        self.update_status(f"✅ Loaded {len(val_images)} validation images")
    
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
🔬 Multi-Task TransUNet - Prediction Interface

Version: Tkinter GUI v2.0 (Prediction)
Platforms: Windows / Linux / macOS
GPU: Supports RTX 5080 and all NVIDIA GPUs

═══════════════════════════════════════════════════════════

⚠️ IMPORTANT NOTICE

This version focuses on MODEL LOADING AND PREDICTION.
For training, please use: python app_train.py (Gradio Training Interface)

═══════════════════════════════════════════════════════════

📖 QUICK START GUIDE

1️⃣ Load Model
   • Navigate to the "Model Management" tab
   • Select your model file (e.g., outputs/models/best_model.pth)
   • Choose computing device (GPU recommended)
   • Click "Load Model"

2️⃣ Single Prediction
   • Navigate to "Single Prediction" tab
   • Select an image file
   • Choose the task type (Cell/Blood/Root)
   • Adjust the threshold value
   • Click "Start Prediction"

3️⃣ Batch Processing
   • Navigate to "Batch Processing" tab
   • Select multiple images or a folder
   • Configure parameters
   • Click "Start Batch Processing"
   • Results are automatically saved to outputs/batch_predictions/

4️⃣ View Training Results
   • Navigate to "Training Monitor" tab
   • Load training curves
   • View validation images

═══════════════════════════════════════════════════════════

🎯 TASK DESCRIPTIONS

🌿 Cell (Plant Cell)
   • For: Plant cell wall images
   • Characteristics: Polygonal structures
   • Recommended threshold: 0.5-0.7

🩸 Blood (Blood Cell)
   • For: Blood cell images
   • Characteristics: Circular structures
   • Recommended threshold: 0.4-0.6

🌱 Root (Root System)
   • For: Plant root system images
   • Characteristics: Linear structures
   • Recommended threshold: 0.3-0.5

═══════════════════════════════════════════════════════════

🚀 TRAINING (Using Dedicated Interface)

Training functionality has been moved to a dedicated interface.
Please run:

    python app_train.py

This will launch a web interface (Gradio) specifically for training:
• Browser opens automatically with training interface
• More intuitive parameter configuration
• Better training status monitoring
• Avoids encoding and compatibility issues

Data structure required:
  data/
  ├── train/
  │   ├── cell/images/ + masks/
  │   ├── blood/images/ + masks/
  │   └── root/images/ + masks/
  └── val/ (same structure)

After training, return to this interface for prediction!

═══════════════════════════════════════════════════════════

📁 BATCH PREDICTION OUTPUT

Output directory:
  outputs/batch_predictions/YYYYMMDD_HHMMSS_TaskName/

Each image generates 4 files:
  • *_original.png  - Original image
  • *_heatmap.png   - Probability heatmap
  • *_binary.png    - Binary mask
  • *_overlay.png   - Overlay visualisation

Example:
  outputs/batch_predictions/20251225_163045_Cell/
  ├── image1_original.png
  ├── image1_heatmap.png
  ├── image1_binary.png
  └── image1_overlay.png

═══════════════════════════════════════════════════════════

⚙️ SYSTEM REQUIREMENTS

Minimum:
  • RAM: 8GB
  • CPU: Intel i5 or equivalent
  • OS: Windows 10 / Ubuntu 18.04 / macOS 10.14

Recommended:
  • RAM: 16GB
  • GPU: NVIDIA RTX 3060 or higher (including RTX 5080)
  • VRAM: 8GB
  • Python: 3.8 or higher

═══════════════════════════════════════════════════════════

❓ FREQUENTLY ASKED QUESTIONS

Q: Prediction result is completely black?
A: Try lowering the threshold to 0.3, or verify the model loaded correctly

Q: GPU not available?
A: Ensure you have installed the CUDA version of PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Q: Programme running slowly?
A: Use GPU mode for 10-50x speed improvement

Q: Is RTX 5080 supported?
A: Fully supported! Just ensure your PyTorch version supports your CUDA version

Q: How do I train a model?
A: Use the dedicated training interface: python app_train.py

Q: Where are my batch prediction images?
A: outputs/batch_predictions/YYYYMMDD_HHMMSS_TaskName/

═══════════════════════════════════════════════════════════

💡 TIPS AND TRICKS

1. Threshold Adjustment
   • Use lower thresholds for root systems (0.3-0.5)
   • Use medium thresholds for cells (0.5-0.7)
   • Adjust in real-time to observe effects

2. Batch Processing
   • Process any number of images at once
   • Use the same settings for similar image types
   • Results are saved automatically without overwriting

3. GPU Usage
   • First prediction may be slower (initialisation)
   • Subsequent predictions will be faster
   • Close other GPU-intensive programmes

═══════════════════════════════════════════════════════════

📁 FILE LOCATIONS

Models: outputs/models/
  • best_model.pth - Best model
  • final_model.pth - Final model
  • checkpoint_epoch*.pth - Checkpoints

Batch Predictions: outputs/batch_predictions/
  • YYYYMMDD_HHMMSS_TaskName/ - Results from each batch run

Training Results: outputs/
  • training_history.json - Training history
  • predictions/ - Validation images

═══════════════════════════════════════════════════════════

🔧 TROUBLESHOOTING

1. Module Import Error
   Ensure all required files are in the same directory:
   • model_multitask.py
   • dataset_multitask.py
   • losses_multitask.py

2. Model Loading Failed
   • Check model file exists
   • Verify file path is correct
   • Try re-downloading the model

3. GPU Memory Insufficient
   • Close other GPU-intensive programmes
   • Use CPU mode
   • Process smaller images

═══════════════════════════════════════════════════════════

🔗 RELATED TOOLS

• Training Interface: python app_train.py
• Prediction Interface: python app_gui.py (current)

═══════════════════════════════════════════════════════════

📞 SUPPORT

For questions or suggestions, please provide feedback!

Version Information:
• Tkinter GUI v2.0 - Prediction
• Training: Use app_train.py
• Last Updated: 2025-01-27
"""
        
        help_text.insert('1.0', help_content)
        help_text.config(state='disabled')


def main():
    """Main programme"""
    root = tk.Tk()
    app = MultiTaskGUI(root)
    
    print("\n" + "="*60)
    print("🚀 Multi-Task TransUNet - Desktop Edition")
    print("="*60)
    print("\n✅ GUI Launched!")
    print("💡 Tips:")
    print("   • No Gradio required, uses Python standard library")
    print("   • Supports all GPUs, including RTX 5080")
    print("   • Please operate within the GUI window")
    print("\n" + "="*60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    # Create necessary directories
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/predictions').mkdir(parents=True, exist_ok=True)
    
    main()
