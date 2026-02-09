"""
TransUNet Training Web Interface (Gradio)
Dedicated to model training functionality
"""

import gradio as gr
import subprocess
import sys
import yaml
from pathlib import Path
import json
import time
import os

# Global variables
training_process = None
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'val_iou': 0.0,
    'message': 'Training not yet started',
    'log_file': '',
    'error_message': ''
}

def start_training(batch_size, epochs, lr, patch_size, num_layers, data_path, 
                  use_pretrained, pretrained_path):
    """Start training"""
    global training_process, training_status
    
    # Check if already training
    if training_status['is_training']:
        return "⚠️ Training is already in progress! Please wait for the current training to complete or stop it first."
    
    try:
        # Update configuration
        config = {
            'batch_size': int(batch_size),
            'epochs': int(epochs),
            'lr': float(lr),
            'patch_size': int(patch_size),
            'num_decoder_conv_layers': int(num_layers),
            'data_path': data_path,
            'task_structure': 'subfolder',
            'boundary_weights': {0: 2.0, 1: 3.0, 2: 5.0},
            'foreground_weights': {0: 1.0, 1: 1.5, 2: 3.0}
        }
        
        # Add pre-trained settings
        if use_pretrained and pretrained_path:
            config['pretrained_model_path'] = pretrained_path
        
        # Save configuration
        config_path = Path('config_training_ui.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        # Create training log file
        log_file = Path('outputs/training_ui.log')
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialise training status
        training_status = {
            'is_training': True,
            'current_epoch': 0,
            'total_epochs': int(epochs),
            'train_loss': 0.0,
            'val_loss': 0.0,
            'val_iou': 0.0,
            'message': 'Starting training...',
            'log_file': str(log_file),
            'error_message': ''
        }
        
        # Start training in background thread
        def run_training():
            global training_process, training_status
            
            try:
                print("\n" + "="*60)
                print("🚀 Starting Training...")
                print("="*60)
                print(f"Configuration file: {config_path}")
                print(f"Log file: {log_file}")
                print(f"Batch Size: {batch_size}")
                print(f"Epochs: {epochs}")
                print(f"Learning Rate: {lr}")
                print("="*60 + "\n")
                
                # Start training process
                training_process = subprocess.Popen(
                    [sys.executable, 'train_multitask.py', '--config', str(config_path)],
                )
                
                training_status['message'] = '✅ Training has started! Please check the terminal window for detailed output.'
                
                # Wait for training to complete
                training_process.wait()
                
                # Check return code
                if training_process.returncode == 0:
                    training_status['message'] = '✅ Training completed successfully!'
                else:
                    training_status['message'] = f'❌ Training failed (return code: {training_process.returncode})'
                
            except Exception as e:
                training_status['error_message'] = str(e)
                training_status['message'] = f'❌ Failed to start training: {str(e)}'
                print(f"Training error: {e}")
            finally:
                training_status['is_training'] = False
                training_process = None
        
        # Start training thread
        import threading
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        return f"""
✅ Training has been successfully started!

📊 Training Configuration:
• Batch Size: {batch_size}
• Epochs: {epochs}
• Learning Rate: {lr}
• Patch Size: {patch_size}
• Decoder Layers: {num_layers}
• Data Path: {data_path}
• Pre-trained Model: {'Yes (' + pretrained_path + ')' if use_pretrained else 'No'}

💡 Tips:
• Detailed training output will be displayed in the terminal (CMD/Terminal)
• Please keep the terminal window open to view training progress
• Models will be automatically saved to outputs/models/
• Training history will be saved to outputs/training_history.json

🔄 Click "Refresh Training Status" to view current progress
"""
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def stop_training():
    """Stop training"""
    global training_process, training_status
    
    if not training_status['is_training']:
        return "⚠️ No training is currently in progress."
    
    try:
        if training_process and training_process.poll() is None:
            training_process.terminate()
            training_process.wait(timeout=5)
            training_status['is_training'] = False
            training_status['message'] = '⏹️ Training stopped'
            return "✅ Training has been stopped successfully."
        else:
            training_status['is_training'] = False
            return "⚠️ Training process has already finished."
    except Exception as e:
        return f"❌ Error occurred whilst stopping training: {str(e)}"


def get_training_status():
    """Get training status"""
    global training_status
    
    if not training_status['is_training']:
        return training_status['message']
    
    # Try to read training history
    history_file = Path('outputs/training_history.json')
    if history_file.exists():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if history and len(history.get('train_loss', [])) > 0:
                current_epoch = len(history['train_loss'])
                train_loss = history['train_loss'][-1]
                val_loss = history['val_loss'][-1]
                val_iou = history['val_iou'][-1]
                
                return f"""
🔄 Training in progress...

📊 Current Progress:
• Epoch: {current_epoch}/{training_status['total_epochs']}
• Train Loss: {train_loss:.4f}
• Val Loss: {val_loss:.4f}
• Val IoU: {val_iou:.4f}

💡 Tip: Detailed training output is displayed in the terminal window
"""
        except:
            pass
    
    return training_status['message']


def check_data_structure(data_path):
    """Check data structure"""
    try:
        data_path = Path(data_path)
        
        if not data_path.exists():
            return f"❌ Data path does not exist: {data_path}"
        
        # Check training and validation folders
        train_path = data_path / 'train'
        val_path = data_path / 'val'
        
        if not train_path.exists():
            return f"❌ Training folder not found: {train_path}"
        if not val_path.exists():
            return f"❌ Validation folder not found: {val_path}"
        
        # Detect available task folders
        available_tasks = []
        for task_folder in train_path.iterdir():
            if task_folder.is_dir():
                available_tasks.append(task_folder.name)
        
        if not available_tasks:
            return f"❌ No task folders found in {train_path}"
        
        result = f"✅ Data Structure Check Results:\n\n"
        result += f"📋 Detected {len(available_tasks)} tasks: {', '.join(available_tasks)}\n\n"
        
        for split in ['train', 'val']:
            split_path = data_path / split
            result += f"📁 {split}/\n"
            
            for task in available_tasks:
                task_path = split_path / task
                if not task_path.exists():
                    result += f"  ❌ {task}/ - Does not exist\n"
                    continue
                
                images_path = task_path / 'images'
                masks_path = task_path / 'masks'
                
                if not images_path.exists():
                    result += f"  ❌ {task}/images/ - Does not exist\n"
                elif not masks_path.exists():
                    result += f"  ❌ {task}/masks/ - Does not exist\n"
                else:
                    num_images = len(list(images_path.glob('*')))
                    num_masks = len(list(masks_path.glob('*')))
                    result += f"  ✅ {task}/ - {num_images} images, {num_masks} masks\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error occurred during check: {str(e)}"


def load_training_curve():
    """Load training curve"""
    curve_path = Path('outputs/training_history.png')
    if curve_path.exists():
        return str(curve_path)
    return None


def load_validation_image(epoch):
    """Load validation image"""
    try:
        epoch = int(epoch)
        val_image_path = Path(f'outputs/predictions/val_epoch{epoch:03d}.png')
        if val_image_path.exists():
            return str(val_image_path)
    except:
        pass
    return None


def get_training_stats():
    """Get training statistics (supports dynamic number of tasks)"""
    history_path = Path('outputs/training_history.json')
    
    if not history_path.exists():
        return "❌ Training history file not found\n\nPlease complete at least one training session first."
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Debug: show available keys
        available_keys = list(history.keys()) if isinstance(history, dict) else "Not a dict"
        
        # Handle different possible formats
        train_loss = history.get('train_loss', history.get('loss', []))
        val_loss = history.get('val_loss', history.get('validation_loss', []))
        val_iou = history.get('val_iou', history.get('iou', []))
        val_dice = history.get('val_dice', history.get('dice', []))
        
        total_epochs = len(train_loss) if train_loss else 0
        
        if total_epochs == 0:
            return f"❌ Training history is empty or has unexpected format\n\nAvailable keys: {available_keys}"
        
        # Calculate statistics with safe access
        stats = f"""
{'='*60}
TRAINING STATISTICS
{'='*60}

Total Epochs: {total_epochs}
"""
        
        # Loss progression (with safety checks)
        if train_loss and len(train_loss) > 0:
            stats += f"""
📉 Loss Progression:
  Initial Train Loss: {train_loss[0]:.4f}
  Final Train Loss:   {train_loss[-1]:.4f}"""
        
        if val_loss and len(val_loss) > 0:
            stats += f"""
  Initial Val Loss:   {val_loss[0]:.4f}
  Final Val Loss:     {val_loss[-1]:.4f}
"""
        
        # IoU progression
        if val_iou and len(val_iou) > 0:
            stats += f"""
📈 IoU Progression:
  Initial Val IoU: {val_iou[0]:.4f}
  Final Val IoU:   {val_iou[-1]:.4f}
  Improvement:     {val_iou[-1] - val_iou[0]:.4f}
"""
        
        # Dice progression
        if val_dice and len(val_dice) > 0:
            stats += f"""
📊 Dice Progression:
  Initial Val Dice: {val_dice[0]:.4f}
  Final Val Dice:   {val_dice[-1]:.4f}
  Improvement:      {val_dice[-1] - val_dice[0]:.4f}
"""
        
        stats += f"""
{'='*60}

Per-Task Performance (Final Epoch):
"""
        
        # Get task names from history (if available) or use defaults
        task_names = history.get('task_names', {})
        if not task_names:
            task_names = {'0': 'Cell', '1': 'Blood', '2': 'Root'}
        
        # Convert keys to strings if needed
        if isinstance(task_names, dict):
            task_names = {str(k): v for k, v in task_names.items()}
        
        # Per-task performance (dynamic)
        task_metrics = history.get('task_metrics', {})
        
        if task_metrics:
            for task_id_str, metrics in task_metrics.items():
                task_name = task_names.get(str(task_id_str), f'Task_{task_id_str}')
                if metrics and len(metrics) > 0:
                    final_metric = metrics[-1]
                    if isinstance(final_metric, dict):
                        stats += f"\n{task_name}:"
                        stats += f"\n  IoU: {final_metric.get('iou', 0):.4f}"
                        stats += f"\n  Dice: {final_metric.get('dice', 0):.4f}"
                        stats += f"\n  Precision: {final_metric.get('precision', 0):.4f}"
                        stats += f"\n  Recall: {final_metric.get('recall', 0):.4f}\n"
        else:
            stats += "\n(No per-task metrics available)"
        
        return stats
        
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON format in training history: {str(e)}"
    except Exception as e:
        # More detailed error info
        import traceback
        return f"❌ Error reading training history: {str(e)}\n\nType: {type(e).__name__}\n\nTraceback:\n{traceback.format_exc()}"


def refresh_monitoring():
    """Refresh all monitoring data"""
    curve = load_training_curve()
    stats = get_training_stats()
    
    # Check if predictions directory exists
    predictions_dir = Path('outputs/predictions')
    if not predictions_dir.exists():
        return curve, None, gr.update(maximum=200, value=0), stats
    
    # Find all available validation images
    try:
        val_images = sorted(list(predictions_dir.glob('val_epoch*.png')))
    except Exception as e:
        return curve, None, gr.update(maximum=200, value=0), stats
    
    if val_images:
        # Extract all epoch numbers
        available_epochs = []
        for img_path in val_images:
            try:
                epoch_num = int(img_path.stem.replace('val_epoch', ''))
                available_epochs.append(epoch_num)
            except:
                pass
        
        if available_epochs:
            # Find the latest epoch
            latest_epoch = max(available_epochs)
            max_epoch = max(available_epochs)
            
            # Return: training curve, latest validation image, updated slider, statistics
            latest_val_path = predictions_dir / f'val_epoch{latest_epoch:03d}.png'
            
            slider_update = gr.update(maximum=max_epoch, value=latest_epoch)
            
            return curve, str(latest_val_path), slider_update, stats
    
    # If no validation images found, return default values
    return curve, None, gr.update(maximum=200, value=0), stats


# Create Gradio interface
with gr.Blocks(title="TransUNet Training Interface") as demo:
    
    gr.Markdown("# 🚀 TransUNet Multi-Task Training Interface")
    gr.Markdown("Dedicated to model training | For prediction, please use Tkinter GUI (app_gui.py)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Training Settings")
            
            batch_size = gr.Slider(
                minimum=1, maximum=16, value=4, step=1,
                label="Batch Size",
                info="Batch size, depends on GPU memory"
            )
            
            epochs = gr.Slider(
                minimum=1, maximum=500, value=200, step=1,
                label="Epochs",
                info="Number of training epochs"
            )
            
            lr = gr.Textbox(
                value="1e-5",
                label="Learning Rate",
                info="Learning rate (recommended: 1e-5 to 1e-4)"
            )
            
            patch_size = gr.Slider(
                minimum=128, maximum=512, value=400, step=32,
                label="Patch Size",
                info="Image patch size"
            )
            
            num_layers = gr.Slider(
                minimum=20, maximum=120, value=80, step=10,
                label="Decoder Conv Layers",
                info="Number of decoder convolutional layers"
            )
            
            data_path = gr.Textbox(
                value="data/",
                label="Data Path",
                info="Dataset path"
            )
            
            check_data_btn = gr.Button("🔍 Check Data Structure", variant="secondary")
            data_check_output = gr.Textbox(
                label="Data Check Results",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("### Pre-trained Model (Optional)")
            
            use_pretrained = gr.Checkbox(
                label="Use Pre-trained Model",
                value=False
            )
            
            pretrained_path = gr.Textbox(
                value="",
                label="Pre-trained Model Path",
                placeholder="e.g., outputs/models/best_model.pth"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 🎮 Training Control")
            
            with gr.Row():
                start_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop", size="lg")
            
            refresh_btn = gr.Button("🔄 Refresh Training Status", variant="secondary")
            
            training_output = gr.Textbox(
                label="Training Messages",
                lines=15,
                interactive=False
            )
    
    # Training monitoring tab
    with gr.Tab("📊 Training Monitoring"):
        gr.Markdown("## 📊 Training Results Monitoring")
        
        with gr.Row():
            with gr.Column():
                refresh_monitor_btn = gr.Button("🔄 Refresh Monitoring", variant="primary")
                
                gr.Markdown("### Training Curves")
                training_curve = gr.Image(
                    label="Training History Curves",
                    type="filepath"
                )
                
            with gr.Column():
                gr.Markdown("### Validation Images")
                epoch_slider = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=0,
                    step=10,
                    label="Select Epoch"
                )
                
                val_image = gr.Image(
                    label="Validation Results",
                    type="filepath"
                )
        
        gr.Markdown("### Training Statistics")
        stats_output = gr.Textbox(
            label="Training Data Statistics",
            lines=10,
            interactive=False
        )
    
    # User guide
    with gr.Tab("📖 User Guide"):
        gr.Markdown("""
## 📝 User Guide

### 1. Prepare Data
Ensure your data structure is as follows:
```
data/
├── train/
│   ├── cell/
│   │   ├── images/
│   │   └── masks/
│   ├── blood/
│   │   ├── images/
│   │   └── masks/
│   └── root/
│       ├── images/
│       └── masks/
└── val/
    └── (same structure)
```

**Note:** The system will automatically detect all task folders in your data directory.
You can have 2, 3, or more tasks - just create the appropriate folders.

### 2. Configure Parameters
- Adjust training parameters on the left panel
- Click "Check Data Structure" to verify your data is correct

### 3. Start Training
- Click "Start Training"
- View detailed output in the terminal window
- Periodically click "Refresh Training Status" to check progress

### 4. Output Locations
- Models: `outputs/models/`
- Training history: `outputs/training_history.json`
- Logs: `outputs/training_ui.log`

### 💡 Tips
- Training process will display detailed information in the terminal
- Please keep the terminal window open
- You can stop training at any time
- Use a pre-trained model to continue training

### 🔗 Related Tools
- **Prediction and Inference**: Use `app_gui.py` (Tkinter GUI)
- **Training Monitoring**: View `outputs/training_history.json`
        """)
    
    # Event handlers
    start_btn.click(
        fn=start_training,
        inputs=[batch_size, epochs, lr, patch_size, num_layers, data_path, 
                use_pretrained, pretrained_path],
        outputs=training_output
    )
    
    stop_btn.click(
        fn=stop_training,
        outputs=training_output
    )
    
    refresh_btn.click(
        fn=get_training_status,
        outputs=training_output
    )
    
    check_data_btn.click(
        fn=check_data_structure,
        inputs=data_path,
        outputs=data_check_output
    )
    
    # Training monitoring events
    refresh_monitor_btn.click(
        fn=refresh_monitoring,
        outputs=[training_curve, val_image, epoch_slider, stats_output]
    )
    
    epoch_slider.change(
        fn=load_validation_image,
        inputs=epoch_slider,
        outputs=val_image
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TransUNet Training Interface")
    print("="*60)
    print("Dedicated to model training functionality")
    print("For prediction, please use: python app_gui.py")
    print("="*60)
    print("\nLaunching browser...")
    print("If the browser does not open automatically, please visit: http://localhost:7860")
    print("\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        quiet=False
    )
