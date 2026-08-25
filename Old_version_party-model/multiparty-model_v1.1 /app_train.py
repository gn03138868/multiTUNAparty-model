"""
TransUNet 訓練專用 Web UI (Gradio)
專注於模型訓練功能
"""

import gradio as gr
import subprocess
import sys
import yaml
from pathlib import Path
import json
import time
import os

# 全局變量
training_process = None
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'val_iou': 0.0,
    'message': '尚未開始訓練',
    'log_file': '',
    'error_message': ''
}

def start_training(batch_size, epochs, lr, patch_size, num_layers, data_path, 
                  use_pretrained, pretrained_path):
    """開始訓練"""
    global training_process, training_status
    
    # 檢查是否已在訓練
    if training_status['is_training']:
        return "⚠️ 訓練已在進行中！請等待當前訓練完成或先停止訓練。"
    
    try:
        # 更新配置
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
        
        # 添加預訓練設定
        if use_pretrained and pretrained_path:
            config['pretrained_model_path'] = pretrained_path
        
        # 保存配置
        config_path = Path('config_training_ui.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        # 創建訓練日誌檔案
        log_file = Path('outputs/training_ui.log')
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化訓練狀態
        training_status = {
            'is_training': True,
            'current_epoch': 0,
            'total_epochs': int(epochs),
            'train_loss': 0.0,
            'val_loss': 0.0,
            'val_iou': 0.0,
            'message': '正在啟動訓練...',
            'log_file': str(log_file),
            'error_message': ''
        }
        
        # 在後台線程啟動訓練
        def run_training():
            global training_process, training_status
            
            try:
                print("\n" + "="*60)
                print("🚀 開始訓練...")
                print("="*60)
                print(f"配置檔案: {config_path}")
                print(f"日誌檔案: {log_file}")
                print(f"Batch Size: {batch_size}")
                print(f"Epochs: {epochs}")
                print(f"Learning Rate: {lr}")
                print("="*60 + "\n")
                
                # 啟動訓練進程
                training_process = subprocess.Popen(
                    [sys.executable, 'train_multitask.py', '--config', str(config_path)],
                )
                
                training_status['message'] = '✅ 訓練已啟動！請查看 CMD 視窗的訓練輸出。'
                
                # 等待訓練完成
                training_process.wait()
                
                # 檢查返回碼
                if training_process.returncode == 0:
                    training_status['message'] = '✅ 訓練成功完成！'
                else:
                    training_status['message'] = f'❌ 訓練失敗 (返回碼: {training_process.returncode})'
                
            except Exception as e:
                training_status['error_message'] = str(e)
                training_status['message'] = f'❌ 訓練啟動失敗: {str(e)}'
                print(f"訓練錯誤: {e}")
            finally:
                training_status['is_training'] = False
                training_process = None
        
        # 啟動訓練線程
        import threading
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        return f"""
✅ 訓練已成功啟動！

📊 訓練配置：
• Batch Size: {batch_size}
• Epochs: {epochs}
• Learning Rate: {lr}
• Patch Size: {patch_size}
• Decoder Layers: {num_layers}
• Data Path: {data_path}
• 預訓練模型: {'是 (' + pretrained_path + ')' if use_pretrained else '否'}

💡 提示：
• 訓練過程的詳細輸出會顯示在終端機 (CMD/Terminal) 中
• 請保持終端機視窗開啟以查看訓練進度
• 模型會自動保存到 outputs/models/ 目錄
• 訓練歷史會保存到 outputs/training_history.json

🔄 點擊「刷新訓練狀態」查看當前進度
"""
        
    except Exception as e:
        return f"❌ 錯誤: {str(e)}"


def stop_training():
    """停止訓練"""
    global training_process, training_status
    
    if not training_status['is_training']:
        return "⚠️ 目前沒有正在進行的訓練。"
    
    try:
        if training_process and training_process.poll() is None:
            training_process.terminate()
            training_process.wait(timeout=5)
            training_status['is_training'] = False
            training_status['message'] = '⏹️ 訓練已停止'
            return "✅ 訓練已成功停止。"
        else:
            training_status['is_training'] = False
            return "⚠️ 訓練進程已結束。"
    except Exception as e:
        return f"❌ 停止訓練時發生錯誤: {str(e)}"


def get_training_status():
    """獲取訓練狀態"""
    global training_status
    
    if not training_status['is_training']:
        return training_status['message']
    
    # 嘗試讀取訓練歷史
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
🔄 訓練進行中...

📊 當前進度：
• Epoch: {current_epoch}/{training_status['total_epochs']}
• Train Loss: {train_loss:.4f}
• Val Loss: {val_loss:.4f}
• Val IoU: {val_iou:.4f}

💡 提示：訓練詳細輸出在終端機視窗中
"""
        except:
            pass
    
    return training_status['message']


def check_data_structure(data_path):
    """檢查資料結構"""
    try:
        data_path = Path(data_path)
        
        if not data_path.exists():
            return f"❌ 資料路徑不存在: {data_path}"
        
        # 檢查訓練和驗證資料夾
        train_path = data_path / 'train'
        val_path = data_path / 'val'
        
        if not train_path.exists():
            return f"❌ 找不到訓練資料夾: {train_path}"
        if not val_path.exists():
            return f"❌ 找不到驗證資料夾: {val_path}"
        
        # 檢查任務資料夾
        tasks = ['cell', 'blood', 'root']
        result = "✅ 資料結構檢查結果：\n\n"
        
        for split in ['train', 'val']:
            split_path = data_path / split
            result += f"📁 {split}/\n"
            
            for task in tasks:
                task_path = split_path / task
                if not task_path.exists():
                    result += f"  ❌ {task}/ - 不存在\n"
                    continue
                
                images_path = task_path / 'images'
                masks_path = task_path / 'masks'
                
                if not images_path.exists():
                    result += f"  ❌ {task}/images/ - 不存在\n"
                elif not masks_path.exists():
                    result += f"  ❌ {task}/masks/ - 不存在\n"
                else:
                    num_images = len(list(images_path.glob('*')))
                    num_masks = len(list(masks_path.glob('*')))
                    result += f"  ✅ {task}/ - {num_images} 影像, {num_masks} masks\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"❌ 檢查時發生錯誤: {str(e)}"


def load_training_curve():
    """載入訓練曲線"""
    curve_path = Path('outputs/training_history.png')
    if curve_path.exists():
        return str(curve_path)
    return None


def load_validation_image(epoch):
    """載入驗證影像"""
    val_image_path = Path(f'outputs/predictions/val_epoch{int(epoch):03d}.png')
    if val_image_path.exists():
        return str(val_image_path)
    return None


def get_training_stats():
    """獲取訓練統計"""
    history_path = Path('outputs/training_history.json')
    
    if not history_path.exists():
        return "❌ 尚未找到訓練歷史文件\n\n請先完成至少一次訓練。"
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        total_epochs = len(history['train_loss'])
        
        if total_epochs == 0:
            return "❌ 訓練歷史為空"
        
        # 計算統計
        stats = f"""
📊 訓練統計摘要
{'='*60}

總訓練 Epochs: {total_epochs}

📈 Loss 變化:
  初始 Train Loss: {history['train_loss'][0]:.4f}
  最終 Train Loss: {history['train_loss'][-1]:.4f}
  降低: {history['train_loss'][0] - history['train_loss'][-1]:.4f}
  
  初始 Val Loss: {history['val_loss'][0]:.4f}
  最終 Val Loss: {history['val_loss'][-1]:.4f}
  降低: {history['val_loss'][0] - history['val_loss'][-1]:.4f}

📊 IoU 變化:
  初始 Val IoU: {history['val_iou'][0]:.4f}
  最終 Val IoU: {history['val_iou'][-1]:.4f}
  提升: {history['val_iou'][-1] - history['val_iou'][0]:.4f}
  
  最佳 Val IoU: {max(history['val_iou']):.4f}
  最佳 Epoch: {history['val_iou'].index(max(history['val_iou'])) + 1}

📊 Dice 變化:
  初始 Val Dice: {history['val_dice'][0]:.4f}
  最終 Val Dice: {history['val_dice'][-1]:.4f}
  提升: {history['val_dice'][-1] - history['val_dice'][0]:.4f}

{'='*60}

各任務表現 (最終 Epoch):
"""
        
        # 各任務表現
        task_names = ['Cell', 'Blood', 'Root']
        for task_id in range(3):
            if str(task_id) in history['task_metrics']:
                metrics = history['task_metrics'][str(task_id)]
                if len(metrics) > 0:
                    final_metric = metrics[-1]
                    stats += f"\n{task_names[task_id]}:"
                    stats += f"\n  IoU: {final_metric['iou']:.4f}"
                    stats += f"\n  Dice: {final_metric['dice']:.4f}"
                    stats += f"\n  Precision: {final_metric['precision']:.4f}"
                    stats += f"\n  Recall: {final_metric['recall']:.4f}\n"
        
        return stats
        
    except Exception as e:
        return f"❌ 讀取訓練歷史時發生錯誤: {str(e)}"


def refresh_monitoring():
    """刷新所有監控數據"""
    curve = load_training_curve()
    stats = get_training_stats()
    
    # 找出所有可用的驗證影像
    val_images = sorted(list(Path('outputs/predictions').glob('val_epoch*.png')))
    
    if val_images:
        # 提取所有 epoch 數字
        available_epochs = []
        for img_path in val_images:
            try:
                epoch_num = int(img_path.stem.replace('val_epoch', ''))
                available_epochs.append(epoch_num)
            except:
                pass
        
        if available_epochs:
            # 找到最新的 epoch
            latest_epoch = max(available_epochs)
            max_epoch = max(available_epochs)
            
            # 返回：訓練曲線、最新驗證影像、更新的滑桿（帶新的最大值和當前值）、統計
            latest_val_path = Path(f'outputs/predictions/val_epoch{latest_epoch:03d}.png')
            
            # 使用 gr.update() 來更新滑桿的 maximum 和 value
            import gradio as gr
            slider_update = gr.update(maximum=max_epoch, value=latest_epoch)
            
            return curve, str(latest_val_path), slider_update, stats
    
    # 如果沒有找到驗證影像，返回默認值
    import gradio as gr
    return curve, None, gr.update(maximum=200, value=0), stats


# 創建 Gradio 介面
with gr.Blocks(title="TransUNet 訓練介面") as demo:
    
    gr.Markdown("# 🚀 TransUNet 多任務訓練介面")
    gr.Markdown("專注於模型訓練功能 | 預測功能請使用 Tkinter GUI (app_gui.py)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ 訓練設定")
            
            batch_size = gr.Slider(
                minimum=1, maximum=16, value=4, step=1,
                label="Batch Size",
                info="批次大小，取決於 GPU 記憶體"
            )
            
            epochs = gr.Slider(
                minimum=1, maximum=500, value=200, step=1,
                label="Epochs",
                info="訓練輪數"
            )
            
            lr = gr.Textbox(
                value="1e-5",
                label="Learning Rate",
                info="學習率 (建議: 1e-5 到 1e-4)"
            )
            
            patch_size = gr.Slider(
                minimum=128, maximum=512, value=400, step=32,
                label="Patch Size",
                info="影像大小"
            )
            
            num_layers = gr.Slider(
                minimum=20, maximum=120, value=80, step=10,
                label="Decoder Conv Layers",
                info="Decoder 卷積層數"
            )
            
            data_path = gr.Textbox(
                value="data/",
                label="Data Path",
                info="資料集路徑"
            )
            
            check_data_btn = gr.Button("🔍 檢查資料結構", variant="secondary")
            data_check_output = gr.Textbox(
                label="資料檢查結果",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("### 預訓練模型（可選）")
            
            use_pretrained = gr.Checkbox(
                label="使用預訓練模型",
                value=False
            )
            
            pretrained_path = gr.Textbox(
                value="",
                label="預訓練模型路徑",
                placeholder="例如: outputs/models/best_model.pth"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 🎮 訓練控制")
            
            with gr.Row():
                start_btn = gr.Button("🚀 開始訓練", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ 停止訓練", variant="stop", size="lg")
            
            refresh_btn = gr.Button("🔄 刷新訓練狀態", variant="secondary")
            
            training_output = gr.Textbox(
                label="訓練訊息",
                lines=15,
                interactive=False
            )
    
    # 訓練監控標籤頁
    with gr.Tab("📊 訓練監控"):
        gr.Markdown("## 📊 訓練結果監控")
        
        with gr.Row():
            with gr.Column():
                refresh_monitor_btn = gr.Button("🔄 刷新監控", variant="primary")
                
                gr.Markdown("### 訓練曲線")
                training_curve = gr.Image(
                    label="訓練歷史曲線",
                    type="filepath"
                )
                
            with gr.Column():
                gr.Markdown("### 驗證影像")
                epoch_slider = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=0,
                    step=1,
                    label="選擇 Epoch"
                )
                
                val_image = gr.Image(
                    label="驗證結果",
                    type="filepath"
                )
        
        gr.Markdown("### 訓練統計")
        stats_output = gr.Textbox(
            label="訓練數據統計",
            lines=10,
            interactive=False
        )
    
    # 使用說明
    with gr.Tab("📖 使用說明"):
        gr.Markdown("""
## 📝 使用說明

### 1. 準備資料
確保資料結構如下：
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
    └── (相同結構)
```

### 2. 設定參數
- 調整左側的訓練參數
- 點擊「檢查資料結構」確認資料正確

### 3. 開始訓練
- 點擊「開始訓練」
- 查看終端機視窗的詳細輸出
- 定期點擊「刷新訓練狀態」查看進度

### 4. 輸出位置
- 模型: `outputs/models/`
- 訓練歷史: `outputs/training_history.json`
- 日誌: `outputs/training_ui.log`

### 💡 提示
- 訓練過程會在終端機顯示詳細資訊
- 請保持終端機視窗開啟
- 可隨時停止訓練
- 使用預訓練模型可繼續訓練

### 🔗 相關工具
- **預測和推理**: 使用 `app_gui.py` (Tkinter GUI)
- **訓練監控**: 查看 `outputs/training_history.json`
            """)
    
    # 事件處理
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
    
    # 訓練監控事件
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
    print("🚀 TransUNet 訓練介面")
    print("="*60)
    print("專注於模型訓練功能")
    print("預測功能請使用: python app_gui.py")
    print("="*60)
    print("\n正在啟動瀏覽器...")
    print("如果瀏覽器沒有自動開啟，請手動訪問: http://localhost:7860")
    print("\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,  # 自動開啟瀏覽器
        quiet=False
    )
