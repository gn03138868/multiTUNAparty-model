"""
預訓練權重檢查和轉換工具

用途：
1. 檢查預訓練權重的內容和結構
2. 檢查哪些層可以載入到新模型
3. 轉換權重格式（可選）
"""

import torch
import sys
from pathlib import Path
from collections import OrderedDict


def inspect_pretrained_weights(weight_path):
    """檢查預訓練權重的詳細信息"""
    print("\n" + "="*80)
    print(f"檢查預訓練權重: {weight_path}")
    print("="*80)
    
    if not Path(weight_path).exists():
        print(f"✗ 文件不存在: {weight_path}")
        return None
    
    # 載入權重
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"✗ 載入失敗: {e}")
        return None
    
    # 檢查格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("✓ 檢測到 checkpoint 格式（包含 optimizer、scheduler 等）")
            print(f"  Keys: {list(checkpoint.keys())}")
            
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            
            state_dict = checkpoint['model_state_dict']
        else:
            print("✓ 檢測到 state_dict 格式（僅模型權重）")
            state_dict = checkpoint
    else:
        print("⚠ 未知格式")
        return None
    
    # 統計信息
    total_params = len(state_dict)
    total_size = sum(v.numel() for v in state_dict.values())
    
    print(f"\n總參數層數: {total_params}")
    print(f"總參數量: {total_size:,}")
    
    # 按模組分組
    modules = {}
    for key in state_dict.keys():
        module_name = key.split('.')[0]
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(key)
    
    print(f"\n模組分佈:")
    print("-" * 80)
    for module, keys in sorted(modules.items()):
        module_params = sum(state_dict[k].numel() for k in keys)
        print(f"  {module:30s}: {len(keys):3d} 層, {module_params:10,} 參數")
    
    # 顯示部分層的詳細信息
    print(f"\n前 20 層詳細信息:")
    print("-" * 80)
    for i, (key, value) in enumerate(list(state_dict.items())[:20]):
        print(f"  {i+1:2d}. {key:60s} {str(value.shape):20s} ({value.numel():,} params)")
    
    if total_params > 20:
        print(f"  ... (還有 {total_params - 20} 層)")
    
    return state_dict


def check_compatibility(pretrained_path, current_model_path=None):
    """檢查預訓練權重與當前模型的兼容性"""
    print("\n" + "="*80)
    print("檢查兼容性")
    print("="*80)
    
    # 載入預訓練權重
    pretrained_dict = inspect_pretrained_weights(pretrained_path)
    if pretrained_dict is None:
        return
    
    # 如果提供了當前模型，進行對比
    if current_model_path:
        print(f"\n載入當前模型進行對比...")
        try:
            from model_multitask import MultiTaskTransUNet
            
            current_model = MultiTaskTransUNet(
                img_size=400,
                patch_size=16,
                num_decoder_layers=80,
                num_tasks=3
            )
            
            current_dict = current_model.state_dict()
            
            # 統計匹配情況
            matched = []
            shape_mismatch = []
            only_in_pretrained = []
            only_in_current = []
            
            for key in pretrained_dict.keys():
                if key in current_dict:
                    if pretrained_dict[key].shape == current_dict[key].shape:
                        matched.append(key)
                    else:
                        shape_mismatch.append((key, pretrained_dict[key].shape, current_dict[key].shape))
                else:
                    only_in_pretrained.append(key)
            
            for key in current_dict.keys():
                if key not in pretrained_dict:
                    only_in_current.append(key)
            
            # 顯示結果
            print(f"\n兼容性分析:")
            print("-" * 80)
            print(f"✓ 匹配（可直接載入）:     {len(matched):4d} 層 ({len(matched)/len(current_dict)*100:.1f}%)")
            print(f"✗ 形狀不匹配（跳過）:     {len(shape_mismatch):4d} 層")
            print(f"⚠ 僅在預訓練中（忽略）:   {len(only_in_pretrained):4d} 層")
            print(f"⚠ 僅在當前模型（新初始化）: {len(only_in_current):4d} 層")
            
            # 顯示形狀不匹配的層
            if shape_mismatch:
                print(f"\n形狀不匹配的層:")
                for key, pretrained_shape, current_shape in shape_mismatch[:10]:
                    print(f"  {key:50s} {str(pretrained_shape):20s} → {str(current_shape)}")
                if len(shape_mismatch) > 10:
                    print(f"  ... (還有 {len(shape_mismatch) - 10} 層)")
            
            # 顯示新初始化的重要層
            if only_in_current:
                print(f"\n需要新初始化的重要層:")
                important = [k for k in only_in_current if any(x in k for x in ['task_embedding', 'aspp', 'attention'])]
                for key in important[:10]:
                    print(f"  {key}")
            
            # 建議
            print(f"\n建議:")
            if len(matched) / len(current_dict) > 0.7:
                print("  ✓ 兼容性良好！大部分層可以載入。")
                print("  ✓ 建議使用預訓練權重加速訓練。")
            elif len(matched) / len(current_dict) > 0.3:
                print("  ⚠ 兼容性一般。部分層可以載入（主要是 encoder）。")
                print("  ⚠ 可以使用，但效果可能有限。")
            else:
                print("  ✗ 兼容性較差。很少層可以載入。")
                print("  ✗ 建議從頭訓練。")
            
        except ImportError:
            print("⚠ 無法載入 model_multitask.py，跳過對比")
        except Exception as e:
            print(f"✗ 對比失敗: {e}")


def extract_encoder_weights(pretrained_path, output_path):
    """
    從完整模型中提取 encoder 權重
    用於只想使用 encoder 部分的情況
    """
    print("\n" + "="*80)
    print("提取 Encoder 權重")
    print("="*80)
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 提取 encoder 相關的層
    encoder_keys = ['patch_embed', 'pos_embed', 'blocks', 'norm']
    encoder_dict = OrderedDict()
    
    for key, value in state_dict.items():
        if any(k in key for k in encoder_keys):
            encoder_dict[key] = value
    
    print(f"提取的層數: {len(encoder_dict)} / {len(state_dict)}")
    print(f"提取的參數量: {sum(v.numel() for v in encoder_dict.values()):,}")
    
    # 保存
    torch.save(encoder_dict, output_path)
    print(f"\n✓ Encoder 權重已保存到: {output_path}")


def convert_to_multitask_format(pretrained_path, output_path):
    """
    嘗試將舊模型權重轉換為多任務格式
    （這主要是一個框架，實際轉換可能需要根據具體情況調整）
    """
    print("\n" + "="*80)
    print("轉換為多任務格式")
    print("="*80)
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 新的 state_dict（只包含兼容的層）
    new_state_dict = OrderedDict()
    
    # 複製兼容的層
    for key, value in state_dict.items():
        # 跳過不兼容的層（例如舊的 decoder）
        if any(skip in key for skip in ['final_conv', 'decoder']):
            continue
        new_state_dict[key] = value
    
    print(f"保留的層數: {len(new_state_dict)} / {len(state_dict)}")
    
    # 保存
    torch.save(new_state_dict, output_path)
    print(f"\n✓ 轉換後的權重已保存到: {output_path}")


def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 檢查權重:")
        print("     python inspect_pretrained.py <weight_path>")
        print()
        print("  2. 檢查兼容性:")
        print("     python inspect_pretrained.py <weight_path> --check-compat")
        print()
        print("  3. 提取 encoder:")
        print("     python inspect_pretrained.py <weight_path> --extract-encoder <output_path>")
        print()
        print("  4. 轉換格式:")
        print("     python inspect_pretrained.py <weight_path> --convert <output_path>")
        print()
        print("範例:")
        print("  python inspect_pretrained.py data/pretrained_model.pth")
        print("  python inspect_pretrained.py data/pretrained_model.pth --check-compat")
        return
    
    weight_path = sys.argv[1]
    
    if not Path(weight_path).exists():
        print(f"✗ 文件不存在: {weight_path}")
        return
    
    # 執行相應操作
    if len(sys.argv) == 2:
        # 只檢查
        inspect_pretrained_weights(weight_path)
    
    elif '--check-compat' in sys.argv:
        # 檢查兼容性
        check_compatibility(weight_path, current_model_path=True)
    
    elif '--extract-encoder' in sys.argv:
        # 提取 encoder
        if len(sys.argv) < 4:
            print("✗ 請提供輸出路徑")
            print("   用法: python inspect_pretrained.py <weight_path> --extract-encoder <output_path>")
            return
        output_path = sys.argv[3]
        extract_encoder_weights(weight_path, output_path)
    
    elif '--convert' in sys.argv:
        # 轉換格式
        if len(sys.argv) < 4:
            print("✗ 請提供輸出路徑")
            print("   用法: python inspect_pretrained.py <weight_path> --convert <output_path>")
            return
        output_path = sys.argv[3]
        convert_to_multitask_format(weight_path, output_path)
    
    else:
        print(f"✗ 未知選項: {sys.argv[2]}")


if __name__ == '__main__':
    main()
