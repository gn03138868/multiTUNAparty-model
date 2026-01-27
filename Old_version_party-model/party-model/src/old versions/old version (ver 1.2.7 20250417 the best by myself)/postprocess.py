# ==== src/postprocess.py ====

"""
postprocess 模組

本模組提供分割後處理工具與改善策略，
包含形態學後處理、CRF 細化以及多尺度融合，
以改善分割細節及邊界精度。
"""

import cv2
import numpy as np

def morphology_postprocess(binary_mask, kernel_size=5, min_area=100):
    mask = cv2.medianBlur(binary_mask, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    return mask

def apply_crf(image, probability_mask, iterations=5):
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        raise ImportError("請先安裝 pydensecrf: pip install pydensecrf")

    h, w = probability_mask.shape
    d = dcrf.DenseCRF2D(w, h, 2)
    probs = np.stack([1 - probability_mask, probability_mask], axis=0)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(iterations)
    preds = np.array(Q).reshape((2, h, w))
    refined_mask = (preds[1] > preds[0]).astype(np.uint8) * 255
    return refined_mask

def multi_scale_fusion(predictor, image, scales=[0.8, 1.0, 1.2], threshold=0.4):
    h, w = image.shape[:2]
    fused_mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    for scale in scales:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        pred_mask = predictor.predict_single_return_from_array(resized)
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        fused_mask += pred_mask
        count += 1.0
    
    fused_mask = fused_mask / count
    bin_mask = (fused_mask > threshold).astype(np.uint8) * 255
    return bin_mask

if __name__ == '__main__':
    test_mask = cv2.imread("outputs/predictions/test_mask.png", cv2.IMREAD_GRAYSCALE)
    if test_mask is not None:
        proc_mask = morphology_postprocess(test_mask, kernel_size=5, min_area=100)
        cv2.imwrite("outputs/predictions/test_mask_post.png", proc_mask)
        print("形態學後處理結果已保存。")
    else:
        print("請先產生 test_mask.png 後再測試後處理。")