import matplotlib.pyplot as plt
import numpy as np

def plot_results(image, true_mask, pred_mask):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('Input Image')
    
    plt.subplot(1,3,2)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Ground Truth')
    
    plt.subplot(1,3,3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    
    plt.tight_layout()
    plt.savefig('outputs/predictions/visualization.png')
    plt.close()

def calculate_iou(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    return np.sum(intersection) / np.sum(union)