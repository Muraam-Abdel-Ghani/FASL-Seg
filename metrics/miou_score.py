import torch
import numpy as np

def miou_score(preds, targets, num_classes, return_per_class = False):
    """
    Calculate Mean IoU for each class including the background.
    
    Args:
    - preds: Predictions from the model (after argmax), shape [batch_size, height, width]
    - targets: Ground truth masks, shape [batch_size, height, width]
    - num_classes: Number of classes including the background
    
    Returns:
    - Average mIoU across the batch.
    """
    batch_size = preds.shape[0]
    miou = 0.0
    classes = range(num_classes)  # Include background

    if return_per_class:
        miou_per_class_scores = []

    for c in classes:
        # if c==0:
        #     continue
        pred_i = (preds == c).float()
        target_i = (targets == c).float()

        intersection = (pred_i * target_i).sum(dim=(1, 2))  # Sum over spatial dimensions
        union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2)) - intersection  # Sum over spatial dimensions

        iou_per_class = (intersection + 1e-7) / (union + 1e-7)  # Add small epsilon to avoid division by zero
    
        miou += iou_per_class
        
        if return_per_class:
            miou_per_class_scores.append(iou_per_class.mean().item())

    if return_per_class:
        return miou_per_class_scores, (miou / len(classes)).mean().item()
    else:
        return (miou / len(classes)).mean().item()