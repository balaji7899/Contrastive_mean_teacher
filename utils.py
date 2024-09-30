import torch
import os
from torchvision.ops import box_iou

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mAP (mean Average Precision) at a given IoU threshold (default 0.5).
    predictions: List of dictionaries with 'boxes' and 'labels' keys from the model output
    targets: List of dictionaries with 'boxes' and 'labels' keys from the ground truth
    """
    all_ap = []  # List to store AP for each class

    # Flatten the predictions and targets into a format usable for mAP calculation
    pred_boxes = torch.cat([p['boxes'] for p in predictions])
    pred_labels = torch.cat([p['labels'] for p in predictions])
    target_boxes = torch.cat([t['boxes'] for t in targets])
    target_labels = torch.cat([t['labels'] for t in targets])

    unique_classes = torch.unique(torch.cat([pred_labels, target_labels]))

    for cls in unique_classes:
        cls_pred_mask = pred_labels == cls
        cls_target_mask = target_labels == cls

        # Get predicted boxes and target boxes for the current class
        cls_pred_boxes = pred_boxes[cls_pred_mask]
        cls_target_boxes = target_boxes[cls_target_mask]

        # Compute IoU between predicted and target boxes
        iou_matrix = box_iou(cls_pred_boxes, cls_target_boxes)

        # Determine matches (IoU > threshold)
        matched = (iou_matrix > iou_threshold).sum(dim=1).float()

        # True positives, false positives, false negatives
        tp = matched.sum()
        fp = (cls_pred_boxes.size(0) - tp)
        fn = (cls_target_boxes.size(0) - tp)

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6)  # Add a small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-6)

        # Calculate AP for this class
        ap = (precision * recall) / (precision + recall + 1e-6)
        all_ap.append(ap)

    # Calculate mAP as the mean of all APs
    mean_ap = torch.stack(all_ap).mean().item()
    return mean_ap


# Save model checkpoint
def save_checkpoint(model, epoch, optimizer, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Load model checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

# Log metrics (e.g., loss) to console or TensorBoard
def log_metrics(epoch, avg_loss, log_file='./log.txt'):
    log_msg = f"Epoch [{epoch}]: Avg Loss: {avg_loss:.4f}\n"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg)

# Helper function for calculating mAP at IoU 0.5
# def calculate_map(predictions, targets, iou_threshold=0.5):
#     """
#     Calculate mAP (mean Average Precision) at IoU threshold.
#     Predictions and targets should contain bounding boxes and class labels.
#     """
#     # Placeholder for mAP calculation logic (can be implemented or use torchvision.ops)
#     # This typically involves computing IoU between predicted and target boxes, and
#     # then calculating precision-recall for each class.
#     # For now, we'll assume this function is a placeholder.
#     pass  # Implement mAP calculation logic here
