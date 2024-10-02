import torch
import torch.nn as nn
import torch.nn.functional as F

# Detection loss (combines classification loss and bounding box regression loss)

def detection_loss(cls_out, reg_out, targets, num_classes=2):
    """
    Computes classification and regression losses.
    
    Args:
        cls_out: List of model outputs for classification, one for each batch.
                 Each output should have shape [channels, height, width].
        reg_out: List of model outputs for regression (bounding boxes), one for each batch.
                 Each output should have shape [batch_size, height, width] for predicted bounding boxes.
        targets: List of dictionaries containing target information for each batch.
                 Each target dictionary should have 'labels' and 'boxes' keys.
                 - 'labels': Tensor of shape [batch_size], containing the class indices for each object.
                 - 'boxes': Tensor of shape [batch_size, 4], containing bounding box coordinates.

    Returns:
        cls_loss: Cross-entropy classification loss.
        reg_loss: SmoothL1 loss for bounding box regression.
    """

    # Classification loss: Cross-entropy
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    
    # Regression loss: SmoothL1 for bounding boxes
    regression_loss_fn = torch.nn.SmoothL1Loss()

    cls_loss = 0
    reg_loss = 0

    # Compute classification loss
    for i, (cls_out_batch, reg_out_batch, target) in enumerate(zip(cls_out, reg_out, targets)):
        # Debugging: Print cls_out shape and target shape
        print(f"cls_out shape for batch {i}: {cls_out_batch.shape}")  # Expected: [channels, height, width]
        print(f"target['labels'] shape for batch {i}: {target['labels'].shape}")  # Expected: [batch_size]
        print(f"target['labels'] values for batch {i}: {target['labels']}")  # Actual values of target labels

        # Flatten the spatial dimensions (height, width) of cls_out
        cls_out_flat = cls_out_batch.view(cls_out_batch.size(0), -1).permute(1, 0)  # Flatten to [height * width, channels]
        
        # Create background labels for all spatial locations
        target_labels_flat = torch.zeros(cls_out_flat.size(0), dtype=torch.long, device=cls_out_flat.device)

        # Assign the object labels to the appropriate locations
        # (Assuming object labels are assigned based on spatial position or bounding box info)
        if target['labels'].size(0) > 0:
            # Example: assign the object label to all locations as a placeholder
            target_labels_flat[:] = target['labels'][0]  # Placeholder

        # Debugging: Print shapes after flattening
        print(f"cls_out_flat shape for batch {i}: {cls_out_flat.shape}")  # Should be [height * width, channels]
        print(f"target_labels_flat shape for batch {i}: {target_labels_flat.shape}")  # Should match cls_out_flat

        # Add classification loss for the current batch
        cls_loss += classification_loss_fn(cls_out_flat, target_labels_flat)

        # --- Regression Loss ---
        # Flatten the regression output (height, width)
        reg_out_flat = reg_out_batch.view(reg_out_batch.size(0), -1).permute(1, 0)  # Flatten to [height * width, 4]
        
        # Ensure the number of predictions matches the number of target boxes
        if reg_out_flat.size(0) != target['boxes'].size(0):
            # You need to map the predicted boxes to the target boxes correctly
            # Here, we'll assume we take the top-N predicted boxes (this should ideally be based on anchors or a better matching strategy)
            predicted_boxes = reg_out_flat[:target['boxes'].size(0), :]  # Taking the first N boxes (N = number of target boxes)
        else:
            predicted_boxes = reg_out_flat

        # Debugging: Print the shapes of the predicted and target boxes
        print(f"reg_out_flat shape for batch {i}: {reg_out_flat.shape}")  # Should be [height * width, 4]
        print(f"target['boxes'] shape for batch {i}: {target['boxes'].shape}")  # Should be [batch_size, 4]

        # Add regression loss
        reg_loss += regression_loss_fn(predicted_boxes, target['boxes'])

    return cls_loss + reg_loss


# Contrastive loss for object-level feature alignment

import torch
import torch.nn.functional as F

def contrastive_loss(student_cls_out, teacher_cls_out, margin=1.0, temperature=0.07):
    """
    Contrastive loss between student and teacher features.
    """
    device = student_cls_out.device

    # Normalize the outputs to get embeddings
    student_features = F.normalize(student_cls_out, p=2, dim=1).to(device)  # Shape: [batch_size, channels, H, W]
    teacher_features = F.normalize(teacher_cls_out, p=2, dim=1).to(device)  # Shape: [batch_size, channels, H, W]

    # Flatten the features
    batch_size, channels, height, width = student_features.shape
    num_features = height * width

    # Reshape features to [batch_size, num_features, channels]
    student_features_flat = student_features.view(batch_size, channels, -1).permute(0, 2, 1)  # Shape: [batch_size, num_features, channels]
    teacher_features_flat = teacher_features.view(batch_size, channels, -1).permute(0, 2, 1)  # Shape: [batch_size, num_features, channels]

    # Combine batch and spatial dimensions
    student_features_combined = student_features_flat.reshape(-1, channels)  # Shape: [batch_size * num_features, channels]
    teacher_features_combined = teacher_features_flat.reshape(-1, channels)  # Shape: [batch_size * num_features, channels]

    # Compute positive pair loss (same positions in student and teacher)
    positive_loss = F.mse_loss(student_features_combined, teacher_features_combined)

    # Compute similarity matrix between all student and teacher features
    similarity_matrix = torch.mm(student_features_combined, teacher_features_combined.t())  # Shape: [N, N], where N = batch_size * num_features

    # Create labels for contrastive learning
    labels = torch.arange(similarity_matrix.size(0), device=device)

    # Scale similarities by temperature
    logits = similarity_matrix / temperature

    # Apply cross-entropy loss for contrastive learning
    contrastive_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_value = contrastive_loss_fn(logits, labels)

    # Total loss
    total_loss = positive_loss + contrastive_loss_value

    return total_loss

