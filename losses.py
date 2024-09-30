import torch
import torch.nn as nn
import torch.nn.functional as F

# Detection loss (combines classification loss and bounding box regression loss)

# Detection loss (combines classification loss and bounding box regression loss)
def detection_loss(student_cls_out, student_reg_out, targets):
    classification_loss_fn = nn.CrossEntropyLoss()
    bbox_regression_loss_fn = nn.SmoothL1Loss()

    total_cls_loss = 0.0
    total_reg_loss = 0.0

    # Loop over the batch
    for i, target in enumerate(targets):
        # For classification, we need to extract predictions that correspond to the number of objects (bounding boxes) in the target.
        num_objects = target['labels'].shape[0]  # Number of objects in this image

        # Resize student_cls_out[i] to match the number of objects
        cls_out = student_cls_out[i].view(-1, student_cls_out[i].shape[-1])  # Flatten spatial dimensions (H, W) if necessary
        cls_out = cls_out[:num_objects]  # Select the first `num_objects` predictions

        # Calculate classification loss for the current image
        cls_loss = classification_loss_fn(cls_out, target['labels'])

        # Similarly, adjust bounding box regression output
        reg_out = student_reg_out[i].view(-1, 4)  # Flatten to (num_predictions, 4)
        reg_out = reg_out[:num_objects]  # Select the first `num_objects` predictions

        # Calculate bounding box regression loss for the current image
        reg_loss = bbox_regression_loss_fn(reg_out, target['boxes'])

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss

    # Return the total loss for the batch
    total_loss = total_cls_loss + total_reg_loss
    return total_loss



# Contrastive loss for object-level feature alignment
def contrastive_loss(student_cls_out, teacher_cls_out, margin=1.0):
    """
    Contrastive loss between student and teacher features.
    Margin defines the separation between positive and negative pairs.
    """
    batch_size = student_cls_out.size(0)

    # Normalize the outputs to get embeddings
    student_features = F.normalize(student_cls_out, p=2, dim=1)
    teacher_features = F.normalize(teacher_cls_out, p=2, dim=1)

    # Positive pair loss (same objects across teacher-student)
    positive_loss = F.mse_loss(student_features, teacher_features)

    # Negative pair loss (to ensure separation of different objects)
    neg_mask = (torch.ones_like(positive_loss) - torch.eye(batch_size)).to(student_cls_out.device)
    negative_loss = torch.sum(torch.max(margin - torch.matmul(student_features, teacher_features.T) * neg_mask, torch.tensor(0.0).to(student_cls_out.device)))

    # Total contrastive loss
    total_loss = positive_loss + negative_loss
    return total_loss
