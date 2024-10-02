import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import ContrastiveMeanTeacherModel
from losses import contrastive_loss, detection_loss
from dataset import get_dataset
from utils import save_checkpoint, log_metrics, calculate_map

# Custom collate function to handle batches with different numbers of objects (bounding boxes)
def collate_fn(batch):
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample[0])  # Append image
        targets.append(sample[1])  # Append target (bounding boxes and labels)
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    
    # Targets remain as a list of dictionaries (with 'boxes' and 'labels')
    return images, targets

def train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=10, ema_decay=0.999):
    torch.cuda.empty_cache()
    model = model.to(device)
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            images, targets = batch
            images = images.to(device)

            # Move each target dictionary to the correct device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            student_cls_out, student_reg_out, teacher_cls_out, teacher_reg_out = model(images)

            student_cls_out = student_cls_out.to(device)
            teacher_cls_out = teacher_cls_out.to(device)

            # Calculate losses
            det_loss = detection_loss(student_cls_out, student_reg_out, targets)
            cont_loss = contrastive_loss(student_cls_out, teacher_cls_out)

            # Total loss
            loss = det_loss + cont_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher weights with EMA
            model.update_teacher_weights(model.student, ema_decay=ema_decay)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        save_checkpoint(model, epoch, optimizer)

        # Log metrics
        log_metrics(epoch, avg_loss)

        # Evaluate on validation set
        model.eval()
        predictions = []
        targets_list = []
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_images, val_targets = val_batch
                val_images = val_images.to(device)

                student_cls_out, student_reg_out, _, _ = model(val_images)
                
                # Convert model output to format required for mAP calculation
                predictions.append({
                    'boxes': student_reg_out.cpu(),
                    'labels': torch.argmax(student_cls_out, dim=1).cpu(),
                })
                targets_list.append(val_targets)

        # Calculate mAP at IoU 0.5
        map_score = calculate_map(predictions, targets_list, iou_threshold=0.5)
        print(f"Epoch [{epoch+1}/{num_epochs}], mAP: {map_score:.4f}")

if __name__ == '__main__':
    # Hyperparameters and setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 20  # For Pascal VOC → Clipart1k
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 4

    # Load datasets
    train_dataset = get_dataset('trainval', domain='source')  # Pascal VOC
    val_dataset = get_dataset('val', domain='target')  # Clipart1k

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # Initialize model and optimizer
    model = ContrastiveMeanTeacherModel(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start training
    train(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs)
