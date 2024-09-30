import torch
import os
from models import ContrastiveMeanTeacherModel
from dataset import get_dataset
from utils import load_checkpoint
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model and checkpoint
def load_trained_model(checkpoint_path, num_classes, device):
    model = ContrastiveMeanTeacherModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, checkpoint_path)
    model.to(device)
    return model

# Visualize and save bounding boxes on images
def save_results(image, boxes, labels, filename, save_dir='test_results'):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert from Tensor to numpy

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        label = labels[i]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{label}', color='white', fontsize=12, backgroundcolor='red')

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    filepath = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved result to {filepath}")

# Run inference and save results
def test_model(model, dataloader, device, save_dir='test_results'):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images, targets = batch
            images = images.to(device)

            # Run inference (only student model for testing)
            student_cls_out, student_reg_out, _, _ = model(images)

            # Convert regression output and class output to usable format
            for i, image in enumerate(images):
                pred_boxes = student_reg_out[i].cpu().numpy()  # Predicted boxes
                pred_labels = torch.argmax(student_cls_out[i], dim=0).cpu().numpy()  # Predicted class labels

                # Save the results
                filename = f"image_{idx}_{i}"
                save_results(image, pred_boxes, pred_labels, filename, save_dir)

if __name__ == '__main__':
    # Hyperparameters and setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 20  # For Pascal VOC → Clipart1k
    checkpoint_path = './checkpoints/model_epoch_best.pth'

    # Load the dataset
    test_dataset = get_dataset('test', domain='target')  # Clipart1k test set
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = load_trained_model(checkpoint_path, num_classes, device)

    # Run inference on the test set and save results
    test_model(model, test_loader, device, save_dir='test_results')
