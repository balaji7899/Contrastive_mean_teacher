import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection

# Data augmentation
def get_weak_augmentations():
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def get_strong_augmentations():
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomResizedCrop(size=600, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

# Pascal VOC dataset loader (source domain)
# Class-to-index mapping (for Pascal VOC dataset)
class_to_idx = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

class PascalVOCDataset(Dataset):
    def __init__(self, root, split='trainval', transform=None):
        self.voc_dataset = VOCDetection(root, year='2012', image_set=split, download=False)
        self.transform = transform

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        # Extract bounding boxes and labels from target
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
            labels.append(class_to_idx[obj['name']])
        if self.transform:
            image = self.transform(image)
        return image, {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

# Clipart1k dataset loader (target domain)
class ClipartDataset(Dataset):
    def __init__(self, root, transform=None):
        # Assume Clipart1k is preloaded in a custom format
        self.clipart_data = ...  # Load Clipart1k dataset (e.g., from preprocessed data)
        self.transform = transform

    def __len__(self):
        return len(self.clipart_data)

    def __getitem__(self, idx):
        image, boxes, labels = self.clipart_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels)}

# Function to load datasets
def get_dataset(split='trainval', domain='source'):
    if domain == 'source':  # Pascal VOC (Source)
        return PascalVOCDataset(root='D:\Mtech_pro\cross-domain-detection-master\datasets\dt_watercolor\VOC2012', split=split, transform=get_weak_augmentations())
    elif domain == 'target':  # Clipart1k (Target)
        return ClipartDataset(root='D:\\Mtech_pro\\cross-domain-detection-master\\datasets\\clipart', transform=get_strong_augmentations())
    else:
        raise ValueError("Unknown domain specified")
