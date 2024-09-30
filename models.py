import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(BaseFeatureExtractor, self).__init__()
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Remove the last fully connected layer
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        else:
            raise ValueError("Unsupported backbone architecture")

    def forward(self, x):
        return self.model(x)

class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionHead, self).__init__()
        # Add layers for classification and bounding box regression
        self.cls_head = nn.Conv2d(2048, num_classes, kernel_size=1)  # Classification head
        self.reg_head = nn.Conv2d(2048, 4, kernel_size=1)  # Bounding box regression head

    def forward(self, x):
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()
        self.backbone = BaseFeatureExtractor(backbone='resnet50')
        self.head = ObjectDetectionHead(num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        cls_out, reg_out = self.head(features)
        return cls_out, reg_out

class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()
        self.backbone = BaseFeatureExtractor(backbone='resnet50')
        self.head = ObjectDetectionHead(num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        cls_out, reg_out = self.head(features)
        return cls_out, reg_out

class ContrastiveMeanTeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(ContrastiveMeanTeacherModel, self).__init__()
        self.teacher = TeacherModel(num_classes=num_classes)
        self.student = StudentModel(num_classes=num_classes)

    def forward(self, x):
        # Forward pass through both teacher and student models
        student_cls_out, student_reg_out = self.student(x)
        teacher_cls_out, teacher_reg_out = self.teacher(x)
        
        return student_cls_out, student_reg_out, teacher_cls_out, teacher_reg_out

    def update_teacher_weights(self, student_model, ema_decay=0.999):
        # Update the teacher model's weights using EMA
        for teacher_param, student_param in zip(self.teacher.parameters(), student_model.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1. - ema_decay) * student_param.data
