import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,    64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def get_anomaly_map(teacher, student, images):
    teacher_out = teacher(images)
    student_out = student(images)
    if student_out.shape != teacher_out.shape:
        student_out = F.interpolate(student_out, size=teacher_out.shape[2:],
                                    mode="bilinear", align_corners=False)
    diff = (teacher_out - student_out).pow(2).mean(dim=1)
    H, W = images.shape[2], images.shape[3]
    heatmap = F.interpolate(diff.unsqueeze(1), size=(H, W),
                            mode="bilinear", align_corners=False).squeeze(1)
    return heatmap


@st.cache_resource
def load_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = Teacher().to(device).eval()
    return teacher, device


@st.cache_resource
def load_student(class_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"src/model/student_{class_name}.pth"
    student = Student().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    student.eval()
    return student


_COLORMAP = plt.get_cmap("jet")

TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def compute_score(heatmap: torch.Tensor) -> float:
    flat = heatmap.reshape(-1)
    topk = max(1, int(0.01 * flat.numel()))
    score_raw = flat.topk(topk).values.mean()
    return torch.sigmoid(score_raw).item()


def make_overlay(original_img: Image.Image, heatmap_tensor: torch.Tensor) -> Image.Image:
    heatmap_np = heatmap_tensor.cpu().numpy()
    heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    heatmap_rgb = (_COLORMAP(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(original_img.size, Image.Resampling.BILINEAR)
    return Image.blend(original_img.convert("RGB"), heatmap_pil, alpha=0.5)
