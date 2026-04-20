"""
EfficientAD
============================
Teacher (ResNet18 gelé) + Student (petit CNN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.Dataset import TrainDataset
from src.utils.const import PATH, CLASSES


# =============================================================================
#  TEACHER
# =============================================================================

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


# =============================================================================
#  STUDENT
# =============================================================================

class Student(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,    64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
#  DATASET TEST
# =============================================================================

class TestDataset(Dataset):

    def __init__(self, root_path, class_name, transform, img_size=256):
        self.transform = transform
        self.img_size = img_size
        self.samples = []

        test_root = Path(root_path) / class_name / "test"
        gt_root = Path(root_path) / class_name / "ground_truth"

        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_normal = (defect_dir.name == "good")
            label = 0 if is_normal else 1

            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_normal:
                    candidate = gt_root / defect_dir.name / (img_path.stem + "_mask.png")
                    if candidate.exists():
                        mask_path = candidate
                self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask) / 255.0).float()
        else:
            mask = torch.zeros(self.img_size, self.img_size)

        return img, mask, label


# =============================================================================
#  FONCTIONS
# =============================================================================

def get_anomaly_map(teacher, student, images):
    """Calcule la heatmap d'anomalie pour un batch d'images."""

    teacher_out = teacher(images)
    student_out = student(images)

    # Aligner les tailles si besoin
    if student_out.shape != teacher_out.shape:
        student_out = F.interpolate(student_out, size=teacher_out.shape[2:],
                                    mode="bilinear", align_corners=False)

    # Erreur quadratique par position spatiale
    diff = (teacher_out - student_out).pow(2).mean(dim=1)  # (B, h, w)

    # Remettre à la taille de l'image
    H, W = images.shape[2], images.shape[3]
    heatmap = F.interpolate(diff.unsqueeze(1), size=(H, W),
                            mode="bilinear", align_corners=False).squeeze(1)

    return heatmap


def train(teacher, student, train_loader, epochs, device):
    """Entraîne le Student à imiter le Teacher."""

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, _ in pbar:
            images = images.to(device)

            with torch.no_grad():
                teacher_out = teacher(images)

            student_out = student(images)

            if student_out.shape != teacher_out.shape:
                student_out = F.interpolate(student_out, size=teacher_out.shape[2:],
                                            mode="bilinear", align_corners=False)

            loss = F.mse_loss(student_out, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss = {avg_loss:.5f}")


def compute_norm_stats(teacher, student, train_loader, device):
    """Calcule mean/std des scores sur images normales."""

    student.eval()
    all_scores = []

    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            heatmap = get_anomaly_map(teacher, student, images)
            all_scores.append(heatmap.cpu())

    all_scores = torch.cat([s.reshape(-1) for s in all_scores])
    mean = all_scores.mean()
    std = all_scores.std()
    print(f"  Norm stats — mean: {mean:.4f}, std: {std:.4f}")
    return mean.to(device), std.to(device)


def evaluate(teacher, student, test_loader, device, score_mean, score_std):
    """Évalue et retourne les métriques AUROC et F1."""

    student.eval()
    img_scores, img_labels = [], []
    px_scores, px_labels = [], []

    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="Évaluation", leave=False):
            images = images.to(device)
            heatmap = get_anomaly_map(teacher, student, images)

            # Normaliser
            heatmap = (heatmap - score_mean) / (score_std + 1e-8)

            for i in range(images.shape[0]):
                img_scores.append(heatmap[i].max().item())
                img_labels.append(int(labels[i]))
                px_scores.append(heatmap[i].cpu().numpy().ravel())
                px_labels.append(masks[i].numpy().ravel())

    img_scores = np.array(img_scores)
    img_labels = np.array(img_labels)
    px_scores = np.concatenate(px_scores)
    px_labels = np.concatenate(px_labels).astype(int)

    # AUROC
    img_auroc = roc_auc_score(img_labels, img_scores)
    px_auroc = roc_auc_score(px_labels, px_scores)

    # F1 optimal
    fpr, tpr, thresholds = roc_curve(img_labels, img_scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    img_f1 = f1_score(img_labels, (img_scores >= best_thresh).astype(int))

    return {
        "image_auroc": img_auroc,
        "pixel_auroc": px_auroc,
        "image_f1": img_f1,
        "threshold": best_thresh,
    }


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    CLASS_NAME = "bottle"
    EPOCHS = 30
    BATCH_SIZE = 16
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device    : {DEVICE}")
    print(f"Catégorie : {CLASS_NAME}")
    print(f"Epochs    : {EPOCHS}\n")

    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Pour le test : PAS d'augmentation, juste resize + normalize
    test_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_ds = TrainDataset(root_path=PATH, classes=[CLASS_NAME],
                            transform=train_transform, multiplier=3)
    test_ds = TestDataset(root_path=PATH, class_name=CLASS_NAME,
                          transform=test_transform, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modèles
    teacher = Teacher().to(DEVICE).eval()
    student = Student().to(DEVICE)

    # Entraîner
    print("Entraînement...")
    train(teacher, student, train_loader, EPOCHS, DEVICE)

    # Normaliser
    print("\nNormalisation...")
    score_mean, score_std = compute_norm_stats(teacher, student, train_loader, DEVICE)

    # Évaluer
    print("\nÉvaluation...")
    results = evaluate(teacher, student, test_loader, DEVICE, score_mean, score_std)

    print(f"\n{'='*40}")
    print(f"  RÉSULTATS — {CLASS_NAME}")
    print(f"{'='*40}")
    print(f"  Image AUROC : {results['image_auroc']:.4f}")
    print(f"  Pixel AUROC : {results['pixel_auroc']:.4f}")
    print(f"  Image F1    : {results['image_f1']:.4f}")
    print(f"  Seuil       : {results['threshold']:.4f}")
    print(f"{'='*40}")

    # Sauvegarder
    torch.save(student.state_dict(), f"student_{CLASS_NAME}.pth")
    print(f"\nSauvegardé : student_{CLASS_NAME}.pth")