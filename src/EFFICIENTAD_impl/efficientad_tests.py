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
from src.EFFICIENTAD_impl.visualisation import (save_confusion_matrix, save_roc_curve,
                           save_pixel_roc_curve, save_comparison_roc,
                           save_summary_bar_chart)


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
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
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

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{epochs}  loss = {avg_loss:.5f}  lr = {lr:.6f}")

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

    student.eval()
    img_scores, img_labels = [], []
    px_scores, px_labels = [], []

    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="Évaluation", leave=False):
            images = images.to(device)
            heatmap = get_anomaly_map(teacher, student, images)
            heatmap = (heatmap - score_mean) / (score_std + 1e-8)

            for i in range(images.shape[0]):
                topk = int(0.01 * heatmap[i].numel())
                img_scores.append(heatmap[i].reshape(-1).topk(topk).values.mean().item())
                img_labels.append(int(labels[i]))
                px_scores.append(heatmap[i].cpu().numpy().ravel())
                px_labels.append(masks[i].numpy().ravel())

    img_scores = np.array(img_scores)
    img_labels = np.array(img_labels)
    px_scores = np.concatenate(px_scores)
    px_labels = np.concatenate(px_labels).astype(int)

    fpr, tpr, thresholds = roc_curve(img_labels, img_scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    return {
        "image_auroc": roc_auc_score(img_labels, img_scores),
        "pixel_auroc": roc_auc_score(px_labels, px_scores),
        "image_f1": f1_score(img_labels, (img_scores >= best_thresh).astype(int)),
        "threshold": best_thresh,
        "img_scores": img_scores,
        "img_labels": img_labels,
        "px_scores": px_scores,
        "px_labels": px_labels,
    }


# =============================================================================
#  MAIN
# =============================================================================


if __name__ == "__main__":

    RUN_CLASSES = ["bottle", "carpet", "hazelnut", "screw", "cable"]
    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_metrics = {}
    all_raw = {}

    for CLASS_NAME in RUN_CLASSES:

        print(f"\n{'='*50}")
        print(f"  {CLASS_NAME}")
        print(f"{'='*50}")

        train_ds = TrainDataset(root_path=PATH, classes=[CLASS_NAME],
                                transform=transform)
        test_ds = TestDataset(root_path=PATH, class_name=CLASS_NAME,
                              transform=transform, img_size=IMG_SIZE)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        teacher = Teacher().to(DEVICE).eval()
        student = Student().to(DEVICE)

        print("Entraînement...")
        train(teacher, student, train_loader, EPOCHS, DEVICE)

        print("\nNormalisation...")
        score_mean, score_std = compute_norm_stats(teacher, student, train_loader, DEVICE)

        print("\nÉvaluation...")
        results = evaluate(teacher, student, test_loader, DEVICE, score_mean, score_std)

        all_metrics[CLASS_NAME] = results
        all_raw[CLASS_NAME] = {
            "img_scores": results["img_scores"],
            "img_labels": results["img_labels"],
            "px_scores": results["px_scores"],
            "px_labels": results["px_labels"],
        }

        # Graphes par classe
        save_confusion_matrix(results["img_labels"], results["img_scores"],
                              results["threshold"], CLASS_NAME)
        save_roc_curve(results["img_labels"], results["img_scores"], CLASS_NAME)
        save_pixel_roc_curve(results["px_labels"], results["px_scores"], CLASS_NAME)

        print(f"  Image AUROC : {results['image_auroc']:.4f}")
        print(f"  Pixel AUROC : {results['pixel_auroc']:.4f}")
        print(f"  Image F1    : {results['image_f1']:.4f}")

        torch.save(student.state_dict(), f"student_{CLASS_NAME}.pth")

    # Graphes comparatifs
    print("\nGénération des graphes comparatifs...")
    save_comparison_roc(all_raw)
    save_summary_bar_chart(all_metrics)

    # Tableau récap
    print(f"\n{'='*60}")
    print(f"  {'Classe':<15} {'Img AUROC':>10} {'Px AUROC':>10} {'Img F1':>10}")
    print(f"  {'-'*45}")
    for cls, r in all_metrics.items():
        print(f"  {cls:<15} {r['image_auroc']:>10.4f} {r['pixel_auroc']:>10.4f} {r['image_f1']:>10.4f}")

    mean_img = np.mean([r['image_auroc'] for r in all_metrics.values()])
    mean_px = np.mean([r['pixel_auroc'] for r in all_metrics.values()])
    mean_f1 = np.mean([r['image_f1'] for r in all_metrics.values()])
    print(f"  {'-'*45}")
    print(f"  {'MOYENNE':<15} {mean_img:>10.4f} {mean_px:>10.4f} {mean_f1:>10.4f}")
    print(f"{'='*60}")