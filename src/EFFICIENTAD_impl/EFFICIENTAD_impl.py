import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.utils.Dataset import TrainDataset
from src.utils.const import PATH, CLASSES


# ---------------------------------------------------------------------------
# Réseaux
# ---------------------------------------------------------------------------

class PDN_Small(nn.Module):
    """
    Patch Description Network - architecture "small"
    Input  : (B, 3, H, W)
    Output : (B, 384, ~H/4, ~W/4)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1, padding=0)
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.bn2   = nn.BatchNorm2d(256)
        self.bn3   = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class _TeacherPretrainer(nn.Module):
    """Tête de projection pour aligner PDN_Small sur les features ResNet50."""
    def __init__(self, teacher: PDN_Small):
        super().__init__()
        self.teacher = teacher
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 2048),
        )

    def forward(self, x):
        return self.head(self.teacher(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),   # H/2
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),   # H/4
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),   # H/8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),   # H/16
            nn.BatchNorm2d(256), nn.ReLU(),
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    """Décode de H/16 vers H/4 pour correspondre à la résolution du Teacher."""
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # H/8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 384, 4, stride=2, padding=1),  # H/4
        )

    def forward(self, x):
        return self.dec(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Fonctions de loss
# ---------------------------------------------------------------------------

def loss_autoencoder(teacher_features, ae_output):
    if ae_output.shape[2:] != teacher_features.shape[2:]:
        ae_output = F.interpolate(ae_output, size=teacher_features.shape[2:],
                                  mode="bilinear", align_corners=False)
    return F.mse_loss(F.normalize(ae_output, dim=1),
                      F.normalize(teacher_features, dim=1))


def hard_feature_loss(teacher_out, student_out, ratio=0.999):
    """Ne back-propage que sur les (1-ratio)% de patches les plus difficiles."""
    diff = (F.normalize(teacher_out, dim=1) -
            F.normalize(student_out, dim=1)).pow(2).mean(dim=1)
    k    = max(1, int((1 - ratio) * diff.numel()))
    return diff.view(-1).topk(k).values.mean()


# ---------------------------------------------------------------------------
# Pré-entraînement du Teacher
# ---------------------------------------------------------------------------

def pretrain_teacher(teacher, pretrain_loader, epochs=10, lr=1e-4, device="cuda"):
    """
    Entraîne le Teacher (PDN_Small) à imiter les features de ResNet50
    sur les images normales MVTec avant l'entraînement principal.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False

    pretrainer = _TeacherPretrainer(teacher).to(device)
    optimizer  = torch.optim.Adam(pretrainer.parameters(), lr=lr)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Pré-entraînement Teacher", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0
        pbar = tqdm(pretrain_loader, desc=f"Epoch {epoch}/{epochs}",
                    leave=False, unit="batch")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            with torch.no_grad():
                target = feature_extractor(imgs).squeeze(-1).squeeze(-1)

            loss = F.mse_loss(pretrainer(imgs), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_bar.set_postfix(loss=f"{total_loss / len(pretrain_loader):.4f}")

    return teacher


# ---------------------------------------------------------------------------
# Dataset de test MVTec AD
# ---------------------------------------------------------------------------

class MVTecTestDataset(Dataset):
    """
    Charge le split test MVTec AD pour une catégorie.
    Retourne (image_tensor, pixel_mask, image_label) :
      - pixel_mask  : (H, W) float, 1 = pixel anomal, 0 = normal
      - image_label : 0 = normal, 1 = anomal
    """
    def __init__(self, root_path, class_name, transform=None, img_size=256):
        self.transform = transform
        self.img_size  = img_size
        self.samples   = []

        test_root = Path(root_path) / class_name / "test"
        gt_root   = Path(root_path) / class_name / "ground_truth"

        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_normal = defect_dir.name == "good"
            label = 0 if is_normal else 1
            for img_path in sorted(defect_dir.glob("*.png")):
                if is_normal:
                    mask_path = None
                else:
                    candidate = gt_root / defect_dir.name / (img_path.stem + "_mask.png")
                    mask_path = candidate if candidate.exists() else None
                self.samples.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(TF.resize(img, [self.img_size, self.img_size]))

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L").resize(
                (self.img_size, self.img_size), Image.NEAREST
            )
            mask = torch.from_numpy(np.array(mask) / 255.0).float()
        else:
            mask = torch.zeros(self.img_size, self.img_size)

        return img, mask, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EfficientAD_Trainer:
    def __init__(self, teacher=None, lr=1e-4, device="cuda"):
        self.device  = device
        self.teacher = (teacher if teacher is not None else PDN_Small()).to(device)
        self.student = PDN_Small().to(device)
        self.ae      = Autoencoder().to(device)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            list(self.student.parameters()) +
            list(self.ae.parameters()),
            lr=lr,
        )

        self.mu_s, self.sigma_s = None, None
        self.mu_a, self.sigma_a = None, None

    def train_epoch(self, dataloader, epoch=None, total_epochs=None):
        self.student.train()
        self.ae.train()
        total_loss = 0

        desc = f"Epoch {epoch}/{total_epochs}" if epoch else "Training"
        pbar = tqdm(dataloader, desc=desc, leave=False, unit="batch")
        for imgs, _ in pbar:
            imgs = imgs.to(self.device)

            with torch.no_grad():
                t_feats = self.teacher(imgs)

            s_feats  = self.student(imgs)
            ae_feats = self.ae(imgs)

            loss_s  = hard_feature_loss(t_feats, s_feats)
            loss_ae = loss_autoencoder(t_feats, ae_feats)

            ae_aligned = ae_feats
            if ae_aligned.shape[2:] != s_feats.shape[2:]:
                ae_aligned = F.interpolate(ae_aligned, size=s_feats.shape[2:],
                                           mode="bilinear", align_corners=False)
            loss_penalty = -F.mse_loss(F.normalize(s_feats, dim=1),
                                       F.normalize(ae_aligned.detach(), dim=1))

            loss = loss_s + loss_ae + 0.1 * loss_penalty
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(dataloader)

    def compute_normalization_stats(self, dataloader):
        """Calcule mean/std sur le train set pour normaliser les scores à l'inférence."""
        scores_s, scores_a = [], []
        self.student.eval()
        self.ae.eval()

        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc="Stats normalisation", leave=False):
                imgs    = imgs.to(self.device)
                t_feats = self.teacher(imgs)
                s_feats = self.student(imgs)
                a_feats = self.ae(imgs)

                if a_feats.shape[2:] != t_feats.shape[2:]:
                    a_feats = F.interpolate(a_feats, size=t_feats.shape[2:],
                                            mode="bilinear", align_corners=False)

                scores_s.append((F.normalize(t_feats, dim=1) -
                                  F.normalize(s_feats, dim=1)).pow(2).mean(dim=1))
                scores_a.append((F.normalize(t_feats, dim=1) -
                                  F.normalize(a_feats, dim=1)).pow(2).mean(dim=1))

        scores_s = torch.cat([s.view(-1) for s in scores_s])
        scores_a = torch.cat([s.view(-1) for s in scores_a])
        self.mu_s, self.sigma_s = scores_s.mean(), scores_s.std()
        self.mu_a, self.sigma_a = scores_a.mean(), scores_a.std()

    @torch.no_grad()
    def predict(self, img_tensor, threshold=0.5):
        """
        img_tensor : (1, 3, H, W)
        Retourne   : dict avec 'decision', 'score', 'heatmap'
        """
        self.student.eval()
        self.ae.eval()

        img     = img_tensor.to(self.device)
        t_feats = self.teacher(img)
        s_feats = self.student(img)
        a_feats = self.ae(img)

        if a_feats.shape[2:] != t_feats.shape[2:]:
            a_feats = F.interpolate(a_feats, size=t_feats.shape[2:],
                                    mode="bilinear", align_corners=False)

        map_s = (F.normalize(t_feats, dim=1) - F.normalize(s_feats, dim=1)).pow(2).mean(dim=1)
        map_a = (F.normalize(t_feats, dim=1) - F.normalize(a_feats, dim=1)).pow(2).mean(dim=1)

        if self.mu_s is not None:
            map_s = (map_s - self.mu_s) / (self.sigma_s + 1e-8)
            map_a = (map_a - self.mu_a) / (self.sigma_a + 1e-8)

        anomaly_map = 0.5 * map_s + 0.5 * map_a
        H, W = img.shape[2], img.shape[3]
        heatmap = F.interpolate(anomaly_map.unsqueeze(1), size=(H, W),
                                mode="bilinear", align_corners=False).squeeze()
        score = heatmap.max().item()

        return {
            "decision": "NOK" if score > threshold else "OK",
            "score":    score,
            "heatmap":  heatmap.cpu().numpy(),
        }

    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        Évalue sur le test set MVTec AD.
        Nécessite que compute_normalization_stats() ait été appelé avant.

        Retourne image_auroc, pixel_auroc, image_f1, pixel_f1.
        """
        self.student.eval()
        self.ae.eval()

        img_scores, img_labels     = [], []
        pixel_scores, pixel_labels = [], []

        for imgs, masks, labels in tqdm(test_loader, desc="Évaluation", leave=False):
            imgs = imgs.to(self.device)

            t_feats = self.teacher(imgs)
            s_feats = self.student(imgs)
            a_feats = self.ae(imgs)

            if a_feats.shape[2:] != t_feats.shape[2:]:
                a_feats = F.interpolate(a_feats, size=t_feats.shape[2:],
                                        mode="bilinear", align_corners=False)

            map_s = (F.normalize(t_feats, dim=1) - F.normalize(s_feats, dim=1)).pow(2).mean(dim=1)
            map_a = (F.normalize(t_feats, dim=1) - F.normalize(a_feats, dim=1)).pow(2).mean(dim=1)

            if self.mu_s is not None:
                map_s = (map_s - self.mu_s) / (self.sigma_s + 1e-8)
                map_a = (map_a - self.mu_a) / (self.sigma_a + 1e-8)

            heatmap = F.interpolate(
                (0.5 * map_s + 0.5 * map_a).unsqueeze(1),
                size=(imgs.shape[2], imgs.shape[3]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

            for i in range(imgs.shape[0]):
                img_scores.append(heatmap[i].max().item())
                img_labels.append(labels[i].item())
                pixel_scores.append(heatmap[i].cpu().numpy().ravel())
                pixel_labels.append(masks[i].cpu().numpy().ravel())

        img_scores   = np.array(img_scores)
        img_labels   = np.array(img_labels)
        pixel_scores = np.concatenate(pixel_scores)
        pixel_labels = np.concatenate(pixel_labels).astype(np.int32)

        def _best_f1(labels, scores):
            fpr, tpr, thresholds = roc_curve(labels, scores)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            return f1_score(labels, (scores >= best_thresh).astype(int))

        has_img   = len(np.unique(img_labels))   > 1
        has_pixel = len(np.unique(pixel_labels)) > 1

        return {
            "image_auroc": roc_auc_score(img_labels, img_scores)     if has_img   else float("nan"),
            "pixel_auroc": roc_auc_score(pixel_labels, pixel_scores) if has_pixel else float("nan"),
            "image_f1":    _best_f1(img_labels, img_scores)          if has_img   else float("nan"),
            "pixel_f1":    _best_f1(pixel_labels, pixel_scores)      if has_pixel else float("nan"),
        }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Configuration ────────────────────────────────────────────────────────
    CLASS_NAME      = "bottle"   # ← changer pour une autre catégorie MVTec AD
    PRETRAIN_EPOCHS = 10
    EPOCHS          = 20
    BATCH_SIZE      = 8
    IMG_SIZE        = 256
    LR              = 1e-4
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device     : {DEVICE}")
    print(f"Catégorie  : {CLASS_NAME}")

    # ── Transforms ───────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = TrainDataset(
        root_path  = PATH,
        classes    = [CLASS_NAME],
        transform  = transform,
        multiplier = 3,
    )
    test_dataset = MVTecTestDataset(
        root_path  = PATH,
        class_name = CLASS_NAME,
        transform  = transform,
        img_size   = IMG_SIZE,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Pré-entraînement du Teacher ───────────────────────────────────────────
    teacher = PDN_Small()
    teacher = pretrain_teacher(teacher, train_loader,
                               epochs=PRETRAIN_EPOCHS, lr=LR, device=DEVICE)

    # ── Entraînement Student + AE ─────────────────────────────────────────────
    trainer   = EfficientAD_Trainer(teacher=teacher, lr=LR, device=DEVICE)
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Entraînement", unit="epoch")
    for epoch in epoch_bar:
        loss = trainer.train_epoch(train_loader, epoch=epoch, total_epochs=EPOCHS)
        epoch_bar.set_postfix(loss=f"{loss:.4f}")

    # ── Normalisation des scores ──────────────────────────────────────────────
    trainer.compute_normalization_stats(train_loader)

    # ── Évaluation ───────────────────────────────────────────────────────────
    metrics = trainer.evaluate(test_loader)

    print(f"\n{'═'*45}")
    print(f"  {'Image AUROC':<20} {metrics['image_auroc']:.4f}")
    print(f"  {'Pixel AUROC':<20} {metrics['pixel_auroc']:.4f}")
    print(f"  {'Image F1':<20} {metrics['image_f1']:.4f}")
    print(f"  {'Pixel F1':<20} {metrics['pixel_f1']:.4f}")
    print(f"{'═'*45}")
