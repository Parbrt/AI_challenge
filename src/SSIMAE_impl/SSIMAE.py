"""
    SSIM Autoencoder — Bergmann et al. 2019
    "Improving Unsupervised Defect Segmentation by Applying
     Structural Similarity To Autoencoders"  (arXiv:1807.02011)

    Adapté depuis l'implémentation existante pour :
      - Images RGB 128×128 (MVTec AD)
      - Loss SSIM différentiable (au lieu de MSE)
      - Résidual map SSIM pour la segmentation de défauts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

from src.utils.Dataset import TrainDataset, TestDataset

RANDOM_SEED = 42
LR = 2e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
NB_EPOCHS = 20
D = 100
K = 11
PATCH_SIZE = 128

torch.manual_seed(RANDOM_SEED)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, size=PATCH_SIZE):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :self.size, :self.size]



class SSIMAutoEncoder(nn.Module):
    """
    Autoencoder

    """

    def __init__(self, in_channels = 3, #rgb
                 latent_dim = 100):
        super().__init__()

        lrelu = lambda: nn.LeakyReLU(0.2, inplace=True)

        self.encoder = nn.Sequential(
            #1
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #2
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #3
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #4
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #5
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #6
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #7
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #8
            nn.Conv2d(32, latent_dim, kernel_size=8, stride=1, padding=0),
        )

        self.decoder = nn.Sequential(
            #9
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=8, stride=1, padding=0),
            lrelu(),
            #8
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #7
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #6
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #5
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #3
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            lrelu(),
            #2
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            lrelu(),
            #1
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            Trim(PATCH_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


def _gaussian_kernel(window_size, sigma = 1.5):
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size, channels):
    k1d = _gaussian_kernel(window_size)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    window = k2d.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_map(x: torch.Tensor, y: torch.Tensor,
             window_size: int = D,
             C1: float = 0.01 ** 2,
             C2: float = 0.03 ** 2):
    """
    Calcule la carte SSIM pixel-à-pixel entre x et y.
    Retourne un tenseur de forme (B, 1, H, W) avec des valeurs dans [-1, 1].
    Les valeurs proches de 1 indiquent une forte similarité.
    """
    channels = x.shape[1]
    window = _create_window(window_size, channels).to(x.device)

    pad = window_size // 2
    mu_x  = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y  = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    membre_1 = (2 * mu_xy  + C1) * (2 * sigma_xy  + C2)
    membre_2 = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim  = membre_1 / (membre_2 + 1e-8)

    return ssim.mean(dim=1, keepdim=True)


def ssim_loss(x: torch.Tensor, y: torch.Tensor,
              window_size: int = D) -> torch.Tensor:
    """
    """
    return (1.0 - ssim_map(x, y, window_size)).mean()


def ssim_residual_map(x: torch.Tensor, x_hat: torch.Tensor,
                      window_size: int = D) -> torch.Tensor:
    """
    """
    return 1.0 - ssim_map(x, x_hat, window_size)


#train
def train_ssimae(nb_epochs,model,optimizer,train_loader,device,logging_interval = 50,save_model = None):
    """
    Entraîne le SSIM-AE.

    Returns
    -------
    log_dict : dict avec 'train_loss_per_batch' et 'train_loss_per_epoch'
    """
    log_dict = {
        'train_loss_per_batch': [],
        'train_loss_per_epoch': [],
    }
    model.to(device)
    start = time.time()

    for epoch in range(nb_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward
            reconstructions = model(images)
            loss = ssim_loss(images, reconstructions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            log_dict['train_loss_per_batch'].append(loss_val)
            epoch_losses.append(loss_val)

            if batch_idx % logging_interval == 0:
                print(f"Epoch {epoch+1:03d}/{nb_epochs} | "
                      f"Batch {batch_idx:04d}/{len(train_loader)} | "
                      f"SSIM Loss: {loss_val:.4f}")

        mean_epoch_loss = np.mean(epoch_losses)
        log_dict['train_loss_per_epoch'].append(mean_epoch_loss)
        print(f"*** Epoch {epoch+1:03d}/{nb_epochs} | "
              f"Mean SSIM Loss: {mean_epoch_loss:.4f} | "
              f"Elapsed: {(time.time()-start)/60:.1f} min")

    print(f"Training complete in {(time.time()-start)/60:.2f} min")

    if save_model is not None:
        torch.save(model.state_dict(), save_model)
        print(f"Model saved → {save_model}")

    return log_dict



@torch.no_grad()
def evaluate_defect_segmentation(
        model:       nn.Module,
        test_loader: DataLoader,
        device:      torch.device,
        threshold:   float = 0.3,
) -> None:
    """
    Affiche pour chaque batch :
        - image originale
        - reconstruction
        - carte résiduelle SSIM
        - masque de segmentation (seuillage)
    """
    model.eval()
    model.to(device)

    for images, labels in test_loader:
        images = images.to(device)
        reconstructions = model(images)
        residuals = ssim_residual_map(images, reconstructions)  # (B,1,H,W)

        # Segmentation par seuillage
        seg_masks = (residuals > threshold).float()

        # Affichage du premier batch uniquement
        bs = min(4, images.shape[0])
        fig, axes = plt.subplots(4, bs, figsize=(bs * 3, 12))
        titles = ["Original", "Reconstruction", "Résiduel SSIM", "Segmentation"]

        for i in range(bs):
            img     = images[i].cpu().permute(1, 2, 0).numpy()
            rec     = reconstructions[i].cpu().permute(1, 2, 0).numpy()
            res     = residuals[i, 0].cpu().numpy()
            seg     = seg_masks[i, 0].cpu().numpy()

            for row, (data, cmap) in enumerate(
                zip([img, rec, res, seg],
                    [None, None, 'hot', 'gray'])):
                axes[row, i].imshow(
                    np.clip(data, 0, 1) if cmap is None else data,
                    cmap=cmap
                )
                axes[row, i].axis('off')
                if i == 0:
                    axes[row, i].set_title(titles[row], fontsize=10)

        plt.tight_layout()
        plt.show()
        break   # un seul batch pour la démo



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),        # down-scale avant crop (paper §4.2)
        transforms.RandomCrop(PATCH_SIZE),
        transforms.ToTensor(),                # [0,1]
    ])

    test_transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = TrainDataset(transform=train_transform)
    test_dataset  = TestDataset(transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    model     = SSIMAutoEncoder(in_channels=3, latent_dim=K)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    log_dict = train_ssimae(
        nb_epochs=NB_EPOCHS,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        logging_interval=50,
        save_model="ssim_ae_mvtec.pth",
    )

    plt.figure(figsize=(8, 4))
    plt.plot(log_dict['train_loss_per_epoch'])
    plt.xlabel("Epoch")
    plt.ylabel("SSIM Loss")
    plt.title("SSIM Autoencoder — Courbe d'entraînement")
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()

    evaluate_defect_segmentation(model, test_loader, device, threshold=0.3)
