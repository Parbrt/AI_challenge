"""
Visualisation des résultats - Confusion Matrix + Courbes ROC
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path


def save_confusion_matrix(img_labels, img_scores, threshold, class_name, output_dir="results"):
    """Sauvegarde la matrice de confusion pour une classe."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    preds = (img_scores >= threshold).astype(int)
    cm = confusion_matrix(img_labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Défaut"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion — {class_name}")

    path = f"{output_dir}/confusion_{class_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")


def save_roc_curve(img_labels, img_scores, class_name, output_dir="results"):
    """Sauvegarde la courbe ROC image-level pour une classe."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(img_labels, img_scores)
    auc = roc_auc_score(img_labels, img_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Image — {class_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    path = f"{output_dir}/roc_image_{class_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")


def save_pixel_roc_curve(px_labels, px_scores, class_name, output_dir="results"):
    """Sauvegarde la courbe ROC pixel-level pour une classe."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sous-échantillonner pour éviter les problèmes de mémoire
    n = len(px_labels)
    if n > 500_000:
        idx = np.random.choice(n, 500_000, replace=False)
        px_labels = px_labels[idx]
        px_scores = px_scores[idx]

    fpr, tpr, _ = roc_curve(px_labels, px_scores)
    auc = roc_auc_score(px_labels, px_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="red", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Pixel — {class_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    path = f"{output_dir}/roc_pixel_{class_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")


def save_comparison_roc(all_results, output_dir="results"):
    """Sauvegarde une courbe ROC comparant toutes les classes sur un seul graphe."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Image-level
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, data in all_results.items():
        fpr, tpr, _ = roc_curve(data["img_labels"], data["img_scores"])
        auc = roc_auc_score(data["img_labels"], data["img_scores"])
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Comparaison ROC Image — Toutes les classes")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    path = f"{output_dir}/roc_comparison_image.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")

    # Pixel-level
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, data in all_results.items():
        px_l = data["px_labels"]
        px_s = data["px_scores"]
        n = len(px_l)
        if n > 500_000:
            idx = np.random.choice(n, 500_000, replace=False)
            px_l = px_l[idx]
            px_s = px_s[idx]
        fpr, tpr, _ = roc_curve(px_l, px_s)
        auc = roc_auc_score(px_l, px_s)
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Comparaison ROC Pixel — Toutes les classes")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    path = f"{output_dir}/roc_comparison_pixel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")


def save_summary_bar_chart(all_metrics, output_dir="results"):
    """Sauvegarde un bar chart comparant Img AUROC / Px AUROC / F1 par classe."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    classes = list(all_metrics.keys())
    img_aurocs = [all_metrics[c]["image_auroc"] for c in classes]
    px_aurocs = [all_metrics[c]["pixel_auroc"] for c in classes]
    img_f1s = [all_metrics[c]["image_f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, img_aurocs, width, label="Image AUROC", color="#2196F3")
    ax.bar(x, px_aurocs, width, label="Pixel AUROC", color="#4CAF50")
    ax.bar(x + width, img_f1s, width, label="Image F1", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Comparaison des performances par classe")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)

    path = f"{output_dir}/summary_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvé : {path}")