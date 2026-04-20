"""
    ESAIP
    Computer vision TP3
    18/02/2026
    ROBERT Paul-Aimé
    IR4 ING2027
"""

from torch.utils.data import Dataset

import os

from pathlib import Path
from PIL import Image
from src.utils.const import PATH,CLASSES


#======================


class TrainDataset(Dataset):
    """
    data set for the training 
    """
    def __init__(self, root_path = PATH, classes = CLASSES, transform=None, multiplier=1):
        self.root_path = Path(root_path)
        self.classes = classes
        self.transform = transform
        self.multiplier = multiplier
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        
        self.samples = []
        for cls in classes:
            for img_path in self.root_path.glob(f'{cls}/train/*/*.png'):
                self.samples.append((img_path, self.class_to_idx[cls]))
        
        print(f"Dataset chargé : {len(self.samples)} images physiques.")
        print(f"Dataset virtuel : {len(self.samples) * self.multiplier} images.")

    def __len__(self):
        # On multiplie la longueur par le facteur choisi
        return len(self.samples) * self.multiplier

    def __getitem__(self, idx):
        # On utilise le modulo pour retomber sur un index de chemin valide
        real_idx = idx % len(self.samples)
        path, label = self.samples[real_idx]
        
        # Maintenant path est GARANTI d'être un chemin de fichier
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label


def check_images(path = PATH):
    nb_image = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    img.verify() # Vérifie que le fichier n'est pas cassé
                except (IOError, SyntaxError) as e:
                    print(f'IMAGE CORROMPUE : {img_path}')
                nb_image += 1
    print("all_images (",nb_image,") are valid")

if __name__ == "__main__":
    check_images(PATH)
    mvtecad_dataset = TrainDataset()
