# EfficientAD — Détection de défauts industriels

Implémentation d'un modèle de détection d'anomalies basé sur EfficientAD, appliqué au dataset MVTec AD.

## Architecture

Le modèle repose sur une approche **Teacher-Student** :
- **Teacher** : backbone ResNet18 pré-entraîné (ImageNet), gelé
- **Student** : petit CNN entraîné à imiter le Teacher sur des images *normales*
- **Détection** : l'anomalie est détectée par la différence entre les sorties du Teacher et du Student — plus l'écart est grand, plus la zone est suspecte

## Démo Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application permet de :
1. Sélectionner une classe produit (bottle, cable, carpet, hazelnut, screw)
2. Charger une image PNG ou JPG
3. Ajuster le seuil de détection (slider 0.0–1.0)
4. Obtenir un verdict (Normale / Défectueuse) avec score et heatmap d'anomalie

## Classes disponibles

| Classe | Modèle |
|--------|--------|
| bottle | `src/model/student_bottle.pth` |
| cable | `src/model/student_cable.pth` |
| carpet | `src/model/student_carpet.pth` |
| hazelnut | `src/model/student_hazelnut.pth` |
| screw | `src/model/student_screw.pth` |

## Structure

```
app.py                  # Application Streamlit
src/
  model/                # Poids Student entraînés (.pth)
  EFFICIENTAD_impl/     # Entraînement, évaluation, visualisation
  utils/                # Dataset, constantes
```

## Dépendances

```bash
pip install -r requirements.txt
```

Principales : `torch`, `torchvision`, `streamlit`, `Pillow`, `matplotlib`, `scikit-learn`
