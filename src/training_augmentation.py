"""
training_augmentation.py

Data augmentation utilizzata per il retraining robusto.

Strategia:
- 50%: immagine originale
- 50%: una degradazione scelta uniformemente
        tra le 9 condizioni definite nel protocollo

Le degradazioni vengono applicate al crop facciale
prima del preprocessing specifico del modello.
"""

import random

from robustness_transforms import apply_degradation


# ============================================================
# CONDIZIONI UTILIZZATE DURANTE IL TRAINING
# ============================================================

TRAIN_DEGRADATION_CONDITIONS = [
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "resize_75",
    "resize_50",
    "resize_25",
    "blur_05",
    "blur_10",
    "blur_20",
]


# ============================================================
# AUGMENTATION
# ============================================================

def apply_training_augmentation(image):
    """
    Applica la strategia di augmentation prevista
    per il retraining robusto.

    Con probabilità 0.5 l'immagine viene mantenuta
    nella condizione originale.

    Con probabilità 0.5 viene scelta uniformemente
    una delle nove condizioni degradate.

    Parameters
    ----------
    image : PIL.Image
        Crop facciale originale.

    Returns
    -------
    image : PIL.Image
        Immagine originale oppure degradata.

    condition : str
        Condizione effettivamente applicata.
    """

    # 50% dei casi: nessuna degradazione
    if random.random() < 0.5:
        return image, "original"

    # 50% dei casi: una delle 9 degradazioni,
    # tutte con la stessa probabilità
    condition = random.choice(
        TRAIN_DEGRADATION_CONDITIONS
    )

    degraded_image = apply_degradation(
        image,
        condition,
    )

    return degraded_image, condition