"""
robust_dataloaders.py

DataLoader utilizzati per il retraining robusto.

- TRAIN:
    utilizza RobustTrainingDataset
    con data augmentation on-the-fly.

- VALIDATION:
    utilizza il dataset originale,
    senza degradazioni aggiuntive.

- TEST:
    utilizza il dataset originale,
    senza degradazioni aggiuntive.
"""

import torch
from torch.utils.data import DataLoader

from dataset import FaceForensicsDataset
from robust_training_dataset import RobustTrainingDataset


# ============================================================
# CONFIGURAZIONE
# ============================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0


# ============================================================
# CREAZIONE DEI DATASET
# ============================================================

def create_robust_datasets(model_name):
    """
    Crea i dataset utilizzati nel retraining robusto.

    Soltanto il training set utilizza data augmentation.
    Validation e test rimangono invariati.
    """

    # Training with augmentation
    train_dataset = RobustTrainingDataset(
        model_name=model_name,
    )

    # original Validation 
    val_dataset = FaceForensicsDataset(
        split="val",
        model_name=model_name,
    )

    # originale Test 
    test_dataset = FaceForensicsDataset(
        split="test",
        model_name=model_name,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# CREAZIONE DEI DATALOADER
# ============================================================

def create_robust_dataloaders(
    model_name,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """
    Crea i DataLoader per il retraining robusto.
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_robust_datasets(
        model_name=model_name
    )

    use_pin_memory = torch.cuda.is_available()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    #
    # shuffle=True come nella pipeline baseline.
    # A ogni caricamento RobustTrainingDataset può applicare
    # una nuova degradazione al campione.
    # --------------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------
    #
    # Nessuna augmentation.
    # Manteniamo il validation set nella configurazione
    # originale per monitorare la validation loss e
    # selezionare il checkpoint.
    # --------------------------------------------------------

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    #
    # Nessuna augmentation durante il training.
    # Verrà utilizzato successivamente per la valutazione.
    # --------------------------------------------------------

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )