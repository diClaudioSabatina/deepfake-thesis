"""
dataloaders.py

Creazione dei DataLoader PyTorch per FaceForensics++.

Questo modulo utilizza FaceForensicsDataset definito in dataset.py
e costruisce i loader per:

- training
- validation
- test

Il DataLoader suddivide il dataset in batch e gestisce
il caricamento delle immagini durante training e valutazione.

Il bilanciamento delle classi NON viene gestito in questo file.
Lo affronteremo separatamente nella funzione di loss durante
l'addestramento.
"""

import torch
from torch.utils.data import DataLoader

from dataset import FaceForensicsDataset


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

# Numero di immagini elaborate contemporaneamente.
#
# 32 è un valore iniziale ragionevole.
# Quando utilizzeremo EfficientNet-B4 potremmo doverlo
# ridurre, perché il modello utilizza immagini più grandi
# e richiede più memoria GPU.
DEFAULT_BATCH_SIZE = 32


# Numero di processi utilizzati dal DataLoader
# per caricare le immagini in parallelo.
#
# In locale su Windows partiamo prudentemente da 0.
#
# Quando eseguiremo il training su Colab potremo
# aumentarlo, ad esempio a 2 o 4, dopo aver verificato
# la stabilità della pipeline.
DEFAULT_NUM_WORKERS = 0


# ============================================================
# 2. CREAZIONE DEI DATASET
# ============================================================

def create_datasets(model_name):
    """
    Crea i dataset di training, validation e test.

    Parameters
    ----------
    model_name : str
        Modello per il quale devono essere preparate
        le immagini.

        Valori supportati:
        - "xception"
        - "efficientnet_b4"

    Returns
    -------
    tuple
        train_dataset, val_dataset, test_dataset
    """

    train_dataset = FaceForensicsDataset(
        split="train",
        model_name=model_name,
    )

    val_dataset = FaceForensicsDataset(
        split="val",
        model_name=model_name,
    )

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
# 3. CREAZIONE DEI DATALOADER
# ============================================================

def create_dataloaders(
    model_name,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """
    Crea i DataLoader per training, validation e test.

    Parameters
    ----------
    model_name : str
        Nome del modello:
        "xception" oppure "efficientnet_b4".

    batch_size : int
        Numero di immagini presenti in ogni batch.

    num_workers : int
        Numero di processi utilizzati per caricare
        i dati in parallelo.

    Returns
    -------
    tuple
        train_loader, val_loader, test_loader
    """

    # --------------------------------------------------------
    # Creazione dei tre Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(
        model_name=model_name
    )

    # --------------------------------------------------------
    # Verifica disponibilità CUDA
    # --------------------------------------------------------
    #
    # Se il training viene eseguito su GPU NVIDIA/Colab,
    # possiamo utilizzare pinned memory per facilitare
    # il trasferimento dei tensor dalla RAM alla GPU.
    #
    # Sul PC locale, dove CUDA non è disponibile,
    # questo valore sarà False.

    use_pin_memory = torch.cuda.is_available()

    # ========================================================
    # TRAIN DATALOADER
    # ========================================================
    #
    # shuffle=True:
    #
    # durante il training vogliamo che l'ordine
    # delle immagini venga rimescolato a ogni epoca.
    #
    # In questo modo il modello non osserva sempre
    # i campioni nello stesso ordine.

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # ========================================================
    # VALIDATION DATALOADER
    # ========================================================
    #
    # shuffle=False:
    #
    # durante validation non dobbiamo modificare
    # l'ordine dei dati, perché non stiamo
    # aggiornando i pesi del modello.

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

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


# ============================================================
# 4. TEST DEL MODULO
# ============================================================

def main():
    """
    Verifica che i DataLoader funzionino correttamente.

    Non viene eseguito alcun training.
    Viene semplicemente caricato un batch dal training set.
    """

    print("=" * 60)
    print("TEST DATALOADER FACEFORENSICS++")
    print("=" * 60)

    # Per ora testiamo la configurazione Xception.
    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        model_name="xception",
        batch_size=32,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Numero di immagini
    # --------------------------------------------------------

    print(
        f"\nCampioni train: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Campioni validation: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Campioni test: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Numero di batch
    # --------------------------------------------------------

    print(
        f"\nBatch train: "
        f"{len(train_loader)}"
    )

    print(
        f"Batch validation: "
        f"{len(val_loader)}"
    )

    print(
        f"Batch test: "
        f"{len(test_loader)}"
    )

    # ========================================================
    # Recupero di un singolo batch
    # ========================================================

    batch = next(
        iter(train_loader)
    )

    images = batch["image"]
    labels = batch["label"]

    print(
        f"\nDimensione batch immagini: "
        f"{images.shape}"
    )

    print(
        f"Dimensione batch label: "
        f"{labels.shape}"
    )

    print(
        f"Tipo immagini: "
        f"{images.dtype}"
    )

    print(
        f"Tipo label: "
        f"{labels.dtype}"
    )

    # --------------------------------------------------------
    # Controllo delle label presenti nel batch
    # --------------------------------------------------------

    print(
        f"\nLabel nel primo batch: "
        f"{labels.tolist()}"
    )

    # --------------------------------------------------------
    # CUDA / pinned memory
    # --------------------------------------------------------

    print(
        f"\nCUDA disponibile: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Pin memory utilizzata: "
        f"{torch.cuda.is_available()}"
    )


# ============================================================
# 5. AVVIO DEL TEST
# ============================================================

if __name__ == "__main__":
    main()