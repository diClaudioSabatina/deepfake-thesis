"""
test_robust_training_pipeline.py

Controllo della pipeline di training robusto.

Verifica che:
- il train utilizzi l'augmentation;
- validation e test rimangano invariati;
- le dimensioni dei dataset non cambino;
- il preprocessing produca tensor della dimensione corretta.
"""

import random
import torch

from robust_dataloaders import create_robust_datasets


def main():

    # Rendiamo il controllo riproducibile.
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("TEST PIPELINE ROBUST TRAINING")
    print("=" * 70)

    # ========================================================
    # 1. CREAZIONE DATASET
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_robust_datasets(
        model_name="xception"
    )

    print("\nDIMENSIONI DATASET")

    print(
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Validation: {len(val_dataset)}"
    )

    print(
        f"Test: {len(test_dataset)}"
    )

    # ========================================================
    # 2. CONTROLLO TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAIN - AUGMENTATION")
    print("=" * 70)

    # Carichiamo più volte lo stesso campione.
    # Poiché l'augmentation è on-the-fly,
    # la condizione può cambiare a ogni accesso.

    conditions = []

    for i in range(20):

        sample = train_dataset[0]

        conditions.append(
            sample["augmentation_condition"]
        )

        print(
            f"{i + 1:02d} -> "
            f"{sample['augmentation_condition']}"
        )

    print(
        "\nDimensione tensor train:",
        sample["image"].shape,
    )

    # ========================================================
    # 3. CONTROLLO VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION - NESSUNA AUGMENTATION")
    print("=" * 70)

    val_sample_1 = val_dataset[0]
    val_sample_2 = val_dataset[0]

    print(
        "augmentation_condition presente:",
        "augmentation_condition"
        in val_sample_1,
    )

    print(
        "Immagine identica in due caricamenti:",
        torch.equal(
            val_sample_1["image"],
            val_sample_2["image"],
        ),
    )

    print(
        "Dimensione tensor validation:",
        val_sample_1["image"].shape,
    )

    # ========================================================
    # 4. CONTROLLO TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST - NESSUNA AUGMENTATION")
    print("=" * 70)

    test_sample_1 = test_dataset[0]
    test_sample_2 = test_dataset[0]

    print(
        "augmentation_condition presente:",
        "augmentation_condition"
        in test_sample_1,
    )

    print(
        "Immagine identica in due caricamenti:",
        torch.equal(
            test_sample_1["image"],
            test_sample_2["image"],
        ),
    )

    print(
        "Dimensione tensor test:",
        test_sample_1["image"].shape,
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETATO")
    print("=" * 70)


if __name__ == "__main__":
    main()