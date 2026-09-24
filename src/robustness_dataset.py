"""
robustness_dataset.py

Dataset utilizzato per la valutazione della robustezza.

Le degradazioni sperimentali vengono applicate al crop facciale
prima del preprocessing specifico del modello.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from dataset import (
    DATASET_SPLITS_FILE,
    FACES_DIR,
    get_transforms,
)

from robustness_transforms import apply_degradation


class RobustnessDataset(Dataset):

    def __init__(
        self,
        split,
        model_name,
        condition="original",
        csv_path=DATASET_SPLITS_FILE,
        faces_dir=FACES_DIR,
    ):

        self.split = split
        self.model_name = model_name
        self.condition = condition

        self.csv_path = Path(csv_path)
        self.faces_dir = Path(faces_dir)

        # Lettura metadata
        metadata = pd.read_csv(
            self.csv_path
        )

        # Manteniamo solo lo split richiesto
        self.data = (
            metadata[
                metadata["split"] == split
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(self.data) == 0:
            raise ValueError(
                f"Nessun campione trovato "
                f"per lo split '{split}'."
            )

        # Preprocessing originale del modello
        self.transform = get_transforms(
            model_name
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        filename = row["filename"]

        image_path = (
            self.faces_dir
            / filename
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Immagine non trovata: "
                f"{image_path}"
            )

        # Apertura crop originale
        image = Image.open(
            image_path
        ).convert("RGB")

        # ====================================================
        # DEGRADAZIONE SPERIMENTALE
        # ====================================================

        image = apply_degradation(
            image,
            self.condition,
        )

        # ====================================================
        # PREPROCESSING BASELINE DEL MODELLO
        # ====================================================

        image = self.transform(
            image
        )

        label = int(
            row["label"]
        )

        return {
            "image": image,
            "label": label,
            "filename": filename,
            "manipulation": row["manipulation"],
            "source_video": row["source_video"],
            "split": row["split"],
            "condition": self.condition,
        }