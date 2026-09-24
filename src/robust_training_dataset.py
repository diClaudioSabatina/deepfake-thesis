"""
robust_training_dataset.py

Dataset utilizzato esclusivamente per il training
dei modelli robusti.

La degradazione viene applicata al crop facciale
prima del preprocessing specifico del modello.

Validation e test NON utilizzano questa classe.
"""

from PIL import Image

from dataset import (
    FaceForensicsDataset,
    DATASET_SPLITS_FILE,
    FACES_DIR,
)

from training_augmentation import (
    apply_training_augmentation,
)


class RobustTrainingDataset(FaceForensicsDataset):
    """
    Dataset di training con data augmentation on-the-fly.

    Utilizza esclusivamente lo split "train".
    """

    def __init__(
        self,
        model_name,
        csv_path=DATASET_SPLITS_FILE,
        faces_dir=FACES_DIR,
    ):
        super().__init__(
            split="train",
            model_name=model_name,
            csv_path=csv_path,
            faces_dir=faces_dir,
        )

    def __getitem__(self, index):
        """
        Carica un campione del training set,
        applica eventualmente una degradazione
        e successivamente il preprocessing del modello.
        """

        # Riga corrispondente al campione
        row = self.data.iloc[index]

        filename = row["filename"]

        image_path = (
            self.faces_dir
            / filename
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Immagine non trovata:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # 1. Apertura del crop facciale
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # 2. Data augmentation
        # ----------------------------------------------------
        #
        # 50% originale
        # 50% una delle 9 degradazioni
        #
        # IMPORTANTE:
        # avviene PRIMA del preprocessing del modello.
        # ----------------------------------------------------

        image, augmentation_condition = (
            apply_training_augmentation(image)
        )

        # ----------------------------------------------------
        # 3. Preprocessing specifico del modello
        # ----------------------------------------------------

        image = self.transform(
            image
        )

        # ----------------------------------------------------
        # 4. Label
        # ----------------------------------------------------

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # 5. Restituzione del campione
        # ----------------------------------------------------

        sample = {
            "image": image,
            "label": label,
            "filename": filename,
            "manipulation": row["manipulation"],
            "source_video": row["source_video"],
            "split": row["split"],

            # Ci serve per poter verificare quale
            # trasformazione è stata applicata.
            "augmentation_condition":
                augmentation_condition,
        }

        return sample