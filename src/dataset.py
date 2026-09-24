"""
dataset.py

Gestione del dataset di immagini facciali ottenute da FaceForensics++.

Questo modulo ha il compito di:
- leggere il file dataset_splits.csv;
- selezionare train, validation oppure test;
- caricare il crop facciale corrispondente;
- applicare il preprocessing richiesto dal modello;
- restituire a PyTorch immagine, label e metadati utili.

Il file NON esegue:
- training;
- bilanciamento delle classi;
- degradazioni JPEG, blur o resize sperimentali;
- classificazione.

Queste operazioni verranno gestite in moduli separati.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

# dataset.py si trova in:
#
# deepfake-thesis/src/dataset.py
#
# parents[1] permette di risalire automaticamente
# alla root del progetto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Cartella contenente i crop facciali validi.
FACES_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "faces"
)


# File prodotto da create_splits.py.
#
# Contiene:
# - filename
# - label
# - manipulation
# - source_video
# - split
# - altri metadati
DATASET_SPLITS_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "dataset_splits.csv"
)


# ============================================================
# 2. DIMENSIONI DI INPUT DEI MODELLI
# ============================================================

# Xception viene normalmente utilizzato con input 299x299.
XCEPTION_IMAGE_SIZE = 299


# EfficientNet-B4 utilizza una risoluzione maggiore.
EFFICIENTNET_B4_IMAGE_SIZE = 380
EFFICIENTNET_B4_RESIZE_SIZE = 384


# ============================================================
# 3. NORMALIZZAZIONE
# ============================================================

# Per ora utilizziamo la normalizzazione standard ImageNet.
#
# I modelli verranno inizializzati con pesi pre-addestrati
# su ImageNet, quindi le immagini devono essere convertite
# in un intervallo compatibile con tale pretraining.
#
# Questi valori rappresentano media e deviazione standard
# dei tre canali RGB utilizzate comunemente nei modelli
# pre-addestrati PyTorch.

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# 4. CREAZIONE DEL PREPROCESSING
# ============================================================

def get_transforms(model_name):
    """
    Restituisce il preprocessing appropriato per il modello.

    Parameters
    ----------
    model_name : str
        Nome del modello.

        Valori supportati:
        - "xception"
        - "efficientnet_b4"

    Returns
    -------
    torchvision.transforms.Compose
        Sequenza di trasformazioni da applicare
        all'immagine prima di fornirla alla rete.
    """

    model_name = model_name.lower()

    # --------------------------------------------------------
    # XCEPTION
    # --------------------------------------------------------

    if model_name == "xception":

        transform = transforms.Compose(
            [
                # I pesi legacy_xception utilizzati da timm
                # prevedono un resize a circa 333 pixel.
                transforms.Resize(
                    333, 
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),

                # Successivamente viene effettuato
                # un crop centrale a 299x299.
                transforms.CenterCrop(
                    XCEPTION_IMAGE_SIZE
                ),

                # Conversione dell'immagine PIL
                # in tensor PyTorch nell'intervallo [0, 1].
                transforms.ToTensor(),

                # Normalizzazione prevista dai pesi
                # pre-addestrati legacy_xception.
                #
                # Questa trasformazione porta sostanzialmente
                # i valori dei pixel nell'intervallo [-1, 1].
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        return transform

    # --------------------------------------------------------
    # EFFICIENTNET-B4
    # --------------------------------------------------------

    elif model_name == "efficientnet_b4":

        image_size = EFFICIENTNET_B4_IMAGE_SIZE

    else:

        raise ValueError(
            f"Modello non supportato: {model_name}\n"
            f"Utilizzare 'xception' oppure "
            f"'efficientnet_b4'."
        )

    # --------------------------------------------------------
    # Baseline preprocessing
    # --------------------------------------------------------
    #
    # In questa prima fase NON introduciamo:
    #
    # - JPEG compression
    # - blur
    # - resize casuali
    # - degradazioni
    #
    # perché vogliamo misurare inizialmente
    # le prestazioni del modello baseline.
    #
    # Le degradazioni verranno aggiunte
    # successivamente in una pipeline separata.

    transform = transforms.Compose(
    [
        # Ridimensiona il lato più corto a 384 pixel
        # mantenendo le proporzioni dell'immagine.
        transforms.Resize(
            EFFICIENTNET_B4_RESIZE_SIZE,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),

        # Estrae la regione centrale 380x380,
        # dimensione richiesta da EfficientNet-B4.
        transforms.CenterCrop(
            image_size
        ),

        # Converte l'immagine PIL in tensor PyTorch.
        transforms.ToTensor(),

        # Normalizzazione prevista dai pesi ImageNet
        # utilizzati da EfficientNet-B4.
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)
    return transform


# ============================================================
# 5. DATASET PYTORCH
# ============================================================

class FaceForensicsDataset(Dataset):
    """
    Dataset PyTorch per i crop facciali di FaceForensics++.

    Ogni elemento restituito contiene:
    - immagine preprocessata;
    - label real/fake;
    - nome del file;
    - tecnica di manipolazione;
    - video di origine;
    - split.

    La presenza dei metadati sarà utile anche nelle
    successive analisi sperimentali.
    """

    def __init__(
        self,
        split,
        model_name,
        csv_path=DATASET_SPLITS_FILE,
        faces_dir=FACES_DIR,
    ):
        """
        Parameters
        ----------
        split : str
            Split da utilizzare.

            Valori ammessi:
            - "train"
            - "val"
            - "test"

        model_name : str
            Modello per il quale preparare le immagini.

            Valori ammessi:
            - "xception"
            - "efficientnet_b4"

        csv_path : Path, optional
            Percorso del file dataset_splits.csv.

        faces_dir : Path, optional
            Cartella contenente i crop facciali.
        """

        # ----------------------------------------------------
        # Controllo dello split
        # ----------------------------------------------------

        valid_splits = {
            "train",
            "val",
            "test",
        }

        if split not in valid_splits:

            raise ValueError(
                f"Split non valido: {split}. "
                f"Utilizzare train, val oppure test."
            )

        self.split = split
        self.model_name = model_name

        self.csv_path = Path(csv_path)
        self.faces_dir = Path(faces_dir)

        # ----------------------------------------------------
        # Controllo dei percorsi
        # ----------------------------------------------------

        if not self.csv_path.exists():

            raise FileNotFoundError(
                f"File dataset_splits.csv "
                f"non trovato:\n{self.csv_path}"
            )

        if not self.faces_dir.exists():

            raise FileNotFoundError(
                f"Cartella faces non trovata:\n"
                f"{self.faces_dir}"
            )

        # ----------------------------------------------------
        # Lettura dei metadati
        # ----------------------------------------------------

        metadata = pd.read_csv(
            self.csv_path
        )

        # Manteniamo soltanto i campioni
        # appartenenti allo split richiesto.
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

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        self.transform = get_transforms(
            model_name
        )

    # ========================================================
    # NUMERO DI ELEMENTI DEL DATASET
    # ========================================================

    def __len__(self):
        """
        Restituisce il numero di immagini
        presenti nello split.
        """

        return len(self.data)

    # ========================================================
    # LETTURA DI UN ELEMENTO
    # ========================================================

    def __getitem__(self, index):
        """
        Carica e restituisce un singolo campione.

        Parameters
        ----------
        index : int
            Posizione dell'immagine nel dataset.

        Returns
        -------
        dict
            Dizionario contenente immagine,
            label e metadati.
        """

        # Recuperiamo la riga del CSV.
        row = self.data.iloc[index]

        filename = row["filename"]

        image_path = (
            self.faces_dir
            / filename
        )

        # ----------------------------------------------------
        # Controllo del file
        # ----------------------------------------------------

        if not image_path.exists():

            raise FileNotFoundError(
                f"Immagine non trovata:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Apertura immagine
        # ----------------------------------------------------
        #
        # convert("RGB") garantisce che ogni immagine
        # abbia sempre esattamente tre canali,
        # indipendentemente dal formato originale.

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Applicazione preprocessing
        # ----------------------------------------------------

        image = self.transform(
            image
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------
        #
        # Convenzione:
        #
        # 0 = REAL
        # 1 = FAKE

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Restituzione del campione
        # ----------------------------------------------------
        #
        # Non restituiamo soltanto image e label.
        #
        # Conserviamo anche alcuni metadati perché
        # saranno utili successivamente per:
        #
        # - analisi per tecnica;
        # - studio degli errori;
        # - selective classification;
        # - produzione del report tecnico.

        sample = {

            "image": image,

            "label": label,

            "filename": filename,

            "manipulation": row["manipulation"],

            "source_video": row["source_video"],

            "split": row["split"],
        }

        return sample


# ============================================================
# 6. TEST DEL MODULO
# ============================================================

def main():
    """
    Piccolo test per verificare che il Dataset
    funzioni correttamente.

    Non esegue nessun training.
    """

    print("=" * 60)
    print("TEST DATASET FACEFORENSICS++")
    print("=" * 60)

    # Per il test utilizziamo Xception.
    dataset = FaceForensicsDataset(
        split="train",
        model_name="xception",
    )

    print(
        f"\nNumero campioni train: "
        f"{len(dataset)}"
    )

    # Recuperiamo il primo elemento.
    sample = dataset[0]

    print(
        f"\nFilename: "
        f"{sample['filename']}"
    )

    print(
        f"Label: "
        f"{sample['label']}"
    )

    print(
        f"Manipolazione: "
        f"{sample['manipulation']}"
    )

    print(
        f"Video origine: "
        f"{sample['source_video']}"
    )

    print(
        f"Split: "
        f"{sample['split']}"
    )

    print(
        f"Dimensione tensor immagine: "
        f"{sample['image'].shape}"
    )

    print(
        f"Tipo tensor: "
        f"{sample['image'].dtype}"
    )


# ============================================================
# 7. AVVIO DEL TEST
# ============================================================

if __name__ == "__main__":
    main()