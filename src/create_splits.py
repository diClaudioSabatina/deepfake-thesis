"""
create_splits.py

Creazione degli split train / validation / test
per il dataset preprocessed di FaceForensics++.

Lo script:
- legge i metadati dei frame;
- legge i metadati della face detection;
- conserva solo i frame con volto rilevato;
- legge gli split ufficiali di FaceForensics++;
- assegna ogni video a train, validation o test;
- salva un nuovo CSV con la colonna "split".

Gli split vengono assegnati a livello di VIDEO,
non di singolo frame, per evitare data leakage.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAMES_METADATA_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "frames_metadata.csv"
)

FACES_METADATA_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "faces_metadata.csv"
)

OFFICIAL_SPLITS_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "official_splits"
)

TRAIN_SPLIT_FILE = OFFICIAL_SPLITS_DIR / "train.json"
VAL_SPLIT_FILE = OFFICIAL_SPLITS_DIR / "val.json"
TEST_SPLIT_FILE = OFFICIAL_SPLITS_DIR / "test.json"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "dataset_splits.csv"
)


# ============================================================
# 2. LETTURA DEGLI SPLIT UFFICIALI
# ============================================================

def load_split_pairs(json_path):
    """
    Legge un file JSON dello split ufficiale di FaceForensics++.

    Ogni elemento contiene una coppia di identificativi,
    ad esempio:

        ["071", "054"]

    Questa coppia corrisponde a una sequenza manipolata
    come:

        071_054.mp4

    Returns
    -------
    list[tuple[str, str]]
        Lista di coppie di identificativi.
    """

    with open(json_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    return [
        (str(pair[0]).zfill(3), str(pair[1]).zfill(3))
        for pair in split_data
    ]


# ============================================================
# 3. CREAZIONE DELLE MAPPE DI APPARTENENZA
# ============================================================

def build_split_mapping():
    """
    Costruisce due strutture:

    1. pair_to_split
       associa una coppia di ID allo split corretto.

       Esempio:
           ("071", "054") -> "train"

    2. original_id_to_split
       associa ciascun ID originale allo split.

       Esempio:
           "071" -> "train"

    Questo ci permette di gestire sia:
    - video originali, es. 071.mp4
    - video manipolati, es. 071_054.mp4
    """

    split_files = {
        "train": TRAIN_SPLIT_FILE,
        "val": VAL_SPLIT_FILE,
        "test": TEST_SPLIT_FILE,
    }

    pair_to_split = {}
    original_id_to_split = {}

    for split_name, split_file in split_files.items():

        pairs = load_split_pairs(split_file)

        for first_id, second_id in pairs:

            # Registriamo entrambe le direzioni.
            #
            # Questo perché potrebbero esistere file:
            # 071_054.mp4
            # oppure
            # 054_071.mp4
            pair_to_split[(first_id, second_id)] = split_name
            pair_to_split[(second_id, first_id)] = split_name

            # Entrambi gli ID originali appartengono
            # allo stesso split della coppia.
            for original_id in (first_id, second_id):

                if (
                    original_id in original_id_to_split
                    and
                    original_id_to_split[original_id] != split_name
                ):
                    raise ValueError(
                        f"L'ID originale {original_id} compare "
                        f"in split differenti."
                    )

                original_id_to_split[original_id] = split_name

    return pair_to_split, original_id_to_split


# ============================================================
# 4. ASSEGNAZIONE DELLO SPLIT A UN VIDEO
# ============================================================

def assign_split(
    source_video,
    pair_to_split,
    original_id_to_split,
):
    """
    Determina lo split di appartenenza di un video.

    Esempi
    -------
    Originale:
        071.mp4
        -> usa l'ID "071"

    Manipolato:
        071_054.mp4
        -> usa la coppia ("071", "054")

    Returns
    -------
    str | None
        "train", "val", "test"
        oppure None se il video non è presente negli split.
    """

    # Eliminiamo l'estensione .mp4
    video_id = Path(source_video).stem

    # --------------------------------------------------------
    # Caso 1: video originale
    # --------------------------------------------------------

    if "_" not in video_id:

        original_id = video_id.zfill(3)

        return original_id_to_split.get(
            original_id
        )

    # --------------------------------------------------------
    # Caso 2: video manipolato
    # --------------------------------------------------------

    parts = video_id.split("_")

    # Per il nucleo classico di FaceForensics++
    # ci aspettiamo nomi come 071_054.
    if len(parts) != 2:

        return None

    first_id = parts[0].zfill(3)
    second_id = parts[1].zfill(3)

    return pair_to_split.get(
        (first_id, second_id)
    )


# ============================================================
# 5. FUNZIONE PRINCIPALE
# ============================================================

def main():

    print("=" * 60)
    print("FaceForensics++ - Creazione split ufficiali")
    print("=" * 60)

    # --------------------------------------------------------
    # Controllo dei file necessari
    # --------------------------------------------------------

    required_files = [
        FRAMES_METADATA_FILE,
        FACES_METADATA_FILE,
        TRAIN_SPLIT_FILE,
        VAL_SPLIT_FILE,
        TEST_SPLIT_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"File richiesto non trovato:\n"
                f"{file_path}"
            )

    # --------------------------------------------------------
    # Lettura dei metadati
    # --------------------------------------------------------

    frames_metadata = pd.read_csv(
        FRAMES_METADATA_FILE
    )

    faces_metadata = pd.read_csv(
        FACES_METADATA_FILE
    )

    print(
        f"\nFrame presenti nei metadati: "
        f"{len(frames_metadata)}"
    )

    print(
        f"Record di face detection: "
        f"{len(faces_metadata)}"
    )

    # --------------------------------------------------------
    # Merge dei metadati
    # --------------------------------------------------------
    #
    # Manteniamo soltanto le colonne della fase
    # di face detection che ci servono in questa fase.

    dataset = frames_metadata.merge(
        faces_metadata[
            [
                "filename",
                "face_detected",
                "detection_score",
            ]
        ],
        on="filename",
        how="inner"
    )

    # --------------------------------------------------------
    # Conserviamo solo i crop validi
    # --------------------------------------------------------

    dataset = dataset[
        dataset["face_detected"] == 1
    ].copy()

    print(
        f"\nFrame con volto rilevato: "
        f"{len(dataset)}"
    )

    # ========================================================
    # Costruzione delle mappe degli split
    # ========================================================

    (
        pair_to_split,
        original_id_to_split,
    ) = build_split_mapping()

    # ========================================================
    # Assegnazione degli split
    # ========================================================

    dataset["split"] = dataset[
        "source_video"
    ].apply(
        lambda video: assign_split(
            video,
            pair_to_split,
            original_id_to_split,
        )
    )

    # --------------------------------------------------------
    # Controllo dei video senza split
    # --------------------------------------------------------

    missing_split = dataset[
        dataset["split"].isna()
    ]

    print(
        f"\nFrame senza split assegnato: "
        f"{len(missing_split)}"
    )

    if len(missing_split) > 0:

        print(
            "\nEsempi di video senza split:"
        )

        print(
            missing_split[
                "source_video"
            ]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Manteniamo solo i record correttamente assegnati
    # --------------------------------------------------------

    dataset_with_split = dataset[
        dataset["split"].notna()
    ].copy()

    # ========================================================
    # Salvataggio CSV finale
    # ========================================================

    dataset_with_split.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print("\n" + "=" * 60)
    print("SPLIT COMPLETATI")
    print("=" * 60)

    print(
        f"\nRecord finali: "
        f"{len(dataset_with_split)}"
    )

    print("\nDistribuzione per split:")

    print(
        dataset_with_split[
            "split"
        ].value_counts()
    )

    print(
        "\nDistribuzione split × classe:"
    )

    print(
        pd.crosstab(
            dataset_with_split["split"],
            dataset_with_split["label"]
        )
    )

    print(
        "\nDistribuzione split × tecnica:"
    )

    print(
        pd.crosstab(
            dataset_with_split["split"],
            dataset_with_split["manipulation"]
        )
    )

    print(
        f"\nFile salvato in:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# 6. AVVIO DEL PROGRAMMA
# ============================================================

if __name__ == "__main__":
    main()