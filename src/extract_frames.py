"""
extract_frames.py

Estrazione dei frame dal dataset FaceForensics++ C23.

Obiettivo:
- leggere i video originali e manipolati;
- estrarre 10 frame distribuiti lungo ciascun video;
- salvare i frame in formato PNG (lossless);
- creare un file CSV contenente i metadati di ogni frame.

Il codice NON effettua ancora:
- face detection;
- crop del volto;
- resize;
- train/validation/test split;
- degradazioni JPEG, blur o resize.

Queste operazioni verranno gestite separatamente nelle fasi successive
della pipeline sperimentale.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

# Numero di frame da estrarre da ciascun video.
NUM_FRAMES_PER_VIDEO = 10


# Il file extract_frames.py si trova dentro:
#
# deepfake-thesis/src/extract_frames.py
#
# parents[1] permette quindi di risalire automaticamente
# alla cartella principale "deepfake-thesis".
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Percorso del dataset originale.
#
# La struttura attesa è:
#
# data_raw/
# └── archive/
#     └── FaceForensics++_C23/
#         ├── original/
#         ├── Deepfakes/
#         ├── Face2Face/
#         ├── FaceSwap/
#         └── NeuralTextures/
DATASET_ROOT = (
    PROJECT_ROOT
    / "data_raw"
    / "archive"
    / "FaceForensics++_C23"
)


# Cartella nella quale verranno salvati i frame estratti.
FRAMES_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "frames"
)


# File CSV contenente le informazioni associate a ogni frame.
METADATA_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "frames_metadata.csv"
)


# ============================================================
# 2. CARTELLE DEL DATASET DA UTILIZZARE
# ============================================================

# Per ciascuna categoria specifichiamo:
#
# path:
#   percorso della cartella contenente i video della categoria.
#
# label:
#   0 = immagine reale
#   1 = immagine manipolata
#
# manipulation:
#   tecnica con cui è stato prodotto il video.
#
# La classificazione finale rimane BINARIA (real/fake).
# Il tipo di manipolazione viene conservato solo come metadato,
# in modo da poter analizzare separatamente le prestazioni
# del detector sulle diverse tecniche di manipolazione.

DATASET_CATEGORIES = {

    "original": {
        "path": DATASET_ROOT / "original",
        "label": 0,
        "manipulation": "original",
    },

    "Deepfakes": {
        "path": DATASET_ROOT / "Deepfakes",
        "label": 1,
        "manipulation": "Deepfakes",
    },

    "Face2Face": {
        "path": DATASET_ROOT / "Face2Face",
        "label": 1,
        "manipulation": "Face2Face",
    },

    "FaceSwap": {
        "path": DATASET_ROOT / "FaceSwap",
        "label": 1,
        "manipulation": "FaceSwap",
    },

    "NeuralTextures": {
        "path": DATASET_ROOT / "NeuralTextures",
        "label": 1,
        "manipulation": "NeuralTextures",
    },
}


# ============================================================
# 3. FUNZIONE PER CALCOLARE QUALI FRAME ESTRARRE
# ============================================================

def select_frame_indices(total_frames, num_frames):
    """
    Restituisce gli indici dei frame da estrarre da un video.

    Strategia:
    il video viene suddiviso idealmente in 'num_frames'
    intervalli temporali della stessa dimensione.

    Da ciascun intervallo viene selezionato il frame centrale.

    Esempio:
        video con 400 frame
        num_frames = 10

    Il video viene diviso in circa 10 sezioni da 40 frame
    e viene selezionato un frame rappresentativo da ciascuna sezione.

    Questo evita di:
    - prendere frame consecutivi molto simili;
    - concentrarsi soltanto sull'inizio del video;
    - utilizzare esclusivamente il primo e l'ultimo frame.

    Parameters
    ----------
    total_frames : int
        Numero totale di frame presenti nel video.

    num_frames : int
        Numero di frame che vogliamo estrarre.

    Returns
    -------
    numpy.ndarray
        Array contenente gli indici dei frame selezionati.
    """

    # Se il video non contiene frame validi,
    # restituiamo un array vuoto.
    if total_frames <= 0:
        return np.array([], dtype=int)

    # If the video contains fewer frames than the number
    # we would like to extract, we cannot extract 10 distinct ones.
    # In this case, we use all the available frames.
    actual_num_frames = min(num_frames, total_frames)
    
    # Creating the endpoints of the time intervals.
    # For example, with 400 frames and 10 intervals:
    # 0, 40, 80, 120, …, 400
    interval_edges = np.linspace(
        0,
        total_frames,
        actual_num_frames + 1
    )
    # working out the midpoint of each interval.
    frame_indices = []
    for i in range(actual_num_frames):
        start = interval_edges[i]
        end = interval_edges[i + 1]

        # central frame of the interval
        center = int((start + end) / 2)

        # avoiding that, as a result of rounding,
        # an index equal to `total_frames` is not generated,
        # as this would be outside the valid range.
        center = min(center, total_frames - 1)

        frame_indices.append(center)

    return np.array(frame_indices, dtype=int)


# ============================================================
# 4. FUNZIONE PER ESTRARRE I FRAME DA UN SINGOLO VIDEO
# ============================================================

def extract_frames_from_video(
    video_path,
    output_dir,
    label,
    manipulation,
    num_frames=NUM_FRAMES_PER_VIDEO,
):
    """
    Estrae i frame selezionati da un singolo video.

    Parameters
    ----------
    video_path : Path
        Percorso del video .mp4.

    output_dir : Path
        Cartella nella quale salvare i frame.

    label : int
        0 = real
        1 = fake

    manipulation : str
        Tecnica di manipolazione.
        Per i video autentici sarà "original".

    num_frames : int
        Numero di frame da estrarre.

    Returns
    -------
    list
        Lista di dizionari contenenti i metadati
        dei frame estratti.
    """

    metadata_records = []

    # Apertura del video attraverso OpenCV.
    cap = cv2.VideoCapture(str(video_path))

    # Verifichiamo che OpenCV sia riuscito ad aprire il file.
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire: {video_path}")
        return metadata_records

    # Numero totale di frame presenti nel video.
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Frame per secondo del video.
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Selezioniamo le posizioni temporali da cui
    # estrarre i 10 frame.
    frame_indices = select_frame_indices(
        total_frames,
        num_frames
    )

    # Nome del video senza estensione.
    #
    # Esempio:
    # 000.mp4     -> 000
    # 000_003.mp4 -> 000_003
    video_id = video_path.stem

    # ========================================================
    # Estrazione dei singoli frame
    # ========================================================

    for sample_number, frame_index in enumerate(frame_indices):

        # Spostiamo il cursore del video direttamente
        # alla posizione del frame desiderato.
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        # Leggiamo il frame.
        success, frame = cap.read()

        # Se OpenCV non riesce a leggere il frame,
        # segnaliamo il problema e continuiamo con il successivo.
        if not success:
            print(
                f"[ATTENZIONE] Frame {frame_index} "
                f"non leggibile in {video_path.name}"
            )
            continue

        # Nome univoco del frame.
        #
        # Inseriamo:
        # - tecnica;
        # - video di origine;
        # - numero progressivo del frame campionato.
        #
        # Esempio:
        #
        # original_000_frame_00.png
        #
        # Deepfakes_000_003_frame_04.png

        frame_filename = (
            f"{manipulation}_"
            f"{video_id}_"
            f"frame_{sample_number:02d}.png"
        )

        frame_path = output_dir / frame_filename

        # ----------------------------------------------------
        # Salvataggio in PNG
        # ----------------------------------------------------
        #
        # PNG utilizza una compressione LOSSLESS.
        #
        # Questa scelta è particolarmente importante per
        # il nostro progetto perché successivamente studieremo
        # gli effetti della compressione JPEG.
        #
        # Salvare ora in JPEG introdurrebbe invece una
        # compressione aggiuntiva non controllata.

        saved = cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )

        if not saved:
            print(
                f"[ERRORE] Impossibile salvare: "
                f"{frame_path}"
            )
            continue

        # ----------------------------------------------------
        # Calcolo della posizione temporale
        # ----------------------------------------------------
        #
        # Serve per sapere approssimativamente in quale
        # secondo del video è stato estratto il frame.

        if fps > 0:
            timestamp_seconds = frame_index / fps
        else:
            timestamp_seconds = None

        # ----------------------------------------------------
        # Metadati del frame
        # ----------------------------------------------------

        metadata_records.append({

            # Nome dell'immagine prodotta.
            "filename": frame_filename,

            # Percorso relativo rispetto alla root del progetto.
            "frame_path": str(
                frame_path.relative_to(PROJECT_ROOT)
            ),

            # Ground truth:
            # 0 = real
            # 1 = fake
            "label": label,

            # Tecnica di manipolazione.
            "manipulation": manipulation,

            # Video dal quale proviene il frame.
            "source_video": video_path.name,

            # Posizione reale del frame nel video.
            "frame_index": int(frame_index),

            # Numero progressivo tra i 10 frame selezionati.
            "sample_number": sample_number,

            # Tempo approssimativo in secondi.
            "timestamp_seconds": timestamp_seconds,

            # Numero totale di frame del video.
            "total_video_frames": total_frames,

            # FPS del video.
            "fps": fps,
        })

    # Chiudiamo correttamente il file video.
    cap.release()

    return metadata_records


# ============================================================
# 5. FUNZIONE PRINCIPALE
# ============================================================

def main():
    """
    Funzione principale dello script.

    Scorre tutte le categorie selezionate di FaceForensics++,
    estrae i frame e genera il file CSV finale.
    """

    print("=" * 60)
    print("FaceForensics++ - Estrazione frame")
    print("=" * 60)

    print(f"\nDataset sorgente:")
    print(DATASET_ROOT)

    print(f"\nFrame destinazione:")
    print(FRAMES_DIR)

    print(
        f"\nNumero frame per video: "
        f"{NUM_FRAMES_PER_VIDEO}"
    )

    # --------------------------------------------------------
    # Controllo del dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "\nDataset non trovato.\n"
            f"Percorso atteso:\n{DATASET_ROOT}"
        )

    # Creiamo automaticamente le cartelle di output
    # se non dovessero già esistere.
    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lista che conterrà i metadati
    # di TUTTI i frame estratti.
    all_metadata = []

    # Contatori utili per il riepilogo finale.
    total_videos_processed = 0
    total_frames_extracted = 0

    # ========================================================
    # Elaborazione delle categorie
    # ========================================================

    for folder_name, info in DATASET_CATEGORIES.items():

        category_dir = info["path"]

        print("\n" + "-" * 60)
        print(f"Categoria: {folder_name}")
        print("-" * 60)

        # Verifichiamo che la cartella esista.
        if not category_dir.exists():

            print(
                f"[ATTENZIONE] Cartella non trovata: "
                f"{category_dir}"
            )

            continue

        # Recuperiamo tutti i video MP4.
        #
        # sorted() garantisce un ordine deterministico:
        # ad ogni esecuzione i video verranno processati
        # nello stesso ordine.
        video_files = sorted(
            category_dir.glob("*.mp4")
        )

        print(
            f"Video trovati: {len(video_files)}"
        )

        # ====================================================
        # Elaborazione dei singoli video
        # ====================================================

        for video_number, video_path in enumerate(
            video_files,
            start=1
        ):

            # Stampiamo periodicamente lo stato
            # dell'elaborazione.
            if (
                video_number == 1
                or video_number % 50 == 0
                or video_number == len(video_files)
            ):
                print(
                    f"  Elaborazione video "
                    f"{video_number}/{len(video_files)}"
                )

            records = extract_frames_from_video(

                video_path=video_path,

                output_dir=FRAMES_DIR,

                label=info["label"],

                manipulation=info["manipulation"],

                num_frames=NUM_FRAMES_PER_VIDEO,
            )

            all_metadata.extend(records)

            total_videos_processed += 1

            total_frames_extracted += len(records)

    # ========================================================
    # 6. CREAZIONE DEL FILE CSV
    # ========================================================

    if not all_metadata:

        print(
            "\n[ERRORE] Nessun frame è stato estratto."
        )

        return

    metadata_df = pd.DataFrame(all_metadata)

    # Ordiniamo il CSV per rendere il contenuto
    # più facile da leggere e riproducibile.
    metadata_df = metadata_df.sort_values(
        by=[
            "label",
            "manipulation",
            "source_video",
            "sample_number",
        ]
    ).reset_index(drop=True)

    # Scriviamo il CSV.
    metadata_df.to_csv(
        METADATA_FILE,
        index=False
    )

    # ========================================================
    # 7. RIEPILOGO FINALE
    # ========================================================

    print("\n" + "=" * 60)
    print("ESTRAZIONE COMPLETATA")
    print("=" * 60)

    print(
        f"Video elaborati: "
        f"{total_videos_processed}"
    )

    print(
        f"Frame estratti: "
        f"{total_frames_extracted}"
    )

    print(
        f"\nFrame salvati in:\n"
        f"{FRAMES_DIR}"
    )

    print(
        f"\nMetadati salvati in:\n"
        f"{METADATA_FILE}"
    )

    print("\nDistribuzione delle classi:")

    print(
        metadata_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nDistribuzione per tecnica:")

    print(
        metadata_df["manipulation"]
        .value_counts()
    )


# ============================================================
# 8. AVVIO DEL PROGRAMMA
# ============================================================

# Questa condizione permette di eseguire main()
# soltanto quando lanciamo direttamente questo file:
#
# python src/extract_frames.py
#
# Se invece in futuro importeremo alcune funzioni di questo
# script da un altro file, main() non verrà eseguito
# automaticamente.

if __name__ == "__main__":
    main()