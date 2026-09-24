"""
crop_faces.py

Rilevamento e ritaglio dei volti dai frame estratti da FaceForensics++.

Obiettivo:
- leggere i frame presenti in data_processed/frames;
- rilevare i volti tramite YuNet;
- selezionare il volto principale;
- aggiungere un margine intorno al bounding box;
- salvare il crop in formato PNG;
- creare un file CSV con i metadati relativi al rilevamento.

Il codice NON modifica i frame originali.
I frame completi rimangono conservati in data_processed/frames.
"""

from pathlib import Path

import cv2
import pandas as pd
import random


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

# Cartella principale del progetto.
#
# crop_faces.py si trova in:
# deepfake-thesis/src/crop_faces.py
#
# parents[1] permette di risalire automaticamente
# alla cartella deepfake-thesis.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Cartella contenente i 50.000 frame completi.
FRAMES_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "frames"
)


# Cartella nella quale verranno salvati
# i volti ritagliati.
FACES_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "faces"
)


# Modello YuNet scaricato da OpenCV Zoo.
YUNET_MODEL_PATH = (
    PROJECT_ROOT
    / "face_detector"
    / "face_detection_yunet_2023mar.onnx"
)

FRAMES_METADATA_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "frames_metadata.csv"
)

# File CSV che conterrà i risultati
# della fase di face detection.
FACES_METADATA_FILE = (
    PROJECT_ROOT
    / "metadata"
    / "faces_metadata.csv"
)


# ============================================================
# 2. PARAMETRI DEL FACE DETECTOR
# ============================================================

# Confidence minima richiesta per considerare
# valido un volto rilevato.
#
# YuNet restituisce uno score compreso tra 0 e 1.
SCORE_THRESHOLD = 0.8


# Threshold utilizzata dalla Non-Maximum Suppression.
#
# Serve a eliminare bounding box duplicate o
# fortemente sovrapposte relative allo stesso volto.
NMS_THRESHOLD = 0.3


# Numero massimo di candidati considerati internamente
# prima della Non-Maximum Suppression.
TOP_K = 5000


# Margine aggiuntivo intorno al bounding box del volto.
#
# 0.20 significa che estendiamo il crop del 20%
# rispetto alla larghezza e all'altezza del bounding box.
#
# Questo permette di conservare anche zone vicine al volto,
# come contorni, mascella e parte dei capelli.
FACE_MARGIN = 0.20


# ============================================================
# 3. CREAZIONE DEL DETECTOR YUNET
# ============================================================

def create_face_detector():
    """
    Crea e restituisce il face detector YuNet.

    La dimensione iniziale dell'input viene impostata
    temporaneamente a 320x320.

    Prima di analizzare ogni singola immagine,
    il detector verrà aggiornato con le dimensioni
    reali del frame.
    """

    detector = cv2.FaceDetectorYN.create(
        model=str(YUNET_MODEL_PATH),
        config="",
        input_size=(320, 320),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
    )

    return detector


# ============================================================
# 4. SELEZIONE DEL VOLTO PRINCIPALE
# ============================================================

def select_main_face(faces):
    """
    Se in un'immagine vengono rilevati più volti,
    seleziona quello con bounding box di area maggiore.

    YuNet restituisce per ogni volto una riga contenente:
    - x
    - y
    - width
    - height
    - coordinate dei landmark
    - confidence score

    Parameters
    ----------
    faces : numpy.ndarray
        Array contenente i volti rilevati.

    Returns
    -------
    numpy.ndarray
        Riga relativa al volto principale.
    """

    # Calcoliamo l'area di ogni bounding box.
    # width = faces[:, 2]
    # height = faces[:, 3]
    areas = faces[:, 2] * faces[:, 3]

    # Recuperiamo l'indice del volto con area maggiore.
    largest_face_index = areas.argmax()

    return faces[largest_face_index]


# ============================================================
# 5. CALCOLO DEL CROP CON MARGINE
# ============================================================

def calculate_crop_coordinates(
    x,
    y,
    width,
    height,
    image_width,
    image_height,
    margin=FACE_MARGIN,
):
    """
    Calcola le coordinate del crop aggiungendo
    un margine intorno al bounding box del volto.

    Le coordinate vengono limitate ai bordi dell'immagine,
    così non possiamo ottenere valori negativi
    o superiori alla dimensione del frame.
    """

    # Margine orizzontale e verticale.
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    # Bounding box esteso.
    x1 = int(x - margin_x)
    y1 = int(y - margin_y)

    x2 = int(x + width + margin_x)
    y2 = int(y + height + margin_y)

    # Evitiamo coordinate esterne all'immagine.
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return x1, y1, x2, y2


# ============================================================
# 6. ELABORAZIONE DI UNA SINGOLA IMMAGINE
# ============================================================

def process_image(
    image_path,
    detector,
):
    """
    Esegue face detection e crop su una singola immagine.

    Parameters
    ----------
    image_path : Path
        Percorso del frame da elaborare.

    detector : cv2.FaceDetectorYN
        Face detector YuNet.

    Returns
    -------
    dict
        Dizionario contenente i metadati
        del rilevamento.
    """

    # --------------------------------------------------------
    # Lettura dell'immagine
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"[ERRORE] Impossibile leggere: "
            f"{image_path.name}"
        )

        return {
            "filename": image_path.name,
            "face_detected": 0,
            "detection_score": None,
            "x": None,
            "y": None,
            "width": None,
            "height": None,
            "crop_x1": None,
            "crop_y1": None,
            "crop_x2": None,
            "crop_y2": None,
        }

    # Dimensioni dell'immagine.
    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Aggiornamento della dimensione di input di YuNet
    # --------------------------------------------------------
    #
    # YuNet deve conoscere le dimensioni reali
    # dell'immagine che sta analizzando.

    detector.setInputSize(
        (image_width, image_height)
    )

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    _, faces = detector.detect(image)

    # --------------------------------------------------------
    # Nessun volto trovato
    # --------------------------------------------------------

    if faces is None or len(faces) == 0:

        print(
            f"[NESSUN VOLTO] {image_path.name}"
        )

        return {
            "filename": image_path.name,
            "face_detected": 0,
            "detection_score": None,
            "x": None,
            "y": None,
            "width": None,
            "height": None,
            "crop_x1": None,
            "crop_y1": None,
            "crop_x2": None,
            "crop_y2": None,
        }

    # --------------------------------------------------------
    # Selezione del volto principale
    # --------------------------------------------------------

    main_face = select_main_face(faces)

    # Prime quattro colonne:
    # x, y, width, height.
    x = float(main_face[0])
    y = float(main_face[1])
    width = float(main_face[2])
    height = float(main_face[3])

    # L'ultima colonna contiene
    # la confidence del rilevamento.
    detection_score = float(main_face[-1])

    # --------------------------------------------------------
    # Calcolo crop con margine
    # --------------------------------------------------------

    x1, y1, x2, y2 = calculate_crop_coordinates(
        x=x,
        y=y,
        width=width,
        height=height,
        image_width=image_width,
        image_height=image_height,
    )

    # --------------------------------------------------------
    # Crop effettivo
    # --------------------------------------------------------

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    # Verifica di sicurezza.
    if face_crop.size == 0:

        print(
            f"[ERRORE CROP] {image_path.name}"
        )

        return {
            "filename": image_path.name,
            "face_detected": 0,
            "detection_score": detection_score,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
        }

    # --------------------------------------------------------
    # Salvataggio del crop
    # --------------------------------------------------------
    #
    # Manteniamo lo stesso nome del frame originale.
    #
    # Questo rende immediata la corrispondenza tra:
    #
    # data_processed/frames/
    # e
    # data_processed/faces/

    output_path = (
        FACES_DIR
        / image_path.name
    )

    saved = cv2.imwrite(
        str(output_path),
        face_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    if not saved:

        print(
            f"[ERRORE SALVATAGGIO] "
            f"{output_path}"
        )

        return {
            "filename": image_path.name,
            "face_detected": 0,
            "detection_score": detection_score,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
        }

    # --------------------------------------------------------
    # Restituzione dei metadati
    # --------------------------------------------------------

    return {

        "filename": image_path.name,

        "face_detected": 1,

        "detection_score": detection_score,

        "x": x,
        "y": y,

        "width": width,
        "height": height,

        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
    }


# ============================================================
# 7. FUNZIONE PRINCIPALE
# ============================================================

def main():
    """
    Esegue il face detection su tutti i frame
    presenti in data_processed/frames.
    """

    print("=" * 60)
    print("FaceForensics++ - Face detection e crop")
    print("=" * 60)

    # --------------------------------------------------------
    # Controlli preliminari
    # --------------------------------------------------------

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"Cartella frame non trovata:\n"
            f"{FRAMES_DIR}"
        )

    if not YUNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello YuNet non trovato:\n"
            f"{YUNET_MODEL_PATH}"
        )

    # Creiamo automaticamente le cartelle
    # necessarie se non esistono.
    FACES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FACES_METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Recuperiamo i frame
    # --------------------------------------------------------

    image_files = sorted(
        FRAMES_DIR.glob("*.png")
    )

    print(
        f"\nFrame trovati: "
        f"{len(image_files)}"
    )

    # ========================================================
    # TEST INIZIALE
    # ========================================================
    #
    # IMPORTANTE:
    #
    # Per ora elaboriamo SOLO 20 immagini.
    #
    # Dopo aver controllato visivamente
    # la qualità dei crop, elimineremo questa riga.

   

    image_files = sorted(
        FRAMES_DIR.glob("*.png")
    )

    print(
        f"Frame elaborati nel test: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # Creazione YuNet
    # --------------------------------------------------------

    detector = create_face_detector()

    metadata_records = []

    detected_faces = 0
    missing_faces = 0

    # ========================================================
    # Elaborazione delle immagini
    # ========================================================

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        record = process_image(
            image_path=image_path,
            detector=detector,
        )

        metadata_records.append(record)

        if record["face_detected"] == 1:
            detected_faces += 1
        else:
            missing_faces += 1

    # ========================================================
    # CREAZIONE CSV
    # ========================================================

    metadata_df = pd.DataFrame(
        metadata_records
    )

    metadata_df.to_csv(
        FACES_METADATA_FILE,
        index=False
    )
    # ========================================================
    # ANALISI DEI FALLIMENTI PER TECNICA
    # ========================================================
    #
    # Il file frames_metadata.csv contiene, per ciascun frame,
    # la ground truth e la tecnica di manipolazione.
    #
    # Il file faces_metadata.csv contiene invece l'esito
    # della fase di face detection.
    #
    # Unendo i due file tramite il nome del frame possiamo
    # verificare quanti mancati rilevamenti si sono verificati
    # per ciascuna categoria del dataset.

    if FRAMES_METADATA_FILE.exists():

        # Leggiamo i metadati relativi ai frame originali.
        frames_metadata = pd.read_csv(
            FRAMES_METADATA_FILE
        )

        # Uniamo i metadati dei frame con quelli prodotti
        # dalla fase di face detection.
        #
        # Utilizziamo "filename" come chiave comune.
        merged_metadata = frames_metadata.merge(
            metadata_df[
                ["filename", "face_detected"]
            ],
            on="filename",
            how="inner"
        )

        # Selezioniamo soltanto i frame nei quali
        # YuNet non ha rilevato alcun volto.
        failed_detections = merged_metadata[
            merged_metadata["face_detected"] == 0
        ]

        print("\n" + "=" * 60)
        print("ANALISI DEI MANCATI RILEVAMENTI")
        print("=" * 60)

        print(
            f"\nTotale fallimenti: "
            f"{len(failed_detections)}"
        )

        print("\nFallimenti per tecnica:")

        failures_by_manipulation = (
            failed_detections["manipulation"]
            .value_counts()
        )

        print(
            failures_by_manipulation
        )

        print(
            "\nPercentuale di fallimento per tecnica:"
        )

        failure_percentage = (
            merged_metadata
            .groupby("manipulation")["face_detected"]
            .apply(
                lambda x: (x == 0).mean() * 100
            )
            .sort_values(ascending=False)
        )

        print(
            failure_percentage
        )

    else:

        print(
            "\n[ATTENZIONE] Impossibile analizzare "
            "i fallimenti per tecnica: "
            "frames_metadata.csv non trovato."
        )
    # ========================================================
    # RIEPILOGO
    # ========================================================

    print("\n" + "=" * 60)
    print("CROP COMPLETATO")
    print("=" * 60)

    print(
        f"Immagini elaborate: "
        f"{len(image_files)}"
    )

    print(
        f"Volti rilevati: "
        f"{detected_faces}"
    )

    print(
        f"Volti non rilevati: "
        f"{missing_faces}"
    )

    if len(image_files) > 0:

        detection_rate = (
            detected_faces
            / len(image_files)
            * 100
        )

        print(
            f"Tasso di rilevamento: "
            f"{detection_rate:.2f}%"
        )

    print(
        f"\nCrop salvati in:\n"
        f"{FACES_DIR}"
    )

    print(
        f"\nMetadati salvati in:\n"
        f"{FACES_METADATA_FILE}"
    )


# ============================================================
# 8. AVVIO DEL PROGRAMMA
# ============================================================

if __name__ == "__main__":
    main()