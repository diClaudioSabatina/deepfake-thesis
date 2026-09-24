"""
report_analysis.py

Coordinamento delle informazioni utilizzate
nel report tecnico.

Il modulo combina:

- informazioni tecniche del file;
- metadati EXIF;
- hash SHA-256;
- indicatori di qualità;
- face detection;
- selezione del volto principale;
- qualità tecnica del volto analizzato.

In questa fase NON vengono ancora eseguiti
Xception ed EfficientNet-B4.
"""

from pathlib import Path

import cv2
from PIL import Image

from forensic_indicators import (
    analyze_forensic_indicators,
    compute_blur_score_from_image,
)

from report_face_detection import (
    analyze_face,
)


# ============================================================
# 1. CONVERSIONE CROP OPENCV -> PIL
# ============================================================

def cv2_crop_to_pil(face_crop):
    """
    Converte il crop prodotto da OpenCV
    da BGR a RGB e successivamente in PIL.Image.
    """

    face_crop_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        face_crop_rgb
    )


# ============================================================
# 2. ANALISI COMPLETA PER IL REPORT
# ============================================================

def analyze_image_for_report(
    image_path,
    detector=None,
):
    """
    Esegue le analisi preliminari necessarie
    alla costruzione del report tecnico.

    Parameters
    ----------
    image_path : str or Path
        Percorso dell'immagine da analizzare.

    detector : cv2.FaceDetectorYN, optional
        Detector YuNet già inizializzato.

    Returns
    -------
    dict
        Struttura contenente:
        - stato dell'analisi;
        - indicatori del file;
        - informazioni sulla face detection;
        - indicatori relativi al volto selezionato;
        - crop facciale.
    """

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Controllo file
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"File non trovato:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Indicatori tecnici del file
    # --------------------------------------------------------

    file_indicators = (
        analyze_forensic_indicators(
            image_path
        )
    )

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    face_result = analyze_face(
        image_path=image_path,
        detector=detector,
    )

    # --------------------------------------------------------
    # Informazioni serializzabili della face detection
    # --------------------------------------------------------
    #
    # Non inseriamo face_crop all'interno di questo
    # sotto-dizionario perché è un array NumPy.
    # Il crop viene conservato separatamente.
    # --------------------------------------------------------

    face_detection_info = {
        "face_detected": (
            face_result["face_detected"]
        ),

        "faces_detected": (
            face_result["faces_detected"]
        ),

        "detection_score": (
            face_result["detection_score"]
        ),

        "bounding_box": (
            face_result["bounding_box"]
        ),

        "crop_coordinates": (
            face_result["crop_coordinates"]
        ),
    }

    # --------------------------------------------------------
    # Nessun volto rilevato
    # --------------------------------------------------------

    if not face_result["face_detected"]:

        return {
            "status": "input_not_analyzable",

            "message": (
                "Nessun volto valido rilevato. "
                "La classificazione real/fake "
                "non deve essere eseguita."
            ),

            "file_indicators": (
                file_indicators
            ),

            "face_detection": (
                face_detection_info
            ),

            "face_quality": None,

            "face_crop": None,
        }

    # --------------------------------------------------------
    # Recupero crop facciale
    # --------------------------------------------------------

    face_crop = (
        face_result["face_crop"]
    )

    face_crop_height, face_crop_width = (
        face_crop.shape[:2]
    )

    # --------------------------------------------------------
    # Conversione in PIL
    # --------------------------------------------------------

    face_crop_pil = cv2_crop_to_pil(
        face_crop
    )

    # --------------------------------------------------------
    # Blur score sul volto effettivamente analizzato
    # --------------------------------------------------------
    #
    # Questo valore è particolarmente importante perché
    # viene calcolato proprio sulla regione che verrà
    # successivamente fornita ai detector.
    # --------------------------------------------------------

    face_blur_score = (
        compute_blur_score_from_image(
            face_crop_pil
        )
    )

    # --------------------------------------------------------
    # Indicatori del volto
    # --------------------------------------------------------

    face_quality = {
        "crop_width": (
            face_crop_width
        ),

        "crop_height": (
            face_crop_height
        ),

        "blur_score": (
            face_blur_score
        ),
    }

    # --------------------------------------------------------
    # Risultato completo
    # --------------------------------------------------------

    return {
        "status": "ready_for_classification",

        "message": (
            "Volto rilevato correttamente. "
            "L'immagine può essere sottoposta "
            "ai deepfake detector."
        ),

        "file_indicators": (
            file_indicators
        ),

        "face_detection": (
            face_detection_info
        ),

        "face_quality": (
            face_quality
        ),

        # Conserviamo il crop perché servirà
        # successivamente a Xception,
        # EfficientNet-B4 e Grad-CAM.
        "face_crop": (
            face_crop
        ),
    }


# ============================================================
# 3. STAMPA DEI RISULTATI
# ============================================================

def print_report_analysis(result):
    """
    Stampa nel terminale una versione semplificata
    dell'analisi preliminare.
    """

    print("=" * 65)
    print("ANALISI PRELIMINARE PER REPORT TECNICO")
    print("=" * 65)

    print(
        "\nStato:",
        result["status"]
    )

    print(
        "Messaggio:",
        result["message"]
    )

    # ========================================================
    # FILE
    # ========================================================

    file_info = (
        result["file_indicators"]
    )

    print("\n" + "-" * 65)
    print("INFORMAZIONI FILE")
    print("-" * 65)

    print(
        "Nome:",
        file_info["filename"]
    )

    print(
        "Formato:",
        file_info["format"]
    )

    print(
        "Dimensione:",
        f'{file_info["width"]} x '
        f'{file_info["height"]}'
    )

    print(
        "Peso:",
        f'{file_info["file_size_kb"]} KB'
    )

    print(
        "Aspect ratio:",
        file_info["aspect_ratio"]
    )

    print(
        "Modalità colore:",
        file_info["color_mode"]
    )

    print(
        "SHA-256:",
        file_info["sha256"]
    )

    # ========================================================
    # EXIF
    # ========================================================

    print("\n" + "-" * 65)
    print("METADATI EXIF")
    print("-" * 65)

    if file_info["exif_present"]:

        exif = file_info["exif"]

        if len(exif) == 0:

            print(
                "Metadati EXIF non disponibili."
            )

        else:

            for key, value in exif.items():

                print(
                    f"{key}: {value}"
                )

    else:

        print(
            "Metadati EXIF non presenti."
        )

    # ========================================================
    # QUALITÀ TECNICA DEL FILE
    # ========================================================

    print("\n" + "-" * 65)
    print("INDICATORI TECNICI")
    print("-" * 65)

    print(
        "Blur score immagine completa:",
        round(
            file_info["blur_score"],
            4
        )
    )

    print(
        "JPEG blockiness score:",
        (
            round(
                file_info[
                    "jpeg_blockiness_score"
                ],
                4
            )
            if file_info[
                "jpeg_blockiness_score"
            ] is not None
            else None
        )
    )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_info = (
        result["face_detection"]
    )

    print("\n" + "-" * 65)
    print("FACE DETECTION")
    print("-" * 65)

    print(
        "Volto rilevato:",
        face_info["face_detected"]
    )

    print(
        "Numero di volti:",
        face_info["faces_detected"]
    )

    print(
        "Detection score:",
        face_info["detection_score"]
    )

    print(
        "Bounding box:",
        face_info["bounding_box"]
    )

    print(
        "Coordinate crop:",
        face_info["crop_coordinates"]
    )

    # ========================================================
    # QUALITÀ DEL VOLTO
    # ========================================================

    if result["face_quality"] is not None:

        face_quality = (
            result["face_quality"]
        )

        print("\n" + "-" * 65)
        print("QUALITÀ DEL VOLTO ANALIZZATO")
        print("-" * 65)

        print(
            "Dimensione crop:",
            f'{face_quality["crop_width"]} x '
            f'{face_quality["crop_height"]}'
        )

        print(
            "Blur score crop:",
            round(
                face_quality["blur_score"],
                4
            )
        )


# ============================================================
# 4. TEST DEL MODULO
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    result = analyze_image_for_report(
        image_path
    )

    print_report_analysis(
        result
    )


# ============================================================
# 5. AVVIO TEST
# ============================================================

if __name__ == "__main__":
    main()