"""
forensic_indicators.py

Estrazione di indicatori tecnici e forensi da un'immagine.

Gli indicatori NON determinano se l'immagine sia real o fake.
Servono a contestualizzare il risultato prodotto dai deepfake detector.
"""
import numpy as np
from pathlib import Path
import hashlib

import cv2
from PIL import Image, ExifTags


# ============================================================
# 1. HASH SHA-256
# ============================================================

def compute_sha256(image_path):
    """
    Calcola l'hash SHA-256 del file.

    L'hash identifica il file analizzato e permette
    di verificarne successivamente l'integrità.
    """

    image_path = Path(image_path)

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 2. INFORMAZIONI GENERALI SUL FILE
# ============================================================

def extract_basic_info(image_path):
    """
    Estrae le principali caratteristiche tecniche
    dell'immagine.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"File non trovato:\n{image_path}"
        )

    file_size_bytes = image_path.stat().st_size

    with Image.open(image_path) as image:

        width, height = image.size

        image_format = image.format

        image_mode = image.mode

    if height > 0:
        aspect_ratio = width / height
    else:
        aspect_ratio = None

    return {
        "filename": image_path.name,
        "extension": image_path.suffix.lower(),
        "format": image_format,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(
            file_size_bytes / 1024,
            2
        ),
        "width": width,
        "height": height,
        "aspect_ratio": (
            round(aspect_ratio, 4)
            if aspect_ratio is not None
            else None
        ),
        "color_mode": image_mode,
    }


# ============================================================
# 3. METADATI EXIF
# ============================================================

def extract_exif(image_path):
    """
    Estrae i metadati EXIF disponibili.

    L'assenza di EXIF NON indica automaticamente
    che l'immagine sia stata manipolata.
    """

    image_path = Path(image_path)

    exif_data = {}

    with Image.open(image_path) as image:

        exif = image.getexif()

        if not exif:
            return {
                "exif_present": False,
                "exif": {}
            }

        for tag_id, value in exif.items():

            tag_name = ExifTags.TAGS.get(
                tag_id,
                str(tag_id)
            )

            # Alcuni valori EXIF possono essere
            # oggetti non facilmente serializzabili.
            try:
                value = str(value)
            except Exception:
                value = "<valore non leggibile>"

            exif_data[tag_name] = value

    return {
        "exif_present": True,
        "exif": exif_data
    }


# ============================================================
# 4. BLUR SCORE
# ============================================================

def compute_blur_score_from_image(image):
    """
    Calcola la varianza del Laplaciano
    a partire da un'immagine PIL.
    """

    image_rgb = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )
# ============================================================
# JPEG BLOCKINESS SCORE
# ============================================================

def compute_jpeg_blockiness_from_image(image):
    """
    Calcola un indicatore della presenza di discontinuità
    in corrispondenza dei blocchi 8x8 tipici della
    compressione JPEG.

    Valori maggiori indicano che le differenze tra pixel
    risultano relativamente più marcate sui bordi degli
    8x8 block.

    Il valore NON permette di determinare con certezza
    se un'immagine sia stata compressa in JPEG né di
    ricostruire la qualità JPEG utilizzata.
    """

    gray = np.array(
        image.convert("L"),
        dtype=np.float32
    )

    height, width = gray.shape

    # Immagini troppo piccole non consentono
    # una valutazione significativa.
    if height < 16 or width < 16:
        return None

    # Differenze tra pixel adiacenti.
    vertical_diff = np.abs(
        np.diff(gray, axis=1)
    )

    horizontal_diff = np.abs(
        np.diff(gray, axis=0)
    )

    # In un JPEG i blocchi sono 8x8.
    # Gli indici 7, 15, 23, ... rappresentano
    # la differenza tra la fine di un blocco
    # e l'inizio del successivo.
    vertical_boundaries = np.arange(
        7,
        vertical_diff.shape[1],
        8
    )

    horizontal_boundaries = np.arange(
        7,
        horizontal_diff.shape[0],
        8
    )

    vertical_mask = np.ones(
        vertical_diff.shape[1],
        dtype=bool
    )

    horizontal_mask = np.ones(
        horizontal_diff.shape[0],
        dtype=bool
    )

    vertical_mask[
        vertical_boundaries
    ] = False

    horizontal_mask[
        horizontal_boundaries
    ] = False

    # Differenze sui bordi 8x8.
    boundary_values = np.concatenate(
        [
            vertical_diff[
                :,
                vertical_boundaries
            ].ravel(),

            horizontal_diff[
                horizontal_boundaries,
                :
            ].ravel(),
        ]
    )

    # Differenze nelle altre posizioni.
    non_boundary_values = np.concatenate(
        [
            vertical_diff[
                :,
                vertical_mask
            ].ravel(),

            horizontal_diff[
                horizontal_mask,
                :
            ].ravel(),
        ]
    )

    boundary_mean = float(
        boundary_values.mean()
    )

    non_boundary_mean = float(
        non_boundary_values.mean()
    )

    if non_boundary_mean == 0:
        return None

    blockiness_score = (
        boundary_mean
        / non_boundary_mean
    )

    return float(blockiness_score)


def compute_jpeg_blockiness(image_path):
    """
    Calcola il JPEG blockiness score
    a partire da un file.
    """

    with Image.open(image_path) as image:

        return compute_jpeg_blockiness_from_image(
            image
        )

def compute_blur_score(image_path):
    """
    Calcola il blur score di un file immagine.
    """

    with Image.open(image_path) as image:

        return compute_blur_score_from_image(
            image
        )

# ============================================================
# 5. INFORMAZIONI JPEG
# ============================================================

def extract_jpeg_info(image_path):
    """
    Recupera alcune informazioni disponibili
    direttamente nel file JPEG.

    Non tenta di ricostruire con certezza la qualità
    JPEG originariamente utilizzata.
    """

    image_path = Path(image_path)

    with Image.open(image_path) as image:

        if image.format != "JPEG":

            return {
                "is_jpeg": False,
                "progressive": None,
                "jfif_version": None,
                "dpi": None,
                "quantization_tables": None,
            }

        info = image.info

        quantization = getattr(
            image,
            "quantization",
            None
        )

        return {
            "is_jpeg": True,

            "progressive": bool(
                info.get(
                    "progressive",
                    info.get("progression", False)
                )
            ),

            "jfif_version": info.get(
                "jfif_version"
            ),

            "dpi": info.get(
                "dpi"
            ),

            "quantization_tables": (
                len(quantization)
                if quantization
                else 0
            ),
        }


# ============================================================
# 6. ANALISI COMPLETA
# ============================================================

def analyze_forensic_indicators(image_path):
    """
    Esegue l'estrazione completa degli indicatori
    tecnici disponibili.
    """

    image_path = Path(image_path)

    basic_info = extract_basic_info(
        image_path
    )

    exif_info = extract_exif(
        image_path
    )

    jpeg_info = extract_jpeg_info(
        image_path
    )
    
 

    result = {
        **basic_info,

        "sha256": compute_sha256(
            image_path
        ),

        "blur_score": compute_blur_score(
            image_path
        ),

        **exif_info,

        "jpeg_info": jpeg_info,
        
        "jpeg_blockiness_score": (
            compute_jpeg_blockiness(
            image_path
            )
        ),
    }

    return result


# ============================================================
# 7. TEST
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estrazione degli indicatori "
            "forensi da un'immagine."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Percorso dell'immagine da analizzare."
    )

    args = parser.parse_args()

    results = analyze_forensic_indicators(
        args.image
    )

    print("\n" + "=" * 60)
    print("INDICATORI FORENSI")
    print("=" * 60)

    for key, value in results.items():

        if key == "exif":

            print("\nEXIF:")

            if not value:
                print("  Nessun metadato EXIF disponibile.")

            else:
                for exif_key, exif_value in value.items():

                    print(
                        f"  {exif_key}: "
                        f"{exif_value}"
                    )

        elif key == "jpeg_info":

            print("\nJPEG:")

            for jpeg_key, jpeg_value in value.items():

                print(
                    f"  {jpeg_key}: "
                    f"{jpeg_value}"
                )

        else:

            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()