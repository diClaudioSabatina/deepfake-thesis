"""
robustness_transforms.py

Trasformazioni utilizzate per valutare la robustezza
dei detector baseline.

Le degradazioni vengono applicate al crop facciale
prima del preprocessing specifico del modello.

Condizioni considerate:
- original
- JPEG compression
- isotropic resize
- Gaussian blur
"""

from io import BytesIO

from PIL import Image, ImageFilter


# ============================================================
# 1. CONFIGURAZIONE DELLE DEGRADAZIONI
# ============================================================

ROBUSTNESS_CONDITIONS = {
    "original": {
        "type": "original",
        "value": None,
    },

    "jpeg_q90": {
        "type": "jpeg",
        "value": 90,
    },

    "jpeg_q70": {
        "type": "jpeg",
        "value": 70,
    },

    "jpeg_q50": {
        "type": "jpeg",
        "value": 50,
    },

    "resize_75": {
        "type": "resize",
        "value": 0.75,
    },

    "resize_50": {
        "type": "resize",
        "value": 0.50,
    },

    "resize_25": {
        "type": "resize",
        "value": 0.25,
    },

    "blur_05": {
        "type": "blur",
        "value": 0.5,
    },

    "blur_10": {
        "type": "blur",
        "value": 1.0,
    },

    "blur_20": {
        "type": "blur",
        "value": 2.0,
    },
}


# ============================================================
# 2. JPEG COMPRESSION
# ============================================================

def apply_jpeg_compression(image, quality):
    """
    Applica una compressione JPEG all'immagine mantenendo
    invariate larghezza e altezza.

    Parameters
    ----------
    image : PIL.Image
        Immagine RGB originale.

    quality : int
        Qualità JPEG compresa tra 1 e 100.
        Valori più bassi corrispondono a una compressione
        più intensa.
    """

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    degraded_image = Image.open(
        buffer
    ).convert("RGB")

    # Copia necessaria perché il buffer verrà chiuso.
    degraded_image = degraded_image.copy()

    buffer.close()

    return degraded_image


# ============================================================
# 3. ISOTROPIC RESIZE
# ============================================================

def apply_resize(image, scale):
    """
    Riduce isotropicamente l'immagine e successivamente
    la riporta alle dimensioni originali.

    In questo modo viene simulata la perdita di informazione
    dovuta a una riduzione della risoluzione.

    Parameters
    ----------
    image : PIL.Image
        Immagine originale.

    scale : float
        Fattore di ridimensionamento.
        Esempio: 0.50 = 50% della dimensione originale.
    """

    original_width, original_height = image.size

    reduced_width = max(
        1,
        round(original_width * scale),
    )

    reduced_height = max(
        1,
        round(original_height * scale),
    )

    reduced_image = image.resize(
        (reduced_width, reduced_height),
        resample=Image.Resampling.BICUBIC,
    )

    restored_image = reduced_image.resize(
        (original_width, original_height),
        resample=Image.Resampling.BICUBIC,
    )

    return restored_image


# ============================================================
# 4. GAUSSIAN BLUR
# ============================================================

def apply_blur(image, radius):
    """
    Applica un Gaussian blur all'immagine.

    Parameters
    ----------
    image : PIL.Image
        Immagine originale.

    radius : float
        Intensità del blur.
        Valori maggiori producono una sfocatura più marcata.
    """

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


# ============================================================
# 5. FUNZIONE GENERALE
# ============================================================

def apply_degradation(image, condition):
    """
    Applica la degradazione associata alla condizione
    sperimentale specificata.

    Parameters
    ----------
    image : PIL.Image
        Crop facciale RGB.

    condition : str
        Nome della condizione presente in
        ROBUSTNESS_CONDITIONS.

    Returns
    -------
    PIL.Image
        Immagine originale o degradata.
    """

    if condition not in ROBUSTNESS_CONDITIONS:
        raise ValueError(
            f"Condizione non riconosciuta: {condition}"
        )

    config = ROBUSTNESS_CONDITIONS[
        condition
    ]

    degradation_type = config["type"]
    value = config["value"]

    if degradation_type == "original":
        return image.copy()

    if degradation_type == "jpeg":
        return apply_jpeg_compression(
            image,
            quality=value,
        )

    if degradation_type == "resize":
        return apply_resize(
            image,
            scale=value,
        )

    if degradation_type == "blur":
        return apply_blur(
            image,
            radius=value,
        )

    raise ValueError(
        f"Tipo di degradazione non supportato: "
        f"{degradation_type}"
    )


# ============================================================
# 6. TEST
# ============================================================

if __name__ == "__main__":

    print("Condizioni di robustezza:")

    for condition, config in ROBUSTNESS_CONDITIONS.items():

        print(
            f"{condition:12s} -> "
            f"{config['type']:8s} "
            f"{config['value']}"
        )