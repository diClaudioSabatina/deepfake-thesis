"""
report_models.py

Caricamento e utilizzo dei detector robust
Xception ed EfficientNet-B4 per il report tecnico.

Per ciascun modello vengono calcolati:

- classe predetta;
- probabilità REAL e FAKE;
- confidence non calibrata;
- confidence calibrata mediante Temperature Scaling;
- soglia di selective classification;
- stato ACCEPT / ABSTAIN.

Convenzione:
0 = REAL
1 = FAKE
"""

from pathlib import Path

import cv2
import torch
from PIL import Image

from models import create_model
from dataset import get_transforms


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


CHECKPOINTS = {

    "xception": (
        CHECKPOINTS_DIR
        / "xception_robust_best.pth"
    ),

    "efficientnet_b4": (
        CHECKPOINTS_DIR
        / "efficientnet_b4_robust_best.pth"
    ),
}


# ============================================================
# 2. PARAMETRI DI CALIBRAZIONE
# ============================================================
#
# Temperature Scaling determinato sul validation set.
# ============================================================

TEMPERATURES = {

    "xception": 1.4859,

    "efficientnet_b4": 1.5581,
}


# ============================================================
# 3. SOGLIE DI ASTENSIONE
# ============================================================
#
# Punto operativo scelto nella tesi:
# target empirico di selective risk = 2%.
#
# Le soglie sono state determinate sul validation set.
# ============================================================

ABSTENTION_THRESHOLDS = {

    "xception": 0.862503,

    "efficientnet_b4": 0.879208,
}


# ============================================================
# 4. LABEL
# ============================================================

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE",
}


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 6. CONVERSIONE CROP OPENCV -> PIL
# ============================================================

def face_crop_to_pil(face_crop):
    """
    Converte il crop OpenCV da BGR a RGB
    e restituisce una PIL.Image.
    """

    if face_crop is None:

        raise ValueError(
            "Il crop facciale è None."
        )

    if face_crop.size == 0:

        raise ValueError(
            "Il crop facciale è vuoto."
        )

    rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        rgb
    )


# ============================================================
# 7. CARICAMENTO DI UN MODELLO
# ============================================================

def load_report_model(model_name):
    """
    Ricostruisce il modello e carica il relativo
    checkpoint robust.

    Parameters
    ----------
    model_name : str
        "xception" oppure "efficientnet_b4".

    Returns
    -------
    torch.nn.Module
        Modello pronto per l'inference.
    """

    model_name = model_name.lower()

    if model_name not in CHECKPOINTS:

        raise ValueError(
            f"Modello non supportato: "
            f"{model_name}"
        )

    checkpoint_path = (
        CHECKPOINTS[model_name]
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint non trovato:\n"
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Caricamento checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Ricostruzione architettura
    # --------------------------------------------------------

    model = create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=2,
    )

    # --------------------------------------------------------
    # Caricamento pesi robust
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# 8. CARICAMENTO DI ENTRAMBI I MODELLI
# ============================================================

def load_all_report_models():
    """
    Carica Xception ed EfficientNet-B4.

    Returns
    -------
    dict
        Dizionario dei modelli caricati.
    """

    models = {}

    for model_name in [
        "xception",
        "efficientnet_b4",
    ]:

        print(
            f"Caricamento "
            f"{model_name}..."
        )

        models[model_name] = (
            load_report_model(
                model_name
            )
        )

        print(
            f"{model_name} caricato."
        )

    return models


# ============================================================
# 9. INFERENCE DI UN SINGOLO MODELLO
# ============================================================

def predict_with_model(
    model,
    model_name,
    face_crop,
):
    """
    Esegue la classificazione di un crop facciale
    con un singolo detector.

    Vengono calcolate sia le probabilità originali
    sia quelle calibrate mediante Temperature Scaling.
    """

    model_name = model_name.lower()

    if model_name not in TEMPERATURES:

        raise ValueError(
            f"Temperatura non disponibile per "
            f"{model_name}"
        )

    # --------------------------------------------------------
    # Crop -> PIL
    # --------------------------------------------------------

    face_image = face_crop_to_pil(
        face_crop
    )

    # --------------------------------------------------------
    # Preprocessing specifico del modello
    # --------------------------------------------------------

    transform = get_transforms(
        model_name
    )

    input_tensor = transform(
        face_image
    )

    # Aggiunge la dimensione batch:
    #
    # [C, H, W]
    # ->
    # [1, C, H, W]

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(
            input_tensor
        )

    # --------------------------------------------------------
    # Probabilità NON calibrate
    # --------------------------------------------------------

    raw_probabilities = torch.softmax(
        logits,
        dim=1,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_class = int(
        torch.argmax(
            logits,
            dim=1
        ).item()
    )

    predicted_label = (
        CLASS_NAMES[
            predicted_class
        ]
    )

    raw_confidence = float(
        raw_probabilities[
            0,
            predicted_class
        ].item()
    )

    raw_real_probability = float(
        raw_probabilities[
            0,
            0
        ].item()
    )

    raw_fake_probability = float(
        raw_probabilities[
            0,
            1
        ].item()
    )

    # ========================================================
    # TEMPERATURE SCALING
    # ========================================================

    temperature = (
        TEMPERATURES[
            model_name
        ]
    )

    calibrated_logits = (
        logits
        / temperature
    )

    calibrated_probabilities = (
        torch.softmax(
            calibrated_logits,
            dim=1,
        )
    )

    calibrated_real_probability = float(
        calibrated_probabilities[
            0,
            0
        ].item()
    )

    calibrated_fake_probability = float(
        calibrated_probabilities[
            0,
            1
        ].item()
    )

    calibrated_confidence = float(
        calibrated_probabilities[
            0,
            predicted_class
        ].item()
    )

    # ========================================================
    # SELECTIVE CLASSIFICATION
    # ========================================================

    threshold = (
        ABSTENTION_THRESHOLDS[
            model_name
        ]
    )

    if calibrated_confidence >= threshold:

        selective_status = (
            "ACCEPT"
        )

        human_review_required = False

    else:

        selective_status = (
            "ABSTAIN"
        )

        human_review_required = True

    # --------------------------------------------------------
    # Risultato
    # --------------------------------------------------------

    return {

        "model": model_name,

        "predicted_class": (
            predicted_class
        ),

        "predicted_label": (
            predicted_label
        ),

        "raw_real_probability": (
            raw_real_probability
        ),

        "raw_fake_probability": (
            raw_fake_probability
        ),

        "raw_confidence": (
            raw_confidence
        ),

        "temperature": (
            temperature
        ),

        "calibrated_real_probability": (
            calibrated_real_probability
        ),

        "calibrated_fake_probability": (
            calibrated_fake_probability
        ),

        "calibrated_confidence": (
            calibrated_confidence
        ),

        "abstention_threshold": (
            threshold
        ),

        "selective_status": (
            selective_status
        ),

        "human_review_required": (
            human_review_required
        ),
    }


# ============================================================
# 10. ANALISI CON ENTRAMBI I MODELLI
# ============================================================

def predict_with_all_models(
    models,
    face_crop,
):
    """
    Esegue la classificazione dello stesso crop
    con Xception ed EfficientNet-B4.
    """

    results = {}

    for model_name, model in (
        models.items()
    ):

        results[model_name] = (
            predict_with_model(
                model=model,
                model_name=model_name,
                face_crop=face_crop,
            )
        )

    return results


# ============================================================
# 11. STAMPA RISULTATI
# ============================================================

def print_model_results(
    results,
):
    """
    Stampa nel terminale i risultati
    dei due detector.
    """

    print("\n" + "=" * 65)
    print("RISULTATI DEEPFAKE DETECTION")
    print("=" * 65)

    for model_name, result in (
        results.items()
    ):

        print(
            "\n" + "-" * 65
        )

        print(
            model_name.upper()
        )

        print(
            "-" * 65
        )

        print(
            "Predizione:",
            result[
                "predicted_label"
            ]
        )

        print(
            "Probabilità REAL calibrata:",
            f'{result["calibrated_real_probability"]:.4f}'
        )

        print(
            "Probabilità FAKE calibrata:",
            f'{result["calibrated_fake_probability"]:.4f}'
        )

        print(
            "Confidence non calibrata:",
            f'{result["raw_confidence"]:.4f}'
        )

        print(
            "Confidence calibrata:",
            f'{result["calibrated_confidence"]:.4f}'
        )

        print(
            "Temperatura:",
            result["temperature"]
        )

        print(
            "Soglia astensione:",
            f'{result["abstention_threshold"]:.6f}'
        )

        print(
            "Selective status:",
            result[
                "selective_status"
            ]
        )

        print(
            "Revisione umana richiesta:",
            result[
                "human_review_required"
            ]
        )


# ============================================================
# 12. TEST
# ============================================================

def main():

    # Import locale utilizzato soltanto
    # per il test del modulo.

    from report_face_detection import (
        analyze_face,
    )

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    print("=" * 65)
    print("TEST REPORT MODELS")
    print("=" * 65)

    print(
        "\nDevice:",
        DEVICE
    )

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    face_result = analyze_face(
        image_path
    )

    if not face_result[
        "face_detected"
    ]:

        print(
            "\nNessun volto rilevato."
        )

        print(
            "Classificazione interrotta."
        )

        return

    face_crop = (
        face_result[
            "face_crop"
        ]
    )

    print(
        "\nVolto rilevato."
    )

    print(
        "Dimensione crop:",
        face_crop.shape
    )

    # --------------------------------------------------------
    # Caricamento modelli
    # --------------------------------------------------------

    models = (
        load_all_report_models()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    results = (
        predict_with_all_models(
            models=models,
            face_crop=face_crop,
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print_model_results(
        results
    )


# ============================================================
# 13. AVVIO TEST
# ============================================================

if __name__ == "__main__":
    main()