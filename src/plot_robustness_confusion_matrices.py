"""
plot_robustness_confusion_matrices.py

Genera le confusion matrix per:
- condizione original;
- JPEG 90, 70, 50;
- resize 75%, 50%, 25%;
- blur 0.5, 1.0, 2.0;

per:
- Xception;
- EfficientNet-B4.

Le matrici vengono costruite direttamente dai CSV
delle predizioni già salvati durante le valutazioni.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. PERCORSI
# ============================================================

DRIVE_PROJECT = Path(
    "/content/drive/MyDrive/deepfake-thesis"
)

RESULTS_DIR = (
    DRIVE_PROJECT
    / "results"
)

ROBUSTNESS_DIR = (
    RESULTS_DIR
    / "robustness"
)

OUTPUT_DIR = (
    ROBUSTNESS_DIR
    / "confusion_matrices"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. MODELLI E CONDIZIONI
# ============================================================

MODELS = [
    "xception",
    "efficientnet_b4",
]


CONDITIONS = [
    "original",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "resize_75",
    "resize_50",
    "resize_25",
    "blur_05",
    "blur_10",
    "blur_20",
]


# Nomi più leggibili da mostrare nelle figure
CONDITION_LABELS = {

    "original":
        "Original",

    "jpeg_q90":
        "JPEG quality 90",

    "jpeg_q70":
        "JPEG quality 70",

    "jpeg_q50":
        "JPEG quality 50",

    "resize_75":
        "Resize 75%",

    "resize_50":
        "Resize 50%",

    "resize_25":
        "Resize 25%",

    "blur_05":
        "Gaussian blur 0.5",

    "blur_10":
        "Gaussian blur 1.0",

    "blur_20":
        "Gaussian blur 2.0",
}


MODEL_LABELS = {

    "xception":
        "Xception",

    "efficientnet_b4":
        "EfficientNet-B4",
}


# ============================================================
# 3. RECUPERO DEL CSV
# ============================================================

def get_predictions_file(
    model_name,
    condition,
):

    # --------------------------------------------------------
    # ORIGINAL = predizioni baseline già calcolate
    # --------------------------------------------------------

    if condition == "original":

        return (
            RESULTS_DIR
            / f"{model_name}_baseline_test_predictions.csv"
        )


    # --------------------------------------------------------
    # CONDIZIONI DEGRADATE
    # --------------------------------------------------------

    return (
        ROBUSTNESS_DIR
        / (
            f"{model_name}_"
            f"{condition}_"
            f"predictions.csv"
        )
    )


# ============================================================
# 4. CREAZIONE CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    model_name,
    condition,
):

    predictions_file = (
        get_predictions_file(
            model_name,
            condition,
        )
    )


    if not predictions_file.exists():

        raise FileNotFoundError(
            f"File non trovato:\n"
            f"{predictions_file}"
        )


    # --------------------------------------------------------
    # Lettura delle predizioni
    # --------------------------------------------------------

    df = pd.read_csv(
        predictions_file
    )


    # Controlliamo che ci siano le colonne necessarie
    required_columns = {
        "true_label",
        "predicted_label",
    }


    if not required_columns.issubset(
        df.columns
    ):

        raise ValueError(
            f"Colonne mancanti in "
            f"{predictions_file.name}.\n"
            f"Colonne presenti: "
            f"{list(df.columns)}"
        )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        df["true_label"],
        df["predicted_label"],
        labels=[0, 1],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    print(
        f"\n{MODEL_LABELS[model_name]} "
        f"- {CONDITION_LABELS[condition]}"
    )

    print(
        f"TN={tn} | FP={fp} | "
        f"FN={fn} | TP={tp}"
    )


    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "REAL",
            "FAKE",
        ],
    )


    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )


    ax.set_title(
        f"{MODEL_LABELS[model_name]} - "
        f"{CONDITION_LABELS[condition]}"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # Salvataggio
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / (
            f"{model_name}_"
            f"{condition}_"
            f"confusion_matrix.png"
        )
    )


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print(
        "Salvata in:",
        output_file
    )


# ============================================================
# 5. GENERAZIONE DI TUTTE LE MATRICI
# ============================================================

for model_name in MODELS:

    print("\n" + "=" * 70)

    print(
        MODEL_LABELS[
            model_name
        ]
    )

    print("=" * 70)


    for condition in CONDITIONS:

        create_confusion_matrix(
            model_name,
            condition,
        )


print("\n" + "=" * 70)
print("CONFUSION MATRIX COMPLETATE")
print("=" * 70)

print(
    "\nFigure salvate in:"
)

print(
    OUTPUT_DIR
)