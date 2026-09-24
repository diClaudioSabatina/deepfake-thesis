"""
validate_report_models.py

Verifica che la nuova pipeline di inference utilizzata
dal report riproduca le predizioni ottenute durante
la valutazione sperimentale dei modelli robust.

Il confronto viene eseguito su un crop REAL del test set
già utilizzato negli esperimenti.

IMPORTANTE:
non viene eseguita nuovamente la face detection.
Viene utilizzato direttamente il crop presente in
data_processed/faces, come durante la valutazione originale.
"""

from pathlib import Path

import cv2
import pandas as pd

from report_models import (
    load_all_report_models,
    predict_with_model,
)


# ============================================================
# 1. PERCORSI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACES_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "faces"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "robust_models_evaluation"
)

XCEPTION_CSV = (
    RESULTS_DIR
    / "xception_robust_original_predictions.csv"
)

EFFICIENTNET_CSV = (
    RESULTS_DIR
    / "efficientnet_b4_robust_original_predictions.csv"
)


# ============================================================
# 2. CONTROLLO FILE
# ============================================================

def check_required_files():

    required = [
        XCEPTION_CSV,
        EFFICIENTNET_CSV,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"File non trovato:\n{path}"
            )

    if not FACES_DIR.exists():

        raise FileNotFoundError(
            f"Cartella dei crop non trovata:\n"
            f"{FACES_DIR}"
        )


# ============================================================
# 3. SCELTA CAMPIONE REAL
# ============================================================

def select_real_test_sample():
    """
    Seleziona automaticamente un campione REAL che:

    - è presente in entrambi i CSV;
    - era stato classificato REAL da entrambi i modelli;
    - presenta una probabilità FAKE originale < 0.10
      per entrambi i modelli.

    In questo modo scegliamo un caso lontano dalla
    frontiera decisionale e rendiamo il confronto
    più stabile.
    """

    xception_df = pd.read_csv(
        XCEPTION_CSV
    )

    efficientnet_df = pd.read_csv(
        EFFICIENTNET_CSV
    )

    # --------------------------------------------------------
    # Manteniamo soltanto le colonne necessarie
    # --------------------------------------------------------

    xception_df = xception_df[
        [
            "filename",
            "true_label",
            "predicted_label",
            "fake_probability",
        ]
    ].copy()

    efficientnet_df = efficientnet_df[
        [
            "filename",
            "true_label",
            "predicted_label",
            "fake_probability",
        ]
    ].copy()

    # --------------------------------------------------------
    # Rinomina colonne per distinguere i modelli
    # --------------------------------------------------------

    xception_df = xception_df.rename(
        columns={
            "true_label": "true_label_xception",
            "predicted_label": "predicted_label_xception",
            "fake_probability": "fake_probability_xception",
        }
    )

    efficientnet_df = efficientnet_df.rename(
        columns={
            "true_label": "true_label_efficientnet",
            "predicted_label": "predicted_label_efficientnet",
            "fake_probability": "fake_probability_efficientnet",
        }
    )

    # --------------------------------------------------------
    # Join tramite filename
    # --------------------------------------------------------

    merged = pd.merge(
        xception_df,
        efficientnet_df,
        on="filename",
        how="inner",
    )

    # --------------------------------------------------------
    # Cerchiamo un REAL correttamente classificato
    # da entrambi e lontano da 0.5.
    # --------------------------------------------------------

    candidates = merged[
        (
            merged["true_label_xception"] == 0
        )
        &
        (
            merged["true_label_efficientnet"] == 0
        )
        &
        (
            merged["predicted_label_xception"] == 0
        )
        &
        (
            merged["predicted_label_efficientnet"] == 0
        )
        &
        (
            merged["fake_probability_xception"] < 0.10
        )
        &
        (
            merged["fake_probability_efficientnet"] < 0.10
        )
    ].copy()

    if len(candidates) == 0:

        raise ValueError(
            "Nessun campione REAL adatto trovato "
            "nei due CSV."
        )

    # --------------------------------------------------------
    # Controlliamo anche che il crop esista localmente
    # --------------------------------------------------------

    for _, row in candidates.iterrows():

        image_path = (
            FACES_DIR
            / row["filename"]
        )

        if image_path.exists():

            return row

    raise FileNotFoundError(
        "Sono stati trovati campioni compatibili nei CSV, "
        "ma nessuno dei relativi crop è presente "
        "in data_processed/faces."
    )


# ============================================================
# 4. CONFRONTO
# ============================================================

def compare_prediction(
    model,
    model_name,
    face_crop,
    expected_fake_probability,
    expected_predicted_label,
):
    """
    Confronta l'inference locale con il valore
    salvato durante l'esperimento originale.
    """

    result = predict_with_model(
        model=model,
        model_name=model_name,
        face_crop=face_crop,
    )

    current_fake_probability = (
        result["raw_fake_probability"]
    )

    current_predicted_class = (
        result["predicted_class"]
    )

    difference = abs(
        current_fake_probability
        - expected_fake_probability
    )

    # Durante gli esperimenti originali l'inference
    # su Colab poteva utilizzare mixed precision.
    # Sul PC stiamo lavorando in float32.
    #
    # Non richiediamo quindi uguaglianza bit-a-bit:
    # ci interessa che il risultato sia sostanzialmente
    # lo stesso.
    probability_close = (
        difference < 0.01
    )

    class_equal = (
        current_predicted_class
        == expected_predicted_label
    )

    print("\n" + "-" * 65)
    print(model_name.upper())
    print("-" * 65)

    print(
        "Classe originale CSV:",
        expected_predicted_label
    )

    print(
        "Classe pipeline locale:",
        current_predicted_class
    )

    print(
        "Prob. FAKE originale:",
        f"{expected_fake_probability:.8f}"
    )

    print(
        "Prob. FAKE locale:",
        f"{current_fake_probability:.8f}"
    )

    print(
        "Differenza assoluta:",
        f"{difference:.8f}"
    )

    print(
        "Classe coincidente:",
        class_equal
    )

    print(
        "Probabilità compatibile:",
        probability_close
    )

    return (
        class_equal
        and probability_close
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("VALIDAZIONE PIPELINE REPORT")
    print("=" * 65)

    check_required_files()

    # --------------------------------------------------------
    # Selezione campione
    # --------------------------------------------------------

    sample = (
        select_real_test_sample()
    )

    filename = (
        sample["filename"]
    )

    image_path = (
        FACES_DIR
        / filename
    )

    print(
        "\nCampione selezionato:",
        filename
    )

    print(
        "Classe reale: REAL (0)"
    )

    print(
        "Percorso crop:",
        image_path
    )

    # --------------------------------------------------------
    # Lettura crop
    # --------------------------------------------------------

    face_crop = cv2.imread(
        str(image_path)
    )

    if face_crop is None:

        raise RuntimeError(
            f"Impossibile leggere:\n"
            f"{image_path}"
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
    # Xception
    # --------------------------------------------------------

    xception_ok = compare_prediction(
        model=models["xception"],
        model_name="xception",
        face_crop=face_crop,
        expected_fake_probability=float(
            sample[
                "fake_probability_xception"
            ]
        ),
        expected_predicted_label=int(
            sample[
                "predicted_label_xception"
            ]
        ),
    )

    # --------------------------------------------------------
    # EfficientNet-B4
    # --------------------------------------------------------

    efficientnet_ok = compare_prediction(
        model=models["efficientnet_b4"],
        model_name="efficientnet_b4",
        face_crop=face_crop,
        expected_fake_probability=float(
            sample[
                "fake_probability_efficientnet"
            ]
        ),
        expected_predicted_label=int(
            sample[
                "predicted_label_efficientnet"
            ]
        ),
    )

    # --------------------------------------------------------
    # Risultato finale
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    if (
        xception_ok
        and efficientnet_ok
    ):

        print(
            "TEST SUPERATO"
        )

        print(
            "La pipeline locale riproduce "
            "le predizioni sperimentali."
        )

    else:

        print(
            "ATTENZIONE: DIFFERENZA RILEVATA"
        )

        print(
            "La pipeline locale non riproduce "
            "sufficientemente i risultati originali."
        )

        print(
            "Prima di procedere con il report "
            "dobbiamo individuare la causa."
        )

    print("=" * 65)


# ============================================================
# 6. AVVIO
# ============================================================

if __name__ == "__main__":
    main()