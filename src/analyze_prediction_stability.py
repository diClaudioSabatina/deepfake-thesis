"""
analyze_prediction_stability.py

Analizza la stabilità delle predizioni dei modelli baseline
quando le immagini del test set vengono sottoposte alle
degradazioni definite nell'esperimento di robustezza.

Per ogni condizione confronta la predizione ottenuta
sull'immagine originale con quella ottenuta sulla stessa
immagine degradata.

Metriche calcolate:
- prediction flip count;
- prediction flip rate;
- REAL -> FAKE;
- FAKE -> REAL;
- variazione media assoluta della probabilità fake;
- variazione media con segno della probabilità fake.

I modelli NON vengono rieseguiti: vengono utilizzati
esclusivamente i CSV delle predizioni già salvati.
"""

from pathlib import Path

import pandas as pd


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
    / "stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. CONFIGURAZIONE
# ============================================================

MODELS = [
    "xception",
    "efficientnet_b4",
]


CONDITIONS = [
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


EXPECTED_TEST_SAMPLES = 6894


# ============================================================
# 3. PERCORSI DEI CSV
# ============================================================

def get_baseline_file(model_name):

    return (
        RESULTS_DIR
        / f"{model_name}_baseline_test_predictions.csv"
    )


def get_degraded_file(
    model_name,
    condition,
):

    return (
        ROBUSTNESS_DIR
        / (
            f"{model_name}_"
            f"{condition}_"
            f"predictions.csv"
        )
    )


# ============================================================
# 4. CONTROLLO CSV
# ============================================================

def check_prediction_file(
    df,
    file_path,
):

    required_columns = {
        "true_label",
        "predicted_label",
        "fake_probability",
    }


    missing_columns = (
        required_columns
        - set(df.columns)
    )


    if missing_columns:

        raise ValueError(
            f"\nColonne mancanti in "
            f"{file_path.name}: "
            f"{missing_columns}"
        )


    if len(df) != EXPECTED_TEST_SAMPLES:

        raise ValueError(
            f"\nNumero campioni inatteso "
            f"in {file_path.name}: "
            f"{len(df)} "
            f"(attesi {EXPECTED_TEST_SAMPLES})"
        )


# ============================================================
# 5. ANALISI DI UNA CONDIZIONE
# ============================================================

def analyze_condition(
    model_name,
    condition,
):

    baseline_file = (
        get_baseline_file(
            model_name
        )
    )

    degraded_file = (
        get_degraded_file(
            model_name,
            condition,
        )
    )


    # --------------------------------------------------------
    # Verifica esistenza file
    # --------------------------------------------------------

    if not baseline_file.exists():

        raise FileNotFoundError(
            f"Baseline non trovata:\n"
            f"{baseline_file}"
        )


    if not degraded_file.exists():

        raise FileNotFoundError(
            f"Predizioni degradate non trovate:\n"
            f"{degraded_file}"
        )


    # --------------------------------------------------------
    # Lettura
    # --------------------------------------------------------

    baseline_df = pd.read_csv(
        baseline_file
    )

    degraded_df = pd.read_csv(
        degraded_file
    )


    check_prediction_file(
        baseline_df,
        baseline_file,
    )

    check_prediction_file(
        degraded_df,
        degraded_file,
    )


    # --------------------------------------------------------
    # IMPORTANTE:
    # controlliamo che i campioni siano nello stesso ordine
    # verificando l'intera sequenza delle true label.
    #
    # I CSV baseline non contengono il filename, quindi
    # il confronto viene effettuato riga per riga.
    # --------------------------------------------------------

    same_labels = (
        baseline_df["true_label"]
        .reset_index(drop=True)
        .equals(
            degraded_df["true_label"]
            .reset_index(drop=True)
        )
    )


    if not same_labels:

        raise ValueError(
            f"\nL'ordine dei campioni non coincide "
            f"per {model_name} - {condition}."
        )


    # --------------------------------------------------------
    # DataFrame di confronto
    # --------------------------------------------------------

    comparison = pd.DataFrame()


    comparison["true_label"] = (
        baseline_df["true_label"]
        .astype(int)
    )


    comparison[
        "original_prediction"
    ] = (
        baseline_df[
            "predicted_label"
        ]
        .astype(int)
    )


    comparison[
        "degraded_prediction"
    ] = (
        degraded_df[
            "predicted_label"
        ]
        .astype(int)
    )


    comparison[
        "original_fake_probability"
    ] = (
        baseline_df[
            "fake_probability"
        ]
        .astype(float)
    )


    comparison[
        "degraded_fake_probability"
    ] = (
        degraded_df[
            "fake_probability"
        ]
        .astype(float)
    )


    # Se presenti nel CSV degradato,
    # recuperiamo anche i metadati.
    for column in [
        "filename",
        "manipulation",
        "source_video",
    ]:

        if column in degraded_df.columns:

            comparison[column] = (
                degraded_df[column]
            )


    # --------------------------------------------------------
    # FLIP DELLA PREDIZIONE
    # --------------------------------------------------------

    comparison[
        "prediction_flip"
    ] = (
        comparison[
            "original_prediction"
        ]
        !=
        comparison[
            "degraded_prediction"
        ]
    )


    # REAL = 0
    # FAKE = 1

    comparison[
        "real_to_fake"
    ] = (
        (
            comparison[
                "original_prediction"
            ] == 0
        )
        &
        (
            comparison[
                "degraded_prediction"
            ] == 1
        )
    )


    comparison[
        "fake_to_real"
    ] = (
        (
            comparison[
                "original_prediction"
            ] == 1
        )
        &
        (
            comparison[
                "degraded_prediction"
            ] == 0
        )
    )


    # --------------------------------------------------------
    # VARIAZIONE DELLA PROBABILITÀ FAKE
    # --------------------------------------------------------

    comparison[
        "delta_fake_probability"
    ] = (
        comparison[
            "degraded_fake_probability"
        ]
        -
        comparison[
            "original_fake_probability"
        ]
    )


    comparison[
        "abs_delta_fake_probability"
    ] = (
        comparison[
            "delta_fake_probability"
        ]
        .abs()
    )


    # --------------------------------------------------------
    # METRICHE AGGREGATE
    # --------------------------------------------------------

    n_samples = len(
        comparison
    )


    flip_count = int(
        comparison[
            "prediction_flip"
        ]
        .sum()
    )


    flip_rate = (
        flip_count
        / n_samples
    )


    real_to_fake_count = int(
        comparison[
            "real_to_fake"
        ]
        .sum()
    )


    fake_to_real_count = int(
        comparison[
            "fake_to_real"
        ]
        .sum()
    )


    real_to_fake_rate = (
        real_to_fake_count
        / n_samples
    )


    fake_to_real_rate = (
        fake_to_real_count
        / n_samples
    )


    mean_abs_delta = (
        comparison[
            "abs_delta_fake_probability"
        ]
        .mean()
    )


    mean_signed_delta = (
        comparison[
            "delta_fake_probability"
        ]
        .mean()
    )


    # --------------------------------------------------------
    # SALVATAGGIO CONFRONTO PER-IMMAGINE
    # --------------------------------------------------------

    comparison_file = (
        OUTPUT_DIR
        / (
            f"{model_name}_"
            f"{condition}_"
            f"stability.csv"
        )
    )


    comparison.to_csv(
        comparison_file,
        index=False,
    )


    # --------------------------------------------------------
    # RISULTATO AGGREGATO
    # --------------------------------------------------------

    result = {

        "model":
            model_name,

        "condition":
            condition,

        "n_samples":
            n_samples,

        "flip_count":
            flip_count,

        "flip_rate":
            flip_rate,

        "real_to_fake_count":
            real_to_fake_count,

        "real_to_fake_rate":
            real_to_fake_rate,

        "fake_to_real_count":
            fake_to_real_count,

        "fake_to_real_rate":
            fake_to_real_rate,

        "mean_abs_delta_fake_probability":
            mean_abs_delta,

        "mean_signed_delta_fake_probability":
            mean_signed_delta,
    }


    # --------------------------------------------------------
    # STAMPA
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"{model_name.upper()} "
        f"| {condition}"
    )

    print(
        "-" * 70
    )

    print(
        f"Campioni: {n_samples}"
    )

    print(
        f"Prediction flip: "
        f"{flip_count} "
        f"({flip_rate * 100:.2f}%)"
    )

    print(
        f"REAL -> FAKE: "
        f"{real_to_fake_count} "
        f"({real_to_fake_rate * 100:.2f}%)"
    )

    print(
        f"FAKE -> REAL: "
        f"{fake_to_real_count} "
        f"({fake_to_real_rate * 100:.2f}%)"
    )

    print(
        f"Mean |Δ P(fake)|: "
        f"{mean_abs_delta:.4f}"
    )

    print(
        f"Mean Δ P(fake): "
        f"{mean_signed_delta:+.4f}"
    )

    print(
        "Dettaglio salvato in:",
        comparison_file,
    )


    return result


# ============================================================
# 6. ANALISI COMPLETA
# ============================================================

all_results = []


print(
    "=" * 70
)

print(
    "ANALISI DELLA STABILITÀ DELLE PREDIZIONI"
)

print(
    "=" * 70
)


for model_name in MODELS:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MODELLO: "
        f"{model_name.upper()}"
    )

    print(
        "=" * 70
    )


    for condition in CONDITIONS:

        result = analyze_condition(
            model_name,
            condition,
        )

        all_results.append(
            result
        )


# ============================================================
# 7. TABELLA RIASSUNTIVA
# ============================================================

summary_df = pd.DataFrame(
    all_results
)


summary_file = (
    OUTPUT_DIR
    / "prediction_stability_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False,
)


print(
    "\n"
    + "=" * 70
)

print(
    "ANALISI COMPLETATA"
)

print(
    "=" * 70
)


print(
    "\nTabella riassuntiva:"
)

print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nRisultati salvati in:"
)

print(
    summary_file
)