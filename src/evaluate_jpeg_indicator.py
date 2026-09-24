"""
evaluate_jpeg_indicator.py

Valutazione del JPEG blockiness score sulle stesse
condizioni utilizzate negli esperimenti di robustezza.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

from dataset import (
    FACES_DIR,
    DATASET_SPLITS_FILE,
)

from robustness_transforms import (
    apply_jpeg_compression,
)

from forensic_indicators import (
    compute_jpeg_blockiness_from_image,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "jpeg_indicator_evaluation.csv"
)

# Prima verifica su 500 immagini.
MAX_IMAGES = None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("VALUTAZIONE INDICATORE JPEG")
    print("=" * 65)

    metadata = pd.read_csv(
        DATASET_SPLITS_FILE
    )

    test_data = (
        metadata[
            metadata["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if MAX_IMAGES is not None:

        test_data = test_data.sample(
            n=min(
                MAX_IMAGES,
                len(test_data)
            ),
            random_state=42,
        )

    print(
        f"\nImmagini analizzate: "
        f"{len(test_data)}"
    )

    results = []

    for _, row in test_data.iterrows():

        filename = row["filename"]

        image_path = (
            FACES_DIR
            / filename
        )

        if not image_path.exists():

            print(
                f"[MANCANTE] {filename}"
            )

            continue

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            jpeg_90 = apply_jpeg_compression(
                image,
                quality=90
            )

            jpeg_70 = apply_jpeg_compression(
                image,
                quality=70
            )

            jpeg_50 = apply_jpeg_compression(
                image,
                quality=50
            )

            original_score = (
                compute_jpeg_blockiness_from_image(
                    image
                )
            )

            score_90 = (
                compute_jpeg_blockiness_from_image(
                    jpeg_90
                )
            )

            score_70 = (
                compute_jpeg_blockiness_from_image(
                    jpeg_70
                )
            )

            score_50 = (
                compute_jpeg_blockiness_from_image(
                    jpeg_50
                )
            )

        results.append(
            {
                "filename": filename,
                "label": row["label"],
                "manipulation": row["manipulation"],

                "original": original_score,
                "jpeg_q90": score_90,
                "jpeg_q70": score_70,
                "jpeg_q50": score_50,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("RISULTATI")
    print("=" * 65)

    columns = [
        "original",
        "jpeg_q90",
        "jpeg_q70",
        "jpeg_q50",
    ]

    summary = (
        results_df[columns]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "25%",
                "50%",
                "75%",
            ]
        ]
    )

    print(
        "\n",
        summary.round(4)
    )

    # --------------------------------------------------------
    # Andamento individuale
    # --------------------------------------------------------

    monotonic = (
        (results_df["original"]
         <= results_df["jpeg_q90"])
        &
        (results_df["jpeg_q90"]
         <= results_df["jpeg_q70"])
        &
        (results_df["jpeg_q70"]
         <= results_df["jpeg_q50"])
    )

    print(
        "\nAndamento "
        "original <= q90 <= q70 <= q50:"
    )

    print(
        f"{monotonic.mean() * 100:.2f}% "
        f"dei campioni"
    )

    print(
        f"\nRisultati salvati in:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()