"""
evaluate_blur_indicator.py

Valutazione dell'indicatore di blur sui crop facciali
del test set di FaceForensics++.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

from dataset import (
    FACES_DIR,
    DATASET_SPLITS_FILE,
)

from robustness_transforms import apply_blur

from forensic_indicators import (
    compute_blur_score_from_image,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "blur_indicator_evaluation.csv"
)

# Partiamo con 500 immagini.
# Se tutto funziona, poi possiamo usare tutto il test set.
MAX_IMAGES = None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("VALUTAZIONE INDICATORE BLUR")
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

    for index, row in test_data.iterrows():

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

            original_score = (
                compute_blur_score_from_image(
                    image
                )
            )

            blur_05 = apply_blur(
                image,
                0.5
            )

            blur_10 = apply_blur(
                image,
                1.0
            )

            blur_20 = apply_blur(
                image,
                2.0
            )

            score_05 = (
                compute_blur_score_from_image(
                    blur_05
                )
            )

            score_10 = (
                compute_blur_score_from_image(
                    blur_10
                )
            )

            score_20 = (
                compute_blur_score_from_image(
                    blur_20
                )
            )

        results.append(
            {
                "filename": filename,
                "label": row["label"],
                "manipulation": row["manipulation"],

                "original": original_score,

                "blur_05": score_05,
                "blur_10": score_10,
                "blur_20": score_20,
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
        "blur_05",
        "blur_10",
        "blur_20",
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

    print(
        f"\nRisultati salvati in:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()