from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataset import DATASET_SPLITS_FILE
from robustness_transforms import (
    ROBUSTNESS_CONDITIONS,
    apply_degradation,
)

FACES_DIR = Path("/content/deepfake-thesis/faces")
from robustness_transforms import (
    ROBUSTNESS_CONDITIONS,
    apply_degradation,
)


# ============================================================
# 1. SELEZIONE DI UN CAMPIONE DEL TEST SET
# ============================================================

metadata = pd.read_csv(DATASET_SPLITS_FILE)

test_data = (
    metadata[metadata["split"] == "test"]
    .copy()
    .reset_index(drop=True)
)

if len(test_data) == 0:
    raise ValueError("Nessun campione trovato nel test set.")

# Selezioniamo il primo campione disponibile
row = test_data.iloc[0]

filename = row["filename"]
label = int(row["label"])
manipulation = row["manipulation"]

image_path = Path(FACES_DIR) / filename

if not image_path.exists():
    raise FileNotFoundError(
        f"Immagine non trovata: {image_path}"
    )

image = Image.open(image_path).convert("RGB")

print("Immagine selezionata:")
print(f"Filename: {filename}")
print(f"Label: {label}")
print(f"Manipulation: {manipulation}")
print(f"Dimensione originale: {image.size}")


# ============================================================
# 2. APPLICAZIONE DELLE TRASFORMAZIONI
# ============================================================

results = []

for condition in ROBUSTNESS_CONDITIONS:

    transformed = apply_degradation(
        image,
        condition,
    )

    results.append(
        (condition, transformed)
    )

    print(
        f"{condition:12s} -> "
        f"dimensione finale: {transformed.size}"
    )

    # Tutte le trasformazioni devono restituire
    # un'immagine con le stesse dimensioni del crop originale
    assert transformed.size == image.size


# ============================================================
# 3. VISUALIZZAZIONE DELLE 10 CONDIZIONI
# ============================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18, 8),
)

axes = axes.flatten()

for ax, (condition, transformed) in zip(
    axes,
    results,
):

    ax.imshow(transformed)
    ax.set_title(condition)
    ax.axis("off")


plt.suptitle(
    f"Test trasformazioni - {filename}",
    fontsize=14,
)

plt.tight_layout()

output_path = Path("robustness_transform_test.png")

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print()
print(
    f"Figura salvata in: "
    f"{output_path.resolve()}"
)