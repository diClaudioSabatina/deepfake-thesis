"""
evaluate_robust_model.py

Valutazione dei modelli ROBUST:
- Xception robust
- EfficientNet-B4 robust

Ogni modello viene valutato sull'intero test set nelle
10 condizioni sperimentali:

- original
- JPEG 90
- JPEG 70
- JPEG 50
- resize 75%
- resize 50%
- resize 25%
- blur 0.5
- blur 1.0
- blur 2.0

Per ogni combinazione modello-condizione vengono salvati:
- metriche aggregate;
- confusion matrix;
- predizioni per singola immagine;
- probabilità associata alla classe FAKE.

I risultati vengono salvati progressivamente su Google Drive
in una cartella separata rispetto alla valutazione baseline.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from tqdm.auto import tqdm

from models import create_model
from robustness_dataset import RobustnessDataset
from robustness_transforms import ROBUSTNESS_CONDITIONS


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

DRIVE_PROJECT = Path(
    "/content/drive/MyDrive/deepfake-thesis"
)


# ------------------------------------------------------------
# Crop facciali nel runtime Colab
# ------------------------------------------------------------

FACES_DIR = Path(
    "/content/deepfake-thesis/faces"
)


# ------------------------------------------------------------
# Metadata dataset
# ------------------------------------------------------------

DATASET_SPLITS_FILE = (
    DRIVE_PROJECT
    / "metadata"
    / "dataset_splits.csv"
)


# ------------------------------------------------------------
# Cartella RISULTATI ROBUST
#
# Separata dai risultati baseline:
#
# baseline:
# results/robustness/
#
# robust:
# results/robust_models_evaluation/
# ------------------------------------------------------------

RESULTS_DIR = (
    DRIVE_PROJECT
    / "results"
    / "robust_models_evaluation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "robust_models_summary.csv"
)


# ============================================================
# 2. CHECKPOINT ROBUST
# ============================================================

CHECKPOINTS = {

    "xception": (
        DRIVE_PROJECT
        / "xception_robust"
        / "models"
        / "xception_robust_best.pth"
    ),

    "efficientnet_b4": (
        DRIVE_PROJECT
        / "efficientnet_b4_robust"
        / "models"
        / "efficientnet_b4_robust_best.pth"
    ),
}


# ============================================================
# 3. CONDIZIONI DA VALUTARE
# ============================================================
#
# A differenza della valutazione baseline,
# qui valutiamo ANCHE "original".
#
# I checkpoint robust sono modelli nuovi e dobbiamo verificare:
#
# 1. le loro prestazioni sulle immagini originali;
# 2. le loro prestazioni sulle stesse nove degradazioni
#    utilizzate per i modelli baseline.
# ============================================================

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


# ============================================================
# 4. PARAMETRI DATALOADER
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 2


# ============================================================
# 5. SMOKE TEST
# ============================================================
#
# PRIMA lascia True.
#
# Vengono elaborati soltanto due batch di Xception robust
# nella condizione original.
#
# Lo smoke test NON salva risultati.
#
# Dopo aver verificato che tutto funziona:
#
# SMOKE_TEST = False
#
# e rilancia lo script.
# ============================================================

SMOKE_TEST = True
SMOKE_MAX_BATCHES = 2


# ============================================================
# 6. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("ROBUST MODELS EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "ATTENZIONE: esecuzione su CPU. "
        "L'esperimento completo sarà molto più lento."
    )


# ============================================================
# 7. CONTROLLO FILE
# ============================================================

if not DATASET_SPLITS_FILE.exists():

    raise FileNotFoundError(
        f"dataset_splits.csv non trovato:\n"
        f"{DATASET_SPLITS_FILE}"
    )


if not FACES_DIR.exists():

    raise FileNotFoundError(
        f"Cartella dei crop non trovata:\n"
        f"{FACES_DIR}"
    )


for model_name, checkpoint_path in CHECKPOINTS.items():

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint robust {model_name} non trovato:\n"
            f"{checkpoint_path}"
        )


print("\nFile necessari trovati correttamente.")

print("\nCheckpoint robust:")

for model_name, checkpoint_path in CHECKPOINTS.items():

    print(
        f"- {model_name}: "
        f"{checkpoint_path}"
    )


print(
    "\nCartella risultati:"
)

print(
    RESULTS_DIR
)


# ============================================================
# 8. CARICAMENTO MODELLO E CHECKPOINT
# ============================================================

def load_model_and_checkpoint(
    model_name,
    checkpoint_path,
):

    print("\n" + "=" * 70)

    print(
        f"CARICAMENTO MODELLO ROBUST: "
        f"{model_name.upper()}"
    )

    print("=" * 70)


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    # Durante la valutazione NON vengono caricati
    # nuovi pesi ImageNet.
    #
    # La struttura del modello viene creata vuota
    # e successivamente vengono caricati i parametri
    # del checkpoint robust.
    model = create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=2,
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(
        device
    )

    model.eval()


    print(
        "Checkpoint epoca:",
        checkpoint["epoch"]
    )

    print(
        "Validation loss:",
        checkpoint["validation_loss"]
    )


    # Gli stessi pesi di classe utilizzati
    # durante il training vengono utilizzati
    # per calcolare la loss sul test set.
    class_weights = (
        checkpoint["class_weights"]
        .to(device)
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    return (
        model,
        checkpoint,
        criterion,
    )


# ============================================================
# 9. VALUTAZIONE DI UNA CONDIZIONE
# ============================================================

def evaluate_condition(
    model,
    model_name,
    checkpoint,
    criterion,
    condition,
    max_batches=None,
):

    print("\n" + "-" * 70)

    print(
        f"{model_name.upper()} ROBUST "
        f"| condizione: {condition}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    #
    # RobustnessDataset supporta anche "original".
    #
    # Nella condizione original non viene applicata
    # alcuna degradazione aggiuntiva.
    #
    # Nelle altre condizioni la degradazione viene
    # applicata PRIMA del preprocessing specifico
    # dell'architettura.
    # --------------------------------------------------------

    dataset = RobustnessDataset(
        split="test",
        model_name=model_name,
        condition=condition,
        csv_path=DATASET_SPLITS_FILE,
        faces_dir=FACES_DIR,
    )


    print(
        "Campioni test:",
        len(dataset)
    )


    if len(dataset) != 6894:

        raise ValueError(
            f"Numero inatteso di campioni: "
            f"{len(dataset)}"
        )


    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )


    # --------------------------------------------------------
    # Contenitori risultati
    # --------------------------------------------------------

    running_loss = 0.0
    processed_samples = 0

    all_labels = []
    all_predictions = []
    all_fake_probabilities = []

    all_filenames = []
    all_manipulations = []
    all_source_videos = []


    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc=(
                f"{model_name}_robust "
                f"{condition}"
            ),
        )


        for batch_index, batch in enumerate(
            progress_bar
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )


            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels,
                )


            current_batch_size = (
                images.size(0)
            )


            running_loss += (
                loss.item()
                * current_batch_size
            )

            processed_samples += (
                current_batch_size
            )


            # ------------------------------------------------
            # Probabilità
            # ------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=1,
            )


            # Classe FAKE = 1
            fake_probabilities = (
                probabilities[:, 1]
            )


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            # ------------------------------------------------
            # Salvataggio valori
            # ------------------------------------------------

            all_labels.extend(
                labels
                .cpu()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .tolist()
            )

            all_fake_probabilities.extend(
                fake_probabilities
                .float()
                .cpu()
                .tolist()
            )

            all_filenames.extend(
                batch["filename"]
            )

            all_manipulations.extend(
                batch["manipulation"]
            )

            all_source_videos.extend(
                batch["source_video"]
            )


            # ------------------------------------------------
            # Smoke test
            # ------------------------------------------------

            if (
                max_batches is not None
                and
                batch_index + 1
                >= max_batches
            ):

                break


    # ========================================================
    # 10. METRICHE
    # ========================================================

    epoch_loss = (
        running_loss
        / processed_samples
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )


    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )


    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )


    # Durante lo smoke test potrebbe capitare
    # che nei pochi batch analizzati sia presente
    # una sola classe.
    #
    # In quel caso ROC-AUC non è definita.
    if len(set(all_labels)) == 2:

        roc_auc = roc_auc_score(
            all_labels,
            all_fake_probabilities,
        )

    else:

        roc_auc = float("nan")


    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    print("\nRisultati:")

    print(
        f"Loss:      {epoch_loss:.4f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"TN={tn} | FP={fp} | "
        f"FN={fn} | TP={tp}"
    )


    # ========================================================
    # 11. RISULTATI AGGREGATI
    # ========================================================

    degradation_config = (
        ROBUSTNESS_CONDITIONS[
            condition
        ]
    )


    summary = {

        "model":
            model_name,

        "variant":
            "robust",

        "condition":
            condition,

        "degradation_type":
            degradation_config["type"],

        "degradation_value":
            degradation_config["value"],

        "samples":
            processed_samples,

        "loss":
            epoch_loss,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "checkpoint_epoch":
            checkpoint["epoch"],

        "checkpoint_validation_loss":
            checkpoint[
                "validation_loss"
            ],
    }


    # ========================================================
    # 12. RISULTATI PER SINGOLA IMMAGINE
    # ========================================================

    predictions_df = pd.DataFrame(
        {

            "model":
                model_name,

            "variant":
                "robust",

            "condition":
                condition,

            "filename":
                all_filenames,

            "true_label":
                all_labels,

            "predicted_label":
                all_predictions,

            "fake_probability":
                all_fake_probabilities,

            "manipulation":
                all_manipulations,

            "source_video":
                all_source_videos,
        }
    )


    return (
        summary,
        predictions_df,
    )


# ============================================================
# 13. SALVATAGGIO PROGRESSIVO
# ============================================================

def save_results(
    summary,
    predictions_df,
):

    model_name = summary["model"]
    condition = summary["condition"]


    # --------------------------------------------------------
    # Predizioni per singola condizione
    # --------------------------------------------------------

    predictions_file = (
        RESULTS_DIR
        / (
            f"{model_name}_"
            f"robust_"
            f"{condition}_"
            f"predictions.csv"
        )
    )


    predictions_df.to_csv(
        predictions_file,
        index=False,
    )


    # --------------------------------------------------------
    # Summary generale
    # --------------------------------------------------------

    new_row = pd.DataFrame(
        [summary]
    )


    if SUMMARY_FILE.exists():

        existing = pd.read_csv(
            SUMMARY_FILE
        )


        # Rimuove un'eventuale vecchia esecuzione
        # della stessa combinazione modello-condizione.
        existing = existing[
            ~(
                (
                    existing["model"]
                    == model_name
                )
                &
                (
                    existing["variant"]
                    == "robust"
                )
                &
                (
                    existing["condition"]
                    == condition
                )
            )
        ]


        summary_df = pd.concat(
            [
                existing,
                new_row,
            ],
            ignore_index=True,
        )

    else:

        summary_df = new_row


    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    print(
        "\nPredizioni salvate in:"
    )

    print(
        predictions_file
    )


    print(
        "\nSummary aggiornato:"
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# 14. CONTROLLO CONDIZIONI GIÀ COMPLETATE
# ============================================================

def condition_already_completed(
    model_name,
    condition,
):

    predictions_file = (
        RESULTS_DIR
        / (
            f"{model_name}_"
            f"robust_"
            f"{condition}_"
            f"predictions.csv"
        )
    )


    if not SUMMARY_FILE.exists():

        return False


    if not predictions_file.exists():

        return False


    summary_df = pd.read_csv(
        SUMMARY_FILE
    )


    required_columns = {
        "model",
        "variant",
        "condition",
    }


    if not required_columns.issubset(
        summary_df.columns
    ):

        return False


    completed = (
        (
            summary_df["model"]
            == model_name
        )
        &
        (
            summary_df["variant"]
            == "robust"
        )
        &
        (
            summary_df["condition"]
            == condition
        )
    ).any()


    return bool(
        completed
    )


# ============================================================
# 15. SMOKE TEST
# ============================================================

if SMOKE_TEST:

    print("\n" + "=" * 70)
    print("SMOKE TEST ROBUST MODEL")
    print("=" * 70)


    print(
        "\nVerranno elaborati soltanto "
        f"{SMOKE_MAX_BATCHES} batch."
    )


    smoke_model_name = (
        "xception"
    )

    smoke_condition = (
        "original"
    )


    (
        model,
        checkpoint,
        criterion,
    ) = load_model_and_checkpoint(
        smoke_model_name,
        CHECKPOINTS[
            smoke_model_name
        ],
    )


    summary, _ = evaluate_condition(
        model=model,
        model_name=smoke_model_name,
        checkpoint=checkpoint,
        criterion=criterion,
        condition=smoke_condition,
        max_batches=SMOKE_MAX_BATCHES,
    )


    print("\n" + "=" * 70)

    print(
        "SMOKE TEST COMPLETATO "
        "CORRETTAMENTE"
    )

    print("=" * 70)


    print(
        "\nNessun risultato dello smoke test "
        "è stato salvato."
    )


# ============================================================
# 16. ESPERIMENTO COMPLETO
# ============================================================

else:

    for model_name in [
        "xception",
        "efficientnet_b4",
    ]:

        (
            model,
            checkpoint,
            criterion,
        ) = load_model_and_checkpoint(
            model_name,
            CHECKPOINTS[
                model_name
            ],
        )


        for condition in CONDITIONS:

            if condition_already_completed(
                model_name,
                condition,
            ):

                print(
                    f"\nSKIP: "
                    f"{model_name} robust "
                    f"{condition} "
                    f"già completata."
                )

                continue


            (
                summary,
                predictions_df,
            ) = evaluate_condition(
                model=model,
                model_name=model_name,
                checkpoint=checkpoint,
                criterion=criterion,
                condition=condition,
            )


            save_results(
                summary,
                predictions_df,
            )


        del model


        if device.type == "cuda":

            torch.cuda.empty_cache()


    print("\n" + "=" * 70)

    print(
        "VALUTAZIONE MODELLI ROBUST "
        "COMPLETATA"
    )

    print("=" * 70)


    print(
        "\nRisultati disponibili in:"
    )

    print(
        RESULTS_DIR
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )