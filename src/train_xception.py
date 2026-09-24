"""
train_xception.py

Training baseline di Xception per la classificazione
REAL / FAKE sui crop facciali di FaceForensics++.

Scelte principali:

- transfer learning da pesi ImageNet;
- fine-tuning completo della rete;
- classificazione a 2 classi:
      0 = REAL
      1 = FAKE
- CrossEntropyLoss pesata per compensare
  lo sbilanciamento delle classi;
- optimizer AdamW;
- learning rate iniziale = 1e-4;
- weight decay = 1e-4;
- scheduler ReduceLROnPlateau;
- early stopping;
- salvataggio del modello con migliore validation loss;
- test set NON utilizzato durante il training.

È inoltre disponibile la modalità --smoke-test,
utile per verificare localmente che tutta la pipeline
funzioni senza eseguire un training completo.
"""

# ============================================================
# 1. IMPORT
# ============================================================

from pathlib import Path
import argparse
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataloaders import create_dataloaders
from models import create_model


# ============================================================
# 2. PERCORSI DEL PROGETTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Creiamo automaticamente le cartelle se non esistono.
MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. CONFIGURAZIONE DEL TRAINING
# ============================================================

MODEL_NAME = "xception"

NUM_CLASSES = 2

# Maximum number of epochs.
MAX_EPOCHS = 20

# Initial learning rate.
LEARNING_RATE = 1e-4

# Optimiser tuning.
WEIGHT_DECAY = 1e-4

# Early stopping:
# if the validation loss does not improve for 4
# consecutive epochs, we stop training.
EARLY_STOPPING_PATIENCE = 4

# Scheduler:
# after 2 epochs with no improvement in the validation loss,
# we halve the learning rate.
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5

# Seed used to make the experiments
# as reproducible as possible.
RANDOM_SEED = 42


# ============================================================
# 4. FILE DI OUTPUT
# ============================================================

BEST_MODEL_PATH = (
    MODELS_DIR
    / "xception_baseline_best.pth"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "xception_baseline_history.csv"
)


# ============================================================
# 5. RIPRODUCIBILITÀ
# ============================================================

def set_seed(seed=RANDOM_SEED):
    """
    Imposta i seed delle principali librerie utilizzate.

    Questo riduce la variabilità dovuta alla casualità
    durante il training.

    Non garantisce necessariamente una riproducibilità
    bit-a-bit su ogni hardware, ma permette di controllare
    le principali sorgenti di casualità.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        # Privilegiamo la riproducibilità rispetto
        # alle ottimizzazioni automatiche di cuDNN.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 6. PESI DELLE CLASSI
# ============================================================

def calculate_class_weights(train_dataset, device):
    """
    Calcola automaticamente i pesi delle classi
    utilizzando esclusivamente il training set.

    Formula:

        weight_c = N / (C * N_c)

    dove:

        N   = numero totale di campioni;
        C   = numero delle classi;
        N_c = numero di campioni della classe c.

    Nel nostro caso:

        classe 0 = REAL
        classe 1 = FAKE

    I pesi vengono calcolati dinamicamente, evitando
    di inserire manualmente numeri dipendenti
    dalla specifica versione del dataset.
    """

    labels = (
        train_dataset.data["label"]
        .astype(int)
    )

    class_counts = (
        labels
        .value_counts()
        .sort_index()
    )

    total_samples = len(labels)

    weights = []

    for class_index in range(NUM_CLASSES):

        class_count = class_counts.get(
            class_index,
            0,
        )

        if class_count == 0:

            raise ValueError(
                f"La classe {class_index} "
                f"non contiene campioni."
            )

        class_weight = (
            total_samples
            / (
                NUM_CLASSES
                * class_count
            )
        )

        weights.append(
            class_weight
        )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print("\nDistribuzione training set:")

    print(
        f"REAL (0): "
        f"{class_counts.get(0, 0)}"
    )

    print(
        f"FAKE (1): "
        f"{class_counts.get(1, 0)}"
    )

    print("\nPesi CrossEntropyLoss:")

    print(
        f"REAL: "
        f"{weights_tensor[0].item():.4f}"
    )

    print(
        f"FAKE: "
        f"{weights_tensor[1].item():.4f}"
    )

    return weights_tensor


# ============================================================
# 7. CALCOLO DELLE METRICHE
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
    fake_probabilities,
):
    """
    Calcola le principali metriche di classificazione.

    Parameters
    ----------
    true_labels:
        Etichette corrette.

    predicted_labels:
        Classi previste dal modello.

    fake_probabilities:
        Probabilità assegnata alla classe FAKE.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1 e ROC-AUC.

    Precision, recall e F1 vengono calcolati considerando
    la classe FAKE (label=1) come classe positiva.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    # ROC-AUC utilizza la probabilità della classe FAKE,
    # non la label binaria prodotta dalla soglia decisionale.
    
    if len(set(true_labels)) == 2:

        roc_auc = roc_auc_score(
            true_labels,
            fake_probabilities,
        )

    else:

        roc_auc = float("nan")
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# 8. TRAINING DI UNA SINGOLA EPOCA
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
    max_batches=None,
):
    """
    Esegue una singola epoca di training.

    Durante questa funzione:

    1. vengono caricati i batch;
    2. viene eseguito il forward pass;
    3. viene calcolata la loss;
    4. vengono calcolati i gradienti;
    5. vengono aggiornati i pesi del modello.
    """

    # Modalità training.
    model.train()

    running_loss = 0.0
    processed_samples = 0

    all_labels = []
    all_predictions = []
    all_fake_probabilities = []

    for batch_index, batch in enumerate(dataloader):

        # ----------------------------------------------------
        # Modalità smoke test
        # ----------------------------------------------------

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Azzeramento dei gradienti
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------
        #
        # Su GPU utilizziamo Automatic Mixed Precision.
        #
        # Su CPU viene automaticamente disattivata.

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Statistiche
        # ----------------------------------------------------

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        processed_samples += (
            batch_size
        )

        # Convertiamo i logits in probabilità.
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # Probabilità della classe FAKE.
        fake_probabilities = (
            probabilities[:, 1]
        )

        # Classe con probabilità maggiore.
        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .tolist()
        )

        all_fake_probabilities.extend(
            fake_probabilities
            .detach()
            .float()
            .cpu()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / processed_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_fake_probabilities,
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# 9. VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device,
    use_amp,
    max_batches=None,
):
    """
    Valuta il modello sul validation set.

    IMPORTANTE:

    durante la validation NON vengono aggiornati
    i parametri del modello.
    """

    model.eval()

    running_loss = 0.0
    processed_samples = 0

    all_labels = []
    all_predictions = []
    all_fake_probabilities = []

    # Disabilitiamo il calcolo dei gradienti.
    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

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

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            processed_samples += (
                batch_size
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            fake_probabilities = (
                probabilities[:, 1]
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

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

    epoch_loss = (
        running_loss
        / processed_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_fake_probabilities,
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# 10. SALVATAGGIO DEL CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    validation_loss,
    class_weights,
):
    """
    Salva il miglior modello trovato fino a quel momento.

    Non salviamo soltanto i pesi della rete:
    manteniamo anche informazioni utili per
    ricostruire l'esperimento.
    """

    checkpoint = {

        "model_name": MODEL_NAME,

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "validation_loss":
            validation_loss,

        "class_weights":
            class_weights
            .detach()
            .cpu(),

        "initial_learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "random_seed":
            RANDOM_SEED,

        "num_classes":
            NUM_CLASSES,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


# ============================================================
# 11. TRAINING COMPLETO
# ============================================================

def train(
    smoke_test=False,
    batch_size=32,
    num_workers=0,
):
    """
    Gestisce l'intero training baseline di Xception.
    """

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print("=" * 70)
    print("TRAINING XCEPTION BASELINE")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------
    #
    # In smoke test elaboriamo solamente:
    #
    # - 1 batch train
    # - 1 batch validation
    # - 1 epoca
    #
    # Serve esclusivamente a verificare il codice.

    if smoke_test:

        print(
            "\nMODALITÀ SMOKE TEST ATTIVA"
        )

        batch_size = 2

        epochs = 1

        max_train_batches = 1
        max_val_batches = 1

    else:

        epochs = MAX_EPOCHS

        max_train_batches = None
        max_val_batches = None

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Epoche massime: {epochs}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        _
    ) = create_dataloaders(
        model_name=MODEL_NAME,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(
        f"\nTraining samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Modello
    # --------------------------------------------------------

    model = create_model(
        model_name=MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Pesi delle classi
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader.dataset,
        device,
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    #
    # Se la validation loss non migliora,
    # il learning rate viene progressivamente ridotto.

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
    )

    # --------------------------------------------------------
    # Automatic Mixed Precision
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(
        f"\nMixed precision: "
        f"{use_amp}"
    )

    # --------------------------------------------------------
    # Variabili early stopping
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        start_time = time.time()

        print("\n" + "=" * 70)

        print(
            f"EPOCA {epoch}/{epochs}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=max_train_batches,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=max_val_batches,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Stampa risultati
        # ----------------------------------------------------

        print("\nTRAIN")

        print(
            f"Loss:      "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Accuracy:  "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{train_metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{train_metrics['recall']:.4f}"
        )

        print(
            f"F1:        "
            f"{train_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print("\nVALIDATION")

        print(
            f"Loss:      "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Accuracy:  "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Recall:    "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"F1:        "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{val_metrics['roc_auc']:.4f}"
        )

        print(
            f"\nLearning rate: "
            f"{current_lr:.8f}"
        )

        print(
            f"Tempo epoca: "
            f"{elapsed_time:.1f} secondi"
        )

        # ----------------------------------------------------
        # Salvataggio nella history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,

                "train_loss":
                    train_metrics["loss"],

                "train_accuracy":
                    train_metrics["accuracy"],

                "train_precision":
                    train_metrics["precision"],

                "train_recall":
                    train_metrics["recall"],

                "train_f1":
                    train_metrics["f1"],

                "train_roc_auc":
                    train_metrics["roc_auc"],

                "val_loss":
                    val_metrics["loss"],

                "val_accuracy":
                    val_metrics["accuracy"],

                "val_precision":
                    val_metrics["precision"],

                "val_recall":
                    val_metrics["recall"],

                "val_f1":
                    val_metrics["f1"],

                "val_roc_auc":
                    val_metrics["roc_auc"],

                "learning_rate":
                    current_lr,

                "epoch_seconds":
                    elapsed_time,
            }
        )

        # Salviamo la history ad ogni epoca.
        #
        # In questo modo, se il runtime viene interrotto,
        # non perdiamo i risultati delle epoche già completate.

        history_df = pd.DataFrame(
            history
        )

        if not smoke_test:
            history_df.to_csv(
                HISTORY_PATH,
                index=False,
        )

        # ----------------------------------------------------
        # Miglior modello
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            < best_validation_loss
        ):

            print(
                "\nNuova migliore "
                "validation loss."
            )

            best_validation_loss = (
                val_metrics["loss"]
            )

            epochs_without_improvement = 0

            # Durante uno smoke test non vogliamo
            # sovrascrivere il vero modello baseline.
            if not smoke_test:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    validation_loss=val_metrics["loss"],
                    class_weights=class_weights,
                )

                print(
                    f"Checkpoint salvato in:\n"
                    f"{BEST_MODEL_PATH}"
                )

        else:

            epochs_without_improvement += 1

            print(
                "\nNessun miglioramento "
                "della validation loss."
            )

            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}"
                f"/{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            not smoke_test
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEARLY STOPPING"
            )

            print(
                "Il training viene interrotto "
                "perché la validation loss "
                "non migliora più."
            )

            break

    # ========================================================
    # FINE TRAINING
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TRAINING TERMINATO"
    )

    print("=" * 70)

    if not smoke_test:

        print(
            f"\nMigliore validation loss: "
            f"{best_validation_loss:.4f}"
        )

        print(
            f"\nModello migliore:\n"
            f"{BEST_MODEL_PATH}"
        )

        print(
            f"\nTraining history:\n"
            f"{HISTORY_PATH}"
        )

    else:

        print(
            "\nSmoke test completato."
        )


# ============================================================
# 12. ARGOMENTI DA TERMINALE
# ============================================================

def parse_arguments():
    """
    Permette di avviare il programma in modalità normale
    oppure in modalità smoke test.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Training baseline Xception "
            "per deepfake detection."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Esegue un solo batch di training "
            "e validation per controllare la pipeline."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size del training.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Numero di processi usati "
            "dal DataLoader."
        ),
    )

    return parser.parse_args()


# ============================================================
# 13. AVVIO
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    train(
        smoke_test=args.smoke_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )