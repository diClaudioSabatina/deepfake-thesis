"""
models.py

Definizione dei modelli utilizzati nel progetto di deepfake detection.

Questo modulo contiene:
- Xception
- EfficientNet-B4

Entrambi i modelli vengono inizializzati, quando richiesto,
con pesi pre-addestrati su ImageNet e successivamente adattati
alla classificazione binaria:

    0 = REAL
    1 = FAKE

Il training vero e proprio NON viene eseguito in questo file.
"""

import torch
import torch.nn as nn
import timm

from torchvision.models import (
    efficientnet_b4,
    EfficientNet_B4_Weights,
)


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

# Numero di classi del nostro problema:
#
# 0 = REAL
# 1 = FAKE
NUM_CLASSES = 2


# ============================================================
# 2. XCEPTION
# ============================================================

def create_xception(
    pretrained=True,
    num_classes=NUM_CLASSES,
):
    """
    Crea il modello Xception adattato alla classificazione
    binaria real/fake.

    Parameters
    ----------
    pretrained : bool
        Se True, vengono utilizzati pesi pre-addestrati
        su ImageNet.

    num_classes : int
        Numero di classi di output.
        Nel nostro progetto è pari a 2.

    Returns
    -------
    torch.nn.Module
        Modello Xception pronto per il fine-tuning.
    """

    # --------------------------------------------------------
    # Creazione del modello
    # --------------------------------------------------------
    #
    # timm permette di creare direttamente Xception.
    #
    # num_classes=2 sostituisce automaticamente
    # il classificatore finale originale di ImageNet
    # con un classificatore a due classi.
    #
    # In questo modo l'output sarà:
    #
    # [logit_REAL, logit_FAKE]

    model = timm.create_model(
        "legacy_xception",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


# ============================================================
# 3. EFFICIENTNET-B4
# ============================================================

def create_efficientnet_b4(
    pretrained=True,
    num_classes=NUM_CLASSES,
):
    """
    Crea EfficientNet-B4 adattata alla classificazione
    binaria real/fake.

    Parameters
    ----------
    pretrained : bool
        Se True, utilizza i pesi ImageNet disponibili
        in torchvision.

    num_classes : int
        Numero di classi di output.

    Returns
    -------
    torch.nn.Module
        EfficientNet-B4 pronta per il fine-tuning.
    """

    # --------------------------------------------------------
    # Selezione dei pesi
    # --------------------------------------------------------

    if pretrained:

        weights = (
            EfficientNet_B4_Weights.DEFAULT
        )

    else:

        weights = None

    # --------------------------------------------------------
    # Creating the template
    # --------------------------------------------------------

    model = efficientnet_b4(
        weights=weights
    )

    # --------------------------------------------------------
    # Replacing the final classifier
    # --------------------------------------------------------

    in_features = (
        model.classifier[1].in_features
    )

    # Replacing the classifier with:
    #
    # Original dropout
    # +
    # Linear(... → 2 classes)

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


# ============================================================
# 4. FUNZIONE GENERALE PER CREARE UN MODELLO
# ============================================================

def create_model(
    model_name,
    pretrained=True,
    num_classes=NUM_CLASSES,
):
    """
    Crea il modello richiesto utilizzando una singola
    funzione di interfaccia.

    Parameters
    ----------
    model_name : str
        Valori supportati:
        - "xception"
        - "efficientnet_b4"

    pretrained : bool
        Se True utilizza pesi ImageNet.

    num_classes : int
        Numero di classi finali.

    Returns
    -------
    torch.nn.Module
        Modello richiesto.
    """

    model_name = model_name.lower()

    if model_name == "xception":

        return create_xception(
            pretrained=pretrained,
            num_classes=num_classes,
        )

    elif model_name == "efficientnet_b4":

        return create_efficientnet_b4(
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:

        raise ValueError(
            f"Modello non supportato: "
            f"{model_name}"
        )


# ============================================================
# 5. CONTEGGIO DEI PARAMETRI
# ============================================================

def count_parameters(model):
    """
    Conta il numero totale di parametri e il numero
    di parametri addestrabili del modello.

    Questo sarà utile per documentare e confrontare
    la complessità di Xception ed EfficientNet-B4.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return (
        total_parameters,
        trainable_parameters,
    )


# ============================================================
# 6. TEST DEL MODULO
# ============================================================

def main():
    """
    Esegue un test di Xception senza training.

    Verifica:
    - creazione del modello;
    - numero dei parametri;
    - correttezza della dimensione dell'output.
    """

    print("=" * 60)
    print("TEST MODELLO XCEPTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Creazione Xception
    # --------------------------------------------------------

    model = create_model(
        model_name="xception",
        pretrained=True,
    )

    # Modalità evaluation per il test.
    model.eval()

    # --------------------------------------------------------
    # Conteggio parametri
    # --------------------------------------------------------

    (
        total_parameters,
        trainable_parameters,
    ) = count_parameters(model)

    print(
        f"\nParametri totali: "
        f"{total_parameters:,}"
    )

    print(
        f"Parametri addestrabili: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Input artificiale
    # --------------------------------------------------------
    #
    # Simuliamo un batch composto da 2 immagini:
    #
    # batch = 2
    # canali = 3
    # altezza = 299
    # larghezza = 299

    dummy_input = torch.randn(
        2,
        3,
        299,
        299,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    #
    # Non servono gradienti perché è solo un test.

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        f"\nDimensione input: "
        f"{dummy_input.shape}"
    )

    print(
        f"Dimensione output: "
        f"{output.shape}"
    )

    print(
        "\nOutput atteso: "
        "[batch_size, 2]"
    )


# ============================================================
# 7. AVVIO DEL TEST
# ============================================================

if __name__ == "__main__":
    main()