"""
report_gradcam.py

Generazione delle mappe Grad-CAM per:

- Xception robust
- EfficientNet-B4 robust

La Grad-CAM evidenzia le regioni dell'immagine che hanno
contribuito maggiormente alla classe predetta dal modello.

IMPORTANTE:
la Grad-CAM non identifica automaticamente una manipolazione
e non costituisce una prova dell'autenticità dell'immagine.
È uno strumento di supporto per interpretare il comportamento
del detector.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as tv_transforms

from dataset import get_transforms

from report_models import (
    DEVICE,
    CLASS_NAMES,
    load_all_report_models,
    face_crop_to_pil,
)

from report_face_detection import (
    analyze_face,
)


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "report_gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. RICERCA DELL'ULTIMO LAYER CONVOLUZIONALE
# ============================================================

def find_last_conv_layer(model):
    """
    Individua automaticamente l'ultimo layer Conv2d
    presente nel modello.

    Questo evita di hardcodare il nome del layer e rende
    il codice utilizzabile sia con Xception sia con
    EfficientNet-B4.

    Returns
    -------
    tuple
        nome_layer, layer
    """

    last_name = None
    last_layer = None

    for name, module in (
        model.named_modules()
    ):

        if isinstance(
            module,
            nn.Conv2d,
        ):

            last_name = name
            last_layer = module

    if last_layer is None:

        raise RuntimeError(
            "Nessun layer Conv2d trovato "
            "nel modello."
        )

    return (
        last_name,
        last_layer,
    )


# ============================================================
# 3. IMMAGINE SPAZIALE EFFETTIVAMENTE VISTA DAL MODELLO
# ============================================================

def get_model_display_image(
    face_crop,
    model_name,
):
    """
    Applica soltanto le trasformazioni spaziali
    del preprocessing:

    - Resize
    - CenterCrop

    Non vengono applicate ToTensor e Normalize.

    In questo modo otteniamo l'immagine sulla quale
    sovrapporre correttamente la Grad-CAM.
    """

    image = face_crop_to_pil(
        face_crop
    )

    transform = get_transforms(
        model_name
    )

    spatial_image = (
        image.copy()
    )

    spatial_operations_found = 0

    for operation in (
        transform.transforms
    ):

        if isinstance(
            operation,
            (
                tv_transforms.Resize,
                tv_transforms.CenterCrop,
            ),
        ):

            spatial_image = operation(
                spatial_image
            )

            spatial_operations_found += 1

    if spatial_operations_found == 0:

        raise RuntimeError(
            "Nessuna trasformazione spaziale "
            "trovata nel preprocessing."
        )

    spatial_rgb = np.array(
        spatial_image.convert("RGB")
    )

    spatial_bgr = cv2.cvtColor(
        spatial_rgb,
        cv2.COLOR_RGB2BGR,
    )

    return spatial_bgr


# ============================================================
# 4. PREPARAZIONE INPUT TENSOR
# ============================================================

def prepare_model_input(
    face_crop,
    model_name,
):
    """
    Applica esattamente lo stesso preprocessing
    utilizzato durante training e valutazione.
    """

    image = face_crop_to_pil(
        face_crop
    )

    transform = get_transforms(
        model_name
    )

    input_tensor = transform(
        image
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    return input_tensor


# ============================================================
# 5. CLASSE GRAD-CAM
# ============================================================

class GradCAM:
    """
    Implementazione Grad-CAM tramite hook PyTorch.

    Salva:
    - activation maps del layer target;
    - gradienti della classe rispetto alle activation maps.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

        # Salviamo i gradienti relativi
        # all'output del layer convoluzionale.

        if output.requires_grad:

            output.register_hook(
                self._save_gradients
            )

    # --------------------------------------------------------
    # Gradient hook
    # --------------------------------------------------------

    def _save_gradients(
        self,
        gradients,
    ):

        self.gradients = gradients

    # --------------------------------------------------------
    # Generazione Grad-CAM
    # --------------------------------------------------------

    def generate(
        self,
        input_tensor,
        target_class=None,
    ):
        """
        Genera la Grad-CAM.

        Se target_class è None viene utilizzata
        automaticamente la classe predetta dal modello.

        Returns
        -------
        dict
            heatmap normalizzata, classe predetta e logits.
        """

        self.activations = None
        self.gradients = None

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # IMPORTANTE:
        #
        # qui NON utilizziamo torch.no_grad() oppure
        # torch.inference_mode(), perché Grad-CAM necessita
        # dei gradienti.
        # ----------------------------------------------------

        with torch.enable_grad():

            logits = self.model(
                input_tensor
            )

            predicted_class = int(
                torch.argmax(
                    logits,
                    dim=1,
                ).item()
            )

            if target_class is None:

                target_class = (
                    predicted_class
                )

            target_score = logits[
                0,
                target_class
            ]

            target_score.backward()

        # ----------------------------------------------------
        # Controllo hook
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Activation maps non disponibili."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Gradienti non disponibili."
            )

        # ----------------------------------------------------
        # Activation maps
        #
        # shape:
        # [1, channels, height, width]
        # ----------------------------------------------------

        activations = (
            self.activations
            .detach()
        )

        gradients = (
            self.gradients
            .detach()
        )

        # ----------------------------------------------------
        # Peso di ogni feature map
        #
        # Global Average Pooling dei gradienti.
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Combinazione pesata delle feature maps
        # ----------------------------------------------------

        cam = (
            weights
            * activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Grad-CAM utilizza ReLU:
        # vengono mantenuti i contributi positivi.

        cam = torch.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize alle dimensioni dell'input del modello
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                input_tensor.shape[-2],
                input_tensor.shape[-1],
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = (
            cam[0, 0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Normalizzazione 0-1
        # ----------------------------------------------------

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if (
            cam_max - cam_min
            > 1e-12
        ):

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        return {
            "cam": cam,

            "predicted_class": (
                predicted_class
            ),

            "target_class": int(
                target_class
            ),

            "logits": (
                logits
                .detach()
                .cpu()
            ),
        }

    # --------------------------------------------------------
    # Rimozione hook
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()


# ============================================================
# 6. CREAZIONE HEATMAP
# ============================================================

def create_heatmap(
    cam,
):
    """
    Converte la Grad-CAM normalizzata [0,1]
    in una heatmap OpenCV.
    """

    cam_uint8 = np.uint8(
        np.clip(
            cam,
            0.0,
            1.0,
        )
        * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    return heatmap


# ============================================================
# 7. SOVRAPPOSIZIONE HEATMAP
# ============================================================

def create_overlay(
    image,
    heatmap,
    heatmap_weight=0.45,
):
    """
    Sovrappone la heatmap all'immagine
    effettivamente fornita al modello.
    """

    if (
        image.shape[:2]
        != heatmap.shape[:2]
    ):

        heatmap = cv2.resize(
            heatmap,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    image_float = (
        image.astype(
            np.float32
        )
    )

    heatmap_float = (
        heatmap.astype(
            np.float32
        )
    )

    overlay = cv2.addWeighted(
        image_float,
        1.0 - heatmap_weight,
        heatmap_float,
        heatmap_weight,
        0,
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# 8. GRAD-CAM PER UN MODELLO
# ============================================================

def generate_gradcam_for_model(
    model,
    model_name,
    face_crop,
    target_class=None,
):
    """
    Genera la Grad-CAM per un singolo detector.

    Per default viene spiegata la classe effettivamente
    predetta dal modello.
    """

    model_name = (
        model_name.lower()
    )

    # --------------------------------------------------------
    # Input tensor
    # --------------------------------------------------------

    input_tensor = (
        prepare_model_input(
            face_crop=face_crop,
            model_name=model_name,
        )
    )

    # --------------------------------------------------------
    # Immagine visualizzabile con la stessa geometria
    # dell'input tensor
    # --------------------------------------------------------

    model_input_image = (
        get_model_display_image(
            face_crop=face_crop,
            model_name=model_name,
        )
    )

    expected_height = int(
        input_tensor.shape[-2]
    )

    expected_width = int(
        input_tensor.shape[-1]
    )

    if (
        model_input_image.shape[0]
        != expected_height
        or
        model_input_image.shape[1]
        != expected_width
    ):

        raise RuntimeError(
            "Dimensioni dell'immagine visuale "
            "non coerenti con l'input del modello."
        )

    # --------------------------------------------------------
    # Ultimo layer convoluzionale
    # --------------------------------------------------------

    (
        target_layer_name,
        target_layer,
    ) = find_last_conv_layer(
        model
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:

        gradcam_result = (
            gradcam.generate(
                input_tensor=input_tensor,
                target_class=target_class,
            )
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    cam = (
        gradcam_result["cam"]
    )

    heatmap = create_heatmap(
        cam
    )

    overlay = create_overlay(
        image=model_input_image,
        heatmap=heatmap,
    )

    # --------------------------------------------------------
    # Probabilità NON calibrate
    #
    # Servono soltanto come informazione tecnica.
    # Grad-CAM viene generata dal logit della classe,
    # non dalla probabilità calibrata.
    # --------------------------------------------------------

    logits = (
        gradcam_result["logits"]
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    real_probability = float(
        probabilities[
            0,
            0
        ].item()
    )

    fake_probability = float(
        probabilities[
            0,
            1
        ].item()
    )

    predicted_class = (
        gradcam_result[
            "predicted_class"
        ]
    )

    target_class = (
        gradcam_result[
            "target_class"
        ]
    )

    return {
        "model": model_name,

        "target_layer_name": (
            target_layer_name
        ),

        "predicted_class": (
            predicted_class
        ),

        "predicted_label": (
            CLASS_NAMES[
                predicted_class
            ]
        ),

        "target_class": (
            target_class
        ),

        "target_label": (
            CLASS_NAMES[
                target_class
            ]
        ),

        "raw_real_probability": (
            real_probability
        ),

        "raw_fake_probability": (
            fake_probability
        ),

        "cam": cam,

        "heatmap": heatmap,

        "overlay": overlay,

        "model_input_image": (
            model_input_image
        ),
    }


# ============================================================
# 9. GRAD-CAM PER ENTRAMBI I MODELLI
# ============================================================

def generate_all_gradcams(
    models,
    face_crop,
):
    """
    Genera le Grad-CAM di Xception
    ed EfficientNet-B4.
    """

    results = {}

    for (
        model_name,
        model,
    ) in models.items():

        results[model_name] = (
            generate_gradcam_for_model(
                model=model,
                model_name=model_name,
                face_crop=face_crop,
            )
        )

    return results


# ============================================================
# 10. SALVATAGGIO RISULTATI
# ============================================================

def save_gradcam_results(
    results,
):
    """
    Salva per ogni modello:

    - input visuale;
    - heatmap;
    - overlay Grad-CAM.
    """

    saved_files = {}

    for (
        model_name,
        result,
    ) in results.items():

        input_path = (
            OUTPUT_DIR
            / (
                f"{model_name}_"
                f"model_input.png"
            )
        )

        heatmap_path = (
            OUTPUT_DIR
            / (
                f"{model_name}_"
                f"gradcam_heatmap.png"
            )
        )

        overlay_path = (
            OUTPUT_DIR
            / (
                f"{model_name}_"
                f"gradcam_overlay.png"
            )
        )

        cv2.imwrite(
            str(input_path),
            result[
                "model_input_image"
            ],
        )

        cv2.imwrite(
            str(heatmap_path),
            result[
                "heatmap"
            ],
        )

        cv2.imwrite(
            str(overlay_path),
            result[
                "overlay"
            ],
        )

        saved_files[
            model_name
        ] = {
            "input": input_path,
            "heatmap": heatmap_path,
            "overlay": overlay_path,
        }

    return saved_files


# ============================================================
# 11. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    print("=" * 65)
    print("TEST GRAD-CAM")
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
            "Grad-CAM non eseguita."
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
        "Crop originale:",
        face_crop.shape
    )

    # --------------------------------------------------------
    # Modelli
    # --------------------------------------------------------

    models = (
        load_all_report_models()
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    results = (
        generate_all_gradcams(
            models=models,
            face_crop=face_crop,
        )
    )

    # --------------------------------------------------------
    # Salvataggio
    # --------------------------------------------------------

    saved_files = (
        save_gradcam_results(
            results
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "RISULTATI GRAD-CAM"
    )

    print(
        "=" * 65
    )

    for (
        model_name,
        result,
    ) in results.items():

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
            "Layer utilizzato:",
            result[
                "target_layer_name"
            ]
        )

        print(
            "Classe predetta:",
            result[
                "predicted_label"
            ]
        )

        print(
            "Classe spiegata:",
            result[
                "target_label"
            ]
        )

        print(
            "Probabilità REAL:",
            f'{result["raw_real_probability"]:.4f}'
        )

        print(
            "Probabilità FAKE:",
            f'{result["raw_fake_probability"]:.4f}'
        )

        print(
            "Dimensione input visuale:",
            result[
                "model_input_image"
            ].shape
        )

        print(
            "Overlay salvato in:",
            saved_files[
                model_name
            ]["overlay"]
        )

    print(
        "\nGrad-CAM completata."
    )


# ============================================================
# 12. AVVIO
# ============================================================

if __name__ == "__main__":
    main()