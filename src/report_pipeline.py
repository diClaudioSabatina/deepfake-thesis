"""
report_pipeline.py

Pipeline completa per il report tecnico di supporto
alla valutazione umana.
"""

from report_analysis import analyze_image_for_report

from report_models import (
    load_all_report_models,
    predict_with_all_models,
)

from report_gradcam import (
    generate_all_gradcams,
)


# ============================================================
# 1. CARICAMENTO MODELLI
# ============================================================

def load_pipeline_models():
    """
    Carica una sola volta Xception ed EfficientNet-B4.
    """

    return load_all_report_models()


# ============================================================
# 2. PIPELINE COMPLETA
# ============================================================

def run_report_pipeline(
    image_path,
    models,
):
    """
    Esegue l'intera pipeline di analisi.

    Returns
    -------
    dict
        Risultato completo utilizzabile dal report grafico.
    """

    # --------------------------------------------------------
    # Analisi preliminare:
    # - file
    # - indicatori
    # - face detection
    # - crop
    # --------------------------------------------------------

    analysis = analyze_image_for_report(
        image_path
    )

    # --------------------------------------------------------
    # Caso non analizzabile
    # --------------------------------------------------------

    if (
        analysis["status"]
        != "ready_for_classification"
    ):

        return {
            **analysis,

            "model_results": None,
            "model_agreement": None,
            "gradcam_results": None,
        }

    # --------------------------------------------------------
    # Recupero crop facciale
    # --------------------------------------------------------

    face_crop = analysis[
        "face_crop"
    ]

    # --------------------------------------------------------
    # Deepfake detection
    # --------------------------------------------------------

    model_results = (
        predict_with_all_models(
            models=models,
            face_crop=face_crop,
        )
    )

    # --------------------------------------------------------
    # Confronto tra i due modelli
    # --------------------------------------------------------

    xception_prediction = (
        model_results[
            "xception"
        ][
            "predicted_label"
        ]
    )

    efficientnet_prediction = (
        model_results[
            "efficientnet_b4"
        ][
            "predicted_label"
        ]
    )

    models_agree = (
        xception_prediction
        == efficientnet_prediction
    )

    model_agreement = {
        "agree": models_agree,

        "xception_prediction": (
            xception_prediction
        ),

        "efficientnet_prediction": (
            efficientnet_prediction
        ),
    }

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam_results = (
        generate_all_gradcams(
            models=models,
            face_crop=face_crop,
        )
    )

    # --------------------------------------------------------
    # Stato di supporto
    #
    # ATTENZIONE:
    # non è un verdetto REAL/FAKE.
    # --------------------------------------------------------

    xception_status = (
        model_results[
            "xception"
        ][
            "selective_status"
        ]
    )

    efficientnet_status = (
        model_results[
            "efficientnet_b4"
        ][
            "selective_status"
        ]
    )

    if (
        xception_status == "ABSTAIN"
        or
        efficientnet_status == "ABSTAIN"
    ):

        support_status = (
            "review_recommended"
        )

    elif not models_agree:

        support_status = (
            "review_recommended"
        )

    else:

        support_status = (
            "detector_outputs_available"
        )

    # --------------------------------------------------------
    # Risultato finale
    # --------------------------------------------------------

    return {
        **analysis,

        "model_results": (
            model_results
        ),

        "model_agreement": (
            model_agreement
        ),

        "gradcam_results": (
            gradcam_results
        ),

        "support_status": (
            support_status
        ),
    }


# ============================================================
# 3. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    print("=" * 65)
    print("TEST PIPELINE COMPLETA")
    print("=" * 65)

    models = (
        load_pipeline_models()
    )

    result = (
        run_report_pipeline(
            image_path=image_path,
            models=models,
        )
    )

    print(
        "\nStatus:",
        result["status"]
    )

    print(
        "Support status:",
        result.get(
            "support_status"
        )
    )

    # --------------------------------------------------------
    # Se classificabile
    # --------------------------------------------------------

    if result[
        "model_results"
    ] is not None:

        print(
            "\nXception:",
            result[
                "model_results"
            ][
                "xception"
            ][
                "predicted_label"
            ],
            "-",
            result[
                "model_results"
            ][
                "xception"
            ][
                "selective_status"
            ],
        )

        print(
            "EfficientNet-B4:",
            result[
                "model_results"
            ][
                "efficientnet_b4"
            ][
                "predicted_label"
            ],
            "-",
            result[
                "model_results"
            ][
                "efficientnet_b4"
            ][
                "selective_status"
            ],
        )

        print(
            "\nModelli concordi:",
            result[
                "model_agreement"
            ][
                "agree"
            ]
        )

        print(
            "Grad-CAM Xception:",
            result[
                "gradcam_results"
            ][
                "xception"
            ][
                "overlay"
            ].shape
        )

        print(
            "Grad-CAM EfficientNet:",
            result[
                "gradcam_results"
            ][
                "efficientnet_b4"
            ][
                "overlay"
            ].shape
        )


if __name__ == "__main__":
    main()