"""
report_pipeline.py

Experimental multi-face orchestration layer for the technical report.

This module is intended to be placed in:

    src_multiface/report_pipeline.py

It keeps the stable models and inference pipeline in:

    src/report_pipeline.py

unchanged.

Workflow
--------
1. Run the experimental multi-face report analysis.
2. Obtain every distinct detected face.
3. For each face, run the already validated stable detector pipeline.
4. Attach Xception, EfficientNet-B4, selective-classification and
   Grad-CAM outputs to that specific face.
5. Preserve top-level compatibility fields for the primary face.

Important
---------
The stable training, calibration, thresholds and model weights are not
modified. The experimental branch only applies the same detector pipeline
separately to multiple detected faces.
"""

from pathlib import Path
import importlib.util
import os
import tempfile

import cv2
import numpy as np

from report_analysis import (
    analyze_image_for_report,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

STABLE_REPORT_PIPELINE_PATH = (
    PROJECT_ROOT
    / "src"
    / "report_pipeline.py"
)


# ============================================================
# 2. LOAD STABLE PIPELINE
# ============================================================

def _load_stable_report_pipeline():
    """
    Load src/report_pipeline.py under a private module name.

    Only the experimental multi-face orchestration is changed.
    Model loading and the validated single-face inference logic
    continue to come from the stable pipeline.
    """

    if not STABLE_REPORT_PIPELINE_PATH.exists():

        raise FileNotFoundError(
            "Stable report_pipeline.py not found at: "
            f"{STABLE_REPORT_PIPELINE_PATH}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            "_stable_report_pipeline",
            STABLE_REPORT_PIPELINE_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            "Unable to load the stable "
            "report_pipeline.py module."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    required_functions = [
        "load_pipeline_models",
        "run_report_pipeline",
    ]

    for function_name in required_functions:

        if not hasattr(
            module,
            function_name,
        ):

            raise AttributeError(
                "The stable report_pipeline.py does not expose "
                f"{function_name}()."
            )

    return module


_STABLE_PIPELINE = (
    _load_stable_report_pipeline()
)


# ============================================================
# 3. PUBLIC MODEL LOADER
# ============================================================

def load_pipeline_models():
    """
    Load the same Xception and EfficientNet-B4 models used by
    the stable report pipeline.
    """

    return (
        _STABLE_PIPELINE
        .load_pipeline_models()
    )


# ============================================================
# 4. SUPPORT UTILITIES
# ============================================================

def _save_face_crop_temporarily(
    face_crop,
):
    """
    Save one BGR facial crop to a temporary PNG.

    A temporary file is used only to call the existing stable
    pipeline without changing its public interface.
    """

    if (
        face_crop is None
        or not isinstance(
            face_crop,
            np.ndarray,
        )
        or face_crop.size == 0
    ):

        raise ValueError(
            "Invalid face crop."
        )

    file_descriptor, path_string = (
        tempfile.mkstemp(
            suffix=".png"
        )
    )

    os.close(
        file_descriptor
    )

    temporary_path = Path(
        path_string
    )

    saved = cv2.imwrite(
        str(
            temporary_path
        ),
        face_crop,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not saved:

        try:
            temporary_path.unlink()
        except OSError:
            pass

        raise RuntimeError(
            "Unable to save the temporary face crop."
        )

    return temporary_path


def _compute_model_agreement(
    model_results,
):
    """
    Build a model-agreement block when the stable pipeline
    does not provide one.
    """

    if not isinstance(
        model_results,
        dict,
    ):

        return {
            "agree": None,
            "xception_prediction": None,
            "efficientnet_prediction": None,
        }

    xception_result = (
        model_results.get(
            "xception",
            {},
        )
    )

    efficientnet_result = (
        model_results.get(
            "efficientnet_b4",
            {},
        )
    )

    xception_prediction = (
        xception_result.get(
            "predicted_label"
        )
    )

    efficientnet_prediction = (
        efficientnet_result.get(
            "predicted_label"
        )
    )

    agree = None

    if (
        xception_prediction is not None
        and efficientnet_prediction
        is not None
    ):

        agree = (
            xception_prediction
            == efficientnet_prediction
        )

    return {
        "agree": agree,
        "xception_prediction": (
            xception_prediction
        ),
        "efficientnet_prediction": (
            efficientnet_prediction
        ),
    }


def _compute_face_support_status(
    model_results,
    model_agreement,
):
    """
    Determine whether human review should be highlighted
    for one face.

    Review is recommended when:
    - at least one detector abstains;
    - the detector labels disagree;
    - expected detector outputs are missing.
    """

    if not isinstance(
        model_results,
        dict,
    ):

        return "review_recommended"

    xception_result = (
        model_results.get(
            "xception",
            {},
        )
    )

    efficientnet_result = (
        model_results.get(
            "efficientnet_b4",
            {},
        )
    )

    if (
        not xception_result
        or not efficientnet_result
    ):

        return "review_recommended"

    xception_status = (
        xception_result.get(
            "selective_status"
        )
    )

    efficientnet_status = (
        efficientnet_result.get(
            "selective_status"
        )
    )

    if (
        xception_status
        == "ABSTAIN"
        or efficientnet_status
        == "ABSTAIN"
    ):

        return "review_recommended"

    models_agree = (
        model_agreement.get(
            "agree"
        )
        if isinstance(
            model_agreement,
            dict,
        )
        else None
    )

    if models_agree is not True:

        return "review_recommended"

    return "detector_outputs_available"


# ============================================================
# 5. ANALYZE ONE FACE WITH THE STABLE PIPELINE
# ============================================================

def _run_stable_pipeline_on_face(
    face,
    models,
):
    """
    Run the existing stable report pipeline on one facial crop.

    The face-detection metadata produced by the multi-face stage
    remains authoritative for the face identity, bounding box and
    orientation. The stable result contributes detector outputs,
    calibration/selective-classification information and Grad-CAM.
    """

    face_result = dict(
        face
    )

    face_crop = (
        face_result.get(
            "face_crop"
        )
    )

    temporary_path = None

    try:

        temporary_path = (
            _save_face_crop_temporarily(
                face_crop
            )
        )

        stable_result = (
            _STABLE_PIPELINE
            .run_report_pipeline(
                image_path=str(
                    temporary_path
                ),
                models=models,
            )
        )

        if not isinstance(
            stable_result,
            dict,
        ):

            raise TypeError(
                "Stable pipeline returned a non-dictionary result."
            )

        model_results = (
            stable_result.get(
                "model_results"
            )
            or {}
        )

        if not isinstance(
            model_results,
            dict,
        ):
            model_results = {}

        if (
            not model_results.get(
                "xception"
            )
            or not model_results.get(
                "efficientnet_b4"
            )
        ):
            raise RuntimeError(
                "The stable detector pipeline did not return "
                "classification outputs for this facial crop."
            )

        model_agreement = (
            stable_result.get(
                "model_agreement"
            )
            or {}
        )

        if not isinstance(
            model_agreement,
            dict,
        ):

            model_agreement = (
                _compute_model_agreement(
                    model_results
                )
            )

        support_status = (
            stable_result.get(
                "support_status"
            )
        )

        if support_status is None:

            support_status = (
                _compute_face_support_status(
                    model_results,
                    model_agreement,
                )
            )

        face_result.update(
            {
                "classification_status": (
                    "completed"
                ),
                "model_results": (
                    model_results
                ),
                "model_agreement": (
                    model_agreement
                ),
                "gradcam_results": (
                    stable_result.get(
                        "gradcam_results"
                    )
                    or {}
                ),
                "support_status": (
                    support_status
                ),
            }
        )

    except Exception as error:

        face_result.update(
            {
                "classification_status": (
                    "failed"
                ),
                "classification_error": (
                    str(
                        error
                    )
                ),
                "model_results": {},
                "model_agreement": {
                    "agree": None,
                    "xception_prediction": None,
                    "efficientnet_prediction": None,
                },
                "gradcam_results": {},
                "support_status": (
                    "review_recommended"
                ),
            }
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:
                temporary_path.unlink()

            except OSError:
                pass

    return face_result


# ============================================================
# 6. MAIN MULTI-FACE PIPELINE
# ============================================================

def run_report_pipeline(
    image_path,
    models=None,
):
    """
    Run the complete experimental multi-face report pipeline.

    Parameters
    ----------
    image_path : str or Path
        Uploaded image.

    models : optional
        Already loaded detector models. When None, the same stable
        model loader used by the original application is called.

    Returns
    -------
    dict
        File-level forensic information plus one independent
        detector/Grad-CAM result block for each detected face.
    """

    if models is None:

        models = (
            load_pipeline_models()
        )

    analysis_result = (
        analyze_image_for_report(
            image_path=image_path
        )
    )

    if not isinstance(
        analysis_result,
        dict,
    ):

        raise TypeError(
            "analyze_image_for_report() "
            "returned a non-dictionary result."
        )

    result = dict(
        analysis_result
    )

    detected_faces = (
        analysis_result.get(
            "faces",
            [],
        )
    )

    # --------------------------------------------------------
    # No valid faces
    # --------------------------------------------------------

    if (
        analysis_result.get(
            "status"
        )
        != "ready_for_classification"
        or not detected_faces
    ):

        result.update(
            {
                "face_results": [],
                "faces": [],
                "faces_analyzed": 0,
                "overall_support_status": (
                    "input_not_analyzable"
                ),

                # Compatibility fields
                "model_results": {},
                "model_agreement": {
                    "agree": None,
                    "xception_prediction": None,
                    "efficientnet_prediction": None,
                },
                "gradcam_results": {},
                "support_status": (
                    "input_not_analyzable"
                ),
            }
        )

        return result

    # --------------------------------------------------------
    # Run the stable model pipeline for every distinct face
    # --------------------------------------------------------

    face_results = []

    for face in detected_faces:

        analyzed_face = (
            _run_stable_pipeline_on_face(
                face=face,
                models=models,
            )
        )

        face_results.append(
            analyzed_face
        )

    # Keep both names available.
    result[
        "faces"
    ] = face_results

    result[
        "face_results"
    ] = face_results

    result[
        "faces_analyzed"
    ] = len(
        face_results
    )

    # --------------------------------------------------------
    # Overall review flag
    # --------------------------------------------------------

    any_failed = any(
        face.get(
            "classification_status"
        )
        != "completed"
        for face in face_results
    )

    any_review_recommended = any(
        face.get(
            "support_status"
        )
        == "review_recommended"
        for face in face_results
    )

    if (
        any_failed
        or any_review_recommended
    ):

        overall_support_status = (
            "review_recommended"
        )

    else:

        overall_support_status = (
            "detector_outputs_available"
        )

    result[
        "overall_support_status"
    ] = overall_support_status

    # --------------------------------------------------------
    # Compatibility with the current single-face app
    # --------------------------------------------------------
    #
    # Face 1 remains the largest distinct detected face because
    # report_face_detection.py orders faces by relative area.
    #
    primary_face = (
        face_results[0]
    )

    result[
        "face_crop"
    ] = primary_face.get(
        "face_crop"
    )

    result[
        "detection_score"
    ] = primary_face.get(
        "detection_score"
    )

    result[
        "bounding_box"
    ] = primary_face.get(
        "bounding_box"
    )

    result[
        "crop_coordinates"
    ] = primary_face.get(
        "crop_coordinates"
    )

    result[
        "rotation_applied"
    ] = primary_face.get(
        "rotation_applied"
    )

    result[
        "face_crop_quality"
    ] = primary_face.get(
        "face_crop_quality",
        {},
    )

    result[
        "crop_blur_score"
    ] = (
        primary_face.get(
            "face_crop_quality",
            {},
        ).get(
            "blur_score"
        )
    )

    result[
        "face_crop_blur_score"
    ] = result[
        "crop_blur_score"
    ]

    result[
        "model_results"
    ] = primary_face.get(
        "model_results",
        {},
    )

    result[
        "model_agreement"
    ] = primary_face.get(
        "model_agreement",
        {},
    )

    result[
        "gradcam_results"
    ] = primary_face.get(
        "gradcam_results",
        {},
    )

    # Keep the historical field tied to Face 1 so the current
    # single-face interface remains internally coherent.
    result[
        "support_status"
    ] = primary_face.get(
        "support_status",
        "review_recommended",
    )

    return result


# ============================================================
# 7. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    print(
        "=" * 70
    )

    print(
        "MULTI-FACE REPORT PIPELINE TEST"
    )

    print(
        "=" * 70
    )

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
        "Status:",
        result.get(
            "status"
        ),
    )

    print(
        "Faces detected:",
        result.get(
            "faces_detected"
        ),
    )

    print(
        "Faces analyzed:",
        result.get(
            "faces_analyzed"
        ),
    )

    print(
        "Overall support status:",
        result.get(
            "overall_support_status"
        ),
    )

    for face in result.get(
        "face_results",
        [],
    ):

        print(
            "\n"
            f"Face {face.get('face_id')}"
        )

        print(
            "  classification status:",
            face.get(
                "classification_status"
            ),
        )

        print(
            "  rotation:",
            face.get(
                "rotation_applied"
            ),
        )

        print(
            "  detection score:",
            face.get(
                "detection_score"
            ),
        )

        model_results = (
            face.get(
                "model_results",
                {},
            )
        )

        for model_name in [
            "xception",
            "efficientnet_b4",
        ]:

            model_result = (
                model_results.get(
                    model_name,
                    {},
                )
            )

            print(
                f"  {model_name}:",
                model_result.get(
                    "predicted_label"
                ),
                "-",
                model_result.get(
                    "selective_status"
                ),
            )


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
