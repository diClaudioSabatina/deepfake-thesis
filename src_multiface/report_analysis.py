"""
report_analysis.py

Experimental multi-face analysis layer for the technical report.

This module keeps the existing stable report analysis as the basis
for file-level forensic indicators and extends it with:

- detection of all distinct faces;
- one crop-quality block for each detected face;
- compatibility fields for the previous single-face pipeline.

The experimental version is intended to be placed in:

    src_multiface/report_analysis.py

The stable file in:

    src/report_analysis.py

is not modified.
"""

from pathlib import Path
import importlib.util

import cv2
import numpy as np

from report_face_detection import (
    analyze_faces,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

STABLE_REPORT_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "src"
    / "report_analysis.py"
)


# ============================================================
# 2. LOAD THE STABLE ANALYSIS MODULE
# ============================================================

def _load_stable_report_analysis():
    """
    Load src/report_analysis.py under a private module name.

    This lets the multi-face branch reuse the already validated
    file-level forensic analysis without modifying the stable code.
    """

    if not STABLE_REPORT_ANALYSIS_PATH.exists():

        raise FileNotFoundError(
            "Stable report_analysis.py not found at: "
            f"{STABLE_REPORT_ANALYSIS_PATH}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            "_stable_report_analysis",
            STABLE_REPORT_ANALYSIS_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            "Unable to load the stable "
            "report_analysis.py module."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "analyze_image_for_report",
    ):

        raise AttributeError(
            "The stable report_analysis.py does not expose "
            "analyze_image_for_report()."
        )

    return module


_STABLE_REPORT_ANALYSIS = (
    _load_stable_report_analysis()
)


# ============================================================
# 3. CROP-QUALITY UTILITIES
# ============================================================

def compute_blur_score(
    image,
):
    """
    Compute the variance of the Laplacian.

    This is the same type of descriptive sharpness/blur indicator
    used by the report. A lower value generally corresponds to a
    smoother image, but the value is not an automatic blur verdict.
    """

    if (
        image is None
        or not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        return None

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def analyze_face_crop_quality(
    face_crop,
):
    """
    Compute descriptive indicators for one facial crop.
    """

    if (
        face_crop is None
        or not isinstance(
            face_crop,
            np.ndarray,
        )
        or face_crop.size == 0
    ):

        return {
            "blur_score": None,
            "crop_blur_score": None,
            "width": None,
            "height": None,
        }

    crop_height, crop_width = (
        face_crop.shape[:2]
    )

    blur_score = compute_blur_score(
        face_crop
    )

    return {
        "blur_score": blur_score,
        "crop_blur_score": blur_score,
        "width": int(
            crop_width
        ),
        "height": int(
            crop_height
        ),
    }


# ============================================================
# 4. FACE RECORD ENRICHMENT
# ============================================================

def enrich_face_record(
    face,
):
    """
    Copy one face record and attach crop-quality information.
    """

    enriched_face = dict(
        face
    )

    face_crop = enriched_face.get(
        "face_crop"
    )

    crop_quality = (
        analyze_face_crop_quality(
            face_crop
        )
    )

    enriched_face[
        "face_crop_quality"
    ] = crop_quality

    # Convenience compatibility field.
    enriched_face[
        "crop_blur_score"
    ] = crop_quality.get(
        "blur_score"
    )

    return enriched_face


# ============================================================
# 5. MAIN MULTI-FACE REPORT ANALYSIS
# ============================================================

def analyze_image_for_report(
    image_path,
):
    """
    Analyze one uploaded image before model inference.

    The stable report analysis remains responsible for the
    already validated file-level forensic indicators.

    This experimental layer adds all distinct detected faces
    and a crop-quality block for every face.

    Returns
    -------
    dict
        Report-analysis result containing:

        - file-level indicators from the stable pipeline;
        - faces_detected;
        - faces: list of distinct facial crops;
        - compatibility fields for the primary face;
        - status ready_for_classification when at least one
          valid face is available.
    """

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Stable analysis
    # --------------------------------------------------------
    #
    # Reuse the validated implementation for:
    # - SHA-256
    # - EXIF
    # - image information
    # - blur indicator
    # - JPEG information / blockiness
    # - existing compatibility fields
    #
    stable_result = (
        _STABLE_REPORT_ANALYSIS
        .analyze_image_for_report(
            image_path=str(
                image_path
            )
        )
    )

    if not isinstance(
        stable_result,
        dict,
    ):

        stable_result = {}

    result = dict(
        stable_result
    )

    # --------------------------------------------------------
    # Multi-face detection
    # --------------------------------------------------------

    face_detection = (
        analyze_faces(
            image_path=str(
                image_path
            )
        )
    )

    raw_faces = (
        face_detection.get(
            "faces",
            [],
        )
    )

    faces = [
        enrich_face_record(
            face
        )
        for face in raw_faces
        if (
            face.get(
                "face_crop"
            )
            is not None
        )
    ]

    # Re-number after filtering invalid crops.
    for index, face in enumerate(
        faces,
        start=1,
    ):

        face[
            "face_id"
        ] = index

    # --------------------------------------------------------
    # No valid face
    # --------------------------------------------------------

    if not faces:

        result.update(
            {
                "status": (
                    "input_not_analyzable"
                ),
                "face_detected": False,
                "faces_detected": 0,
                "faces": [],
                "detections_considered": (
                    face_detection.get(
                        "detections_considered",
                        0,
                    )
                ),
                "angles_tested": (
                    face_detection.get(
                        "angles_tested",
                        [],
                    )
                ),

                # Single-face compatibility
                "detection_score": None,
                "bounding_box": None,
                "crop_coordinates": None,
                "face_crop": None,
                "rotation_applied": None,
                "face_crop_quality": {
                    "blur_score": None,
                    "crop_blur_score": None,
                    "width": None,
                    "height": None,
                },
                "crop_blur_score": None,
                "face_crop_blur_score": None,
            }
        )

        return result

    # --------------------------------------------------------
    # At least one valid face
    # --------------------------------------------------------

    primary_face = faces[0]

    primary_crop_quality = (
        primary_face[
            "face_crop_quality"
        ]
    )

    result.update(
        {
            "status": (
                "ready_for_classification"
            ),
            "face_detected": True,
            "faces_detected": len(
                faces
            ),
            "faces": faces,
            "detections_considered": (
                face_detection.get(
                    "detections_considered",
                    0,
                )
            ),
            "angles_tested": (
                face_detection.get(
                    "angles_tested",
                    [],
                )
            ),

            # ------------------------------------------------
            # Single-face compatibility
            # ------------------------------------------------
            #
            # Existing code can still read the largest face
            # from the top-level fields.
            #
            "detection_score": (
                primary_face.get(
                    "detection_score"
                )
            ),
            "bounding_box": (
                primary_face.get(
                    "bounding_box"
                )
            ),
            "crop_coordinates": (
                primary_face.get(
                    "crop_coordinates"
                )
            ),
            "face_crop": (
                primary_face.get(
                    "face_crop"
                )
            ),
            "rotation_applied": (
                primary_face.get(
                    "rotation_applied"
                )
            ),
            "face_crop_quality": (
                primary_crop_quality
            ),
            "crop_blur_score": (
                primary_crop_quality.get(
                    "blur_score"
                )
            ),
            "face_crop_blur_score": (
                primary_crop_quality.get(
                    "blur_score"
                )
            ),
        }
    )

    return result


# ============================================================
# 6. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    result = (
        analyze_image_for_report(
            image_path
        )
    )

    print(
        "=" * 70
    )

    print(
        "MULTI-FACE REPORT ANALYSIS TEST"
    )

    print(
        "=" * 70
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
        "Candidate detections:",
        result.get(
            "detections_considered"
        ),
    )

    for face in result.get(
        "faces",
        [],
    ):

        print(
            "\n"
            f"Face {face['face_id']}"
        )

        print(
            "  detection score:",
            face.get(
                "detection_score"
            ),
        )

        print(
            "  rotation:",
            face.get(
                "rotation_applied"
            ),
        )

        print(
            "  blur score:",
            face.get(
                "face_crop_quality",
                {},
            ).get(
                "blur_score"
            ),
        )

        face_crop = face.get(
            "face_crop"
        )

        print(
            "  crop shape:",
            (
                face_crop.shape
                if face_crop
                is not None
                else None
            ),
        )


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
