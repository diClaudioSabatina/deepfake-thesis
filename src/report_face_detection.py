"""
report_face_detection.py

Face detection used during the analysis of a single image
for the technical report.

The function does NOT modify the dataset and does NOT
automatically save the crop to disk.

Updated version:
- attempts face detection at multiple image orientations;
- supports arbitrary rotations instead of only 90° steps;
- expands the canvas during rotation to avoid cutting image content;
- selects the valid detection with the highest confidence score.
"""

from pathlib import Path

import cv2

from crop_faces import (
    create_face_detector,
    select_main_face,
    calculate_crop_coordinates,
)


# ============================================================
# 1. ROTATION ANGLES
# ============================================================

# Angles tested during face detection.
#
# Positive values rotate the image counterclockwise.
# Negative values rotate it clockwise.
ROTATION_ANGLES = [
    0,
    -15,
    15,
    -30,
    30,
    -45,
    45,
    -60,
    60,
    -75,
    75,
    -90,
    90,
    180,
]


# ============================================================
# 2. IMAGE ROTATION
# ============================================================

def rotate_image(image, angle):
    """
    Rotate an image by an arbitrary angle while expanding
    the canvas so that image content is not cut.

    Positive angles rotate counterclockwise.
    Negative angles rotate clockwise.
    """

    if angle == 0:
        return image.copy()

    image_height, image_width = image.shape[:2]

    center = (
        image_width / 2.0,
        image_height / 2.0,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos_value = abs(
        rotation_matrix[0, 0]
    )

    sin_value = abs(
        rotation_matrix[0, 1]
    )

    new_width = int(
        round(
            image_height * sin_value
            + image_width * cos_value
        )
    )

    new_height = int(
        round(
            image_height * cos_value
            + image_width * sin_value
        )
    )

    # Re-center the rotated image inside the expanded canvas.
    rotation_matrix[0, 2] += (
        new_width / 2.0
        - center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2.0
        - center[1]
    )

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (
            new_width,
            new_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return rotated_image


# ============================================================
# 3. FACE DETECTION ON ONE ORIENTATION
# ============================================================

def analyze_face_on_rotated_image(
    image,
    detector,
):
    """
    Run face detection on one image orientation.

    If more than one face is detected, the main face is selected
    using the same rule already used by crop_faces.py.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    detector.setInputSize(
        (
            image_width,
            image_height,
        )
    )

    _, faces = detector.detect(
        image
    )

    if faces is None or len(faces) == 0:
        return None

    number_of_faces = len(
        faces
    )

    main_face = select_main_face(
        faces
    )

    x = float(main_face[0])
    y = float(main_face[1])
    width = float(main_face[2])
    height = float(main_face[3])

    detection_score = float(
        main_face[-1]
    )

    x1, y1, x2, y2 = (
        calculate_crop_coordinates(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
    )

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None

    return {
        "face_detected": True,
        "faces_detected": number_of_faces,
        "detection_score": detection_score,
        "bounding_box": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "crop_coordinates": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "face_crop": face_crop,
    }


# ============================================================
# 4. MAIN ANALYSIS
# ============================================================

def analyze_face(
    image_path,
    detector=None,
):
    """
    Detect the main face in an image.

    Face detection is attempted on multiple rotated versions
    of the input image. Among all valid detections, the result
    with the highest YuNet detection score is retained.

    Returns
    -------
    dict
        Face detection information, selected crop and
        rotation applied before detection.
    """

    image_path = Path(
        image_path
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        return {
            "face_detected": False,
            "faces_detected": 0,
            "detection_score": None,
            "bounding_box": None,
            "crop_coordinates": None,
            "face_crop": None,
            "rotation_applied": None,
        }

    if detector is None:
        detector = create_face_detector()

    best_result = None
    best_area = -1.0
    best_score = -1.0

    # --------------------------------------------------------
    # Test multiple orientations
    # --------------------------------------------------------

    for angle in ROTATION_ANGLES:

        rotated_image = rotate_image(
            image,
            angle,
        )

        result = (
            analyze_face_on_rotated_image(
                rotated_image,
                detector,
            )
        )

        if result is None:
            continue

        result[
            "rotation_applied"
        ] = angle

        score = result[
            "detection_score"
        ]

        bbox = result[
            "bounding_box"
        ]

        area = (
            bbox["width"]
            * bbox["height"]
        )

        if (
            area > best_area
            or (
                area == best_area
                and score > best_score
            )
        ):

            best_area = area
            best_score = score
            best_result = result

    # --------------------------------------------------------
    # No valid face found at any tested orientation
    # --------------------------------------------------------

    if best_result is None:

        return {
            "face_detected": False,
            "faces_detected": 0,
            "detection_score": None,
            "bounding_box": None,
            "crop_coordinates": None,
            "face_crop": None,
            "rotation_applied": None,
        }

    return best_result


# ============================================================
# 5. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    result = analyze_face(
        image_path
    )

    print("=" * 60)
    print("FACE DETECTION REPORT TEST")
    print("=" * 60)

    print(
        "Face detected:",
        result["face_detected"]
    )

    print(
        "Faces detected:",
        result["faces_detected"]
    )

    print(
        "Detection score:",
        result["detection_score"]
    )

    print(
        "Bounding box:",
        result["bounding_box"]
    )

    print(
        "Crop coordinates:",
        result["crop_coordinates"]
    )

    print(
        "Applied rotation:",
        result["rotation_applied"]
    )

    if result["face_crop"] is not None:

        print(
            "Crop shape:",
            result["face_crop"].shape
        )

    else:

        print(
            "Face crop not available."
        )


# ============================================================
# 6. TEST ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
