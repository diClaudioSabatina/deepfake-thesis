"""
report_face_detection.py

Experimental multi-face face detection for the technical report.

Main goals:
- detect all distinct faces in an uploaded image;
- try multiple image rotations to handle tilted faces;
- map detections back to the original image coordinates;
- deduplicate the same face detected at different rotations;
- keep, for each distinct face, the crop obtained from the
  orientation with the best YuNet detection score.

This module does NOT modify the dataset and does NOT overwrite
the stable single-face pipeline.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Needed when this file is executed directly from src_multiface.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from crop_faces import (
    YUNET_MODEL_PATH,
    NMS_THRESHOLD,
    TOP_K,
    calculate_crop_coordinates,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# The stable dataset preprocessing uses SCORE_THRESHOLD = 0.8.
# In the experimental multi-face REPORT only, candidate generation
# is slightly more permissive so that partially occluded / tilted /
# background faces are less likely to be missed.
#
# This does NOT alter training, validation or test preprocessing.
REPORT_SCORE_THRESHOLD = 0.60


def create_report_face_detector():
    """
    Create the YuNet detector for the experimental multi-face report.

    The only intentional difference from the stable detector is the
    lower candidate score threshold.
    """

    return cv2.FaceDetectorYN.create(
        model=str(
            YUNET_MODEL_PATH
        ),
        config="",
        input_size=(
            320,
            320,
        ),
        score_threshold=(
            REPORT_SCORE_THRESHOLD
        ),
        nms_threshold=(
            NMS_THRESHOLD
        ),
        top_k=TOP_K,
    )


# Positive values rotate counterclockwise.
# Negative values rotate clockwise.
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

# Detections mapped back to the original image are considered
# duplicates when their bounding boxes overlap sufficiently.
DUPLICATE_IOU_THRESHOLD = 0.30

# Additional duplicate check used when two boxes shift slightly
# because they were obtained from different rotations.
CENTER_DISTANCE_FACTOR = 0.30
MIN_AREA_RATIO_FOR_CENTER_MATCH = 0.40


# ============================================================
# 3. ROTATION
# ============================================================

def rotate_image_with_matrix(
    image,
    angle,
):
    """
    Rotate an image by an arbitrary angle while expanding
    the canvas so that image content is not cut.

    Returns
    -------
    tuple
        rotated_image, rotation_matrix
    """

    image_height, image_width = (
        image.shape[:2]
    )

    if angle == 0:

        identity_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        return (
            image.copy(),
            identity_matrix,
        )

    center = (
        image_width / 2.0,
        image_height / 2.0,
    )

    rotation_matrix = (
        cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )
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

    return (
        rotated_image,
        rotation_matrix,
    )


# ============================================================
# 4. BOUNDING-BOX UTILITIES
# ============================================================

def bbox_area(bbox):
    """
    Return the area of a bounding box dictionary.
    """

    return max(
        0.0,
        float(bbox["width"])
        * float(bbox["height"]),
    )


def bbox_iou(
    bbox_a,
    bbox_b,
):
    """
    Compute Intersection over Union between two boxes.
    """

    ax1 = float(bbox_a["x"])
    ay1 = float(bbox_a["y"])
    ax2 = ax1 + float(
        bbox_a["width"]
    )
    ay2 = ay1 + float(
        bbox_a["height"]
    )

    bx1 = float(bbox_b["x"])
    by1 = float(bbox_b["y"])
    bx2 = bx1 + float(
        bbox_b["width"]
    )
    by2 = by1 + float(
        bbox_b["height"]
    )

    intersection_x1 = max(
        ax1,
        bx1,
    )

    intersection_y1 = max(
        ay1,
        by1,
    )

    intersection_x2 = min(
        ax2,
        bx2,
    )

    intersection_y2 = min(
        ay2,
        by2,
    )

    intersection_width = max(
        0.0,
        intersection_x2
        - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2
        - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = bbox_area(
        bbox_a
    )

    area_b = bbox_area(
        bbox_b
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def bbox_overlap_over_smaller(
    bbox_a,
    bbox_b,
):
    """
    Return the intersection area divided by the area of the
    smaller box.

    This is useful when the same physical face is detected with
    slightly different extents at different rotations: IoU can be
    modest even though most of one box lies inside the other.
    """

    ax1 = float(bbox_a["x"])
    ay1 = float(bbox_a["y"])
    ax2 = ax1 + float(bbox_a["width"])
    ay2 = ay1 + float(bbox_a["height"])

    bx1 = float(bbox_b["x"])
    by1 = float(bbox_b["y"])
    bx2 = bx1 + float(bbox_b["width"])
    by2 = by1 + float(bbox_b["height"])

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)

    smaller_area = min(
        area_a,
        area_b,
    )

    if smaller_area <= 0:
        return 0.0

    return (
        intersection_area
        / smaller_area
    )


def bbox_center(
    bbox,
):
    """
    Return the center of a bounding box.
    """

    center_x = (
        float(bbox["x"])
        + float(bbox["width"]) / 2.0
    )

    center_y = (
        float(bbox["y"])
        + float(bbox["height"]) / 2.0
    )

    return (
        center_x,
        center_y,
    )


def boxes_represent_same_face(
    bbox_a,
    bbox_b,
):
    """
    Decide whether two boxes likely represent the same physical face.

    The comparison is performed only after both detections have been
    mapped back to the ORIGINAL image coordinate system.

    Rules:
    1. standard IoU overlap;
    2. close centers + reasonably similar area;
    3. strong containment/overlap of the smaller box;
    4. moderate center shift for similarly sized boxes.

    Rules 3-4 specifically address the case in which the same face is
    detected at different rotations with one box shifted toward the
    upper or lower part of the face.
    """

    iou = bbox_iou(
        bbox_a,
        bbox_b,
    )

    if (
        iou
        >= DUPLICATE_IOU_THRESHOLD
    ):
        return True

    center_a = bbox_center(
        bbox_a
    )

    center_b = bbox_center(
        bbox_b
    )

    center_distance = float(
        np.hypot(
            center_a[0] - center_b[0],
            center_a[1] - center_b[1],
        )
    )

    diagonal_a = float(
        np.hypot(
            float(bbox_a["width"]),
            float(bbox_a["height"]),
        )
    )

    diagonal_b = float(
        np.hypot(
            float(bbox_b["width"]),
            float(bbox_b["height"]),
        )
    )

    area_a = bbox_area(
        bbox_a
    )

    area_b = bbox_area(
        bbox_b
    )

    if (
        area_a <= 0
        or area_b <= 0
        or diagonal_a <= 0
        or diagonal_b <= 0
    ):
        return False

    area_ratio = (
        min(
            area_a,
            area_b,
        )
        / max(
            area_a,
            area_b,
        )
    )

    # Original conservative center rule.
    reference_diagonal = max(
        diagonal_a,
        diagonal_b,
    )

    if (
        center_distance
        <= CENTER_DISTANCE_FACTOR
        * reference_diagonal
        and
        area_ratio
        >= MIN_AREA_RATIO_FOR_CENTER_MATCH
    ):
        return True

    # If most of the smaller detection overlaps the larger one,
    # they are very likely to be the same physical face even when
    # the two bounding boxes have different extents.
    overlap_smaller = (
        bbox_overlap_over_smaller(
            bbox_a,
            bbox_b,
        )
    )

    if (
        overlap_smaller >= 0.55
        and area_ratio >= 0.30
    ):
        return True

    # Cross-rotation detections of the same face can move vertically
    # (for example full-face vs lower-face box).  For similarly sized
    # boxes we therefore allow a moderate center displacement.
    smaller_diagonal = min(
        diagonal_a,
        diagonal_b,
    )

    if (
        area_ratio >= 0.60
        and center_distance
        <= 0.50
        * smaller_diagonal
    ):
        return True

    return False


# ============================================================
# 5. MAP A ROTATED BOX TO ORIGINAL COORDINATES
# ============================================================

def map_bbox_to_original(
    bbox,
    rotation_matrix,
    original_width,
    original_height,
):
    """
    Map a bounding box detected on a rotated image back
    to the coordinate system of the original image.

    The four corners are transformed with the inverse affine
    matrix and then enclosed in an axis-aligned rectangle.
    """

    x = float(
        bbox["x"]
    )

    y = float(
        bbox["y"]
    )

    width = float(
        bbox["width"]
    )

    height = float(
        bbox["height"]
    )

    corners = np.array(
        [
            [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
            ]
        ],
        dtype=np.float32,
    )

    inverse_matrix = (
        cv2.invertAffineTransform(
            rotation_matrix
        )
    )

    original_corners = cv2.transform(
        corners,
        inverse_matrix,
    )[0]

    min_x = float(
        np.min(
            original_corners[:, 0]
        )
    )

    min_y = float(
        np.min(
            original_corners[:, 1]
        )
    )

    max_x = float(
        np.max(
            original_corners[:, 0]
        )
    )

    max_y = float(
        np.max(
            original_corners[:, 1]
        )
    )

    min_x = max(
        0.0,
        min(
            min_x,
            float(original_width),
        ),
    )

    min_y = max(
        0.0,
        min(
            min_y,
            float(original_height),
        ),
    )

    max_x = max(
        0.0,
        min(
            max_x,
            float(original_width),
        ),
    )

    max_y = max(
        0.0,
        min(
            max_y,
            float(original_height),
        ),
    )

    mapped_width = max(
        0.0,
        max_x - min_x,
    )

    mapped_height = max(
        0.0,
        max_y - min_y,
    )

    return {
        "x": min_x,
        "y": min_y,
        "width": mapped_width,
        "height": mapped_height,
    }


# ============================================================
# 6. DETECT ALL FACES AT ONE ORIENTATION
# ============================================================

def detect_faces_on_rotated_image(
    rotated_image,
    detector,
    angle,
    rotation_matrix,
    original_width,
    original_height,
):
    """
    Detect all faces in one rotated version of the image.

    Returns one candidate dictionary for each detected face.
    """

    image_height, image_width = (
        rotated_image.shape[:2]
    )

    detector.setInputSize(
        (
            image_width,
            image_height,
        )
    )

    _, faces = detector.detect(
        rotated_image
    )

    if (
        faces is None
        or len(faces) == 0
    ):
        return []

    candidates = []

    for face in faces:

        x = float(
            face[0]
        )

        y = float(
            face[1]
        )

        width = float(
            face[2]
        )

        height = float(
            face[3]
        )

        detection_score = float(
            face[-1]
        )

        rotated_bbox = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

        original_bbox = (
            map_bbox_to_original(
                bbox=rotated_bbox,
                rotation_matrix=rotation_matrix,
                original_width=original_width,
                original_height=original_height,
            )
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

        face_crop = rotated_image[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        original_area = bbox_area(
            original_bbox
        )

        original_image_area = float(
            original_width
            * original_height
        )

        relative_area = (
            original_area
            / original_image_area
            if original_image_area > 0
            else 0.0
        )

        candidates.append(
            {
                "detection_score": (
                    detection_score
                ),
                "rotation_applied": (
                    angle
                ),
                "bounding_box": (
                    original_bbox
                ),
                "original_bounding_box": (
                    original_bbox
                ),
                "rotated_bounding_box": (
                    rotated_bbox
                ),
                "crop_coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "face_crop": face_crop,
                "relative_face_area": (
                    relative_area
                ),
            }
        )

    return candidates


# ============================================================
# 7. DEDUPLICATION ACROSS ROTATIONS
# ============================================================

def deduplicate_face_candidates(
    candidates,
):
    """
    Merge detections that correspond to the same physical face.

    The previous implementation compared every candidate only with
    the current representative of a cluster.  That can split one face
    into two report entries when bounding boxes drift progressively
    across rotations.

    This version builds connected components:
    if A matches B and B matches C, A/B/C are treated as one physical
    face even when A and C do not directly satisfy the duplicate rule.

    The retained crop is the candidate with the highest YuNet score;
    ties prefer the smallest absolute rotation.
    """

    if not candidates:
        return []

    number_of_candidates = len(
        candidates
    )

    parent = list(
        range(
            number_of_candidates
        )
    )

    def find(index):

        while (
            parent[index]
            != index
        ):

            parent[index] = (
                parent[
                    parent[index]
                ]
            )

            index = parent[
                index
            ]

        return index

    def union(
        index_a,
        index_b,
    ):

        root_a = find(
            index_a
        )

        root_b = find(
            index_b
        )

        if root_a != root_b:
            parent[root_b] = root_a

    # Build duplicate links in ORIGINAL-image coordinates.
    for index_a in range(
        number_of_candidates
    ):

        bbox_a = candidates[
            index_a
        ][
            "original_bounding_box"
        ]

        for index_b in range(
            index_a + 1,
            number_of_candidates,
        ):

            bbox_b = candidates[
                index_b
            ][
                "original_bounding_box"
            ]

            if boxes_represent_same_face(
                bbox_a,
                bbox_b,
            ):

                union(
                    index_a,
                    index_b,
                )

    clusters = {}

    for index, candidate in enumerate(
        candidates
    ):

        root = find(
            index
        )

        clusters.setdefault(
            root,
            [],
        ).append(
            candidate
        )

    unique_faces = []

    for members in clusters.values():

        representative = max(
            members,
            key=lambda item: (
                float(
                    item[
                        "detection_score"
                    ]
                ),
                -abs(
                    float(
                        item[
                            "rotation_applied"
                        ]
                    )
                ),
            ),
        )

        representative = dict(
            representative
        )

        support_angles = sorted(
            {
                float(
                    member[
                        "rotation_applied"
                    ]
                )
                for member in members
            }
        )

        representative[
            "support_angles"
        ] = support_angles

        representative[
            "rotation_support_count"
        ] = len(
            support_angles
        )

        unique_faces.append(
            representative
        )

    # Face 1 remains the largest distinct face in the original image.
    unique_faces.sort(
        key=lambda item: (
            -float(
                item[
                    "relative_face_area"
                ]
            ),
            -float(
                item[
                    "detection_score"
                ]
            ),
        )
    )

    for index, face in enumerate(
        unique_faces,
        start=1,
    ):

        face[
            "face_id"
        ] = index

    return unique_faces


# ============================================================
# 8. MAIN MULTI-FACE ANALYSIS
# ============================================================

def analyze_faces(
    image_path,
    detector=None,
):
    """
    Detect all distinct faces in an image.

    Detection is performed on several rotated versions of the
    same image. Detections are mapped to original coordinates
    and deduplicated across rotations.

    Returns
    -------
    dict
        Multi-face detection result.
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
            "faces": [],
            "detections_considered": 0,
            "angles_tested": (
                ROTATION_ANGLES.copy()
            ),

            # Compatibility fields
            "detection_score": None,
            "bounding_box": None,
            "crop_coordinates": None,
            "face_crop": None,
            "rotation_applied": None,
        }

    original_height, original_width = (
        image.shape[:2]
    )

    if detector is None:
        detector = create_report_face_detector()

    all_candidates = []

    for angle in ROTATION_ANGLES:

        (
            rotated_image,
            rotation_matrix,
        ) = rotate_image_with_matrix(
            image,
            angle,
        )

        angle_candidates = (
            detect_faces_on_rotated_image(
                rotated_image=rotated_image,
                detector=detector,
                angle=angle,
                rotation_matrix=rotation_matrix,
                original_width=original_width,
                original_height=original_height,
            )
        )

        all_candidates.extend(
            angle_candidates
        )

    unique_faces = (
        deduplicate_face_candidates(
            all_candidates
        )
    )

    if not unique_faces:

        return {
            "face_detected": False,
            "faces_detected": 0,
            "faces": [],
            "detections_considered": (
                len(
                    all_candidates
                )
            ),
            "angles_tested": (
                ROTATION_ANGLES.copy()
            ),

            # Compatibility fields
            "detection_score": None,
            "bounding_box": None,
            "crop_coordinates": None,
            "face_crop": None,
            "rotation_applied": None,
        }

    # Compatibility with the existing single-face code:
    # the first face is the largest distinct face.
    primary_face = (
        unique_faces[0]
    )

    return {
        "face_detected": True,
        "faces_detected": len(
            unique_faces
        ),
        "faces": unique_faces,
        "detections_considered": len(
            all_candidates
        ),
        "angles_tested": (
            ROTATION_ANGLES.copy()
        ),

        # Compatibility fields
        "detection_score": (
            primary_face[
                "detection_score"
            ]
        ),
        "bounding_box": (
            primary_face[
                "bounding_box"
            ]
        ),
        "crop_coordinates": (
            primary_face[
                "crop_coordinates"
            ]
        ),
        "face_crop": (
            primary_face[
                "face_crop"
            ]
        ),
        "rotation_applied": (
            primary_face[
                "rotation_applied"
            ]
        ),
    }


# ============================================================
# 9. SINGLE-FACE COMPATIBILITY WRAPPER
# ============================================================

def analyze_face(
    image_path,
    detector=None,
):
    """
    Compatibility wrapper.

    Existing modules that still call analyze_face() receive
    the same top-level structure as before, while the returned
    dictionary also contains the complete "faces" list.

    The top-level face corresponds to the largest distinct face.
    """

    return analyze_faces(
        image_path=image_path,
        detector=detector,
    )


# ============================================================
# 10. TEST
# ============================================================

def main():

    image_path = (
        r"C:\Users\dicla\Desktop\100_1980.JPG"
    )

    result = analyze_faces(
        image_path
    )

    print(
        "=" * 70
    )

    print(
        "MULTI-FACE DETECTION TEST"
    )

    print(
        "=" * 70
    )

    print(
        "Face detected:",
        result[
            "face_detected"
        ],
    )

    print(
        "Distinct faces:",
        result[
            "faces_detected"
        ],
    )

    print(
        "Candidate detections:",
        result[
            "detections_considered"
        ],
    )

    for face in result[
        "faces"
    ]:

        print(
            "\n"
            f"Face {face['face_id']}"
        )

        print(
            "  score:",
            face[
                "detection_score"
            ],
        )

        print(
            "  rotation:",
            face[
                "rotation_applied"
            ],
        )

        print(
            "  original bbox:",
            face[
                "original_bounding_box"
            ],
        )

        print(
            "  relative area:",
            face[
                "relative_face_area"
            ],
        )

        print(
            "  crop shape:",
            face[
                "face_crop"
            ].shape,
        )


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
