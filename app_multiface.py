"""
app_multiface.py

Experimental Streamlit interface for the multi-face version
of the system supporting human assessment of potentially deepfake images.

The application:
- detects all distinct faces in the uploaded image;
- reports the number of detected faces;
- presents one complete technical report for Face 1, Face 2, ..., Face N;
- keeps file-level forensic information separate from face-level results.

The system does NOT provide an automatic certification
of content authenticity.
"""

from pathlib import Path
import sys
import tempfile
import html
from textwrap import dedent

import numpy as np
from PIL import Image
import streamlit as st

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
SRC_MULTIFACE_DIR = PROJECT_ROOT / "src_multiface"

# Stable modules remain available.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

# Experimental multi-face modules have priority.
if str(SRC_MULTIFACE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_MULTIFACE_DIR),
    )

import report_pipeline

print(
    "[APP MULTIFACE] report_pipeline loaded from:",
    report_pipeline.__file__,
)

load_pipeline_models = (
    report_pipeline.load_pipeline_models
)

run_report_pipeline = (
    report_pipeline.run_report_pipeline
)

from report_export import (
    generate_pdf_report,
    compute_sha256_bytes,
    build_report_filename,
)

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 3. STYLE
# ============================================================

st.markdown(
    dedent("""

    <style>

    .stApp {
        background: #f4f6f8;
        color: #1f2933;
    }

    /* Use Streamlit's existing white top header for the report title.
       No additional title band is created in the page body. */
    header[data-testid="stHeader"] {
        background: #ffffff;
        border-bottom: 1px solid #d7dde5;
    }

    header[data-testid="stHeader"]::before {
        content: "Deepfake Analysis – Technical Support Report";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        max-width: 68vw;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #172033;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        pointer-events: none;
        z-index: 1;
    }

    /* Hide Streamlit's native running-status animation/icon.
       The application uses its own neutral rotating loader instead. */
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            Inter,
            "Segoe UI",
            Arial,
            sans-serif;
    }

    h1,
    h2,
    h3 {
        color: #172033;
        letter-spacing: -0.01em;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 750 !important;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    p,
    li {
        line-height: 1.6;
    }

    .subtitle {
        color: #5c6675;
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1.6rem;
        font-weight: 500;
    }

    .section-title {
        font-size: 1.18rem;
        font-weight: 750;
        color: #172033;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        padding: 0.65rem 0.9rem;
        border-left: 5px solid #243b5a;
        border-bottom: 1px solid #cfd6df;
        background: #e9edf2;
        border-radius: 4px 4px 0 0;
        letter-spacing: 0.01em;
    }

    .face-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 3rem;
        margin-bottom: 0;
        padding: 0.8rem 1rem;
        background: #243b5a;
        border: 1px solid #1d3049;
        border-radius: 6px 6px 0 0;
    }

    .face-subtitle {
        color: #4f5b6b;
        margin-bottom: 1.3rem;
        padding: 0.7rem 1rem;
        background: #eef1f5;
        border-left: 1px solid #cbd3dd;
        border-right: 1px solid #cbd3dd;
        border-bottom: 1px solid #cbd3dd;
        border-radius: 0 0 6px 6px;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #c7d0db;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        min-height: 108px;
        box-shadow: 0 1px 2px rgba(28, 39, 54, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #53606f;
        font-size: 0.82rem;
        font-weight: 650;
        letter-spacing: 0.01em;
    }

    div[data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 750;
        font-size: 1.55rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 5px;
        border-width: 1px;
        box-shadow: none;
    }

    details {
        background: #ffffff;
        border: 1px solid #c7d0db !important;
        border-radius: 5px !important;
        margin-top: 0.75rem;
    }

    details > summary {
        font-weight: 700;
        color: #28384e;
        background: #eef1f5;
        padding: 0.75rem 0.9rem !important;
    }

    .technical-note {
        border: 1px solid #c5ced8;
        border-left: 5px solid #526a86;
        padding: 1rem 1.1rem;
        background: #f7f9fb;
        border-radius: 5px;
        margin-top: 1rem;
        color: #364152;
        line-height: 1.6;
    }

    .hash-box {
        font-family:
            "Cascadia Mono",
            "Consolas",
            monospace;
        font-size: 0.82rem;
        overflow-wrap: anywhere;
        border: 1px solid #aeb9c5;
        border-radius: 4px;
        padding: 0.9rem;
        background: #ffffff;
        color: #1f2933;
    }

    pre,
    code {
        font-family:
            "Cascadia Mono",
            "Consolas",
            monospace !important;
    }

    div[data-testid="stImage"] img {
        border: 1px solid #c7d0db;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(28, 39, 54, 0.08);
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #c7d0db;
        border-radius: 6px;
        padding: 0.8rem;
    }

    div[data-testid="stProgress"] > div > div > div {
        border-radius: 2px;
    }

    .report-loader {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: #ffffff;
        color: #28384e;
        border: 1px solid #c7d0db;
        border-left: 5px solid #243b5a;
        border-radius: 5px;
        padding: 0.85rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        font-weight: 650;
        box-shadow: 0 1px 2px rgba(28, 39, 54, 0.04);
    }

    .report-spinner {
        width: 22px;
        height: 22px;
        flex: 0 0 22px;
        border: 3px solid #d9e0e8;
        border-top-color: #243b5a;
        border-radius: 50%;
        animation: report-spin 0.8s linear infinite;
    }

    @keyframes report-spin {
        from {
            transform: rotate(0deg);
        }

        to {
            transform: rotate(360deg);
        }
    }

    hr {
        border: none;
        border-top: 1px solid #bcc6d1;
        margin-top: 2.2rem;
        margin-bottom: 2.2rem;
    }

    button {
        border-radius: 4px !important;
        font-weight: 650 !important;
    }

    div[data-testid="stJson"] {
        background: #ffffff;
        border: 1px solid #c7d0db;
        border-radius: 5px;
        padding: 0.4rem;
    }

    .stCaption,
    div[data-testid="stCaptionContainer"] {
        color: #657180 !important;
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
    }

    @media (max-width: 900px) {

        header[data-testid="stHeader"]::before {
            font-size: 0.95rem;
            max-width: 62vw;
        }

        .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        .face-title {
            font-size: 1.25rem;
        }

        div[data-testid="stMetric"] {
            min-height: auto;
        }
    }

    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# 4. REFERENCE CONSTANTS
# ============================================================

# Distribution observed on original FaceForensics++ test-set crops.
# These values are descriptive references, not classification thresholds.

BLUR_REFERENCE = {
    "q1": 48.1727,
    "median": 98.1901,
    "q3": 197.2486,
}


# ============================================================
# 5. MODEL CACHE
# ============================================================

@st.cache_resource
def get_models():
    """
    Load Xception and EfficientNet-B4 only once
    during the Streamlit session.
    """

    return load_pipeline_models()


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def bgr_to_rgb(image):
    """
    Convert an OpenCV BGR image to RGB.
    """

    if image is None:
        return None

    if not isinstance(
        image,
        np.ndarray,
    ):
        return image

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        return image[:, :, ::-1]

    return image
def resize_image_for_display(
    image,
    max_width=1100,
    max_height=550,
):
    """
    Resize an image only for display in the Streamlit interface.

    The original image used by the analysis pipeline is not modified.
    The aspect ratio is preserved.
    """

    if image is None:
        return None

    display_image = image.copy()

    display_image.thumbnail(
        (
            max_width,
            max_height,
        ),
        Image.Resampling.LANCZOS,
    )

    return display_image

def safe_number(
    value,
    decimals=4,
):
    """
    Format a numeric value safely.
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def safe_percent(value):
    """
    Convert a 0-1 value to percentage text.
    """

    if value is None:
        return "N/A"

    try:
        return (
            f"{float(value) * 100:.2f}%"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def first_available(
    dictionary,
    *keys,
    default=None,
):
    """
    Return the first available non-None dictionary value.
    """

    if not isinstance(
        dictionary,
        dict,
    ):
        return default

    for key in keys:

        if (
            key in dictionary
            and dictionary[key]
            is not None
        ):
            return dictionary[key]

    return default


def format_rotation(angle):
    """
    Convert the rotation used for face detection
    into a readable label.

    Positive angles are counterclockwise.
    Negative angles are clockwise.
    """

    if angle is None:
        return "N/A"

    try:
        angle = float(angle)

    except (
        TypeError,
        ValueError,
    ):
        return str(angle)

    if abs(angle) < 1e-9:
        return "None"

    if angle.is_integer():
        angle_text = str(
            int(
                abs(angle)
            )
        )

    else:
        angle_text = (
            f"{abs(angle):.1f}"
            .rstrip("0")
            .rstrip(".")
        )

    if angle > 0:
        return (
            f"{angle_text}° counterclockwise"
        )

    return (
        f"{angle_text}° clockwise"
    )


def describe_blur_score(score):
    """
    Place the face blur score relative to the
    reference distribution.

    This is descriptive only.
    """

    if score is None:
        return (
            "Reference not available."
        )

    score = float(score)

    q1 = BLUR_REFERENCE["q1"]
    median = BLUR_REFERENCE["median"]
    q3 = BLUR_REFERENCE["q3"]

    if score < q1:

        return (
            "Value below the 25th percentile "
            "of the reference distribution."
        )

    if score < median:

        return (
            "Value between the 25th percentile "
            "and the median of the reference distribution."
        )

    if score < q3:

        return (
            "Value between the median and the "
            "75th percentile of the reference distribution."
        )

    return (
        "Value above the 75th percentile "
        "of the reference distribution."
    )


def get_file_info(result):
    """
    Return file-level forensic information.
    """

    return (
        result.get(
            "file_indicators"
        )
        or result.get(
            "forensic_indicators"
        )
        or result
    )


def get_crop_quality(face):
    """
    Return crop-quality information for one face.
    """

    return (
        face.get(
            "face_crop_quality"
        )
        or face.get(
            "crop_quality"
        )
        or {}
    )


def show_custom_loader(message):
    """
    Display a formal rotating loader and return the placeholder
    so it can be removed when the operation is complete.
    """

    placeholder = st.empty()

    placeholder.markdown(
        (
            '<div class="report-loader">'
            '<div class="report-spinner"></div>'
            f'<div>{html.escape(str(message))}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    return placeholder


# ============================================================
# 7. DETECTOR CARD
# ============================================================

def display_detector_card(
    model_name,
    detector_result,
):
    """
    Display one detector result using native Streamlit components.
    """

    prediction = first_available(
        detector_result,
        "predicted_label",
        default="N/A",
    )

    confidence = first_available(
        detector_result,
        "calibrated_confidence",
    )

    threshold = first_available(
        detector_result,
        "threshold",
        "abstention_threshold",
    )

    selective_status = first_available(
        detector_result,
        "selective_status",
        default="N/A",
    )

    temperature = first_available(
        detector_result,
        "temperature",
    )

    real_probability = first_available(
        detector_result,
        "calibrated_real_probability",
        "real_probability_calibrated",
    )

    fake_probability = first_available(
        detector_result,
        "calibrated_fake_probability",
        "fake_probability_calibrated",
    )

    raw_confidence = first_available(
        detector_result,
        "raw_confidence",
        "uncalibrated_confidence",
    )

    display_name = (
        "EfficientNet-B4"
        if model_name
        == "efficientnet_b4"
        else "Xception"
    )

    with st.container(
        border=True
    ):

        st.subheader(
            display_name
        )

        st.caption(
            "Detector prediction"
        )

        st.markdown(
            f"## {html.escape(str(prediction))}"
        )

        metric_col1, metric_col2 = (
            st.columns(2)
        )

        with metric_col1:

            st.metric(
                "Calibrated confidence",
                safe_percent(
                    confidence
                ),
            )

        with metric_col2:

            st.metric(
                "Selective threshold",
                safe_percent(
                    threshold
                ),
            )

        if (
            selective_status
            == "ABSTAIN"
        ):

            st.warning(
                "ABSTAIN — human review recommended"
            )

        elif (
            selective_status
            == "ACCEPT"
        ):

            st.success(
                "ACCEPT"
            )

        else:

            st.info(
                str(
                    selective_status
                )
            )

        if confidence is not None:

            progress_value = int(
                max(
                    0,
                    min(
                        100,
                        round(
                            float(
                                confidence
                            )
                            * 100
                        ),
                    ),
                )
            )

            st.progress(
                progress_value
            )

        with st.expander(
            "Technical details"
        ):

            if (
                real_probability
                is not None
            ):

                st.write(
                    "Calibrated REAL probability:",
                    safe_percent(
                        real_probability
                    ),
                )

            if (
                fake_probability
                is not None
            ):

                st.write(
                    "Calibrated FAKE probability:",
                    safe_percent(
                        fake_probability
                    ),
                )

            if (
                raw_confidence
                is not None
            ):

                st.write(
                    "Uncalibrated confidence:",
                    safe_percent(
                        raw_confidence
                    ),
                )

            if temperature is not None:

                st.write(
                    "Temperature:",
                    safe_number(
                        temperature,
                        4,
                    ),
                )

            st.caption(
                "ACCEPT only means that the calibrated confidence "
                "exceeds the experimentally defined selective threshold. "
                "It does not certify that the prediction is correct."
            )

            st.caption(
                "Prediction relative to the distribution learned from "
                "FaceForensics++. This output should not be interpreted "
                "as a general detector of AI-generated or synthetic images."
            )


# ============================================================
# 8. ONE FACE REPORT
# ============================================================

def display_face_report(
    face,
    face_number,
):
    """
    Display the complete report for one detected face.
    """

    st.divider()

    st.markdown(
        (
            '<div class="face-title">'
            f"Face {face_number}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="face-subtitle">'
            f"Individual analysis for detected face {face_number}."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FACE CROP + DETECTION INFORMATION
    # --------------------------------------------------------

    crop_column, info_column = (
        st.columns(
            [1, 1.35]
        )
    )

    face_crop = face.get(
        "face_crop"
    )

    with crop_column:

        st.subheader(
            "Analyzed face"
        )

        if face_crop is not None:

            st.image(
                bgr_to_rgb(
                    face_crop
                ),
                width="stretch",
            )

        else:

            st.info(
                "Face crop unavailable."
            )

    with info_column:

        st.subheader(
            "Face-detection information"
        )

        detection_score = (
            face.get(
                "detection_score"
            )
        )

        rotation_applied = (
            face.get(
                "rotation_applied"
            )
        )

        relative_area = (
            face.get(
                "relative_face_area"
            )
        )

        crop_quality = (
            get_crop_quality(
                face
            )
        )

        crop_blur = first_available(
            crop_quality,
            "blur_score",
            "crop_blur_score",
        )

        i1, i2 = st.columns(2)

        with i1:

            st.metric(
                "Detection score",
                safe_number(
                    detection_score,
                    4,
                ),
            )

        with i2:

            st.metric(
                "Orientation adjustment",
                format_rotation(
                    rotation_applied
                ),
            )

        i3, i4 = st.columns(2)

        with i3:

            st.metric(
                "Face blur score",
                safe_number(
                    crop_blur,
                    2,
                ),
            )

        with i4:

            st.metric(
                "Relative face area",
                (
                    safe_percent(
                        relative_area
                    )
                    if relative_area
                    is not None
                    else "N/A"
                ),
            )

        if crop_blur is not None:

            st.caption(
                describe_blur_score(
                    crop_blur
                )
                + " The comparison is descriptive and does not "
                  "automatically determine whether blur was applied."
            )

    # --------------------------------------------------------
    # CLASSIFICATION FAILURE
    # --------------------------------------------------------

    if (
        face.get(
            "classification_status"
        )
        != "completed"
    ):

        st.error(
            "The detector pipeline could not be completed "
            f"for Face {face_number}."
        )

        error_text = face.get(
            "classification_error"
        )

        if error_text:

            st.caption(
                error_text
            )

        return

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Detector results</div>',
        unsafe_allow_html=True,
    )

    model_results = (
        face.get(
            "model_results"
        )
        or {}
    )

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

    detector_col1, detector_col2 = (
        st.columns(2)
    )

    with detector_col1:

        display_detector_card(
            model_name="xception",
            detector_result=(
                xception_result
            ),
        )

    with detector_col2:

        display_detector_card(
            model_name=(
                "efficientnet_b4"
            ),
            detector_result=(
                efficientnet_result
            ),
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    agreement = (
        face.get(
            "model_agreement"
        )
        or {}
    )

    models_agree = (
        agreement.get(
            "agree"
        )
    )

    xception_prediction = (
        agreement.get(
            "xception_prediction"
        )
    )

    efficientnet_prediction = (
        agreement.get(
            "efficientnet_prediction"
        )
    )

    if models_agree is True:

        st.info(
            "Both detectors produce the same class: "
            f"{xception_prediction}. "
            "Agreement does not constitute an automatic "
            "verification of authenticity."
        )

    elif models_agree is False:

        st.warning(
            "The detectors disagree: "
            f"Xception = {xception_prediction}, "
            f"EfficientNet-B4 = {efficientnet_prediction}. "
            "The disagreement requires particular caution "
            "in interpretation."
        )

    else:

        st.warning(
            "Model agreement could not be determined."
        )

    support_status = (
        face.get(
            "support_status"
        )
    )

    if (
        support_status
        == "review_recommended"
    ):

        st.warning(
            "Human review recommended for this face: "
            "at least one detector abstained, the models disagree, "
            "or a detector output is unavailable."
        )

    else:

        st.info(
            "Both detectors produced an output accepted by their "
            "respective selective criteria for this face. "
            "The result remains support for human assessment "
            "and is not a final decision."
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Grad-CAM</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The maps highlight the regions that contributed "
        "to the class predicted by each detector."
    )

    gradcam_results = (
        face.get(
            "gradcam_results"
        )
        or {}
    )

    grad_col1, grad_col2 = (
        st.columns(2)
    )

    with grad_col1:

        st.subheader(
            "Xception"
        )

        xception_gradcam = (
            gradcam_results.get(
                "xception"
            )
            or {}
        )

        overlay = (
            xception_gradcam.get(
                "overlay"
            )
        )

        if overlay is not None:

            st.image(
                bgr_to_rgb(
                    overlay
                ),
                width="stretch",
            )

        else:

            st.info(
                "Grad-CAM unavailable."
            )

        st.caption(
            "Grad-CAM for class "
            f"{xception_gradcam.get('target_label', 'predicted')}."
        )

    with grad_col2:

        st.subheader(
            "EfficientNet-B4"
        )

        efficientnet_gradcam = (
            gradcam_results.get(
                "efficientnet_b4"
            )
            or {}
        )

        overlay = (
            efficientnet_gradcam.get(
                "overlay"
            )
        )

        if overlay is not None:

            st.image(
                bgr_to_rgb(
                    overlay
                ),
                width="stretch",
            )

        else:

            st.info(
                "Grad-CAM unavailable."
            )

        st.caption(
            "Grad-CAM for class "
            f"{efficientnet_gradcam.get('target_label', 'predicted')}."
        )

    st.markdown(
        dedent("""
        <div class="technical-note">
            <b>Interpretation:</b>
            Grad-CAM represents the regions most strongly associated
            with the model decision. It does not necessarily localize
            a manipulation and should not be interpreted as evidence
            that a deepfake is present.
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# 9. HEADER
# ============================================================

st.markdown(
    dedent("""
    <div class="subtitle">
        Experimental multi-face system supporting human assessment
        of potentially manipulated images.
    </div>
    """),
    unsafe_allow_html=True,
)


st.info(
    "The system provides technical evidence to support human assessment. "
    "Each detected face is analyzed separately. "
    "The system does not autonomously determine whether the image "
    "or any person depicted in it is authentic or manipulated."
)


# ============================================================
# 10. UPLOAD
# ============================================================

# The uploader key is changed when "Clear analysis" is pressed.
# This resets the uploaded file and therefore clears all results
# without touching the cached detector models.
if "image_uploader_version" not in st.session_state:
    st.session_state.image_uploader_version = 0

uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
    ],
    key=(
        "image_uploader_"
        f"{st.session_state.image_uploader_version}"
    ),
)


if uploaded_file is None:

    st.markdown(
        """
        Upload an image to start the analysis.
        The system will detect distinct faces and generate
        an individual report for each detected face.
        """
    )

    st.stop()


# ============================================================
# 11. IMAGE LOADING
# ============================================================

try:

    uploaded_file.seek(0)

    original_image = (
        Image.open(
            uploaded_file
        )
        .convert(
            "RGB"
        )
    )

except Exception as error:

    st.error(
        "Unable to read the uploaded image."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 12. TEMPORARY FILE
# ============================================================

suffix = (
    Path(
        uploaded_file.name
    )
    .suffix
    .lower()
)

if not suffix:
    suffix = ".jpg"


uploaded_file.seek(0)

temporary_path = None


try:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:

        temporary_file.write(
            uploaded_file.read()
        )

        temporary_path = Path(
            temporary_file.name
        )

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    model_loader = show_custom_loader(
        "Loading detectors..."
    )

    try:

        models = get_models()

    finally:

        model_loader.empty()

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    analysis_loader = show_custom_loader(
        "Detecting and analyzing faces..."
    )

    try:

        result = (
            run_report_pipeline(
                image_path=str(
                    temporary_path
                ),
                models=models,
            )
        )

    finally:

        analysis_loader.empty()

except Exception as error:

    st.error(
        "An error occurred during the analysis."
    )

    st.exception(
        error
    )

    st.stop()


finally:

    if (
        temporary_path
        is not None
        and temporary_path.exists()
    ):

        try:
            temporary_path.unlink()

        except OSError:
            pass


# ============================================================
# 13. ANALYSIS CONTROLS
# ============================================================

control_left, control_right = st.columns(
    [1, 5]
)

with control_left:

    if st.button(
        "Clear analysis",
        type="secondary",
        width="stretch",
    ):

        st.session_state.image_uploader_version += 1
        st.rerun()


# ============================================================
# 14. ORIGINAL IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">Analyzed input</div>',
    unsafe_allow_html=True,
)

display_image = resize_image_for_display(
    original_image,
    max_width=1100,
    max_height=550,
)

image_left, image_center, image_right = st.columns(
    [1, 10, 1]
)

with image_center:

    st.image(
        display_image,
        caption="Original image",
        width=display_image.width,
    )


# ============================================================
# 15. FACE COUNT
# ============================================================

faces = (
    result.get(
        "face_results"
    )
    or result.get(
        "faces"
    )
    or []
)

faces_detected = first_available(
    result,
    "faces_detected",
    "faces_analyzed",
    default=len(
        faces
    ),
)

st.markdown(
    '<div class="section-title">Face detection</div>',
    unsafe_allow_html=True,
)

st.metric(
    "Faces detected",
    faces_detected,
)


# ============================================================
# 16. ANALYZABILITY CHECK
# ============================================================

if (
    result.get(
        "status"
    )
    != "ready_for_classification"
    or not faces
):

    st.warning(
        "The image could not be classified because no valid face "
        "was available for the detector pipeline."
    )

    st.markdown(
        dedent("""
        <div class="technical-note">
            The absence of a classification is not a detector
            abstention. In this case the pipeline does not have
            a valid facial crop on which to perform REAL/FAKE
            classification.
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# 17. INDIVIDUAL FACE REPORTS
# ============================================================

st.markdown(
    '<div class="section-title">Individual face reports</div>',
    unsafe_allow_html=True,
)

st.write(
    "Each detected face is processed independently by "
    "Xception and EfficientNet-B4."
)


for position, face in enumerate(
    faces,
    start=1,
):

    face_number = face.get(
        "face_id",
        position,
    )

    display_face_report(
        face=face,
        face_number=face_number,
    )


# ============================================================
# 18. OVERALL SUPPORT STATUS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Overall support status</div>',
    unsafe_allow_html=True,
)

overall_support_status = (
    result.get(
        "overall_support_status"
    )
)

if (
    overall_support_status
    == "review_recommended"
):

    st.warning(
        "Human review recommended: at least one detected face "
        "contains an abstention, detector disagreement, or an "
        "incomplete detector output."
    )

elif (
    overall_support_status
    == "detector_outputs_available"
):

    st.info(
        "Detector outputs are available for all detected faces. "
        "This does not constitute an automatic conclusion about "
        "the authenticity of the image."
    )

else:

    st.info(
        str(
            overall_support_status
            or "No overall status available."
        )
    )


# ============================================================
# 19. FILE-LEVEL TECHNICAL INDICATORS
# ============================================================

file_info = (
    get_file_info(
        result
    )
)

jpeg_blockiness = first_available(
    file_info,
    "jpeg_blockiness_score",
)

full_blur = first_available(
    file_info,
    "blur_score",
)

jpeg_info = first_available(
    file_info,
    "jpeg_info",
    default={},
)

if not isinstance(
    jpeg_info,
    dict,
):
    jpeg_info = {}

is_jpeg = bool(
    jpeg_info.get(
        "is_jpeg",
        False,
    )
)


st.markdown(
    '<div class="section-title">Image-level technical indicators</div>',
    unsafe_allow_html=True,
)


indicator_col1, indicator_col2 = (
    st.columns(2)
)


with indicator_col1:

    st.metric(
        "Full-image blur score",
        safe_number(
            full_blur,
            2,
        ),
    )


with indicator_col2:

    if is_jpeg:

        st.metric(
            "JPEG blockiness",
            safe_number(
                jpeg_blockiness,
                4,
            ),
        )

    else:

        st.metric(
            "JPEG blockiness",
            "N/A",
        )

        st.caption(
            "Input is not JPEG; JPEG-specific blockiness "
            "analysis is not applicable."
        )


if is_jpeg:

    st.caption(
        "These are descriptive file-level indicators. "
        "The JPEG blockiness score does not allow the original JPEG "
        "quality or the complete file-processing history to be "
        "reconstructed with certainty."
    )

else:

    st.caption(
        "The full-image blur score is a descriptive image-level "
        "indicator. No JPEG-specific interpretation is applied "
        "because the uploaded file is not JPEG."
    )


# ============================================================
# 20. FILE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">File information</div>',
    unsafe_allow_html=True,
)


# Always report the original user-uploaded filename.
# The forensic pipeline works on a temporary local file, whose
# generated tmp name must not appear in the final report.
filename = uploaded_file.name

file_format = first_available(
    file_info,
    "format",
    default="N/A",
)

width = first_available(
    file_info,
    "width",
)

height = first_available(
    file_info,
    "height",
)

file_size_kb = first_available(
    file_info,
    "file_size_kb",
    "size_kb",
)

color_mode = first_available(
    file_info,
    "color_mode",
    "mode",
    default="N/A",
)


resolution = (
    f"{width} × {height}"
    if (
        width is not None
        and height is not None
    )
    else "N/A"
)


file_col1, file_col2, file_col3, file_col4 = (
    st.columns(4)
)


with file_col1:

    st.metric(
        "Format",
        file_format,
    )


with file_col2:

    st.metric(
        "Resolution",
        resolution,
    )


with file_col3:

    st.metric(
        "File size",
        (
            f"{float(file_size_kb):.2f} KB"
            if file_size_kb
            is not None
            else "N/A"
        ),
    )


with file_col4:

    st.metric(
        "Color mode",
        color_mode,
    )


st.write(
    "**File name:**",
    filename,
)


sha256 = first_available(
    file_info,
    "sha256",
)


if sha256 is not None:

    st.write(
        "**SHA-256**"
    )

    st.markdown(
        dedent(f"""
        <div class="hash-box">
            {html.escape(str(sha256))}
        </div>
        """),
        unsafe_allow_html=True,
    )


# ============================================================
# 21. EXIF
# ============================================================

exif_data = first_available(
    file_info,
    "exif",
    default={},
)


with st.expander(
    "EXIF metadata"
):

    if exif_data:

        st.json(
            exif_data
        )

    else:

        st.write(
            "No EXIF metadata available."
        )

        st.caption(
            "The absence of EXIF metadata is not an automatic "
            "indication of manipulation."
        )


# ============================================================
# 22. JPEG INFO
# ============================================================

with st.expander(
    "JPEG information"
):

    if not is_jpeg:

        st.write(
            "Not applicable — the uploaded file is not JPEG."
        )

    elif jpeg_info:

        st.json(
            jpeg_info
        )

    else:

        st.write(
            "JPEG information is unavailable."
        )


# ============================================================
# 23. METHODOLOGICAL NOTE
# ============================================================

st.markdown(
    '<div class="section-title">Methodological note</div>',
    unsafe_allow_html=True,
)


st.warning(
    "The confidence values and selective-classification thresholds "
    "were determined experimentally using FaceForensics++. "
    "They do not represent universal probabilities of authenticity "
    "or manipulation. Performance may change on images drawn from "
    "distributions different from the experimental dataset."
)


st.markdown(
    dedent("""
    <div class="technical-note">
        <b>Multi-face report extension.</b><br><br>

        Each detected face is analyzed separately using the same
        Xception and EfficientNet-B4 detector pipeline. The multi-face
        functionality is an application-level extension of the report:
        it does not retrain the models, modify their calibrated
        probabilities, or change the selective-classification thresholds.
        A result concerning one facial crop should not automatically
        be generalized to the entire image or to other detected faces.
    </div>
    """),
    unsafe_allow_html=True,
)


st.markdown(
    dedent("""
    <div class="technical-note">
        <b>Intended use of the report.</b><br><br>

        The report combines detector outputs, confidence values,
        abstention status, Grad-CAM visualizations and technical
        indicators to support human assessment. None of these
        elements, individually or together, constitutes an automatic
        certification of image authenticity. In particular, the
        REAL/FAKE predictions refer to patterns learned from
        FaceForensics++ and must not be interpreted as a general
        determination of whether the whole image was generated by AI.
    </div>
    """),
    unsafe_allow_html=True,
)

# ============================================================
# REPORT EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">Report export</div>',
    unsafe_allow_html=True,
)

try:

    pdf_bytes = generate_pdf_report(
        result=result,
        original_image=original_image,
        original_filename=uploaded_file.name,
    )

    pdf_filename = build_report_filename(
        uploaded_file.name
    )

    pdf_sha256 = compute_sha256_bytes(
        pdf_bytes
    )

    st.download_button(
        label="Download PDF technical report",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        width="stretch",
    )

    st.caption(
        f"PDF SHA-256: {pdf_sha256}"
    )

except Exception as error:

    st.error(
        "The PDF technical report could not be generated."
    )

    st.exception(
        error
    )