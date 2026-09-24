
"""
report_export.py

PDF export utilities for the experimental multi-face technical report.

Intended location:
    src_multiface/report_export.py

The module does NOT run face detection or model inference.
It only receives the already completed report result and converts it
into a human-readable PDF.

Main public functions
---------------------
generate_pdf_report(...)
    Return the complete technical report as PDF bytes.

compute_sha256_bytes(...)
    Return the SHA-256 digest of generated PDF bytes.

build_report_filename(...)
    Build a safe PDF filename from the original uploaded filename.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import unicodedata
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# 1. CONSTANTS
# ============================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

MARGIN_LEFT = 18 * mm
MARGIN_RIGHT = 18 * mm
MARGIN_TOP = 18 * mm
MARGIN_BOTTOM = 18 * mm

CONTENT_WIDTH = (
    PAGE_WIDTH
    - MARGIN_LEFT
    - MARGIN_RIGHT
)

DARK_BLUE = colors.HexColor("#243B5A")
DARK_TEXT = colors.HexColor("#172033")
MID_TEXT = colors.HexColor("#4F5B6B")
LIGHT_BORDER = colors.HexColor("#C7D0DB")
LIGHT_FILL = colors.HexColor("#EEF1F5")
VERY_LIGHT_FILL = colors.HexColor("#F7F9FB")
WARNING_FILL = colors.HexColor("#FFF7E6")
ERROR_FILL = colors.HexColor("#FDECEC")
SUCCESS_FILL = colors.HexColor("#EEF7EF")

MODEL_ORDER = (
    "xception",
    "efficientnet_b4",
)


# ============================================================
# 2. GENERIC HELPERS
# ============================================================

def _safe_text(
    value: Any,
    default: str = "N/A",
) -> str:
    """
    Convert a value to safe plain text.
    """

    if value is None:
        return default

    text = str(value).strip()

    return text if text else default


def _safe_number(
    value: Any,
    decimals: int = 4,
) -> str:
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
        return _safe_text(value)


def _safe_percent(
    value: Any,
) -> str:
    """
    Format a 0-1 numeric value as a percentage.
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"

    except (
        TypeError,
        ValueError,
    ):
        return _safe_text(value)


def _first_available(
    dictionary: Any,
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Return the first non-None value associated with the supplied keys.
    """

    if not isinstance(
        dictionary,
        dict,
    ):
        return default

    for key in keys:

        if (
            key in dictionary
            and dictionary[key] is not None
        ):
            return dictionary[key]

    return default


def _format_rotation(
    angle: Any,
) -> str:
    """
    Convert the face-detection rotation into readable report text.

    Positive values are counterclockwise.
    Negative values are clockwise.
    """

    if angle is None:
        return "N/A"

    try:
        angle = float(angle)

    except (
        TypeError,
        ValueError,
    ):
        return _safe_text(angle)

    if abs(angle) < 1e-9:
        return "None"

    angle_abs = abs(angle)

    if angle_abs.is_integer():
        angle_text = str(
            int(angle_abs)
        )
    else:
        angle_text = (
            f"{angle_abs:.1f}"
            .rstrip("0")
            .rstrip(".")
        )

    direction = (
        "counterclockwise"
        if angle > 0
        else "clockwise"
    )

    return (
        f"{angle_text} deg {direction}"
    )


def _display_model_name(
    model_name: str,
) -> str:
    """
    Convert internal model names to report labels.
    """

    if model_name == "efficientnet_b4":
        return "EfficientNet-B4"

    if model_name == "xception":
        return "Xception"

    return model_name


def _get_file_info(
    result: dict,
) -> dict:
    """
    Return file-level forensic information.
    """

    file_info = (
        result.get(
            "file_indicators"
        )
        or result.get(
            "forensic_indicators"
        )
        or result
    )

    return (
        file_info
        if isinstance(
            file_info,
            dict,
        )
        else {}
    )


def _get_faces(
    result: dict,
) -> list:
    """
    Return face-level report blocks.
    """

    faces = (
        result.get(
            "face_results"
        )
        or result.get(
            "faces"
        )
        or []
    )

    return (
        faces
        if isinstance(
            faces,
            list,
        )
        else []
    )


def _clean_for_pdf(
    text: Any,
) -> str:
    """
    Escape text for ReportLab Paragraph XML.
    """

    return escape(
        _safe_text(
            text
        )
    )


# ============================================================
# 3. OUTPUT FILENAME / HASH
# ============================================================

def build_report_filename(
    original_filename: str,
) -> str:

    stem = (
        Path(
            original_filename
            or "image"
        )
        .stem
    )

    stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        stem,
    ).strip(
        "._"
    )

    if not stem:
        stem = "image"

    return (
        f"technical_report_{stem}.pdf"
    )


def compute_sha256_bytes(
    data: bytes,
) -> str:
    """
    Compute SHA-256 for generated bytes.
    """

    return sha256(
        data
    ).hexdigest()


def _sanitize_metadata_value(
    value: Any,
) -> str:
    """
    Convert EXIF / technical metadata to printable text suitable for
    the built-in ReportLab Helvetica font.

    Binary values and strings dominated by unsupported/control
    characters are replaced with a neutral explanatory label instead
    of producing black squares in the PDF.
    """

    if value is None:
        return "N/A"

    if isinstance(
        value,
        bytes,
    ):

        if not value:
            return "N/A"

        # EXIF textual byte fields are commonly null-terminated.
        value = value.replace(
            b"\\x00",
            b"",
        )

        decoded = None

        for encoding in (
            "utf-8",
            "latin-1",
        ):

            try:
                decoded = value.decode(
                    encoding
                )
                break

            except UnicodeDecodeError:
                pass

        if decoded is None:
            return (
                f"Binary metadata ({len(value)} bytes)"
            )

        value = decoded

    text = str(
        value
    )

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove control / formatting characters except normal spaces.
    text = "".join(
        character
        for character in text
        if (
            unicodedata.category(
                character
            )[0]
            != "C"
        )
    ).strip()

    if not text:
        return "Unavailable / unsupported characters"

    # Helvetica in ReportLab uses a WinAnsi-like character set.
    encoded = text.encode(
        "cp1252",
        errors="ignore",
    )

    printable = encoded.decode(
        "cp1252",
        errors="ignore",
    ).strip()

    if not printable:
        return "Unavailable / unsupported characters"

    retained_ratio = (
        len(printable)
        / max(
            len(text),
            1,
        )
    )

    if retained_ratio < 0.70:
        return "Unavailable / unsupported characters"

    return printable


def _get_exif_orientation(
    result: dict,
) -> int | None:
    """
    Read the EXIF Orientation value from the report result.
    """

    file_info = _get_file_info(
        result
    )

    exif_data = _first_available(
        file_info,
        "exif",
        default={},
    )

    if not isinstance(
        exif_data,
        dict,
    ):
        return None

    orientation = _first_available(
        exif_data,
        "Orientation",
        "orientation",
    )

    try:
        orientation = int(
            orientation
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if orientation not in range(
        1,
        9,
    ):
        return None

    return orientation


def _apply_exif_orientation(
    pil_image: PILImage.Image | None,
    result: dict,
) -> PILImage.Image | None:
    """
    Apply EXIF orientation ONLY to the image displayed in the PDF.

    The analyzed source file, its SHA-256 and detector inputs are not
    modified.  If the PIL object still carries EXIF metadata,
    ImageOps.exif_transpose() is preferred.  Otherwise the Orientation
    value already extracted by the report pipeline is used.
    """

    if pil_image is None:
        return None

    image = pil_image.copy()

    try:

        pil_orientation = (
            image.getexif().get(
                274
            )
        )

    except Exception:
        pil_orientation = None

    if pil_orientation in range(
        2,
        9,
    ):

        try:
            return (
                ImageOps.exif_transpose(
                    image
                )
                .convert(
                    "RGB"
                )
            )

        except Exception:
            pass

    orientation = _get_exif_orientation(
        result
    )

    if orientation is None:
        return image.convert(
            "RGB"
        )

    transpose_map = {
        2: PILImage.Transpose.FLIP_LEFT_RIGHT,
        3: PILImage.Transpose.ROTATE_180,
        4: PILImage.Transpose.FLIP_TOP_BOTTOM,
        5: PILImage.Transpose.TRANSPOSE,
        6: PILImage.Transpose.ROTATE_270,
        7: PILImage.Transpose.TRANSVERSE,
        8: PILImage.Transpose.ROTATE_90,
    }

    operation = transpose_map.get(
        orientation
    )

    if operation is not None:

        image = image.transpose(
            operation
        )

    return image.convert(
        "RGB"
    )


# ============================================================
# 4. IMAGE CONVERSION
# ============================================================

def _to_pil_rgb(
    image: Any,
    *,
    assume_bgr: bool = False,
) -> PILImage.Image | None:
    """
    Convert PIL / NumPy image data to a detached RGB PIL image.

    face_crop and Grad-CAM arrays generated by OpenCV are expected
    to be BGR, therefore assume_bgr=True is used for those inputs.
    """

    if image is None:
        return None

    if isinstance(
        image,
        PILImage.Image,
    ):
        return image.convert(
            "RGB"
        ).copy()

    if not isinstance(
        image,
        np.ndarray,
    ):
        return None

    if image.size == 0:
        return None

    array = image

    if array.dtype != np.uint8:

        array = np.clip(
            array,
            0,
            255,
        ).astype(
            np.uint8
        )

    if array.ndim == 2:

        return PILImage.fromarray(
            array,
            mode="L",
        ).convert(
            "RGB"
        )

    if (
        array.ndim == 3
        and array.shape[2] >= 3
    ):

        rgb = array[
            :,
            :,
            :3,
        ]

        if assume_bgr:
            rgb = rgb[
                :,
                :,
                ::-1
            ]

        return PILImage.fromarray(
            rgb
        ).convert(
            "RGB"
        )

    return None


def _pil_to_flowable(
    pil_image: PILImage.Image | None,
    *,
    max_width: float,
    max_height: float,
    jpeg_quality: int = 88,
) -> Image | Paragraph:
    """
    Convert a PIL image to a proportionally scaled ReportLab Image.
    """

    if pil_image is None:

        return Paragraph(
            "Image unavailable.",
            _styles()["small_muted"],
        )

    width_px, height_px = (
        pil_image.size
    )

    if (
        width_px <= 0
        or height_px <= 0
    ):

        return Paragraph(
            "Image unavailable.",
            _styles()["small_muted"],
        )

    scale = min(
        max_width / width_px,
        max_height / height_px,
        1.0,
    )

    draw_width = (
        width_px
        * scale
    )

    draw_height = (
        height_px
        * scale
    )

    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )

    buffer.seek(0)

    flowable = Image(
        buffer,
        width=draw_width,
        height=draw_height,
    )

    # Keep the BytesIO alive for the lifetime of the flowable.
    flowable._source_buffer = buffer

    return flowable


# ============================================================
# 5. STYLES
# ============================================================

_STYLE_CACHE = None


def _styles() -> dict:
    """
    Return cached Paragraph styles.
    """

    global _STYLE_CACHE

    if _STYLE_CACHE is not None:
        return _STYLE_CACHE

    sample = getSampleStyleSheet()

    _STYLE_CACHE = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=DARK_TEXT,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=MID_TEXT,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=DARK_TEXT,
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
        ),
        "face_title": ParagraphStyle(
            "FaceTitle",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=DARK_BLUE,
            spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=DARK_TEXT,
            spaceAfter=2.5 * mm,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=DARK_TEXT,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=DARK_TEXT,
        ),
        "small_muted": ParagraphStyle(
            "SmallMuted",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MID_TEXT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=DARK_TEXT,
        ),
        "table_cell_bold": ParagraphStyle(
            "TableCellBold",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=DARK_TEXT,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=DARK_TEXT,
            borderColor=LIGHT_BORDER,
            borderWidth=0.5,
            borderPadding=7,
            backColor=VERY_LIGHT_FILL,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "warning": ParagraphStyle(
            "Warning",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=DARK_TEXT,
            borderColor=colors.HexColor("#D5A545"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=WARNING_FILL,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "center_small": ParagraphStyle(
            "CenterSmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MID_TEXT,
            alignment=TA_CENTER,
        ),
    }

    return _STYLE_CACHE


# ============================================================
# 6. PAGE TEMPLATE
# ============================================================

def _draw_page_header_footer(
    canvas,
    doc,
):
    """
    Draw a restrained header/footer on every PDF page.
    """

    canvas.saveState()

    canvas.setStrokeColor(
        LIGHT_BORDER
    )

    canvas.setLineWidth(
        0.5
    )

    header_y = (
        PAGE_HEIGHT
        - 11 * mm
    )

    canvas.line(
        MARGIN_LEFT,
        header_y,
        PAGE_WIDTH - MARGIN_RIGHT,
        header_y,
    )

    canvas.setFont(
        "Helvetica-Bold",
        7.5,
    )

    canvas.setFillColor(
        DARK_BLUE
    )

    canvas.drawString(
        MARGIN_LEFT,
        PAGE_HEIGHT - 8 * mm,
        "DEEPFAKE ANALYSIS - TECHNICAL SUPPORT REPORT",
    )

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(
        MID_TEXT
    )

    footer_text = (
        f"Page {doc.page}"
    )

    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        9 * mm,
        footer_text,
    )

    canvas.drawString(
        MARGIN_LEFT,
        9 * mm,
        "Human-assessment support document - not an authenticity certification",
    )

    canvas.restoreState()


class _TechnicalReportDocument(
    BaseDocTemplate
):
    """
    Base document with a single A4 frame and fixed header/footer.
    """

    def __init__(
        self,
        buffer,
        **kwargs,
    ):

        super().__init__(
            buffer,
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            **kwargs,
        )

        frame = Frame(
            MARGIN_LEFT,
            MARGIN_BOTTOM + 4 * mm,
            CONTENT_WIDTH,
            PAGE_HEIGHT
            - MARGIN_TOP
            - MARGIN_BOTTOM
            - 8 * mm,
            id="main_frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        template = PageTemplate(
            id="technical_report",
            frames=[
                frame
            ],
            onPage=_draw_page_header_footer,
        )

        self.addPageTemplates(
            [
                template
            ]
        )


# ============================================================
# 7. REPORT BUILDING BLOCKS
# ============================================================

def _section_heading(
    text: str,
) -> Table:
    """
    Return a formal section heading band.
    """

    style = _styles()

    table = Table(
        [
            [
                Paragraph(
                    _clean_for_pdf(
                        text
                    ),
                    style[
                        "section"
                    ],
                )
            ]
        ],
        colWidths=[
            CONTENT_WIDTH
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_FILL,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    LIGHT_BORDER,
                ),
                (
                    "LINEBEFORE",
                    (0, 0),
                    (0, -1),
                    4,
                    DARK_BLUE,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
            ]
        )
    )

    return table


def _key_value_table(
    rows: list[tuple[Any, Any]],
    *,
    label_width: float = 48 * mm,
    value_width: float | None = None,
) -> Table:
    """
    Create a two-column technical key/value table.
    """

    style = _styles()

    if value_width is None:
        value_width = (
            CONTENT_WIDTH
            - label_width
        )

    data = []

    for label, value in rows:

        data.append(
            [
                Paragraph(
                    _clean_for_pdf(
                        label
                    ),
                    style[
                        "table_cell_bold"
                    ],
                ),
                Paragraph(
                    _clean_for_pdf(
                        value
                    ),
                    style[
                        "table_cell"
                    ],
                ),
            ]
        )

    table = Table(
        data,
        colWidths=[
            label_width,
            value_width,
        ],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    LIGHT_BORDER,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    VERY_LIGHT_FILL,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


def _model_result_table(
    model_results: dict,
) -> Table:
    """
    Create the compact Xception / EfficientNet comparison table.
    """

    style = _styles()

    header = [
        Paragraph(
            "Detector",
            style[
                "table_header"
            ],
        ),
        Paragraph(
            "Prediction",
            style[
                "table_header"
            ],
        ),
        Paragraph(
            "Calibrated confidence",
            style[
                "table_header"
            ],
        ),
        Paragraph(
            "Selective threshold",
            style[
                "table_header"
            ],
        ),
        Paragraph(
            "Status",
            style[
                "table_header"
            ],
        ),
    ]

    data = [
        header
    ]

    for model_name in MODEL_ORDER:

        detector_result = (
            model_results.get(
                model_name
            )
            or {}
        )

        data.append(
            [
                Paragraph(
                    _clean_for_pdf(
                        _display_model_name(
                            model_name
                        )
                    ),
                    style[
                        "table_cell_bold"
                    ],
                ),
                Paragraph(
                    _clean_for_pdf(
                        _first_available(
                            detector_result,
                            "predicted_label",
                            default="N/A",
                        )
                    ),
                    style[
                        "table_cell"
                    ],
                ),
                Paragraph(
                    _clean_for_pdf(
                        _safe_percent(
                            _first_available(
                                detector_result,
                                "calibrated_confidence",
                            )
                        )
                    ),
                    style[
                        "table_cell"
                    ],
                ),
                Paragraph(
                    _clean_for_pdf(
                        _safe_percent(
                            _first_available(
                                detector_result,
                                "threshold",
                                "abstention_threshold",
                            )
                        )
                    ),
                    style[
                        "table_cell"
                    ],
                ),
                Paragraph(
                    _clean_for_pdf(
                        _first_available(
                            detector_result,
                            "selective_status",
                            default="N/A",
                        )
                    ),
                    style[
                        "table_cell"
                    ],
                ),
            ]
        )

    table = Table(
        data,
        repeatRows=1,
        colWidths=[
            34 * mm,
            26 * mm,
            38 * mm,
            36 * mm,
            28 * mm,
        ],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    DARK_BLUE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    LIGHT_BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


def _model_technical_details(
    model_name: str,
    detector_result: dict,
) -> Table:
    """
    Create the detailed calibration/inference block for one detector.
    """

    rows = [
        (
            "Detector",
            _display_model_name(
                model_name
            ),
        ),
        (
            "Calibrated REAL probability",
            _safe_percent(
                _first_available(
                    detector_result,
                    "calibrated_real_probability",
                    "real_probability_calibrated",
                )
            ),
        ),
        (
            "Calibrated FAKE probability",
            _safe_percent(
                _first_available(
                    detector_result,
                    "calibrated_fake_probability",
                    "fake_probability_calibrated",
                )
            ),
        ),
        (
            "Uncalibrated confidence",
            _safe_percent(
                _first_available(
                    detector_result,
                    "raw_confidence",
                    "uncalibrated_confidence",
                )
            ),
        ),
        (
            "Temperature",
            _safe_number(
                _first_available(
                    detector_result,
                    "temperature",
                ),
                4,
            ),
        ),
    ]

    return _key_value_table(
        rows,
        label_width=55 * mm,
        value_width=(
            CONTENT_WIDTH
            - 55 * mm
        ),
    )


def _dictionary_table(
    dictionary: dict,
    *,
    empty_message: str,
) -> list:
    """
    Render a dictionary as a readable two-column table.
    """

    style = _styles()

    if not dictionary:

        return [
            Paragraph(
                _clean_for_pdf(
                    empty_message
                ),
                style[
                    "body"
                ],
            )
        ]

    rows = []

    for key in sorted(
        dictionary.keys(),
        key=lambda item: str(
            item
        ).lower(),
    ):

        value = dictionary.get(
            key
        )

        if isinstance(
            value,
            (
                dict,
                list,
                tuple,
            ),
        ):
            value = repr(
                value
            )

        rows.append(
            (
                _sanitize_metadata_value(
                    key
                ),
                _sanitize_metadata_value(
                    value
                ),
            )
        )

    return [
        _key_value_table(
            rows,
            label_width=48 * mm,
            value_width=(
                CONTENT_WIDTH
                - 48 * mm
            ),
        )
    ]


def _face_detection_summary(
    face: dict,
) -> Table:
    """
    Create the face-detection summary table.
    """

    crop_quality = (
        face.get(
            "face_crop_quality"
        )
        or face.get(
            "crop_quality"
        )
        or {}
    )

    crop_blur = _first_available(
        crop_quality,
        "blur_score",
        "crop_blur_score",
    )

    relative_area = face.get(
        "relative_face_area"
    )

    rows = [
        (
            "Detection score",
            _safe_number(
                face.get(
                    "detection_score"
                ),
                4,
            ),
        ),
        (
            "Orientation adjustment",
            _format_rotation(
                face.get(
                    "rotation_applied"
                )
            ),
        ),
        (
            "Face blur score",
            _safe_number(
                crop_blur,
                2,
            ),
        ),
        (
            "Relative face area",
            (
                _safe_percent(
                    relative_area
                )
                if relative_area
                is not None
                else "N/A"
            ),
        ),
    ]

    return _key_value_table(
        rows,
        label_width=43 * mm,
        value_width=(
            70 * mm
            - 43 * mm
        ),
    )


def _face_top_block(
    face: dict,
) -> Table:
    """
    Place face crop and detection information side by side.
    """

    face_crop = _to_pil_rgb(
        face.get(
            "face_crop"
        ),
        assume_bgr=True,
    )

    face_image = _pil_to_flowable(
        face_crop,
        max_width=82 * mm,
        max_height=70 * mm,
        jpeg_quality=90,
    )

    detection_table = (
        _face_detection_summary(
            face
        )
    )

    table = Table(
        [
            [
                face_image,
                detection_table,
            ]
        ],
        colWidths=[
            88 * mm,
            70 * mm,
        ],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    0,
                ),
            ]
        )
    )

    return table


def _gradcam_table(
    face: dict,
) -> Table:
    """
    Place Xception and EfficientNet Grad-CAM overlays side by side.
    """

    style = _styles()

    gradcam_results = (
        face.get(
            "gradcam_results"
        )
        or {}
    )

    cells = []

    for model_name in MODEL_ORDER:

        grad_result = (
            gradcam_results.get(
                model_name
            )
            or {}
        )

        overlay = _to_pil_rgb(
            grad_result.get(
                "overlay"
            ),
            assume_bgr=True,
        )

        image = _pil_to_flowable(
            overlay,
            max_width=72 * mm,
            max_height=58 * mm,
            jpeg_quality=90,
        )

        label = (
            f"{_display_model_name(model_name)} - "
            f"class {_safe_text(grad_result.get('target_label'), 'predicted')}"
        )

        # A ReportLab table cell can contain a list of flowables.
        # Keeping label + image in the same cell avoids horizontal
        # overflow and keeps each Grad-CAM panel visually grouped.
        cells.append(
            [
                Paragraph(
                    _clean_for_pdf(
                        label
                    ),
                    style[
                        "center_small"
                    ],
                ),
                Spacer(
                    1,
                    2 * mm,
                ),
                image,
            ]
        )

    table = Table(
        [
            [
                cells[0],
                cells[1],
            ]
        ],
        colWidths=[
            80 * mm,
            80 * mm,
        ],
        hAlign="CENTER",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    return table


# ============================================================
# 8. FACE REPORT
# ============================================================

def _append_face_report(
    story: list,
    face: dict,
    face_number: int,
):
    """
    Append one complete face-level report to the document story.
    """

    style = _styles()

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            f"Face {face_number}",
            style[
                "face_title"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Individual analysis for detected face "
                f"{face_number}."
            ),
            style[
                "subtitle"
            ],
        )
    )

    story.append(
        _face_top_block(
            face
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    if (
        face.get(
            "classification_status"
        )
        != "completed"
    ):

        error_text = (
            face.get(
                "classification_error"
            )
            or "Detector pipeline did not return a complete result."
        )

        story.append(
            Paragraph(
                (
                    "<b>Classification unavailable.</b><br/>"
                    + _clean_for_pdf(
                        error_text
                    )
                ),
                style[
                    "warning"
                ],
            )
        )

        return

    model_results = (
        face.get(
            "model_results"
        )
        or {}
    )

    story.append(
        _section_heading(
            "Detector results"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    story.append(
        _model_result_table(
            model_results
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    for model_name in MODEL_ORDER:

        detector_result = (
            model_results.get(
                model_name
            )
            or {}
        )

        story.append(
            Paragraph(
                (
                    "<b>"
                    + _clean_for_pdf(
                        _display_model_name(
                            model_name
                        )
                    )
                    + " technical details</b>"
                ),
                style[
                    "body"
                ],
            )
        )

        story.append(
            _model_technical_details(
                model_name,
                detector_result,
            )
        )

        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

    agreement = (
        face.get(
            "model_agreement"
        )
        or {}
    )

    models_agree = agreement.get(
        "agree"
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

        agreement_text = (
            "Both detectors produce the same class: "
            f"{_safe_text(xception_prediction)}. "
            "Agreement does not constitute an automatic verification "
            "of authenticity."
        )

        agreement_style = (
            style[
                "note"
            ]
        )

    elif models_agree is False:

        agreement_text = (
            "The detectors disagree: "
            f"Xception = {_safe_text(xception_prediction)}, "
            f"EfficientNet-B4 = {_safe_text(efficientnet_prediction)}. "
            "The disagreement requires particular caution in interpretation."
        )

        agreement_style = (
            style[
                "warning"
            ]
        )

    else:

        agreement_text = (
            "Model agreement could not be determined."
        )

        agreement_style = (
            style[
                "warning"
            ]
        )

    story.append(
        Paragraph(
            _clean_for_pdf(
                agreement_text
            ),
            agreement_style,
        )
    )

    support_status = face.get(
        "support_status"
    )

    if (
        support_status
        == "review_recommended"
    ):

        support_text = (
            "Human review recommended for this face: at least one detector "
            "abstained, the models disagree, or a detector output is unavailable."
        )

        support_style = (
            style[
                "warning"
            ]
        )

    else:

        support_text = (
            "Both detectors produced an output accepted by their respective "
            "selective criteria for this face. The result remains support for "
            "human assessment and is not a final decision."
        )

        support_style = (
            style[
                "note"
            ]
        )

    story.append(
        Paragraph(
            _clean_for_pdf(
                support_text
            ),
            support_style,
        )
    )

    story.append(
        Paragraph(
            (
                "Prediction relative to the distribution learned from "
                "FaceForensics++. This output should not be interpreted "
                "as a general detector of AI-generated or synthetic images."
            ),
            style[
                "small_muted"
            ],
        )
    )

    story.append(
        _section_heading(
            "Grad-CAM"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "The maps highlight the regions that contributed to the "
                "class predicted by each detector."
            ),
            style[
                "body"
            ],
        )
    )

    story.append(
        _gradcam_table(
            face
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Interpretation:</b> Grad-CAM represents the regions most "
                "strongly associated with the model decision. It does not "
                "necessarily localize a manipulation and should not be "
                "interpreted as evidence that a deepfake is present."
            ),
            style[
                "note"
            ],
        )
    )


# ============================================================
# 9. FILE-LEVEL REPORT
# ============================================================

def _append_file_level_report(
    story: list,
    result: dict,
    original_filename: str,
):
    """
    Append image-level indicators, file metadata and methodological notes.
    """

    style = _styles()

    file_info = (
        _get_file_info(
            result
        )
    )

    story.append(
        PageBreak()
    )

    story.append(
        _section_heading(
            "Image-level technical indicators"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    jpeg_info = _first_available(
        file_info,
        "jpeg_info",
        default={},
    )

    if not isinstance(
        jpeg_info,
        dict,
    ):
        jpeg_info = {}

    file_format = _safe_text(
        _first_available(
            file_info,
            "format",
            default="N/A",
        )
    )

    filename_suffix = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )

    format_upper = (
        file_format.upper()
    )

    is_mpo = (
        format_upper
        == "MPO"
    )

    is_jpeg_compatible = (
        bool(
            jpeg_info.get(
                "is_jpeg",
                False,
            )
        )
        or format_upper
        in {
            "JPEG",
            "JPG",
            "MPO",
        }
        or filename_suffix
        in {
            ".jpg",
            ".jpeg",
            ".mpo",
        }
    )

    jpeg_blockiness_value = (
        _first_available(
            file_info,
            "jpeg_blockiness_score",
        )
    )

    jpeg_blockiness = (
        _safe_number(
            jpeg_blockiness_value,
            4,
        )
        if (
            is_jpeg_compatible
            and jpeg_blockiness_value
            is not None
        )
        else (
            "N/A - indicator unavailable"
            if is_jpeg_compatible
            else "N/A - input is not JPEG-compatible"
        )
    )

    story.append(
        _key_value_table(
            [
                (
                    "Full-image blur score",
                    _safe_number(
                        _first_available(
                            file_info,
                            "blur_score",
                        ),
                        2,
                    ),
                ),
                (
                    "JPEG blockiness",
                    jpeg_blockiness,
                ),
            ]
        )
    )

    story.append(
        Paragraph(
            (
                "These indicators are descriptive. JPEG blockiness is reported "
                "for JPEG-compatible inputs, including MPO containers whose primary "
                "image is JPEG-based. The score cannot reconstruct the original "
                "compression quality or the complete processing history with certainty."
            ),
            style[
                "small_muted"
            ],
        )
    )

    story.append(
        _section_heading(
            "File information"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    width = _first_available(
        file_info,
        "width",
    )

    height = _first_available(
        file_info,
        "height",
    )

    resolution = (
        f"{width} x {height}"
        if (
            width is not None
            and height is not None
        )
        else "N/A"
    )

    file_size_kb = _first_available(
        file_info,
        "file_size_kb",
        "size_kb",
    )

    file_size_text = (
        f"{float(file_size_kb):.2f} KB"
        if file_size_kb
        is not None
        else "N/A"
    )

    file_rows = [
        (
            "File name",
            original_filename,
        ),
        (
            "Format",
            file_format,
        ),
        (
            "Resolution",
            resolution,
        ),
        (
            "File size",
            file_size_text,
        ),
        (
            "Color mode",
            _safe_text(
                _first_available(
                    file_info,
                    "color_mode",
                    "mode",
                    default="N/A",
                )
            ),
        ),
        (
            "SHA-256",
            _safe_text(
                _first_available(
                    file_info,
                    "sha256",
                    default="N/A",
                )
            ),
        ),
    ]

    story.append(
        _key_value_table(
            file_rows
        )
    )

    story.append(
        _section_heading(
            "EXIF metadata"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    exif_data = _first_available(
        file_info,
        "exif",
        default={},
    )

    if not isinstance(
        exif_data,
        dict,
    ):
        exif_data = {}

    story.extend(
        _dictionary_table(
            exif_data,
            empty_message=(
                "No EXIF metadata available. The absence of EXIF metadata "
                "is not an automatic indication of manipulation."
            ),
        )
    )

    story.append(
        _section_heading(
            "JPEG information"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    if not is_jpeg_compatible:

        story.append(
            Paragraph(
                "Not applicable - the uploaded file is not JPEG-compatible.",
                style[
                    "body"
                ],
            )
        )

    elif is_mpo:

        story.append(
            Paragraph(
                (
                    "MPO container detected. The primary image is JPEG-compatible, "
                    "therefore the descriptive blockiness indicator can be reported. "
                    "Container-specific metadata may differ from a standard JPEG file."
                ),
                style[
                    "body"
                ],
            )
        )

        meaningful_jpeg_info = {
            key: value
            for key, value in jpeg_info.items()
            if (
                key != "is_jpeg"
                and value is not None
            )
        }

        if meaningful_jpeg_info:

            story.extend(
                _dictionary_table(
                    meaningful_jpeg_info,
                    empty_message=(
                        "No additional JPEG-compatible metadata is available."
                    ),
                )
            )

    else:

        story.extend(
            _dictionary_table(
                jpeg_info,
                empty_message=(
                    "JPEG information is unavailable."
                ),
            )
        )

    story.append(
        _section_heading(
            "Methodological note"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    methodological_paragraphs = [
        (
            "The confidence values and selective-classification thresholds "
            "were determined experimentally using FaceForensics++. They do "
            "not represent universal probabilities of authenticity or "
            "manipulation. Performance may change on images drawn from "
            "distributions different from the experimental dataset."
        ),
        (
            "<b>Multi-face report extension.</b><br/>"
            "Each detected face is analyzed separately using the same "
            "Xception and EfficientNet-B4 detector pipeline. The multi-face "
            "functionality is an application-level extension of the report: "
            "it does not retrain the models, modify calibrated probabilities, "
            "or change selective-classification thresholds. A result concerning "
            "one facial crop should not automatically be generalized to the "
            "entire image or to other detected faces."
        ),
        (
            "<b>Intended use of the report.</b><br/>"
            "The report combines detector outputs, confidence values, "
            "abstention status, Grad-CAM visualizations and technical "
            "indicators to support human assessment. None of these elements, "
            "individually or together, constitutes an automatic certification "
            "of image authenticity."
        ),
        (
            "<b>Scope limitation.</b><br/>"
            "REAL/FAKE predictions refer to patterns learned from "
            "FaceForensics++ and must not be interpreted as a general "
            "determination of whether the whole image was generated by AI."
        ),
    ]

    for paragraph_text in methodological_paragraphs:

        story.append(
            Paragraph(
                paragraph_text,
                style[
                    "note"
                ],
            )
        )


# ============================================================
# 10. PUBLIC PDF GENERATOR
# ============================================================

def generate_pdf_report(
    *,
    result: dict,
    original_image: Any,
    original_filename: str,
    generated_at: datetime | None = None,
) -> bytes:
    """
    Generate the complete multi-face technical report as PDF bytes.

    Parameters
    ----------
    result
        Final dictionary returned by the multi-face report pipeline.

    original_image
        Original uploaded image. A PIL RGB image is recommended.

    original_filename
        Original filename supplied by the user. This avoids exposing
        temporary filenames created internally by the Streamlit app.

    generated_at
        Optional datetime used in the report. When omitted, the current
        local timezone-aware datetime is used.

    Returns
    -------
    bytes
        Complete PDF document.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "result must be a dictionary."
        )

    if generated_at is None:

        generated_at = (
            datetime.now()
            .astimezone()
        )

    style = _styles()

    buffer = BytesIO()

    doc = _TechnicalReportDocument(
        buffer,
        title=(
            "Deepfake Analysis - "
            "Technical Support Report"
        ),
        author="Deepfake analysis support system",
        subject=(
            "Experimental multi-face technical report "
            "supporting human assessment."
        ),
    )

    story = []

    # --------------------------------------------------------
    # Front page
    # --------------------------------------------------------

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "Deepfake Analysis - Technical Support Report",
            style[
                "title"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Experimental multi-face system supporting human assessment "
                "of potentially manipulated images."
            ),
            style[
                "subtitle"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "The system provides technical evidence to support human "
                "assessment. Each detected face is analyzed separately. "
                "The system does not autonomously determine whether the image "
                "or any person depicted in it is authentic or manipulated."
            ),
            style[
                "note"
            ],
        )
    )

    story.append(
        _section_heading(
            "Report information"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    faces = (
        _get_faces(
            result
        )
    )

    faces_detected = _first_available(
        result,
        "faces_detected",
        "faces_analyzed",
        default=len(
            faces
        ),
    )

    overall_support_status = _safe_text(
        result.get(
            "overall_support_status"
        ),
        default=(
            _safe_text(
                result.get(
                    "support_status"
                )
            )
        ),
    )

    story.append(
        _key_value_table(
            [
                (
                    "Generated at",
                    generated_at.strftime(
                        "%Y-%m-%d %H:%M:%S %z"
                    ),
                ),
                (
                    "Original filename",
                    original_filename,
                ),
                (
                    "Faces detected",
                    faces_detected,
                ),
                (
                    "Overall support status",
                    overall_support_status,
                ),
            ]
        )
    )

    story.append(
        _section_heading(
            "Analyzed input"
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    original_pil = _to_pil_rgb(
        original_image,
        assume_bgr=False,
    )

    original_pil = (
        _apply_exif_orientation(
            original_pil,
            result,
        )
    )

    story.append(
        _pil_to_flowable(
            original_pil,
            max_width=CONTENT_WIDTH,
            max_height=105 * mm,
            jpeg_quality=88,
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                f"Faces detected: <b>{_clean_for_pdf(faces_detected)}</b>"
            ),
            style[
                "body"
            ],
        )
    )

    # --------------------------------------------------------
    # Individual faces
    # --------------------------------------------------------

    if faces:

        for position, face in enumerate(
            faces,
            start=1,
        ):

            face_number = face.get(
                "face_id",
                position,
            )

            _append_face_report(
                story,
                face,
                face_number,
            )

    else:

        story.append(
            Paragraph(
                (
                    "No valid facial crop was available for REAL/FAKE "
                    "classification."
                ),
                style[
                    "warning"
                ],
            )
        )

    # --------------------------------------------------------
    # File-level information and notes
    # --------------------------------------------------------

    _append_file_level_report(
        story,
        result,
        original_filename,
    )

    doc.build(
        story
    )

    pdf_bytes = buffer.getvalue()

    buffer.close()

    return pdf_bytes
