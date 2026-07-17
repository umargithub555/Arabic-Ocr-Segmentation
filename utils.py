import cv2
import fitz
import hashlib
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ocr import OCRPageResult


def _block_bbox(block):
    if isinstance(block, dict):
        return block.get("bbox", [])
    return getattr(block, "bbox", [])


def deskew(image):
    """
    Automatically deskew an image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    M = cv2.getRotationMatrix2D(
        (image.shape[1] // 2, image.shape[0] // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def crop_white_border(image):
    """
    Remove large white borders.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        250,
        255,
        cv2.THRESH_BINARY_INV
    )

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    return image[y:y + h, x:x + w]


def enhance_contrast(gray):
    """
    CLAHE improves faded scans.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def resize_for_ocr(gray, target_width=2500):
    """
    Upscale if image is too small.
    """

    _, w = gray.shape

    if w >= target_width:
        return gray

    scale = target_width / w

    return cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def sharpen(gray):
    """
    Mild sharpening.
    """

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(gray, -1, kernel)


def preprocess_for_google_vision(image_bytes):
    """
    Returns PNG bytes ready for Google Vision OCR.
    """

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid image.")

    image = crop_white_border(image)
    image = deskew(image)

    image = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        3,
        3,
        7,
        21
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = enhance_contrast(gray)
    gray = resize_for_ocr(gray)
    gray = sharpen(gray)

    success, encoded = cv2.imencode(".png", gray)

    if not success:
        raise RuntimeError("Failed to encode processed image.")

    return encoded.tobytes()


def preprocess_for_colored_images(image_bytes):
    """
    Preprocessing path tuned for colorful textbook pages with Arabic text.
    """
    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("Invalid image.")

    image = deskew(image)
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    gray = resize_for_ocr(gray, target_width=2000)

    success, encoded = cv2.imencode(".png", gray)
    if not success:
        raise RuntimeError("Failed to encode processed image.")

    return encoded.tobytes()


def detect_page_profile(image_bytes: bytes) -> dict:
    """
    Detect whether a page behaves more like a plain text scan or a color/layout-heavy textbook page.
    This only chooses preprocessing strategy; it does not alter deliverable semantics.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    colored_ratio = float(np.mean(saturation > 35))
    mean_saturation = float(np.mean(saturation))

    if colored_ratio >= 0.03 or mean_saturation >= 18:
        profile = "colored_textbook"
        preprocess_mode = "colored"
    else:
        profile = "plain_arabic_scan"
        preprocess_mode = "plain"

    return {
        "profile": profile,
        "ocr_preprocess_mode": preprocess_mode,
        "color_signal": {
            "colored_pixel_ratio": round(colored_ratio, 4),
            "mean_saturation": round(mean_saturation, 2),
        },
    }


def compute_quality(ocr_result: "OCRPageResult", segments: List[dict], page_profile: str) -> dict:
    all_words = [w for b in ocr_result.blocks for w in b.words]
    confidences = [w.confidence for w in all_words]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < 0.75)
    low_conf_ratio = (low_conf_count / len(confidences)) if confidences else 1.0
    unknown_segment_count = sum(1 for s in segments if s.get("type") == "unknown")
    formula_segment_count = sum(1 for s in segments if s.get("type") == "formula")
    body_segment_count = sum(1 for s in segments if s.get("type") == "body")

    notes = []
    if not ocr_result.full_text.strip():
        notes.append("empty_page_no_text_detected")
    if mean_conf < 0.80 and confidences:
        notes.append("low_mean_confidence")
    if low_conf_ratio > 0.20 and confidences:
        notes.append("high_low_confidence_ratio")
    if segments and unknown_segment_count / len(segments) > 0.30:
        notes.append("high_unknown_segment_ratio")
    if page_profile == "plain_arabic_scan" and body_segment_count == 0:
        notes.append("no_body_segment_detected")
    if formula_segment_count > 0:
        notes.append("formula_segments_require_visual_review")

    return {
        "mean_confidence": round(mean_conf, 4),
        "word_count": len(all_words),
        "low_confidence_word_count": low_conf_count,
        "low_confidence_word_ratio": round(low_conf_ratio, 4),
        "ocr_block_count": len(ocr_result.blocks),
        "segment_count": len(segments),
        "unknown_segment_count": unknown_segment_count,
        "formula_segment_count": formula_segment_count,
        "flag_needs_review": bool(notes),
        "notes": notes,
    }


def reconstruction_check(raw_text: str, segments: List[dict]) -> dict:
    joined = "\n".join(s["text"] for s in segments)
    raw_len = len(raw_text.replace("\n", "").replace(" ", ""))
    joined_len = len(joined.replace("\n", "").replace(" ", ""))
    diff = abs(raw_len - joined_len)
    return {
        "raw_char_count": raw_len,
        "segmented_char_count": joined_len,
        "char_diff": diff,
        "passed": diff <= max(5, int(raw_len * 0.01)),
    }


def draw_ocr_bounding_boxes(image_bytes: bytes, blocks, box_color=(0, 0, 255), label_color=(255, 255, 255)) -> bytes:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image.")

    for idx, block in enumerate(blocks, start=1):
        bbox = _block_bbox(block)
        if not bbox:
            continue

        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=box_color, thickness=3)

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, y0 = min(xs), min(ys)
        label = str(idx)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_top = max(0, y0 - text_h - baseline - 6)
        label_bottom = max(0, y0 - 2)
        cv2.rectangle(
            image,
            (x0, label_top),
            (x0 + text_w + 8, label_bottom),
            box_color,
            thickness=-1,
        )
        cv2.putText(
            image,
            label,
            (x0 + 4, max(12, label_bottom - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode OCR overlay image.")
    return encoded.tobytes()


def pdf_to_page_images(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        yield i, pix.tobytes("png"), pix.width, pix.height
    doc.close()


def image_file_to_page(image_path: str):
    path = Path(image_path)
    image_bytes = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid image file: {image_path}")
    height, width = image.shape[:2]
    yield 1, image_bytes, width, height


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
