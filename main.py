import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from ocr import GoogleVisionProvider, prepare_image_for_ocr
from segmentation import GeminiSegmentationProvider
from utils import (
    compute_quality,
    detect_page_profile,
    draw_ocr_bounding_boxes,
    image_file_to_page,
    pdf_to_page_images,
    reconstruction_check,
    sha256_of_file,
)


load_dotenv()


FOOTNOTE_MARKER_PATTERN = __import__('re').compile(r"\(\d+\)|[\u0660-\u0669]+|[A-Za-z]\)|[\u0621-\u064A]\)")


def build_ocr_provider() -> GoogleVisionProvider:
    api_key = os.getenv("GOOGLE_VISION_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_VISION_KEY in environment.")
    return GoogleVisionProvider(api_key=api_key)


def build_segmenter() -> GeminiSegmentationProvider:
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_KEY in environment.")
    return GeminiSegmentationProvider(api_key=api_key)


def flag_multi_marker_footnotes(segments: List[dict]) -> List[dict]:
    updated = []
    for segment in segments:
        text = segment.get("text", "")
        markers = FOOTNOTE_MARKER_PATTERN.findall(text)
        segment_copy = dict(segment)
        if segment_copy.get("type") == "footnote" and len(markers) > 1:
            segment_copy["needs_review"] = True
            review_reasons = list(segment_copy.get("review_reasons", []))
            review_reasons.append("multiple_footnote_markers_detected")
            segment_copy["review_reasons"] = review_reasons
        updated.append(segment_copy)
    return updated


def iter_document_pages(input_path: str, render_dpi: int = 300):
    suffix = Path(input_path).suffix.lower()
    if suffix == ".pdf":
        yield from pdf_to_page_images(input_path, dpi=render_dpi)
        return

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        yield from image_file_to_page(input_path)
        return

    raise ValueError(f"Unsupported input type: {suffix}")


def process_book(
    input_path: str,
    book_id: str,
    language_hints: List[str],
    output_root: str = "output",
    render_dpi: int = 300,
):
    provider = build_ocr_provider()
    segmenter = build_segmenter()

    out_dir = Path(output_root) / book_id
    pages_dir = out_dir / "pages"
    debug_dir = out_dir / "debug"
    pages_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(input_path)
    original_copy = out_dir / f"{book_id}_original{source_path.suffix.lower()}"
    if not original_copy.exists():
        shutil.copy(input_path, original_copy)

    file_hash = sha256_of_file(input_path)
    book_pages = []

    for page_num, png_bytes, width, height in iter_document_pages(input_path, render_dpi=render_dpi):
        page_id = f"{book_id}_p{page_num:04d}"
        txt_path = pages_dir / f"{page_id}.txt"
        json_path = pages_dir / f"{page_id}.json"

        if json_path.exists():
            print(f"[skip] {page_id} already processed")
            with open(json_path, "r", encoding="utf-8") as f:
                book_pages.append(json.load(f))
            continue

        page_profile = detect_page_profile(png_bytes)
        processed_bytes = prepare_image_for_ocr(png_bytes, page_type=page_profile["ocr_preprocess_mode"])

        preprocessed_path = debug_dir / f"{page_id}_preprocessed.png"
        if not preprocessed_path.exists():
            with open(preprocessed_path, "wb") as f:
                f.write(processed_bytes)

        print(f"[ocr]  {page_id} ({page_profile['profile']}) ...")
        ocr_result = provider.recognize(
            png_bytes,
            language_hints=language_hints,
            page_type=page_profile["ocr_preprocess_mode"],
            preprocessed_image_bytes=processed_bytes,
        )

        overlay_bytes = draw_ocr_bounding_boxes(processed_bytes, ocr_result.blocks)
        overlay_path = debug_dir / f"{page_id}_ocr_boxes.png"
        if not overlay_path.exists():
            with open(overlay_path, "wb") as f:
                f.write(overlay_bytes)

        print(f"[seg]  {page_id} ({page_profile['profile']}) ...")
        seg_envelope = segmenter.classify(
            ocr_result.blocks,
            width,
            height,
            image_bytes=png_bytes,
            page_profile=page_profile["profile"],
        )
        segments = flag_multi_marker_footnotes(seg_envelope["source_segments"])

        quality = compute_quality(ocr_result, segments, page_profile["profile"])
        recon = reconstruction_check(ocr_result.full_text, segments)

        page_record = {
            "source_metadata": {
                "book_id": book_id,
                "page_id": page_id,
                "page_number": page_num,
                "source_file_name": source_path.name,
                "source_file_hash_sha256": file_hash,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "language_hints": language_hints,
                "image_dimensions": {"width": width, "height": height},
                "render_dpi": render_dpi,
                "page_profile": page_profile,
                "processing_stack": {
                    "ocr_provider": ocr_result.provider,
                    "ocr_provider_version": ocr_result.provider_version,
                    "segmentation_provider": "gemini",
                    "segmentation_model": segmenter.model_name,
                },
            },
            "quality": quality,
            "reconstruction_check": recon,
            "raw_text": ocr_result.full_text,
            "ocr_blocks": [
                {
                    "block_id": f"ocr_{idx + 1:03d}",
                    "text": block.text,
                    "bounding_box": block.bbox,
                    "confidence": round(block.confidence, 4),
                    "word_count": len(block.words),
                }
                for idx, block in enumerate(ocr_result.blocks)
            ],
            "segments": segments,
        }

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(ocr_result.full_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_record, f, ensure_ascii=False, indent=2)

        book_pages.append(page_record)

    pages_needing_review = [
        p["source_metadata"]["page_id"]
        for p in book_pages
        if p["quality"]["flag_needs_review"]
        or not p["reconstruction_check"]["passed"]
        or any(s.get("needs_review") for s in p["segments"])
    ]

    book_level = {
        "book_id": book_id,
        "source_file_name": source_path.name,
        "source_file_hash_sha256": file_hash,
        "page_count": len(book_pages),
        "pages_needing_review": pages_needing_review,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deliverables": {
            "original_source_copy": str(original_copy.name),
            "page_text_format": "txt",
            "page_json_format": "layer0-review-json",
            "debug_images": ["preprocessed", "ocr_boxes"],
        },
    }
    with open(out_dir / f"{book_id}_book_level.json", "w", encoding="utf-8") as f:
        json.dump(book_level, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(book_pages)} pages -> {out_dir}")
    print(f"Pages flagged for review: {pages_needing_review}")
    return book_pages


def parse_args():
    parser = argparse.ArgumentParser(description="Layer 0 OCR + segmentation pipeline for Arabic source pages.")
    parser.add_argument("input_path", help="Path to a PDF or page image.")
    parser.add_argument("book_id", help="Stable identifier for the source document.")
    parser.add_argument(
        "--language-hint",
        dest="language_hints",
        action="append",
        default=None,
        help="OCR language hint. Repeat to pass multiple hints. Defaults to ar.",
    )
    parser.add_argument("--output-root", default="output", help="Root directory for generated outputs.")
    parser.add_argument("--render-dpi", type=int, default=300, help="Render DPI for PDF inputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_book(
        input_path=args.input_path,
        book_id=args.book_id,
        language_hints=args.language_hints or ["ar"],
        output_root=args.output_root,
        render_dpi=args.render_dpi,
    )
