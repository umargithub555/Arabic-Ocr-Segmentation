import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


class BlockClassification(BaseModel):
    index: int = Field(description="The index of the OCR block being classified")
    type: str = Field(description="One of: header, page_number, body, footnote, footnote_marker, formula, table, unknown")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class ClassificationResponse(BaseModel):
    classifications: List[BlockClassification]


def _field(block: Any, name: str, fallback: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, fallback)
    return getattr(block, name, fallback)


class SegmentationProvider(ABC):
    @abstractmethod
    def classify(
        self,
        blocks: List[Union[dict, Any]],
        page_width: int,
        page_height: int,
        image_bytes: bytes,
        page_profile: str = "plain_arabic_scan",
    ) -> Dict[str, Any]:
        ...


class GeminiSegmentationProvider(SegmentationProvider):
    ALLOWED_TYPES = [
        "header",
        "page_number",
        "body",
        "footnote",
        "footnote_marker",
        "formula",
        "table",
        "unknown",
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def classify(
        self,
        blocks: List[Union[dict, Any]],
        page_width: int,
        page_height: int,
        image_bytes: bytes,
        page_profile: str = "plain_arabic_scan",
    ) -> Dict[str, Any]:
        if not blocks:
            return self._wrap_qcmf_envelope([], page_width, page_height, page_profile)

        block_descriptions = []
        for idx, block in enumerate(blocks):
            bbox = _field(block, "bbox", [])
            text = _field(block, "text", "")

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            block_descriptions.append({
                "index": idx,
                "text_preview": text[:200],
                "bbox_pixels": {
                    "x0": min(xs) if xs else 0,
                    "y0": min(ys) if ys else 0,
                    "x1": max(xs) if xs else 0,
                    "y1": max(ys) if ys else 0,
                },
            })

        prompt = f"""
You are an expert layout parser. Analyze the provided page image alongside the coordinates of the OCR-extracted text blocks.

Your ONLY task is to classify the physical or logical layout type of each OCR block.
DO NOT modify, translate, correct, or interpret the Arabic text.
This is a Layer 0 source-preservation task. Only classify text-bearing OCR blocks.

Allowed types:
- header (titles, chapter headings, running headers)
- page_number (isolated page numbers)
- body (main paragraphs and standard running text)
- footnote (text at the bottom of the page, including scholarly notes)
- footnote_marker (isolated reference markers such as (1), ١, or letter markers)
- formula (mathematical expressions or equation-like OCR blocks)
- table (structured tabular data)
- unknown (any other text-bearing segment)

Page Details:
Width: {page_width}px, Height: {page_height}px
Page profile: {page_profile}
"""

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationResponse,
            temperature=0.0,
            system_instruction="You are a layout classifier for Arabic source pages. Output only structured JSON matching the requested schema.",
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                prompt,
                json.dumps(block_descriptions, ensure_ascii=False),
                image_part,
            ],
            config=config,
        )

        classifications: List[dict] = []
        try:
            parsed: ClassificationResponse = response.parsed
            if parsed is not None:
                classifications = [c.model_dump() for c in parsed.classifications]
        except Exception:
            classifications = []

        if not classifications:
            try:
                raw = json.loads(response.text)
                classifications = raw.get("classifications", [])
            except Exception:
                classifications = []

        lookup = {
            item["index"]: {
                "type": item.get("type", "unknown") if item.get("type") in self.ALLOWED_TYPES else "unknown",
                "confidence": max(0.0, min(float(item.get("confidence", 0.0)), 1.0)),
            }
            for item in classifications
        }

        segmented_blocks = []
        for idx, block in enumerate(blocks):
            bbox = _field(block, "bbox", [])
            text = _field(block, "text", "")
            ocr_conf = _field(block, "confidence", 1.0)

            match = lookup.get(idx, {"type": "unknown", "confidence": 0.0})
            review_reasons = self._review_reasons(match["type"], match["confidence"])

            segmented_blocks.append({
                "segment_id": f"seg_{idx + 1:03d}",
                "type": match["type"],
                "reading_order": idx + 1,
                "text": text,
                "bounding_box": bbox,
                "confidence_scores": {
                    "ocr": round(ocr_conf, 4),
                    "layout_classification": round(match["confidence"], 4),
                },
                "needs_review": bool(review_reasons),
                "review_reasons": review_reasons,
            })

        return self._wrap_qcmf_envelope(segmented_blocks, page_width, page_height, page_profile)

    def _wrap_qcmf_envelope(self, segments: List[dict], width: int, height: int, page_profile: str) -> Dict[str, Any]:
        return {
            "qcmf_version": "L1-Layer0",
            "metadata": {
                "stewardship_station": "Layer 0 - Acquisition & Segment Preservation",
                "page_dimensions": {"width": width, "height": height},
                "page_profile": page_profile,
                "processed_by": f"GeminiSegmentationProvider ({self.model_name})",
            },
            "source_segments": segments,
        }

    @staticmethod
    def _review_reasons(segment_type: str, confidence: float) -> List[str]:
        reasons = []
        if segment_type == "unknown":
            reasons.append("unknown_segment_type")
        if segment_type == "formula":
            reasons.append("formula_requires_source_fidelity_review")
        if confidence < 0.55:
            reasons.append("low_layout_confidence")
        return reasons
