import base64
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from dotenv import load_dotenv

from utils import preprocess_for_colored_images, preprocess_for_google_vision


load_dotenv()


GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_KEY")


@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: List[List[int]]


@dataclass
class OCRBlock:
    text: str
    confidence: float
    bbox: List[List[int]]
    words: List[OCRWord] = field(default_factory=list)


@dataclass
class OCRPageResult:
    full_text: str
    blocks: List[OCRBlock]
    provider: str
    provider_version: str
    raw_response: Optional[dict] = None


class OCRProvider(ABC):
    @abstractmethod
    def recognize(
        self,
        image_bytes: bytes,
        language_hints: Optional[List[str]] = None,
        page_type: str = "plain",
        preprocessed_image_bytes: Optional[bytes] = None,
    ) -> OCRPageResult:
        ...


def prepare_image_for_ocr(image_bytes: bytes, page_type: str = "plain") -> bytes:
    if page_type == "plain":
        return preprocess_for_google_vision(image_bytes)
    return preprocess_for_colored_images(image_bytes)


class GoogleVisionProvider(OCRProvider):
    NAME = "google_cloud_vision"
    VERSION = "v1"
    ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries

    def recognize(
        self,
        image_bytes: bytes,
        language_hints: Optional[List[str]] = None,
        page_type: str = "plain",
        preprocessed_image_bytes: Optional[bytes] = None,
    ) -> OCRPageResult:
        language_hints = language_hints or ["ar"]

        if preprocessed_image_bytes is None:
            image_bytes = prepare_image_for_ocr(image_bytes, page_type=page_type)
        else:
            image_bytes = preprocessed_image_bytes

        if page_type == "plain":
            print("applying complete preprocessings for plain image (black & white)")
        else:
            print("applying partial preprocessings for colored image (graphical-heavy layout)")

        payload = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": language_hints},
            }]
        }

        last_error = None
        for attempt in range(self.max_retries):
            resp = requests.post(f"{self.ENDPOINT}?key={self.api_key}", json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()["responses"][0]
                if "error" in data:
                    raise RuntimeError(f"Vision API error: {data['error']}")
                return self._parse(data)
            last_error = resp.text
            time.sleep(2 ** attempt)

        raise RuntimeError(f"Vision API failed after {self.max_retries} attempts: {last_error}")

    def _parse(self, data: dict) -> OCRPageResult:
        full_text = data.get("fullTextAnnotation", {}).get("text", "")
        blocks = []

        for page in data.get("fullTextAnnotation", {}).get("pages", []):
            for block in page.get("blocks", []):
                block_words = []
                block_text_parts = []
                confidences = []

                for para in block.get("paragraphs", []):
                    for word in para.get("words", []):
                        w_text = ""
                        symbols = word.get("symbols", [])

                        for idx, symbol in enumerate(symbols):
                            w_text += symbol.get("text", "")

                            detected_break = symbol.get("property", {}).get("detectedBreak", {})
                            break_type = detected_break.get("type", "")

                            if break_type in ("SPACE", "SURE_SPACE") and idx < len(symbols) - 1:
                                w_text += " "

                        w_conf = word.get("confidence", 0.0)
                        block_words.append(OCRWord(
                            text=w_text,
                            confidence=w_conf,
                            bbox=[[v.get("x", 0), v.get("y", 0)] for v in word["boundingBox"]["vertices"]],
                        ))

                        block_text_parts.append(w_text)
                        confidences.append(w_conf)

                        if symbols:
                            last_symbol_break = symbols[-1].get("property", {}).get("detectedBreak", {})
                            last_break_type = last_symbol_break.get("type", "")

                            if last_break_type in ("SPACE", "SURE_SPACE"):
                                block_text_parts.append(" ")
                            elif last_break_type in ("LINE_BREAK", "EOL_SURE_SPACE"):
                                block_text_parts.append("\n")
                            else:
                                block_text_parts.append(" ")
                        else:
                            block_text_parts.append(" ")

                    block_text_parts.append("\n")

                block_text = "".join(block_text_parts).strip()

                blocks.append(OCRBlock(
                    text=block_text,
                    confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                    bbox=[[v.get("x", 0), v.get("y", 0)] for v in block["boundingBox"]["vertices"]],
                    words=block_words,
                ))

        return OCRPageResult(
            full_text=full_text,
            blocks=blocks,
            provider=self.NAME,
            provider_version=self.VERSION,
            raw_response=data,
        )
