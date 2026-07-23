import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = "output/plain_arabic_cloudvision/pages/plain_arabic_cloudvision_p0001.json"
DEFAULT_OUTPUT_DIR = "output/labina_demo"
REVIEW_ARTIFACTS = [
    "knowledge_units_approved.json",
    "knowledge_units_approved.csv",
    "knowledge_units_rejected.json",
    "knowledge_units_rejected.csv",
    "knowledge_unit_proposals_remaining.json",
    "knowledge_unit_proposals_remaining.csv",
    "approval_log.json",
    "approval_log.csv",
    "rejection_log.json",
    "rejection_log.csv",
    "review_log.json",
    "review_log.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_page_record(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, rows: Any) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def body_text_from_page(page_record: Dict[str, Any]) -> str:
    body_segments = [
        segment.get("text", "")
        for segment in page_record.get("segments", [])
        if segment.get("type") == "body" and compact_text(segment.get("text", ""))
    ]
    return "\n".join(body_segments) if body_segments else page_record.get("raw_text", "")


def split_sentences(text: str) -> List[str]:
    text = compact_text(text)
    if not text:
        return []

    # Arabic punctuation and common discourse markers keep custom mode deterministic.
    parts = re.split(r"(?<=[.!؟؛])\s+|(?<=\])\s+(?=و|ف|وقيل|وحكمة)", text)
    return [compact_text(part) for part in parts if len(compact_text(part)) >= 20]


def source_page_row(page_record: Dict[str, Any], input_path: Path, book_title: str) -> Dict[str, Any]:
    meta = page_record["source_metadata"]
    quality = page_record.get("quality", {})
    return {
        "id": meta["page_id"],
        "book_id": meta["book_id"],
        "book_title": book_title,
        "source_file_name": meta.get("source_file_name", input_path.name),
        "source_file_hash_sha256": meta.get("source_file_hash_sha256", ""),
        "page_number": meta.get("page_number"),
        "raw_text": page_record.get("raw_text", ""),
        "ocr_provider": meta.get("processing_stack", {}).get("ocr_provider", ""),
        "ocr_mean_confidence": quality.get("mean_confidence", ""),
        "reviewed_at": utc_now(),
        "layer": "Layer 0 - raw OCR page",
    }


def make_unit(
    index: int,
    page_row: Dict[str, Any],
    source_text: str,
    extractor: str,
    unit_type: str = "claim",
    grounding_status: str = "verified_source_span",
) -> Dict[str, Any]:
    unit_id = f"{page_row['id']}_labina_{index:03d}"
    return {
        "id": unit_id,
        "source_page_id": page_row["id"],
        "book_id": page_row["book_id"],
        "book_title": page_row["book_title"],
        "page_number": page_row["page_number"],
        "source_reference": f"{page_row['book_id']} p.{page_row['page_number']}",
        "source_text": source_text,
        "unit_text": source_text,
        "unit_type": unit_type,
        "status": "proposed",
        "proposed_by": extractor,
        "proposed_at": utc_now(),
        "approved_by": "",
        "approved_at": "",
        "rejected_by": "",
        "rejected_at": "",
        "rejection_reason": "",
        "grounding_status": grounding_status,
        "review_note": "Human approval required before this becomes an approved sourced unit.",
        "layer": "Layer 1 - proposed knowledge unit",
    }


def custom_extract(page_record: Dict[str, Any], page_row: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    text = body_text_from_page(page_record)
    candidates = split_sentences(text)

    units: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        if len(units) >= limit:
            break
        if candidate in seen:
            continue
        seen.add(candidate)
        units.append(make_unit(len(units) + 1, page_row, candidate, "custom_extractor"))

    return units


def llm_extract(page_record: Dict[str, Any], page_row: Dict[str, Any], limit: int, model_name: str) -> List[Dict[str, Any]]:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY is required for --mode llm.")

    client = genai.Client(api_key=api_key)
    text = body_text_from_page(page_record)
    compact_source = compact_text(text)
    prompt = f"""
You are proposing Layer 1 labina candidates from a reviewed Layer 0 Arabic OCR page.

Rules:
- Return 5 to {limit} atomic knowledge units.
- One labina must contain one self-contained claim or idea.
- Do not translate.
- Do not add outside knowledge.
- Keep source_text as an exact direct span from the supplied OCR text.
- unit_text may lightly trim spacing but must not change meaning.
- Every unit must remain status="proposed"; a human approves later.

Return JSON only:
{{"units":[{{"source_text":"...","unit_text":"...","unit_type":"claim"}}]}}

OCR body text:
{text[:6000]}
"""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=config,
    )
    payload = json.loads(response.text)
    raw_units = payload.get("units", [])[:limit]

    units = []
    rejected_spans = 0
    for raw in raw_units:
        source_text = compact_text(raw.get("source_text", ""))
        if not source_text:
            continue
        if source_text not in compact_source:
            rejected_spans += 1
            continue
        unit = make_unit(
            len(units) + 1,
            page_row,
            source_text,
            f"llm:{model_name}",
            raw.get("unit_type", "claim"),
        )
        unit["unit_text"] = compact_text(raw.get("unit_text", source_text))
        units.append(unit)

    if rejected_spans:
        print(f"[grounding] Rejected {rejected_spans} LLM unit(s) because source_text was not found in OCR text.")
    return units


def build_demo(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in REVIEW_ARTIFACTS:
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    page_record = read_page_record(input_path)
    page_row = source_page_row(page_record, input_path, args.book_title)

    if args.mode == "custom":
        units = custom_extract(page_record, page_row, args.limit)
    else:
        units = llm_extract(page_record, page_row, args.limit, args.model)

    if not units:
        raise RuntimeError("No labina proposals were generated.")

    raw_rows = [page_row]
    manifest = {
        "demo_name": "Layer 1 labina proposal demo",
        "created_at": utc_now(),
        "input_page_json": str(input_path),
        "extraction_mode": args.mode,
        "client_requirement_covered": [
            "Layer 0 raw OCR stored separately",
            "Layer 1 labina proposals stored in a separate table",
            "Each labina has book and page source reference",
            "Labinas are proposed, not approved automatically",
            "Human approval or rejection is a separate action",
            "LLM source_text spans are checked against OCR text before storage",
        ],
        "tables": {
            "raw_ocr_pages": "raw_ocr_pages.json / raw_ocr_pages.csv",
            "knowledge_unit_proposals": "knowledge_unit_proposals.json / knowledge_unit_proposals.csv",
        },
    }

    write_json(output_dir / "raw_ocr_pages.json", raw_rows)
    write_csv(output_dir / "raw_ocr_pages.csv", raw_rows)
    write_json(output_dir / "knowledge_unit_proposals.json", units)
    write_csv(output_dir / "knowledge_unit_proposals.csv", units)
    write_json(output_dir / "manifest.json", manifest)

    print(f"[done] Raw OCR table rows: {len(raw_rows)}")
    print(f"[done] Proposed labinas: {len(units)}")
    print(f"[done] Output: {output_dir.resolve()}")
    print("[next] Human approval example:")
    print(f"       python approve_labinas_demo.py \"{output_dir}\" --reviewer \"human_reviewer\" --approve-all")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create separate Layer 0 and Layer 1 labina demo tables.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Reviewed Layer 0 page JSON.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for demo table outputs.")
    parser.add_argument("--book-title", default="Plain Arabic OCR Trial Book", help="Human-readable source title.")
    parser.add_argument("--mode", choices=["custom", "llm"], default="custom", help="Labina proposal strategy.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum proposed labinas. Use 5-10 for the client demo.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model for --mode llm.")
    return parser.parse_args()


if __name__ == "__main__":
    build_demo(parse_args())
