# Arabic OCR Segmentation

Layer 0 OCR and segmentation pipeline for Arabic source pages.

This project is set up for source-preserving processing of Arabic PDFs or single-page images. It keeps the original file, produces raw OCR text, writes reviewable JSON, and saves debug images for inspection.

## What it does

- Preserves the original input file
- Runs Arabic-first OCR with Google Cloud Vision
- Classifies OCR blocks with Gemini for Layer 0 segmentation
- Saves per-page `.txt` and `.json` outputs
- Saves debug images:
  - OCR-preprocessed image
  - OCR overlay with bounding boxes
- Generates a book-level review summary

## Requirements

- Python 3.12+
- Google Cloud Vision API key
- Gemini API key

Create a `.env` file with:

```env
GOOGLE_VISION_KEY=your_google_vision_key
GEMINI_KEY=your_gemini_key
```

## Install

```bash
pip install -r requirements.txt
```

## Run a single image

```bash
python main.py "test.png" plain_sample
```

Example with a math page:

```bash
python main.py "test2.png" math_sample
```

## Run a PDF

```bash
python main.py "sample_math_5pages.pdf" math_book
```

## Output structure

```text
output/
  book_id/
    book_id_original.png|pdf
    book_id_book_level.json
    pages/
      book_id_p0001.txt
      book_id_p0001.json
    debug/
      book_id_p0001_preprocessed.png
      book_id_p0001_ocr_boxes.png
```

## Notes

- `book_id` is just a stable name for the source item you are processing.
- The pipeline auto-detects whether a page looks more like a plain scan or a color-heavy textbook page.
- The OCR preprocessing path is chosen automatically, but you can inspect the saved debug images to review how the page changed before OCR.

