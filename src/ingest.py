"""
ingest.py — Document ingestion layer.

Supports:
  - .pdf  → text extraction via pypdf
  - .txt  → plain read
  - .jpg/.jpeg/.png → OCR via easyocr (only when --ocr flag is passed)

Returns a list of dicts:
  {
    "filename": str,       # basename only
    "filepath": str,       # full path
    "file_type": str,      # "pdf" | "txt" | "image"
    "text": str,           # extracted + cleaned text
    "ocr_used": bool       # whether OCR was applied
  }
"""

from pathlib import Path
from src.utils import clean_text, setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(filepath: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        import pypdf

        reader = pypdf.PdfReader(str(filepath))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        raw = "\n".join(pages_text)
        return clean_text(raw)
    except Exception as e:
        logger.warning(f"PDF extraction failed for '{filepath.name}': {e}")
        return ""


# ---------------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------------

def ingest_txt(filepath: Path) -> str:
    """Read a plain-text file."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        return clean_text(text)
    except Exception as e:
        logger.warning(f"TXT read failed for '{filepath.name}': {e}")
        return ""


# ---------------------------------------------------------------------------
# Image / OCR ingestion (optional)
# ---------------------------------------------------------------------------

def ingest_image(filepath: Path) -> str:
    """
    Run OCR on a JPG/PNG image using easyocr.

    NOTE: easyocr downloads model weights (~100MB) on the very first call.
    Subsequent calls are fully offline (models are cached locally).
    """
    try:
        import easyocr

        logger.info(f"Running OCR on '{filepath.name}' (easyocr)…")
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(str(filepath), detail=0, paragraph=True)
        raw = "\n".join(results)
        return clean_text(raw)
    except ImportError:
        logger.error(
            "easyocr is not installed. Install it with: pip install easyocr"
        )
        return ""
    except Exception as e:
        logger.warning(f"OCR failed for '{filepath.name}': {e}")
        return ""


# ---------------------------------------------------------------------------
# Unified ingest dispatcher
# ---------------------------------------------------------------------------

def ingest_file(filepath: Path, use_ocr: bool = False) -> dict:
    """
    Ingest a single file and return its metadata + extracted text.

    Args:
        filepath:   Path to the document.
        use_ocr:    If True, process image files with easyocr.
                    If False, image files are skipped (text = "").

    Returns:
        dict with keys: filename, filepath, file_type, text, ocr_used
    """
    ext = filepath.suffix.lower()
    result = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "file_type": "unknown",
        "text": "",
        "ocr_used": False,
    }

    if ext == ".pdf":
        result["file_type"] = "pdf"
        result["text"] = ingest_pdf(filepath)

    elif ext == ".txt":
        result["file_type"] = "txt"
        result["text"] = ingest_txt(filepath)

    elif ext in {".jpg", ".jpeg", ".png"}:
        result["file_type"] = "image"
        if use_ocr:
            result["text"] = ingest_image(filepath)
            result["ocr_used"] = True
        else:
            logger.info(
                f"Skipping image '{filepath.name}' (pass --ocr to enable OCR)"
            )

    else:
        logger.warning(f"Unsupported file type '{ext}' for '{filepath.name}'")

    if result["text"]:
        logger.info(
            f"Ingested '{filepath.name}' [{result['file_type']}] — "
            f"{len(result['text'])} chars"
        )
    else:
        logger.warning(f"No text extracted from '{filepath.name}'")

    return result


def ingest_all(input_folder: str, use_ocr: bool = False) -> list[dict]:
    """
    Ingest all supported files in input_folder.

    Args:
        input_folder: Path to the directory containing input documents.
        use_ocr:      Enable OCR for image files.

    Returns:
        List of ingested document dicts.
    """
    from src.utils import get_input_files

    files = get_input_files(input_folder, include_images=use_ocr)
    if not files:
        logger.warning("No files found to ingest.")
        return []

    docs = []
    for f in files:
        doc = ingest_file(f, use_ocr=use_ocr)
        docs.append(doc)

    logger.info(f"Ingestion complete: {len(docs)} document(s) processed.")
    return docs
