"""
utils.py — Shared helpers: text cleaning, file discovery, logging setup.
"""

import os
import re
import logging
import unicodedata
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".jpg", ".jpeg", ".png"}


def setup_logger(name: str = "docproc", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text:
    - Normalize unicode characters to ASCII-safe form.
    - Collapse multiple whitespace/newlines.
    - Strip leading/trailing whitespace.
    """
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Replace non-ASCII characters with a space
    text = text.encode("ascii", "ignore").decode("ascii")
    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_input_files(input_folder: str, include_images: bool = False) -> list[Path]:
    """
    Return a sorted list of supported files in input_folder.

    Args:
        input_folder: Path to the directory containing input documents.
        include_images: If True, include .jpg/.jpeg/.png files (requires OCR).

    Returns:
        List of Path objects for each supported file found.
    """
    folder = Path(input_folder)
    if not folder.exists():
        logger.warning(f"Input folder does not exist: {folder}")
        return []

    files = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in {".pdf", ".txt"}:
            files.append(f)
        elif include_images and ext in {".jpg", ".jpeg", ".png"}:
            files.append(f)

    logger.info(f"Found {len(files)} file(s) in '{input_folder}'")
    return files


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_nonempty_line(text: str) -> str:
    """Return the first non-empty, non-whitespace line from text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
