"""
main.py — CLI entry point for the Local AI Document Processing pipeline.

Commands:
  python main.py process          Process all docs in data/input/ → output.json
  python main.py process --ocr    Also run OCR on image files (JPG/PNG)
  python main.py search "query"   Semantic search over processed documents
  python main.py search "query" --top 10  Return top 10 results

All processing is local. No external APIs are called.
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is in sys.path so 'src' package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, ensure_dir
from src.ingest import ingest_all
from src.classify import classify, classify_by_filename
from src.extract import extract_fields
from src.retrieval import build_index, search as semantic_search

logger = setup_logger()

INPUT_FOLDER = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = PROJECT_ROOT / "output.json"


# ---------------------------------------------------------------------------
# Process command
# ---------------------------------------------------------------------------

def cmd_process(use_ocr: bool = False) -> None:
    """
    Full pipeline:
      1. Ingest all documents from data/input/
      2. Classify each document
      3. Extract structured fields
      4. Build semantic search index
      5. Write output.json
    """
    logger.info("=" * 60)
    logger.info("Starting document processing pipeline")
    logger.info(f"  Input folder : {INPUT_FOLDER}")
    logger.info(f"  Output file  : {OUTPUT_FILE}")
    logger.info(f"  OCR enabled  : {use_ocr}")
    logger.info("=" * 60)

    ensure_dir(str(INPUT_FOLDER))

    # Step 1: Ingest
    docs = ingest_all(str(INPUT_FOLDER), use_ocr=use_ocr)
    if not docs:
        logger.error("No documents found in data/input/. Run prepare_sample_dataset.py first.")
        sys.exit(1)

    # Step 2 & 3: Classify + Extract
    output: dict = {}
    for doc in docs:
        filename = doc["filename"]
        text = doc["text"]

        # Try fast filename-based classification first
        doc_class = classify_by_filename(filename) or classify(text)

        # Extract fields based on class
        fields = extract_fields(doc_class, text)

        # Build output record
        record = {"class": doc_class}
        record.update({k: v for k, v in fields.items() if v is not None})

        output[filename] = record
        logger.info(f"  {filename:45s} → {doc_class}")

    # Step 4: Write output.json FIRST — always save results before attempting model download
    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"output.json written: {OUTPUT_FILE}")

    # Step 5: Build semantic index (requires model download on first run)
    logger.info("Building semantic search index…")
    try:
        build_index(docs)
    except Exception as e:
        logger.warning(f"Semantic index build failed: {e}")
        logger.warning(
            "output.json has been saved. Re-run 'python main.py process' "
            "when your internet connection is stable to build the search index."
        )

    logger.info("=" * 60)
    logger.info(f"Done! Processed {len(output)} document(s).")
    logger.info(f"Results written to: {OUTPUT_FILE}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

def cmd_search(query: str, top_k: int = 5) -> None:
    """Semantic search over the processed document index."""
    logger.info(f"Searching for: '{query}' (top {top_k})")
    logger.info("-" * 60)

    try:
        results = semantic_search(query, top_k=top_k)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not results:
        print("No results found.")
        return

    print(f"\nTop {len(results)} result(s) for: \"{query}\"\n")
    print(f"{'Rank':<6} {'Score':<8} {'Filename'}")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"  {i:<4} {r['score']:<8.4f} {r['filename']}")
    print()


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Local AI Document Processing — classify, extract, and search documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process
  python main.py process --ocr
  python main.py search "payments due in January"
  python main.py search "software engineer Python experience" --top 3
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # process subcommand
    proc = subparsers.add_parser(
        "process",
        help="Ingest, classify, extract fields, and build search index."
    )
    proc.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for image files (JPG/PNG). Requires easyocr."
    )

    # search subcommand
    srch = subparsers.add_parser(
        "search",
        help="Semantic search over processed documents."
    )
    srch.add_argument("query", type=str, help="Natural-language search query.")
    srch.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="K",
        help="Number of top results to return (default: 5)."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process":
        cmd_process(use_ocr=args.ocr)
    elif args.command == "search":
        cmd_search(args.query, top_k=args.top)


if __name__ == "__main__":
    main()
