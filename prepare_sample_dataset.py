"""
prepare_sample_dataset.py

Populates data/input/ with 10–15 sample documents drawn from the assessment datasets:
  - 5 Invoice .txt files  (generated from rows in invoices.csv)
  - 5 Resume PDFs         (one from each of 5 categories in Resume Dataset/data/)
  - 3 Utility Bill JPGs   (copied from Financial Dataset/Utility/)
  - 2 Other JPGs          (1 from Bank Statement/, 1 from Check/)

Usage:
  python prepare_sample_dataset.py --dataset-root "C:\\Users\\hashi\\OneDrive\\Desktop\\Assesment Task"
  python prepare_sample_dataset.py  # uses default path from environment variable DATASET_ROOT
"""

import argparse
import os
import shutil
import random
import textwrap
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    r"C:\Users\hashi\OneDrive\Desktop\Assesment Task"
)

PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "data" / "input"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Invoice TXT generation
# ---------------------------------------------------------------------------

def generate_invoice_txt(row: pd.Series, index: int) -> str:
    """Format a CSV row as a realistic invoice text file."""
    invoice_number = f"INV-{1000 + index}"
    return textwrap.dedent(f"""\
        INVOICE
        ========================================
        Invoice No  : {invoice_number}
        Invoice Date: {row.get('invoice_date', 'N/A')}

        Bill To:
          {row.get('first_name', '')} {row.get('last_name', '')}
          {row.get('address', '')}, {row.get('city', '')}
          Email: {row.get('email', '')}

        From / Seller:
          {row.get('job', 'Unknown Company')} Services

        ----------------------------------------
        Product ID  : {row.get('product_id', 'N/A')}
        Stock Code  : {row.get('stock_code', 'N/A')}
        Quantity    : {row.get('qty', 0)}
        Unit Price  : ${float(row.get('amount', 0)) / max(int(row.get('qty', 1)), 1):.2f}
        ----------------------------------------
        Total Amount: ${float(row.get('amount', 0)):.2f}

        Payment Due : 30 days from invoice date
        ========================================
        Thank you for your business!
    """)


def prepare_invoices(dataset_root: Path, n: int = 5) -> None:
    """Generate n invoice .txt files from invoices.csv."""
    csv_path = dataset_root / "Invoices Dataset" / "invoices.csv"
    if not csv_path.exists():
        print(f"[WARN] invoices.csv not found at: {csv_path}")
        return

    print(f"[INFO] Reading invoices.csv…")
    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(n, len(df)), random_state=RANDOM_SEED)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        txt_content = generate_invoice_txt(row, i)
        out_path = INPUT_DIR / f"invoice_{i:03d}.txt"
        out_path.write_text(txt_content, encoding="utf-8")
        print(f"  [+] Created: {out_path.name}")


# ---------------------------------------------------------------------------
# Resume PDFs
# ---------------------------------------------------------------------------

def prepare_resumes(dataset_root: Path, n: int = 5) -> None:
    """Copy n resume PDFs from Resume Dataset/data/ (one per category)."""
    data_dir = dataset_root / "Resume Dataset" / "data"
    if not data_dir.exists():
        print(f"[WARN] Resume data dir not found: {data_dir}")
        return

    categories = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    random.seed(RANDOM_SEED)
    selected_categories = random.sample(categories, min(n, len(categories)))

    count = 0
    for cat in selected_categories:
        pdfs = list(cat.glob("*.pdf"))
        if not pdfs:
            continue
        chosen = random.choice(pdfs)
        dest = INPUT_DIR / f"resume_{cat.name.lower()}_{chosen.name}"
        shutil.copy2(chosen, dest)
        print(f"  [+] Copied: {dest.name}  (category: {cat.name})")
        count += 1

    print(f"[INFO] Copied {count} resume PDF(s).")


# ---------------------------------------------------------------------------
# Utility Bill JPGs
# ---------------------------------------------------------------------------

def prepare_utility_images(dataset_root: Path, n: int = 3) -> None:
    """Copy n JPG images from Financial Dataset/Utility/."""
    utility_dir = dataset_root / "Financial Dataset" / "Utility"
    if not utility_dir.exists():
        print(f"[WARN] Utility dir not found: {utility_dir}")
        return

    jpgs = sorted(utility_dir.glob("*.jpg"))
    random.seed(RANDOM_SEED + 1)
    selected = random.sample(jpgs, min(n, len(jpgs)))

    for i, jpg in enumerate(selected, start=1):
        dest = INPUT_DIR / f"utility_bill_{i:02d}.jpg"
        shutil.copy2(jpg, dest)
        print(f"  [+] Copied: {dest.name}")

    print(f"[INFO] Copied {len(selected)} utility JPG(s).")


# ---------------------------------------------------------------------------
# "Other" category images
# ---------------------------------------------------------------------------

def prepare_other_images(dataset_root: Path) -> None:
    """Copy 1 bank statement JPG and 1 check JPG as "Other" examples."""
    sources = [
        (dataset_root / "Financial Dataset" / "Bank Statement", "bank_statement_01.jpg"),
        (dataset_root / "Financial Dataset" / "Check", "check_01.jpg"),
    ]

    for source_dir, dest_name in sources:
        if not source_dir.exists():
            print(f"[WARN] Dir not found: {source_dir}")
            continue
        jpgs = sorted(source_dir.glob("*.jpg"))
        if not jpgs:
            continue
        random.seed(RANDOM_SEED + 2)
        chosen = random.choice(jpgs)
        dest = INPUT_DIR / dest_name
        shutil.copy2(chosen, dest)
        print(f"  [+] Copied: {dest.name}")

    print("[INFO] Copied 'Other' category samples.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_sample_dataset.py",
        description="Populate data/input/ with sample documents from the assessment datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_sample_dataset.py
  python prepare_sample_dataset.py --dataset-root "C:\\path\\to\\Assesment Task"
  python prepare_sample_dataset.py --invoices 3 --resumes 4
        """,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=f"Path to the assessment dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--invoices", type=int, default=5, help="Number of invoice TXTs to generate (default: 5)")
    parser.add_argument("--resumes", type=int, default=5, help="Number of resume PDFs to copy (default: 5)")
    parser.add_argument("--utility", type=int, default=3, help="Number of utility bill JPGs to copy (default: 3)")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear data/input/ before populating."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}")
        print("Pass the correct path with: --dataset-root 'C:\\path\\to\\Assesment Task'")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"  Dataset root : {dataset_root}")
    print(f"  Output dir   : {INPUT_DIR}")
    print(f"{'='*60}\n")

    # Optionally clear destination
    if args.clear and INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
        print(f"[INFO] Cleared: {INPUT_DIR}")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Generating Invoice TXT files…")
    prepare_invoices(dataset_root, n=args.invoices)

    print("\n[STEP 2] Copying Resume PDFs…")
    prepare_resumes(dataset_root, n=args.resumes)

    print("\n[STEP 3] Copying Utility Bill JPGs…")
    prepare_utility_images(dataset_root, n=args.utility)

    print("\n[STEP 4] Copying 'Other' category samples…")
    prepare_other_images(dataset_root)

    # Summary
    all_files = list(INPUT_DIR.iterdir())
    print(f"\n{'='*60}")
    print(f"  Done! {len(all_files)} file(s) ready in: {INPUT_DIR}")
    print(f"{'='*60}")
    for f in sorted(all_files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<50s} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
