"""
extract.py — Structured field extraction per document class.

Uses regex patterns to pull known fields from document text.
Returns a flat dict of extracted fields (all values are strings or None).

Field schemas:
  Invoice      → invoice_number, date, company, total_amount
  Resume       → name, email, phone, experience_years
  Utility Bill → account_number, date, usage_kwh, amount_due
  Other / Unclassifiable → {} (no fields required)
"""

import re
from src.utils import first_nonempty_line, setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------------
# Shared regex helpers
# ---------------------------------------------------------------------------

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",         # 01/15/2024 or 1-5-24
    r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",            # 2024-01-15
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b",  # 15 Jan 2024
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",  # Jan 15, 2024
]


def _find_date(text: str) -> str | None:
    """Return the first date-like string found in text."""
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _find_amount(text: str, label_pattern: str) -> str | None:
    """
    Find an amount following a label.
    e.g. label_pattern = r'total' → searches for 'Total: $1,200.00'
    """
    pattern = label_pattern + r"[\s\w]*?[\$₹£€]?\s*([\d,]+\.?\d*)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


# ---------------------------------------------------------------------------
# Invoice extraction
# ---------------------------------------------------------------------------

def extract_invoice(text: str) -> dict:
    fields: dict = {}

    # invoice_number
    m = re.search(
        r"invoice\s*(?:no|number|#|num)?\s*[:\-#]?\s*([A-Za-z0-9\-_/]+)",
        text, re.IGNORECASE
    )
    fields["invoice_number"] = m.group(1).strip() if m else None

    # date
    fields["date"] = _find_date(text)

    # company — try labeled patterns first, then first clean line heuristic
    company = None
    for pattern in [
        r"(?:from|sold by|company|vendor|supplier)\s*[:\-]\s*(.+)",
        r"(?:billed from|issued by)\s*[:\-]\s*(.+)",
        r"(?:seller|issued to)\s*[:\-]\s*(.+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            company = m.group(1).strip().split("\n")[0]
            break
    # Fallback: first non-empty line that isn't "Invoice", a separator, or a date
    if not company:
        for line in text.splitlines():
            stripped = line.strip()
            if (
                stripped
                and not re.match(r"invoice", stripped, re.IGNORECASE)
                and not re.match(r"^[=\-_*#]{3,}$", stripped)   # skip separator lines
                and not re.match(r"^\d", stripped)              # skip lines starting with digits
                and len(stripped) > 3
            ):
                company = stripped
                break
    fields["company"] = company

    # total_amount — try several label variants (including $ prefix)
    total = None
    for label in [r"total\s*amount", r"grand\s*total", r"total\s*due", r"total", r"amount"]:
        total = _find_amount(text, label)
        if total:
            break
    # Also try direct $ amount pattern (e.g. "Total Amount: $1250.00")
    if not total:
        m = re.search(r"(?:total\s*amount|grand\s*total|total\s*due)[^\n]*\$([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            total = m.group(1).replace(",", "")
    fields["total_amount"] = total

    return fields


# ---------------------------------------------------------------------------
# Resume extraction
# ---------------------------------------------------------------------------

def extract_resume(text: str) -> dict:
    fields: dict = {}

    # email
    m = re.search(r"[\w\.\+\-]+@[\w\.\-]+\.\w{2,}", text)
    fields["email"] = m.group(0).strip() if m else None

    # phone — flexible: +1-555-1234, (555) 123-4567, +91 9876543210, etc.
    m = re.search(
        r"(\+?[\d][\d\s\-\.\(\)]{8,17}[\d])",
        text
    )
    fields["phone"] = m.group(1).strip() if m else None

    # experience_years — "5 years of experience", "10+ years", "3 yrs"
    exp = None
    for pattern in [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
        r"(\d+)\+?\s*yrs?\s+(?:of\s+)?experience",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            exp = int(m.group(1))
            break
    fields["experience_years"] = exp

    # name — heuristic: first short line (≤ 5 words) not containing "@" or digits
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped
            and len(stripped.split()) <= 6
            and "@" not in stripped
            and not re.search(r"\d{5,}", stripped)
            and not re.match(r"(resume|curriculum|cv|objective|summary|skills|education|experience)", stripped, re.IGNORECASE)
        ):
            name = stripped
            break
    fields["name"] = name

    return fields


# ---------------------------------------------------------------------------
# Utility Bill extraction
# ---------------------------------------------------------------------------

def extract_utility(text: str) -> dict:
    fields: dict = {}

    # account_number
    m = re.search(
        r"account\s*(?:no|number|#|num)?\s*[:\-#]?\s*([A-Za-z0-9\-_]+)",
        text, re.IGNORECASE
    )
    fields["account_number"] = m.group(1).strip() if m else None

    # date
    fields["date"] = _find_date(text)

    # usage_kwh — "123.5 kwh", "123 units", "456 kw"
    m = re.search(
        r"([\d,]+\.?\d*)\s*(?:kwh|kw|kilowatt|units\s+consumed|units)",
        text, re.IGNORECASE
    )
    fields["usage_kwh"] = m.group(1).replace(",", "").strip() if m else None

    # amount_due
    amount = None
    for label in [r"amount\s*due", r"total\s*due", r"payable\s*amount", r"net\s*amount", r"amount"]:
        amount = _find_amount(text, label)
        if amount:
            break
    fields["amount_due"] = amount

    return fields


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_fields(doc_class: str, text: str) -> dict:
    """
    Extract structured fields from text based on the document class.

    Args:
        doc_class:  One of Invoice | Resume | Utility Bill | Other | Unclassifiable
        text:       Cleaned document text.

    Returns:
        Dict of extracted fields (values may be None if not found).
    """
    if doc_class == "Invoice":
        fields = extract_invoice(text)
    elif doc_class == "Resume":
        fields = extract_resume(text)
    elif doc_class == "Utility Bill":
        fields = extract_utility(text)
    else:
        fields = {}  # Other / Unclassifiable — no fields required

    logger.debug(f"Extracted [{doc_class}]: {fields}")
    return fields
