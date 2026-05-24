"""
classify.py — Keyword/rule-based document classification.

Classification is purely deterministic and explainable:
  - Each class has a set of weighted keywords.
  - The text of each document is scored against every class.
  - The class with the highest score wins.
  - If no class scores above the minimum threshold, "Unclassifiable" is returned.

Classes:
  Invoice | Resume | Utility Bill | Other | Unclassifiable
"""

import re
from src.utils import setup_logger

logger = setup_logger()

# Minimum number of keyword hits required to claim a class.
MIN_SCORE_THRESHOLD = 2

# Keyword lists per class.  Each keyword is a regex pattern (case-insensitive).
KEYWORD_RULES: dict[str, list[str]] = {
    "Invoice": [
        r"\binvoice\b",
        r"\binvoice\s*(?:no|number|#)",
        r"\bbill\s*to\b",
        r"\bpayment\s*due\b",
        r"\bdue\s*date\b",
        r"\bqty\b",
        r"\bquantity\b",
        r"\bunit\s*price\b",
        r"\bsubtotal\b",
        r"\btax\b",
        r"\bstock\s*code\b",
        r"\bproduct\s*id\b",
        r"\bamount\b",
        r"\btotal\s*amount\b",
        r"\bpayable\b",
        r"\bseller\b",
        r"\bbuyer\b",
        r"\breceipt\b",
        r"\border\s*(?:no|number|#)\b",
    ],
    "Resume": [
        r"\bresume\b",
        r"\bcurriculum\s*vitae\b",
        r"\bc\.?v\.?\b",
        r"\bexperience\b",
        r"\bwork\s*history\b",
        r"\beducation\b",
        r"\bskills\b",
        r"\bobjective\b",
        r"\bsummary\b",
        r"\bcertification\b",
        r"\bachievement\b",
        r"\bemployment\b",
        r"\bproject\b",
        r"\bprofessional\b",
        r"\blanguage[s]?\b",
        r"\breference[s]?\b",
        r"\bhobby\b",
        r"\blinkedin\b",
        r"\bgithub\b",
    ],
    "Utility Bill": [
        r"\belectricity\b",
        r"\belectric\s*bill\b",
        r"\bpower\s*bill\b",
        r"\bwater\s*bill\b",
        r"\bgas\s*bill\b",
        r"\bkwh\b",
        r"\bkilowatt\b",
        r"\bmeter\s*(?:reading|no|number)\b",
        r"\bunits\s*consumed\b",
        r"\bconsumption\b",
        r"\baccount\s*(?:no|number|#)\b",
        r"\bbilling\s*period\b",
        r"\bdue\s*date\b",
        r"\bamount\s*due\b",
        r"\butility\b",
        r"\bservice\s*address\b",
        r"\btariff\b",
    ],
    "Other": [
        r"\bbank\s*statement\b",
        r"\bopening\s*balance\b",
        r"\bclosing\s*balance\b",
        r"\bdebit\b",
        r"\bcredit\b",
        r"\btransaction\b",
        r"\bsalary\s*slip\b",
        r"\bpayslip\b",
        r"\bbasic\s*salary\b",
        r"\bnet\s*salary\b",
        r"\bgross\s*salary\b",
        r"\bitr\b",
        r"\bincome\s*tax\s*return\b",
        r"\bform\s*16\b",
        r"\bpan\b",
        r"\btan\b",
        r"\bcheque\b",
        r"\bcheck\s*no\b",
        r"\bpayee\b",
        r"\bdrawer\b",
    ],
}


def classify(text: str) -> str:
    """
    Classify document text into one of:
      Invoice | Resume | Utility Bill | Other | Unclassifiable

    Args:
        text: Cleaned, extracted text from a document.

    Returns:
        String class label.
    """
    if not text or not text.strip():
        logger.debug("Empty text → Unclassifiable")
        return "Unclassifiable"

    text_lower = text.lower()
    scores: dict[str, int] = {cls: 0 for cls in KEYWORD_RULES}

    for cls, patterns in KEYWORD_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                scores[cls] += 1

    logger.debug(f"Classification scores: {scores}")

    best_class = max(scores, key=lambda c: scores[c])
    best_score = scores[best_class]

    if best_score < MIN_SCORE_THRESHOLD:
        return "Unclassifiable"

    return best_class


def classify_by_filename(filename: str) -> str | None:
    """
    Optional fast-path classification using the filename alone.
    Returns None if no strong signal found (falls back to text classification).

    This is useful for files we generated ourselves (e.g. invoice_001.txt).
    """
    name = filename.lower()
    if "invoice" in name:
        return "Invoice"
    if "resume" in name or "cv" in name:
        return "Resume"
    if "utility" in name or "electric" in name or "water" in name or "gas" in name:
        return "Utility Bill"
    if any(kw in name for kw in ["bank", "statement", "salary", "slip", "itr", "form16", "check", "cheque"]):
        return "Other"
    return None
