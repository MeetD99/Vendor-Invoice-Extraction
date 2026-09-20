import re
from datetime import date, datetime, timedelta


CASE_ID_RE = re.compile(r"\bV[IL1]\s*[-_]?\s*[O0-9IL]{3,}(?=$|[^A-Z0-9])", re.IGNORECASE)
REVISION_RE = re.compile(r"(?:[-_ ](?:R|REV)\s*(\d+))\b", re.IGNORECASE)


def repair_numeric_token(value: str) -> str:
    """Repair common OCR-like substitutions only inside numeric-looking values."""
    candidate = value.strip()
    candidate = re.sub(r"(?<=\d)[Il](?=\d)", "1", candidate)
    candidate = re.sub(r"(?<=\D)[Il](?=\d)", "1", candidate)
    candidate = re.sub(r"(?<=\d)[Il](?=\D)", "1", candidate)
    candidate = re.sub(r"(?<=\d)[Oo](?=\d)", "0", candidate)
    candidate = re.sub(r"(?<=\D)[Oo](?=\d)", "0", candidate)
    return candidate


def canonical_case_id(value: str) -> str | None:
    match = CASE_ID_RE.search(value)
    if not match:
        return None
    raw = re.sub(r"[\s_]", "-", match.group(0).upper())
    raw = re.sub(r"-+", "-", raw)
    _, number = raw.split("-", 1)
    number = number.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    return f"VI-{number}"


def parse_date(value: str) -> date | None:
    # This function is called only on date fields, so translating all common
    # lookalikes is safer and more complete than changing arbitrary prose.
    repaired = value.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    candidates = [
        (r"\b\d{4}-\d{1,2}-\d{1,2}\b", "%Y-%m-%d"),
        (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%m/%d/%Y"),
    ]
    for pattern, fmt in candidates:
        match = re.search(pattern, repaired)
        if match:
            try:
                return datetime.strptime(match.group(0), fmt).date()
            except ValueError:
                continue
    return None


def extract_labeled_value(text: str, labels: list[str]) -> str | None:
    for label in labels:
        match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
        if match:
            return match.group(1).strip()
    return None


def normalize_company(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9 ]", " ", value.lower())
    value = re.sub(r"\b(?:inc|incorporated|llc|ltd|limited|corp|corporation|company|co)\b", "", value)
    return re.sub(r"\s+", " ", value).strip()


def classify_document(text: str, filename: str) -> str:
    heading = "\n".join(line.strip() for line in text.splitlines()[:4]).lower()
    sample = f"{filename}\n{text[:800]}".lower()
    if "vendor invoice" in heading or re.search(r"(?im)^\s*invoice\s*(?:number|no\.?):", text):
        return "VENDOR_INVOICE"
    if "purchase order" in heading or re.search(r"(?im)^\s*po\s*(?:number|no\.?):", text):
        return "PURCHASE_ORDER"
    if "delivery confirmation" in sample or "proof of delivery" in sample:
        return "DELIVERY_CONFIRMATION"
    if "credit note" in sample or "credit memo" in sample:
        return "CREDIT_NOTE"
    return "UNKNOWN"


def document_metadata(text: str, filename: str, document_type: str) -> dict[str, object]:
    label_map = {
        "VENDOR_INVOICE": ["Invoice Number", "Invoice No"],
        "PURCHASE_ORDER": ["PO Number", "PO No"],
        "DELIVERY_CONFIRMATION": ["Delivery Confirmation Number", "Delivery Number", "DC Number"],
    }
    number = extract_labeled_value(text, label_map.get(document_type, []))
    if number:
        number = repair_numeric_token(number).strip()

    revision_match = REVISION_RE.search(number or filename)
    revision = int(revision_match.group(1)) if revision_match else (1 if "revised" in text[:100].lower() else 0)
    base_number = REVISION_RE.sub("", number or "").rstrip("-_ ") or None

    date_labels = {
        "VENDOR_INVOICE": ["Invoice Date"],
        "PURCHASE_ORDER": ["PO Date"],
        "DELIVERY_CONFIRMATION": ["Delivery Date", "Confirmation Date"],
    }
    raw_date = extract_labeled_value(text, date_labels.get(document_type, []))
    parsed_date = parse_date(raw_date or "")
    return {
        "case_key": canonical_case_id(filename) or canonical_case_id(text),
        "document_number": number,
        "base_document_number": base_number,
        "revision_number": revision,
        "document_date": parsed_date.isoformat() if parsed_date else None,
    }


def net_days(terms: str | None) -> int | None:
    if not terms:
        return None
    match = re.search(r"(?i)\bnet\s*(\d{1,3})\b", repair_numeric_token(terms))
    return int(match.group(1)) if match else None


def derive_due_date(base_date: date, terms: str) -> date | None:
    days = net_days(terms)
    return base_date + timedelta(days=days) if days is not None else None
