from datetime import date
from app.services.parsers import (
    canonical_case_id,
    classify_document,
    derive_due_date,
    document_metadata,
    normalize_company,
    parse_date,
)


SAMPLE_INVOICE = """VENDOR INVOICE

Invoice Number: VI-5004
Bill From: Amberline Hardware Group
Remit To: Amberline Hardware Group

Invoice Date: 2026-02-10
Payment Terms: Net 30 from the date of the Purchase Order on file for this vendor.

Line Items:
1. Structural hardware, mixed lot -- $6,300.00
2. Delivery surcharge -- $150.00

Total Due: $6,450.00
Due Date: Per the terms above, calculated from the date of the referenced Purchase Order.
"""


def test_classifies_invoice_even_when_terms_mention_purchase_order():
    assert classify_document(SAMPLE_INVOICE, "VI-5004_invoice.txt") == "VENDOR_INVOICE"


def test_us_date_and_obfuscated_date_are_supported():
    assert parse_date("03/01/2026") == date(2026, 3, 1)
    assert parse_date("O3/O1/2O26") == date(2026, 3, 1)


def test_obfuscated_vi_number_is_canonicalized():
    assert canonical_case_id("Vl-5OO4") == "VI-5004"
    assert canonical_case_id("VI-5004_purchase_order.txt") == "VI-5004"


def test_company_normalization_is_only_used_for_policy_comparison():
    assert normalize_company("Amberline Hardware Group Ltd.") == normalize_company("Amberline Hardware Group")


def test_revision_metadata_and_due_date_calculation():
    revised_po = """PURCHASE ORDER (REVISED)
PO Number: PO-8810-R1
Vendor: Amberline Hardware Group
PO Date: 2026-01-30
"""
    metadata = document_metadata(revised_po, "VI-5004_PO_revised.txt", "PURCHASE_ORDER")
    assert metadata["base_document_number"] == "PO-8810"
    assert metadata["revision_number"] == 1
    assert derive_due_date(date(2026, 1, 30), "Net 30 from PO date") == date(2026, 3, 1)
