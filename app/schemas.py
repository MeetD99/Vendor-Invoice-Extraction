from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    amount: Decimal
    quantity: Decimal | None = None
    unit_price: Decimal | None = None


class FieldEvidence(BaseModel):
    field_name: str
    value: str | None
    document_id: int
    excerpt: str
    verification: Literal["MATCHED", "DERIVED", "MISSING", "CONFLICT"]


class ExtractionEvidence(BaseModel):
    field_name: Literal["vendor_name", "invoice_number", "line_items", "total_due", "due_date", "remit_to"]
    document_id: int
    excerpt: str


class InvoiceExtraction(BaseModel):
    vendor_name: str | None = None
    remit_to: str | None = None
    remit_to_explanation: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    payment_terms: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    total_due: Decimal | None = None
    total_due_basis: Literal["EXPLICIT", "DERIVED_FROM_LINE_ITEMS", "MISSING"] = "MISSING"
    due_date: date | None = None
    due_date_basis: Literal[
        "EXPLICIT",
        "PURCHASE_ORDER_DATE",
        "DELIVERY_CONFIRMATION_DATE",
        "UNRESOLVED",
    ] = "UNRESOLVED"
    evidence: list[ExtractionEvidence] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    derivations: list[str] = Field(default_factory=list)


class ValidationDecision(BaseModel):
    decision: Literal["COMMITTED", "FLAGGED"]
    reasons: list[str] = Field(default_factory=list)
    vendor_status: str | None = None
    record_id: int | None = None


class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any]


class ToolResult(BaseModel):
    name: str
    output: dict[str, Any]
