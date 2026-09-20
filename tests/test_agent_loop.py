import json
import re
import unittest
import uuid
from pathlib import Path
from typing import Any

from app.agent.adapter import LLMAdapter, OpenAIChatCompletionsAdapter, ToolDefinition, strict_json_schema
from app.agent.messages import AIMessage, AgentMessage, HumanMessage, ToolCall, ToolMessage
from app.agent.orchestrator import AGENT_TOOLS, InvoiceAgent
from app.config import ROOT_DIR, Settings
from app.database import Database
from app.schemas import InvoiceExtraction
from app.services.ingestion import IngestionService


INVOICE = b"""VENDOR INVOICE
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

PO_OLD = b"""PURCHASE ORDER
PO Number: PO-8810
Vendor: Amberline Hardware Group
PO Date: 2026-01-12
"""

PO_REVISED = b"""PURCHASE ORDER (REVISED)
PO Number: PO-8810-R1
Vendor: Amberline Hardware Group
PO Date: 2026-01-30
"""


class ScriptedAdapter(LLMAdapter):
    """Simulates a compliant model while testing the real Python tool loop."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[ToolDefinition],
        response_model: type[InvoiceExtraction] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AIMessage:
        self.calls += 1
        if "independent invoice validation" in system_prompt:
            if isinstance(tool_choice, dict):
                name = tool_choice["function"]["name"]
                if name == "finalize_validation_decision":
                    validation_results = [
                        json.loads(message.content)
                        for message in messages
                        if isinstance(message, ToolMessage) and message.name in {
                            "validate_source_evidence",
                            "reconcile_invoice_total",
                            "enforce_due_date_policy",
                            "validate_remit_to",
                            "lookup_vendor_status",
                        }
                    ]
                    decision = "COMMIT" if validation_results and all(
                        result.get("passed") for result in validation_results
                    ) else "FLAG"
                    return AIMessage(tool_calls=[ToolCall(
                        id=f"validation-{self.calls}",
                        name=name,
                        arguments={"decision": decision, "reason": "Deterministic checks evaluated."},
                    )])
                return AIMessage(tool_calls=[ToolCall(id=f"validation-{self.calls}", name=name, arguments={})])
            validation_results = [
                json.loads(message.content)
                for message in messages
                if isinstance(message, ToolMessage) and message.name in {
                    "validate_source_evidence",
                    "reconcile_invoice_total",
                    "enforce_due_date_policy",
                    "validate_remit_to",
                    "lookup_vendor_status",
                }
            ]
            if validation_results and all(result.get("passed") for result in validation_results):
                return AIMessage(
                    tool_calls=[
                        ToolCall(
                            id=f"validation-{self.calls}",
                            name="commit_invoice_record",
                            arguments={"decision_reason": "All deterministic checks and vendor approval passed."},
                        )
                    ]
                )
            return AIMessage(
                tool_calls=[
                    ToolCall(
                        id=f"validation-{self.calls}",
                        name="flag_vendor_invoice",
                        arguments={"reason": "One or more deterministic checks failed."},
                    )
                ]
            )
        if response_model is not None:
            source_tool = next(message for message in messages if isinstance(message, ToolMessage) and message.name == "read_source_document")
            related_tool = next(message for message in messages if isinstance(message, ToolMessage) and message.name == "search_related_documents")
            source_id = json.loads(source_tool.content)["document_id"]
            po_id = json.loads(related_tool.content)["current_documents"][0]["document_id"]
            extraction = InvoiceExtraction.model_validate(
                {
                    "vendor_name": "Amberline Hardware Group",
                    "remit_to": "Amberline Hardware Group",
                    "remit_to_explanation": None,
                    "invoice_number": "VI-5004",
                    "invoice_date": "2026-02-10",
                    "payment_terms": "Net 30 from the date of the Purchase Order on file for this vendor.",
                    "line_items": [
                        {"description": "Structural hardware, mixed lot", "amount": "6300.00", "quantity": None, "unit_price": None},
                        {"description": "Delivery surcharge", "amount": "150.00", "quantity": None, "unit_price": None},
                    ],
                    "total_due": "6450.00",
                    "total_due_basis": "EXPLICIT",
                    "due_date": "2026-03-01",
                    "due_date_basis": "PURCHASE_ORDER_DATE",
                    "evidence": [
                        {"field_name": "vendor_name", "document_id": source_id, "excerpt": "Bill From: Amberline Hardware Group"},
                        {"field_name": "remit_to", "document_id": source_id, "excerpt": "Remit To: Amberline Hardware Group"},
                        {"field_name": "invoice_number", "document_id": source_id, "excerpt": "Invoice Number: VI-5004"},
                        {"field_name": "line_items", "document_id": source_id, "excerpt": "1. Structural hardware, mixed lot -- $6,300.00\n2. Delivery surcharge -- $150.00"},
                        {"field_name": "total_due", "document_id": source_id, "excerpt": "Total Due: $6,450.00"},
                        {"field_name": "due_date", "document_id": po_id, "excerpt": "PO Date: 2026-01-30"},
                    ],
                    "flags": [],
                    "derivations": ["Net 30 from revised PO date 2026-01-30 gives 2026-03-01."],
                }
            )
            return AIMessage(content=extraction.model_dump_json())

        if not any(isinstance(message, ToolMessage) and message.name == "read_source_document" for message in messages):
            match = re.search(r"document_id is (\d+)", messages[0].content)
            assert match
            return AIMessage(
                tool_calls=[ToolCall(id="call-read", name="read_source_document", arguments={"document_id": int(match.group(1))})]
            )
        if not any(isinstance(message, ToolMessage) and message.name == "search_related_documents" for message in messages):
            return AIMessage(
                tool_calls=[
                    ToolCall(
                        id="call-related",
                        name="search_related_documents",
                        arguments={"document_type": "PURCHASE_ORDER", "reason": "Due date depends on PO date"},
                    )
                ]
            )
        return AIMessage(content="All required source and related-document evidence is available.")


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_completions_adapter_uses_tools_then_json_schema(self):
        settings = Settings(
            llm_provider="openai_compatible",
            llm_model="company-invoice-model",
            llm_api_key="test-key",
            llm_chat_completions_url="https://company.example/chat/completions",
        )
        adapter = OpenAIChatCompletionsAdapter(settings)
        captured: list[dict[str, Any]] = []

        async def fake_tool_post(payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            captured.append(payload)
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_source_document", "arguments": '{"document_id":1}'},
                        }],
                    }
                }]
            }

        adapter._post = fake_tool_post  # type: ignore[method-assign]
        tool_message = await adapter.complete(
            system_prompt="test",
            messages=[HumanMessage(content="Read document 1")],
            tools=AGENT_TOOLS,
            tool_choice="auto",
        )
        self.assertEqual(tool_message.tool_calls[0].name, "read_source_document")
        self.assertIn("tools", captured[0])
        self.assertNotIn("response_format", captured[0])

        extraction = InvoiceExtraction(
            vendor_name=None,
            remit_to=None,
            remit_to_explanation=None,
            invoice_number=None,
            invoice_date=None,
            payment_terms=None,
            line_items=[],
            total_due=None,
            total_due_basis="MISSING",
            due_date=None,
            due_date_basis="UNRESOLVED",
            evidence=[],
            flags=[],
            derivations=[],
        )

        async def fake_final_post(payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
            captured.append(payload)
            return {"choices": [{"message": {"role": "assistant", "content": extraction.model_dump_json()}}]}

        adapter._post = fake_final_post  # type: ignore[method-assign]
        final_message = await adapter.complete(
            system_prompt="test",
            messages=[HumanMessage(content="Finalize")],
            tools=[],
            response_model=InvoiceExtraction,
            tool_choice="none",
        )
        self.assertIsNotNone(final_message.content)
        self.assertEqual(captured[1]["response_format"]["type"], "json_schema")
        self.assertTrue(captured[1]["response_format"]["json_schema"]["strict"])

    async def test_tool_loop_then_schema_extraction_and_deterministic_commit(self):
        root = ROOT_DIR / "data" / "test_runs" / uuid.uuid4().hex
        settings = Settings(database_path=str(root / "test.db"), upload_dir=str(root / "uploads"))
        database = Database(settings.database_file)
        database.initialize()
        ingestion = IngestionService(database, settings)
        batch_id = ingestion.create_batch("agent-loop-test")
        results = [
            ingestion.ingest_document(batch_id, "VI-5004_invoice.txt", INVOICE),
            ingestion.ingest_document(batch_id, "VI-5004_po.txt", PO_OLD),
            ingestion.ingest_document(batch_id, "VI-5004_po_revised.txt", PO_REVISED),
        ]
        ingestion.import_registry(b"Invoice Number,Verified Status\nVI-5004,Verified\n")
        ingestion.finish_batch(batch_id, results)
        case_id = int(results[0]["case_id"])

        adapter = ScriptedAdapter()
        await InvoiceAgent(database, adapter, max_agent_steps=6).process_case(case_id)

        case = database.query_one("SELECT * FROM invoice_cases WHERE id = ?", (case_id,))
        self.assertEqual(case["status"], "PAYMENT_SCHEDULED")
        self.assertEqual(adapter.calls, 11)
        messages = database.query_all(
            """SELECT agent_name, role, tool_name FROM agent_messages
               WHERE case_id = ? ORDER BY created_at, agent_name, sequence_number""",
            (case_id,),
        )
        extraction_tools = [
            message["tool_name"] for message in messages
            if message["agent_name"] == "extraction" and message["role"] == "tool"
        ]
        validation_tools = [
            message["tool_name"] for message in messages
            if message["agent_name"] == "validation" and message["role"] == "tool"
        ]
        self.assertEqual(extraction_tools, [
            "read_source_document",
            "search_related_documents",
        ])
        self.assertEqual(validation_tools, [*[
            "get_extraction_candidate",
            "validate_source_evidence",
            "reconcile_invoice_total",
            "enforce_due_date_policy",
            "validate_remit_to",
            "lookup_vendor_status",
        ], "finalize_validation_decision"])
        current_po = database.query_one(
            """SELECT document_number FROM documents
               WHERE case_id = ? AND document_type = 'PURCHASE_ORDER' AND is_current = 1""",
            (case_id,),
        )
        self.assertEqual(current_po["document_number"], "PO-8810-R1")

    async def test_unapproved_vendor_is_flagged_and_not_committed(self):
        root = ROOT_DIR / "data" / "test_runs" / uuid.uuid4().hex
        settings = Settings(database_path=str(root / "test.db"), upload_dir=str(root / "uploads"))
        database = Database(settings.database_file)
        database.initialize()
        ingestion = IngestionService(database, settings)
        batch_id = ingestion.create_batch("vendor-flag-test")
        results = [
            ingestion.ingest_document(batch_id, "VI-5004_invoice.txt", INVOICE),
            ingestion.ingest_document(batch_id, "VI-5004_po.txt", PO_OLD),
            ingestion.ingest_document(batch_id, "VI-5004_po_revised.txt", PO_REVISED),
        ]
        ingestion.import_registry(b"Invoice Number,Verified Status\nVI-5004,On Hold\n")
        ingestion.finish_batch(batch_id, results)
        case_id = int(results[0]["case_id"])

        await InvoiceAgent(database, ScriptedAdapter(), max_agent_steps=6).process_case(case_id)

        case = database.query_one("SELECT * FROM invoice_cases WHERE id = ?", (case_id,))
        self.assertEqual(case["status"], "VENDOR_FLAGGED")
        self.assertEqual(database.query_one("SELECT COUNT(*) AS n FROM committed_records")["n"], 0)
        followup = database.query_one("SELECT * FROM followups WHERE case_id = ?", (case_id,))
        self.assertEqual(followup["followup_type"], "VENDOR_STATUS")

    def test_structured_output_schema_is_strict(self):
        schema = strict_json_schema(InvoiceExtraction.model_json_schema())
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))


if __name__ == "__main__":
    unittest.main()
