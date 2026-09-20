import json
import re
import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from app.agent.adapter import LLMAdapter, ToolDefinition
from app.agent.messages import AIMessage, AgentMessage, HumanMessage, ToolCall, ToolMessage
from app.database import Database, utc_now
from app.schemas import InvoiceExtraction, ValidationDecision
from app.services.parsers import derive_due_date, normalize_company


VALIDATION_PROMPT = """You are the independent invoice validation and commit-decision agent.
You do not re-extract fields. You must inspect the extraction through tools and use every required
deterministic validation tool. Python owns all calculations and pass/fail rules. Do not override tool
results. After the required checks, call finalize_validation_decision with COMMIT only when every
check passes and the vendor status is approved. Otherwise choose FLAG with a concise evidence-based
reason. Your prose cannot commit or flag anything; only the decision tool changes state."""


def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ToolDefinition:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


VALIDATION_TOOLS: list[ToolDefinition] = [
    function_tool("get_extraction_candidate", "Load the extraction candidate and canonical document manifest.", {}, []),
    function_tool("validate_source_evidence", "Verify evidence document IDs and exact excerpts against source text.", {}, []),
    function_tool("reconcile_invoice_total", "Deterministically compare line-item amounts with Total Due.", {}, []),
    function_tool("enforce_due_date_policy", "Calculate and enforce due date using canonical related documents.", {}, []),
    function_tool("validate_remit_to", "Compare Bill From and Remit To and validate any stated explanation.", {}, []),
    function_tool("lookup_vendor_status", "Look up the current vendor approval status using the invoice registry.", {}, []),
]

ACTION_TOOLS: list[ToolDefinition] = [
    function_tool(
        "finalize_validation_decision",
        (
            "Choose COMMIT only when every deterministic result passed, otherwise choose FLAG. "
            "Python independently enforces the commit gate and executes the selected action."
        ),
        {
            "decision": {"type": "string", "enum": ["COMMIT", "FLAG"]},
            "reason": {"type": "string"},
        },
        ["decision", "reason"],
    ),
]

REQUIRED_CHECKS = [
    "get_extraction_candidate",
    "validate_source_evidence",
    "reconcile_invoice_total",
    "enforce_due_date_policy",
    "validate_remit_to",
    "lookup_vendor_status",
]


class ValidationAgent:
    def __init__(self, db: Database, llm: LLMAdapter, max_steps: int = 10) -> None:
        self.db = db
        self.llm = llm
        self.max_steps = max_steps

    async def run(
        self,
        *,
        case_id: int,
        extraction: InvoiceExtraction,
        documents: list[dict[str, Any]],
        current_supporting: list[dict[str, Any]],
    ) -> ValidationDecision:
        run_id = f"validation-{uuid.uuid4().hex}"
        messages: list[AgentMessage] = [
            HumanMessage(
                content=(
                    f"Validate extraction candidate for case {case_id}. Run every deterministic check, "
                    "look up vendor status, then call exactly one commit or flag action tool."
                )
            )
        ]
        sequence = 0
        self._log_message(case_id, run_id, sequence, messages[0])
        sequence += 1
        results: dict[str, dict[str, Any]] = {}

        for _ in range(self.max_steps):
            missing = [name for name in REQUIRED_CHECKS if name not in results]
            tools = (
                [next(tool for tool in VALIDATION_TOOLS if tool["function"]["name"] == missing[0])]
                if missing else ACTION_TOOLS
            )
            tool_choice: str | dict[str, Any]
            if missing:
                tool_choice = {"type": "function", "function": {"name": missing[0]}}
            else:
                tool_choice = {
                    "type": "function",
                    "function": {"name": "finalize_validation_decision"},
                }

            ai_message = await self.llm.complete(
                system_prompt=VALIDATION_PROMPT,
                messages=messages,
                tools=tools,
                response_model=None,
                tool_choice=tool_choice,
            )
            expected_tool = missing[0] if missing else "finalize_validation_decision"
            selected_call = next(
                (call for call in ai_message.tool_calls if call.name == expected_tool),
                None,
            )
            if selected_call is None and missing:
                ai_message = AIMessage(
                    content=(
                        (ai_message.content or "")
                        + "\n[Controller fallback: the model omitted the forced read-only validation call.]"
                    ),
                    tool_calls=[ToolCall(
                        id=f"controller-{uuid.uuid4().hex}",
                        name=expected_tool,
                        arguments={},
                    )],
                )
                selected_call = ai_message.tool_calls[0]
            messages.append(ai_message)
            self._log_message(case_id, run_id, sequence, ai_message)
            sequence += 1

            if selected_call is None:
                reminder = HumanMessage(
                    content=f"You must call {expected_tool}. Prose or a different tool does not advance validation."
                )
                messages.append(reminder)
                self._log_message(case_id, run_id, sequence, reminder)
                sequence += 1
                continue

            for call in [selected_call]:
                result = self._execute_tool(
                    case_id=case_id,
                    extraction=extraction,
                    documents=documents,
                    current_supporting=current_supporting,
                    results=results,
                    call=call,
                )
                tool_message = ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=call.id,
                    name=call.name,
                )
                messages.append(tool_message)
                self._log_message(case_id, run_id, sequence, tool_message)
                sequence += 1

                if call.name == "finalize_validation_decision" and result.get("committed"):
                    return ValidationDecision(
                        decision="COMMITTED",
                        reasons=[str(call.arguments.get("reason", "All checks passed."))],
                        vendor_status=results["lookup_vendor_status"].get("status"),
                        record_id=int(result["record_id"]),
                    )
                if call.name == "finalize_validation_decision" and result.get("flagged"):
                    return ValidationDecision(
                        decision="FLAGGED",
                        reasons=[str(result["reason"])],
                        vendor_status=results.get("lookup_vendor_status", {}).get("status"),
                        record_id=None,
                    )
                if call.name in REQUIRED_CHECKS:
                    results[call.name] = result

        raise RuntimeError(f"Validation agent exceeded {self.max_steps} tool steps.")

    def _execute_tool(
        self,
        *,
        case_id: int,
        extraction: InvoiceExtraction,
        documents: list[dict[str, Any]],
        current_supporting: list[dict[str, Any]],
        results: dict[str, dict[str, Any]],
        call: ToolCall,
    ) -> dict[str, Any]:
        handlers = {
            "get_extraction_candidate": lambda: self._candidate(extraction, documents),
            "validate_source_evidence": lambda: self._validate_evidence(extraction, documents),
            "reconcile_invoice_total": lambda: self._reconcile_total(extraction),
            "enforce_due_date_policy": lambda: self._enforce_due_date(extraction, current_supporting),
            "validate_remit_to": lambda: self._validate_remit(extraction),
            "lookup_vendor_status": lambda: self._lookup_status(extraction),
            "finalize_validation_decision": lambda: self._finalize_decision(
                case_id, extraction, results, call.arguments
            ),
        }
        handler = handlers.get(call.name)
        if not handler:
            result = {"passed": False, "error": f"Unknown validation tool: {call.name}"}
            self.db.log_tool(case_id, call.name, call.arguments, result, "ERROR")
            return result
        result = handler()
        status = "SUCCESS" if not result.get("error") else "ERROR"
        self.db.log_tool(case_id, call.name, call.arguments, result, status)
        return result

    def _finalize_decision(
        self,
        case_id: int,
        extraction: InvoiceExtraction,
        results: dict[str, dict[str, Any]],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        decision = str(arguments.get("decision", "")).upper()
        reason = str(arguments.get("reason") or "Validation decision completed.")
        if decision == "COMMIT":
            result = self._commit(case_id, extraction, results)
            if result.get("committed"):
                return result
            return self._flag(
                case_id,
                f"The model requested COMMIT but the Python commit gate rejected it. {reason}",
                results,
            )
        if decision == "FLAG":
            return self._flag(case_id, reason, results)
        return {"error": "Decision must be COMMIT or FLAG."}

    @staticmethod
    def _candidate(extraction: InvoiceExtraction, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "passed": True,
            "extraction": extraction.model_dump(mode="json"),
            "document_manifest": [
                {
                    "document_id": doc["id"],
                    "filename": doc["original_filename"],
                    "type": doc["document_type"],
                    "number": doc["document_number"],
                    "revision": doc["revision_number"],
                    "is_current": bool(doc["is_current"]),
                }
                for doc in documents
            ],
        }

    @staticmethod
    def _validate_evidence(
        extraction: InvoiceExtraction,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        documents_by_id = {int(document["id"]): document for document in documents}
        errors: list[str] = []
        evidenced_fields: set[str] = set()
        for item in extraction.evidence:
            document = documents_by_id.get(item.document_id)
            if not document:
                errors.append(f"Evidence for {item.field_name} references an out-of-case document.")
                continue
            excerpt = "\n".join(
                re.sub(r"^\s*\d{1,6}:\s*", "", line) for line in item.excerpt.strip().splitlines()
            )
            if not excerpt or excerpt not in document["raw_text"]:
                errors.append(f"Evidence excerpt for {item.field_name} was not found in document {item.document_id}.")
                continue
            evidenced_fields.add(item.field_name)
        for field in ("vendor_name", "invoice_number", "line_items", "total_due"):
            if field not in evidenced_fields:
                errors.append(f"Required evidence is missing for {field}.")
        return {"passed": not errors, "errors": errors, "verified_fields": sorted(evidenced_fields)}

    @staticmethod
    def _reconcile_total(extraction: InvoiceExtraction) -> dict[str, Any]:
        if extraction.total_due is None:
            return {"passed": False, "errors": ["Total Due is missing."], "line_item_total": None}
        if not extraction.line_items:
            return {"passed": False, "errors": ["No line items were extracted."], "line_item_total": None}
        line_item_total = sum((item.amount for item in extraction.line_items), Decimal("0"))
        passed = line_item_total == extraction.total_due
        if extraction.total_due_basis == "DERIVED_FROM_LINE_ITEMS" and passed:
            if "TOTAL_DUE_MISSING_DERIVED_FROM_LINE_ITEMS" not in extraction.flags:
                extraction.flags.append("TOTAL_DUE_MISSING_DERIVED_FROM_LINE_ITEMS")
        return {
            "passed": passed,
            "line_item_total": str(line_item_total),
            "total_due": str(extraction.total_due),
            "errors": [] if passed else ["Line-item total does not equal Total Due."],
        }

    @staticmethod
    def _referenced_document_type(terms: str | None) -> str | None:
        value = terms or ""
        if re.search(r"(?i)purchase order|\bPO\b", value):
            return "PURCHASE_ORDER"
        if re.search(r"(?i)delivery|confirmation", value):
            return "DELIVERY_CONFIRMATION"
        return None

    def _enforce_due_date(
        self,
        extraction: InvoiceExtraction,
        current_supporting: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target_type = self._referenced_document_type(extraction.payment_terms)
        if not target_type:
            return {
                "passed": extraction.due_date is not None,
                "due_date": str(extraction.due_date) if extraction.due_date else None,
                "errors": [] if extraction.due_date else ["Due date is missing and no calculable related-document terms were found."],
            }
        candidates = [
            doc for doc in current_supporting
            if doc["document_type"] == target_type and doc["document_date"]
        ]
        if len(candidates) != 1:
            return {
                "passed": False,
                "errors": [f"Expected one canonical {target_type}; found {len(candidates)}."],
            }
        base_date = date.fromisoformat(candidates[0]["document_date"])
        calculated = derive_due_date(base_date, extraction.payment_terms or "")
        if not calculated:
            return {"passed": False, "errors": ["Payment terms could not be converted to a due date."]}
        proposed = extraction.due_date
        extraction.due_date = calculated
        extraction.due_date_basis = (
            "PURCHASE_ORDER_DATE" if target_type == "PURCHASE_ORDER" else "DELIVERY_CONFIRMATION_DATE"
        )
        if proposed and proposed != calculated and "DUE_DATE_NORMALIZED_BY_POLICY" not in extraction.flags:
            extraction.flags.append("DUE_DATE_NORMALIZED_BY_POLICY")
        extraction.derivations.append(
            f"Python due-date policy used {candidates[0]['document_number']} dated {base_date.isoformat()} "
            f"with '{extraction.payment_terms}' to produce {calculated.isoformat()}."
        )
        return {
            "passed": True,
            "due_date": calculated.isoformat(),
            "source_document_id": candidates[0]["id"],
            "source_document_number": candidates[0]["document_number"],
            "model_value_overridden": proposed is not None and proposed != calculated,
        }

    @staticmethod
    def _validate_remit(extraction: InvoiceExtraction) -> dict[str, Any]:
        if not extraction.remit_to or normalize_company(extraction.vendor_name) == normalize_company(extraction.remit_to):
            return {"passed": True, "warning": None}
        remit_evidence = any(item.field_name == "remit_to" and item.excerpt for item in extraction.evidence)
        if extraction.remit_to_explanation and remit_evidence:
            if "REMIT_TO_MISMATCH_EXPLAINED" not in extraction.flags:
                extraction.flags.append("REMIT_TO_MISMATCH_EXPLAINED")
            return {"passed": True, "warning": "Bill From and Remit To differ but a sourced explanation exists."}
        return {"passed": False, "errors": ["Bill From and Remit To differ without a sourced explanation."]}

    def _lookup_status(self, extraction: InvoiceExtraction) -> dict[str, Any]:
        invoice_number = (extraction.invoice_number or "").upper()
        row = self.db.query_one(
            "SELECT verified_status FROM registry_status WHERE UPPER(invoice_number) = ?",
            (invoice_number,),
        )
        status = row["verified_status"] if row else None
        normalized = (status or "").strip().lower()
        approved = normalized in {"verified", "active", "approved", "yes", "true", "active - not on hold"}
        return {
            "passed": approved,
            "approved": approved,
            "status": status,
            "invoice_number": invoice_number,
            "errors": [] if approved else ["Vendor is not approved or is missing from the status registry."],
        }

    @staticmethod
    def _failed_checks(results: dict[str, dict[str, Any]]) -> list[str]:
        return [name for name in REQUIRED_CHECKS[1:] if not results.get(name, {}).get("passed")]

    def _commit(
        self,
        case_id: int,
        extraction: InvoiceExtraction,
        results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        missing = [name for name in REQUIRED_CHECKS if name not in results]
        failed = self._failed_checks(results)
        if missing or failed:
            return {
                "committed": False,
                "error": "Commit gate rejected the action.",
                "missing_checks": missing,
                "failed_checks": failed,
            }
        if not extraction.invoice_number or not extraction.vendor_name or not extraction.due_date:
            return {"committed": False, "error": "Required committed-record fields are missing."}
        invoice_number = extraction.invoice_number.upper()
        existing = self.db.query_one(
            "SELECT id FROM committed_records WHERE invoice_number = ? AND is_current = 1",
            (invoice_number,),
        )
        if existing:
            record_id = int(existing["id"])
        else:
            record_id = self.db.execute(
                """INSERT INTO committed_records
                   (case_id, invoice_number, vendor_name, data_json, committed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (case_id, invoice_number, extraction.vendor_name, extraction.model_dump_json(), utc_now()),
            )
        status = results["lookup_vendor_status"].get("status")
        self.db.execute(
            """INSERT INTO payment_actions
               (committed_record_id, registry_status, decision, scheduled_date, created_at)
               VALUES (?, ?, 'SCHEDULED', ?, ?)
               ON CONFLICT(committed_record_id) DO NOTHING""",
            (record_id, status, extraction.due_date.isoformat(), utc_now()),
        )
        self.db.execute(
            "UPDATE invoice_cases SET status = 'PAYMENT_SCHEDULED', review_reason = NULL, updated_at = ? WHERE id = ?",
            (utc_now(), case_id),
        )
        return {"committed": True, "record_id": record_id, "payment_scheduled": True}

    def _flag(
        self,
        case_id: int,
        requested_reason: Any,
        results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        failed = self._failed_checks(results)
        reason = str(requested_reason or "Validation or vendor approval did not pass.")
        if failed:
            reason = f"{reason} Failed checks: {', '.join(failed)}."
        followup_type = "VENDOR_STATUS" if "lookup_vendor_status" in failed else "VALIDATION_REVIEW"
        existing = self.db.query_one(
            """SELECT id FROM followups WHERE case_id = ? AND followup_type = ?
               AND description = ? AND status = 'OPEN'""",
            (case_id, followup_type, reason),
        )
        if not existing:
            self.db.execute(
                """INSERT INTO followups
                   (case_id, followup_type, description, status, created_at)
                   VALUES (?, ?, ?, 'OPEN', ?)""",
                (case_id, followup_type, reason, utc_now()),
            )
        case_status = "VENDOR_FLAGGED" if followup_type == "VENDOR_STATUS" else "NEEDS_REVIEW"
        self.db.execute(
            "UPDATE invoice_cases SET status = ?, review_reason = ?, updated_at = ? WHERE id = ?",
            (case_status, reason, utc_now(), case_id),
        )
        return {"flagged": True, "reason": reason, "failed_checks": failed}

    def _log_message(
        self,
        case_id: int,
        run_id: str,
        sequence: int,
        message: AgentMessage,
    ) -> None:
        tool_calls = [call.model_dump() for call in message.tool_calls] if isinstance(message, AIMessage) else None
        self.db.log_agent_message(
            case_id=case_id,
            agent_name="validation",
            run_id=run_id,
            sequence_number=sequence,
            role=message.role,
            content=message.content,
            tool_calls=tool_calls,
            tool_call_id=message.tool_call_id if isinstance(message, ToolMessage) else None,
            tool_name=message.name if isinstance(message, ToolMessage) else None,
        )
