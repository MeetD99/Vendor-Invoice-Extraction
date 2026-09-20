# Vendor Invoice Extraction

A small FastAPI application that stages folders of text documents, groups them into invoice cases, runs separate extraction and validation/decision agents, and persists the complete review and action history in SQLite.

## Processing boundary

Safety-critical indexing remains deterministic: Python handles paths, hashes, coarse document types, VI grouping, and revision lineage. This index is a manifest, not field extraction. Ambiguous document meaning can still be investigated by an agent through document tools without allowing a model to redefine file ownership or revision precedence.

The extraction LLM receives no invoice text in the initial request and must request documents through tools:

```text
HumanMessage: process case; main document_id is N
    ↓
AIMessage: read_source_document tool call
    ↓
Python executes the tool
    ↓
ToolMessage: line-numbered invoice text
    ↓
AIMessage: optional search_related_documents tool call
    ↓
Python returns its canonical current revision plus superseded metadata
    ↓
AIMessage: ready to finalize
    ↓
Chat Completions JSON Schema response
    ↓
Pydantic validation
    ↓
Independent Validation/Decision Agent
    ├── get_extraction_candidate
    ├── validate_source_evidence
    ├── reconcile_invoice_total
    ├── enforce_due_date_policy
    ├── validate_remit_to
    └── lookup_vendor_status
    ↓
AI calls finalize_validation_decision with COMMIT or FLAG
    ↓
Python commit gate rechecks every tool result before changing state
```

The LLM handles semantic extraction, including:

- Vendor and Remit To names and any stated explanation for a mismatch.
- Line-item interpretation.
- Amounts written in words.
- Missing totals derived from complete line items.
- `MM/DD/YYYY` date normalization.
- Numeric `1/l/I` and `0/O` ambiguity in numeric fields.
- Exact source excerpts for field evidence.
- Recognizing when another document must be consulted.

Python exclusively controls:

- File safety, hashing, decoding, persistence, and duplicate detection.
- VI case grouping and document classification/indexing.
- Revision families and precedence.
- Which related document is current.
- Due-date recalculation from canonical documents and payment terms.
- Line-item reconciliation and evidence verification.
- Deterministic validation tool results and commit authorization gates.
- Idempotent commits, vendor-status enforcement, and payment scheduling.

## What is implemented

- Folder and multi-file browser upload.
- Complete-batch staging before processing starts.
- SQLite-backed queue with retries.
- VI-number grouping from filenames or content.
- Vendor invoice, purchase order, delivery confirmation, and credit-note indexing.
- Immutable original/revised document lineage.
- Provider-neutral `LLMAdapter` interface.
- Exact configurable Chat Completions URL.
- OpenAI-compatible function tools and strict JSON Schema output.
- Typed `HumanMessage`, `AIMessage`, and `ToolMessage` history.
- Independent extraction and validation/decision message loops.
- Mandatory validation and vendor-status tool calls before any commit action.
- Vendor flagging with no committed record when approval is missing or blocked.
- Python-side tool execution and persisted message/tool audit trail.
- Human review, field correction, follow-ups, commit history, and payment actions.

## Start the application

```powershell
cd "C:\Users\Meet\Desktop\Vendor Invoice Extraction"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env`, then run:

```powershell
python run.py
```

Open <http://127.0.0.1:8000>.

## Configure OpenAI Chat Completions

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_CHAT_COMPLETIONS_URL=https://api.openai.com/v1/chat/completions
LLM_EXTRA_HEADERS_JSON={}
LLM_MAX_AGENT_STEPS=8
```

## Configure a company-compatible endpoint

```dotenv
LLM_PROVIDER=openai_compatible
LLM_MODEL=company-model-name
LLM_API_KEY=company-api-key
LLM_CHAT_COMPLETIONS_URL=https://company.example/api/v1/chat/completions
LLM_EXTRA_HEADERS_JSON={"X-Application":"vendor-invoice-quest"}
```

The configured service must support Chat Completions-style `messages`, function `tools`, assistant `tool_calls`, tool-role messages, and `response_format.type=json_schema`.

## Configure local Ollama

Ollama exposes an OpenAI-compatible endpoint. With `mistral:latest` installed, use:

```dotenv
LLM_PROVIDER=openai_compatible
LLM_MODEL=mistral:latest
LLM_API_KEY=
LLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:11434/v1/chat/completions
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=120
```

The controller exposes only the expected tool on forced turns. If a small local model describes a mandatory read-only tool in prose instead of emitting a function call, the controller records that fallback and safely materializes only the already-required document-read or deterministic-validation call. The final `COMMIT` or `FLAG` choice still comes from the model and remains subject to the Python commit gate.

Keys and headers are read server-side and are not sent to the browser or persisted in SQLite.

If a company endpoint uses a different wire protocol, implement `LLMAdapter.complete` and register the adapter in `create_llm_adapter`. The agent workflow and deterministic controls remain unchanged.

## Status registry

The CSV expects `Invoice Number` and `Verified Status`. Payment scheduling currently accepts these normalized values:

- `Verified`
- `Active`
- `Approved`
- `Yes`
- `True`
- `Active - Not On Hold`

Any other or missing status makes the commit gate reject the record. The validation agent selects `FLAG` through `finalize_validation_decision`, which creates a vendor-status follow-up and leaves the invoice uncommitted.

## Tests

```powershell
python -m unittest tests.test_agent_loop -v
python -c "import runpy; n=runpy.run_path('tests/test_parsers.py'); [v() for k,v in n.items() if k.startswith('test_') and callable(v)]"
```

The agent tests use a scripted model adapter to prove both message loops, Python tool execution, revised-PO selection, JSON-schema extraction, deterministic validation, approved-vendor commitment, and unapproved-vendor flagging without making an external API call.
