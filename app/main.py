import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agent.adapter import create_llm_adapter
from app.agent.orchestrator import InvoiceAgent, QueueWorker
from app.config import ROOT_DIR, get_settings
from app.database import db, utc_now
from app.schemas import InvoiceExtraction
from app.services.ingestion import IngestionService


settings = get_settings()
ingestion = IngestionService(db, settings)
llm = create_llm_adapter(settings)
agent = InvoiceAgent(db, llm)
worker = QueueWorker(db, agent)
templates = Jinja2Templates(directory=ROOT_DIR / "app" / "templates")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.initialize()
    task = asyncio.create_task(worker.run())
    yield
    worker.stop()
    await task


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")


class CorrectionRequest(BaseModel):
    field_name: str
    value: Any
    reason: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"app_name": settings.app_name})


@app.post("/api/imports/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    registry: UploadFile | None = File(default=None),
):
    batch_id = ingestion.create_batch(f"Folder upload {utc_now()}")
    results: list[dict[str, object]] = []
    for upload in files:
        content = await upload.read()
        results.append(ingestion.ingest_document(batch_id, upload.filename or "unnamed.txt", content))
    registry_rows = 0
    if registry:
        registry_rows = ingestion.import_registry(await registry.read())
    ingestion.finish_batch(batch_id, results)
    return {"batch_id": batch_id, "files": results, "registry_rows": registry_rows}


@app.post("/api/registry/upload")
async def upload_registry(registry: UploadFile = File(...)):
    return {"imported_rows": ingestion.import_registry(await registry.read())}


@app.get("/api/dashboard")
async def dashboard():
    counts = db.query_all("SELECT status, COUNT(*) AS count FROM invoice_cases GROUP BY status")
    jobs = db.query_all("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status")
    return {
        "cases": {item["status"]: item["count"] for item in counts},
        "jobs": {item["status"]: item["count"] for item in jobs},
        "documents": db.query_one("SELECT COUNT(*) AS count FROM documents")["count"],
        "committed": db.query_one("SELECT COUNT(*) AS count FROM committed_records WHERE is_current = 1")["count"],
    }


@app.get("/api/cases")
async def list_cases():
    return db.query_all(
        """SELECT c.*,
                  COUNT(DISTINCT d.id) AS document_count,
                  dr.validation_status,
                  cr.invoice_number,
                  cr.vendor_name,
                  pa.decision AS payment_decision
           FROM invoice_cases c
           LEFT JOIN documents d ON d.case_id = c.id
           LEFT JOIN invoice_drafts dr ON dr.case_id = c.id
           LEFT JOIN committed_records cr ON cr.case_id = c.id AND cr.is_current = 1
           LEFT JOIN payment_actions pa ON pa.committed_record_id = cr.id
           GROUP BY c.id
           ORDER BY c.updated_at DESC"""
    )


@app.get("/api/queue")
async def list_queue():
    return db.query_all(
        """SELECT j.*, c.case_key FROM jobs j
           JOIN invoice_cases c ON c.id = j.case_id
           ORDER BY j.created_at DESC, j.id DESC LIMIT 100"""
    )


@app.get("/api/cases/{case_id}")
async def case_detail(case_id: int):
    case = db.query_one("SELECT * FROM invoice_cases WHERE id = ?", (case_id,))
    if not case:
        raise HTTPException(404, "Case not found")
    draft = db.query_one("SELECT * FROM invoice_drafts WHERE case_id = ?", (case_id,))
    if draft:
        draft["data"] = json.loads(draft.pop("data_json"))
        draft["evidence"] = db.query_all(
            "SELECT * FROM field_evidence WHERE draft_id = ? ORDER BY id", (draft["id"],)
        )
    committed = db.query_one(
        "SELECT * FROM committed_records WHERE case_id = ? AND is_current = 1", (case_id,)
    )
    if committed:
        committed["data"] = json.loads(committed.pop("data_json"))
        committed["payment"] = db.query_one(
            "SELECT * FROM payment_actions WHERE committed_record_id = ?", (committed["id"],)
        )
    return {
        "case": case,
        "documents": db.query_all(
            "SELECT * FROM documents WHERE case_id = ? ORDER BY document_type, revision_number DESC", (case_id,)
        ),
        "relationships": db.query_all(
            """SELECT r.*, d.original_filename AS related_filename
               FROM document_relationships r
               JOIN documents d ON d.id = r.related_document_id
               WHERE r.case_id = ? ORDER BY r.id""",
            (case_id,),
        ),
        "draft": draft,
        "committed": committed,
        "followups": db.query_all("SELECT * FROM followups WHERE case_id = ? ORDER BY id DESC", (case_id,)),
        "tool_calls": db.query_all("SELECT * FROM tool_calls WHERE case_id = ? ORDER BY id", (case_id,)),
        "agent_messages": db.query_all(
            """SELECT * FROM agent_messages WHERE case_id = ?
               ORDER BY created_at, run_id, sequence_number""",
            (case_id,),
        ),
        "corrections": db.query_all(
            "SELECT * FROM review_corrections WHERE case_id = ? ORDER BY id DESC", (case_id,)
        ),
    }


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: int):
    job = db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if not job:
        raise HTTPException(404, "Job not found")
    db.execute(
        "UPDATE jobs SET status = 'PENDING', error_message = NULL, completed_at = NULL WHERE id = ?",
        (job_id,),
    )
    return {"status": "PENDING"}


@app.post("/api/cases/{case_id}/reprocess")
async def reprocess_case(case_id: int):
    if not db.query_one("SELECT id FROM invoice_cases WHERE id = ?", (case_id,)):
        raise HTTPException(404, "Case not found")
    job_id = db.execute(
        "INSERT INTO jobs (case_id, job_type, status, created_at) VALUES (?, 'PROCESS_CASE', 'PENDING', ?)",
        (case_id, utc_now()),
    )
    db.execute("UPDATE invoice_cases SET status = 'QUEUED', updated_at = ? WHERE id = ?", (utc_now(), case_id))
    return {"job_id": job_id, "status": "PENDING"}


@app.patch("/api/cases/{case_id}/field")
async def correct_field(case_id: int, correction: CorrectionRequest):
    draft = db.query_one("SELECT * FROM invoice_drafts WHERE case_id = ?", (case_id,))
    if not draft:
        raise HTTPException(404, "Invoice draft not found")
    payload = json.loads(draft["data_json"])
    if correction.field_name not in InvoiceExtraction.model_fields:
        raise HTTPException(400, "Unknown or non-editable field")
    previous = payload.get(correction.field_name)
    payload[correction.field_name] = correction.value
    try:
        validated = InvoiceExtraction.model_validate(payload)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    flag_map = {
        "vendor_name": {"VENDOR_NAME_MISSING"},
        "invoice_number": {"INVOICE_NUMBER_MISSING"},
        "line_items": {"LINE_ITEMS_MISSING", "LINE_ITEMS_DO_NOT_MATCH_TOTAL"},
        "total_due": {"TOTAL_DUE_MISSING", "LINE_ITEMS_DO_NOT_MATCH_TOTAL"},
        "due_date": {"DUE_DATE_MISSING", "DUE_DATE_REQUIRES_RESOLUTION", "RELATED_DATE_DOCUMENT_MISSING_OR_AMBIGUOUS"},
        "remit_to": {"REMIT_TO_MISMATCH_UNEXPLAINED"},
    }
    validated.flags = [flag for flag in validated.flags if flag not in flag_map.get(correction.field_name, set())]
    status = "PENDING_VALIDATION"
    validated.flags = list(dict.fromkeys(validated.flags))
    db.execute(
        "UPDATE invoice_drafts SET data_json = ?, validation_status = ?, updated_at = ? WHERE id = ?",
        (validated.model_dump_json(), status, utc_now(), draft["id"]),
    )
    db.execute(
        """INSERT INTO review_corrections
           (case_id, field_name, previous_value, corrected_value, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            case_id,
            correction.field_name,
            json.dumps(previous, default=str),
            json.dumps(correction.value, default=str),
            correction.reason,
            utc_now(),
        ),
    )
    db.execute(
        "UPDATE invoice_cases SET status = ?, review_reason = NULL, updated_at = ? WHERE id = ?",
        (status, utc_now(), case_id),
    )
    return {"validation_status": status, "data": validated.model_dump(mode="json")}


@app.post("/api/cases/{case_id}/commit")
async def commit_reviewed_case(case_id: int):
    draft = db.query_one("SELECT * FROM invoice_drafts WHERE case_id = ?", (case_id,))
    if not draft:
        raise HTTPException(404, "Invoice draft not found")
    try:
        decision = await agent.validate_reviewed_draft(case_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return decision.model_dump(mode="json")
