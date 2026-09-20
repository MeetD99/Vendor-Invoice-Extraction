import csv
import hashlib
import io
import re
import uuid
from pathlib import Path, PurePath

from app.config import Settings
from app.database import Database, utc_now
from app.services.parsers import classify_document, document_metadata


def safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    parts = [part for part in PurePath(normalized).parts if part not in {"", ".", "..", "/"}]
    return "/".join(parts) or "unnamed.txt"


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


class IngestionService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.settings.upload_path.mkdir(parents=True, exist_ok=True)

    def create_batch(self, name: str) -> int:
        return self.db.execute(
            "INSERT INTO import_batches (name, status, created_at) VALUES (?, 'UPLOADING', ?)",
            (name, utc_now()),
        )

    def ingest_document(self, batch_id: int, filename: str, content: bytes) -> dict[str, object]:
        relative_path = safe_relative_path(filename)
        if Path(relative_path).suffix.lower() != ".txt":
            return {"filename": relative_path, "status": "UNSUPPORTED"}

        digest = hashlib.sha256(content).hexdigest()
        existing = self.db.query_one(
            "SELECT id FROM documents WHERE sha256 = ? AND original_filename = ?",
            (digest, Path(relative_path).name),
        )
        if existing:
            return {"filename": relative_path, "status": "DUPLICATE", "document_id": existing["id"]}

        text = decode_text(content)
        document_type = classify_document(text, relative_path)
        metadata = document_metadata(text, relative_path, document_type)
        case_key = metadata["case_key"] or f"UNASSIGNED-{uuid.uuid4().hex[:8].upper()}"
        case = self.db.query_one("SELECT id FROM invoice_cases WHERE case_key = ?", (case_key,))
        if case:
            case_id = int(case["id"])
        else:
            case_id = self.db.execute(
                """INSERT INTO invoice_cases (case_key, status, created_at, updated_at)
                   VALUES (?, 'UPLOADED', ?, ?)""",
                (case_key, utc_now(), utc_now()),
            )

        batch_folder = self.settings.upload_path / str(batch_id)
        batch_folder.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}_{Path(relative_path).name}"
        stored_path = batch_folder / stored_name
        stored_path.write_bytes(content)

        document_id = self.db.execute(
            """INSERT INTO documents
               (batch_id, case_id, original_filename, relative_path, stored_path, sha256,
                raw_text, document_type, document_number, base_document_number,
                revision_number, document_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                case_id,
                Path(relative_path).name,
                relative_path,
                str(stored_path),
                digest,
                text,
                document_type,
                metadata["document_number"],
                metadata["base_document_number"],
                metadata["revision_number"],
                metadata["document_date"],
                utc_now(),
            ),
        )
        return {
            "filename": relative_path,
            "status": "INGESTED",
            "document_id": document_id,
            "case_id": case_id,
            "case_key": case_key,
            "document_type": document_type,
        }

    def finish_batch(self, batch_id: int, results: list[dict[str, object]]) -> None:
        case_ids = sorted({int(item["case_id"]) for item in results if item.get("case_id")})
        with self.db.transaction() as connection:
            connection.execute(
                """UPDATE import_batches SET status = 'QUEUED', total_files = ?, completed_at = ?
                   WHERE id = ?""",
                (len(results), utc_now(), batch_id),
            )
            for case_id in case_ids:
                pending = connection.execute(
                    """SELECT 1 FROM jobs WHERE case_id = ? AND job_type = 'PROCESS_CASE'
                       AND status IN ('PENDING', 'RUNNING')""",
                    (case_id,),
                ).fetchone()
                if not pending:
                    connection.execute(
                        """INSERT INTO jobs (case_id, job_type, status, created_at)
                           VALUES (?, 'PROCESS_CASE', 'PENDING', ?)""",
                        (case_id, utc_now()),
                    )

    def import_registry(self, content: bytes) -> int:
        text = decode_text(content)
        reader = csv.DictReader(io.StringIO(text))
        count = 0
        for row in reader:
            normalized = {re.sub(r"[^a-z]", "", key.lower()): (value or "").strip() for key, value in row.items()}
            invoice = normalized.get("invoicenumber")
            status = normalized.get("verifiedstatus")
            if not invoice or not status:
                continue
            self.db.execute(
                """INSERT INTO registry_status (invoice_number, verified_status, imported_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(invoice_number) DO UPDATE SET
                       verified_status = excluded.verified_status,
                       imported_at = excluded.imported_at""",
                (invoice.upper(), status, utc_now()),
            )
            count += 1
        return count

