import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS import_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'UPLOADING',
    total_files INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS invoice_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'UPLOADED',
    review_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES import_batches(id),
    case_id INTEGER REFERENCES invoice_cases(id),
    original_filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    document_type TEXT NOT NULL,
    document_number TEXT,
    base_document_number TEXT,
    revision_number INTEGER NOT NULL DEFAULT 0,
    document_date TEXT,
    is_current INTEGER NOT NULL DEFAULT 1,
    superseded_by_id INTEGER REFERENCES documents(id),
    created_at TEXT NOT NULL,
    UNIQUE(batch_id, relative_path),
    UNIQUE(sha256, original_filename)
);

CREATE TABLE IF NOT EXISTS document_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    source_document_id INTEGER NOT NULL REFERENCES documents(id),
    related_document_id INTEGER NOT NULL REFERENCES documents(id),
    relationship_type TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence TEXT,
    UNIQUE(source_document_id, related_document_id, relationship_type)
);

CREATE TABLE IF NOT EXISTS invoice_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL UNIQUE REFERENCES invoice_cases(id),
    source_document_id INTEGER NOT NULL REFERENCES documents(id),
    data_json TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS field_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL REFERENCES invoice_drafts(id),
    field_name TEXT NOT NULL,
    value TEXT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    excerpt TEXT NOT NULL,
    verification TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS committed_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    invoice_number TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    data_json TEXT NOT NULL,
    record_version INTEGER NOT NULL DEFAULT 1,
    is_current INTEGER NOT NULL DEFAULT 1,
    committed_at TEXT NOT NULL,
    UNIQUE(invoice_number, record_version)
);

CREATE TABLE IF NOT EXISTS registry_status (
    invoice_number TEXT PRIMARY KEY,
    verified_status TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    committed_record_id INTEGER NOT NULL REFERENCES committed_records(id),
    registry_status TEXT,
    decision TEXT NOT NULL,
    scheduled_date TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(committed_record_id)
);

CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    followup_type TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    tool_name TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    agent_name TEXT NOT NULL DEFAULT 'extraction',
    run_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS review_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    field_name TEXT NOT NULL,
    previous_value TEXT,
    corrected_value TEXT,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES invoice_cases(id),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().database_file
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(agent_messages)").fetchall()
            }
            if "agent_name" not in columns:
                connection.execute(
                    "ALTER TABLE agent_messages ADD COLUMN agent_name TEXT NOT NULL DEFAULT 'extraction'"
                )
            connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> int:
        with closing(self.connect()) as connection:
            cursor = connection.execute(sql, parameters)
            connection.commit()
            return int(cursor.lastrowid)

    def query_one(self, sql: str, parameters: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(sql, parameters).fetchone()
            return dict(row) if row else None

    def query_all(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(sql, parameters).fetchall()
            return [dict(row) for row in rows]

    def log_tool(
        self,
        case_id: int,
        name: str,
        request: dict[str, Any],
        response: dict[str, Any],
        status: str = "SUCCESS",
    ) -> None:
        self.execute(
            """INSERT INTO tool_calls
               (case_id, tool_name, request_json, response_json, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (case_id, name, json.dumps(request), json.dumps(response, default=str), status, utc_now()),
        )

    def log_agent_message(
        self,
        *,
        case_id: int,
        agent_name: str,
        run_id: str,
        sequence_number: int,
        role: str,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        self.execute(
            """INSERT INTO agent_messages
               (case_id, agent_name, run_id, sequence_number, role, content, tool_calls_json,
                tool_call_id, tool_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                agent_name,
                run_id,
                sequence_number,
                role,
                content,
                json.dumps(tool_calls or [], default=str),
                tool_call_id,
                tool_name,
                utc_now(),
            ),
        )


db = Database()
