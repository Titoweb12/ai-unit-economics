import datetime
import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "telemetry.db"

COLUMNS = (
    "ts", "run_id", "system", "question_id", "trial", "step", "provider", "model",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "latency_ms", "cost_usd", "status", "error_type",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    system TEXT,
    question_id TEXT,
    trial INTEGER,
    step TEXT,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER,
    cost_usd REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
"""


def connect(path=DB_PATH):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def log_call(conn, **row):
    """Enregistre un appel. Aucun texte de question, de réponse ou de prompt."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"colonnes inconnues : {sorted(unknown)}")
    row.setdefault("ts", datetime.datetime.now(datetime.timezone.utc).isoformat())
    cols = list(row)
    placeholders = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO calls ({', '.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()