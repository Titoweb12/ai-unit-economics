import datetime
import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "telemetry.db"

COLUMNS = (
    "ts", "run_id", "system", "question_id", "trial", "step", "provider", "model",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "latency_ms", "cost_usd", "status", "error_type",
)

OUTCOME_COLUMNS = (
    "ts", "run_id", "system", "question_id", "trial", "qtype", "lang",
    "expect_answer", "hit5", "hit_used", "refused", "unsupported",
    "same_language", "error",
)

# Si la définition de la vue v_outcomes change un jour : DROP VIEW v_outcomes,
# puis relancer connect() pour la recréer.
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

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT NOT NULL,
    system TEXT NOT NULL,
    question_id TEXT NOT NULL,
    trial INTEGER NOT NULL,
    qtype TEXT,
    lang TEXT,
    expect_answer INTEGER,
    hit5 INTEGER,
    hit_used INTEGER,
    refused INTEGER,
    unsupported INTEGER,
    same_language INTEGER,
    error TEXT,
    UNIQUE (run_id, system, question_id, trial)
);

-- Définition du succès (un seul endroit) :
--  question à réponse : réponse donnée, aucune affirmation non soutenue,
--                       bon passage utilisé (si une attente existe) ;
--  question à refuser : refus correct ;
--  une erreur d'exécution est un échec.
CREATE VIEW IF NOT EXISTS v_outcomes AS
SELECT *,
  CASE
    WHEN error IS NOT NULL THEN 0
    WHEN expect_answer = 1 THEN
      CASE WHEN refused = 0 AND unsupported = 0 AND COALESCE(hit_used, 1) = 1
           THEN 1 ELSE 0 END
    ELSE
      CASE WHEN refused = 1 THEN 1 ELSE 0 END
  END AS success
FROM outcomes;
"""


def connect(path=DB_PATH):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _insert(conn, table, allowed, row):
    unknown = set(row) - set(allowed)
    if unknown:
        raise ValueError(f"colonnes inconnues : {sorted(unknown)}")
    row.setdefault("ts", datetime.datetime.now(datetime.timezone.utc).isoformat())
    cols = list(row)
    placeholders = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()


def log_call(conn, **row):
    """Enregistre un appel. Aucun texte de question, de réponse ou de prompt."""
    _insert(conn, "calls", COLUMNS, row)


def log_outcome(conn, **row):
    """Enregistre le verdict d'un passage (système, question, essai). Aucun texte."""
    _insert(conn, "outcomes", OUTCOME_COLUMNS, row)


def outcome_exists(conn, run_id, system, question_id, trial):
    return conn.execute(
        "SELECT 1 FROM outcomes WHERE run_id=? AND system=? AND question_id=? AND trial=?",
        (run_id, system, question_id, trial),
    ).fetchone() is not None