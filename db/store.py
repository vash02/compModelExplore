import sqlite3
import json
import textwrap
import uuid
from datetime import datetime

# db/store.py
# ──────────────────────────────────────────────────────────────
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_DEFAULT = Path("mcp.db")

# helper – open conn in row-dict mode
def _conn(db_path: str | Path = DB_DEFAULT) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

# ──────────────────────────────────────────────────────────────
def store_simulation_script(
    model_name: str,
    metadata: Dict[str, Any],
    script_path: str,
    db_path: str | Path = DB_DEFAULT,
) -> str:
    """
    Insert or update a simulation entry and return its model_id (string).

    Columns: id (PK TEXT), name TEXT, metadata TEXT, script_path TEXT
    """
    model_id = model_name        # ← slug already unique; adjust if needed
    with _conn(db_path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS simulations (
                   id          TEXT PRIMARY KEY,
                   name        TEXT,
                   metadata    TEXT,
                   script_path TEXT
               )"""
        )
        c.execute(
            """INSERT OR REPLACE INTO simulations
               (id, name, metadata, script_path)
               VALUES (?,  ?,    ?,        ?)""",
            (model_id, model_name, json.dumps(metadata), script_path),
        )
    return model_id


# ★ NEW helper ----------------------------------------------------------
def get_simulation_path(model_id: str,
                        db_path: str | Path = DB_DEFAULT) -> str:
    """
    Return the absolute path to `simulate.py` for the given model_id.

    Raises KeyError if the model_id is unknown.
    """
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT script_path FROM simulations WHERE id = ?", (model_id,)
        ).fetchone()

    if row is None:  # → unknown ID
        raise KeyError(f"model_id '{model_id}' not found in DB {db_path}")

    return row["script_path"]


def get_simulation_script(model_id: str, db_path="mcp.db") -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT script_path FROM simulations WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No script found for model_id={model_id}")
    return textwrap.dedent(row[0])


def get_simulation_script_code(model_id: str, db_path: str = "mcp.db") -> str:
    """
    Fetch the saved path for this model_id, read that file,
    dedent it, and return the actual Python code as a string.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT script_path FROM simulations WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No script found for model_id={model_id!r}")

    script_path = row[0]
    code = Path(script_path).read_text(encoding="utf-8")
    return textwrap.dedent(code)
# db/store.py  (append at the end of the file)

# ────────────────────────────────────────────────────────────
#  STORE BATCH RESULTS
# ------------------------------------------------------------------

def store_simulation_results(
    model_id: str,
    rows: List[Dict[str, Any]],
    param_keys: List[str] | None = None,
    db_path: str | Path = DB_DEFAULT,
) -> None:
    """
    Persist a list of experiment rows *rows* for a given model_id.

    Schema
    ------
    CREATE TABLE IF NOT EXISTS results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id    TEXT,            -- FK → simulations.id   (no ON DELETE)
        ts          TEXT,            -- ISO 8601 timestamp
        params      TEXT,            -- JSON blob (input dict)
        outputs     TEXT             -- JSON blob (returned by simulate)
    )
    """
    if param_keys is None and rows:
        # infer params = keys that appeared in the original grid
        param_keys = [k for k in rows[0].keys() if k != "error"]

    with _conn(db_path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS results (
                   id       INTEGER PRIMARY KEY AUTOINCREMENT,
                   model_id TEXT,
                   ts       TEXT,
                   params   TEXT,
                   outputs  TEXT
               )"""
        )

        ts_now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for row in rows:
            params  = {k: row[k] for k in param_keys if k in row}
            outputs = {k: v for k, v in row.items() if k not in params}
            c.execute(
                "INSERT INTO results (model_id, ts, params, outputs) VALUES (?,?,?,?)",
                (
                    model_id,
                    ts_now,
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(outputs, ensure_ascii=False),
                ),
            )
def get_model_metadata(model_id: str, db_path: str | Path = DB_DEFAULT) -> str:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT metadata FROM simulations WHERE id = ?", (model_id,)
        )
        row = row.fetchone()
        if row is None:
            raise ValueError(f"No metadata found for model_id={model_id}")
        return json.loads(row["metadata"])
