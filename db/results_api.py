import json
import sqlite3
from pathlib import Path

import pandas as pd


def _safe_parse(d: str | None) -> dict:
    try:
        val = json.loads(d)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def load_results(
    db_path: str | Path = "mcp.db",
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Load the `results` table, expand JSON columns, and optionally
    filter to a single model_id.

    Returns a DataFrame with columns:
      model_id, ts, <all params fields>, <all output fields>
    """
    # 1) fetch raw rows
    con = sqlite3.connect(str(db_path))
    raw_df = pd.read_sql("SELECT model_id, params, outputs, ts FROM results", con)
    con.close()

    # 2) optionally filter to only the given model_id
    if model_id is not None:
        raw_df = raw_df[raw_df["model_id"] == model_id]

    # 3) parse the JSON columns safely
    raw_df["params"] = raw_df["params"].apply(_safe_parse)
    raw_df["outputs"] = raw_df["outputs"].apply(_safe_parse)

    # 4) drop any rows where parsing failed (empty dict)
    filtered = raw_df[
        raw_df["params"].apply(bool) & raw_df["outputs"].apply(bool)
    ].reset_index(drop=True)

    # 5) expand the dict columns into separate DataFrame columns
    params_df = pd.json_normalize(filtered["params"])
    outputs_df = pd.json_normalize(filtered["outputs"])

    # 6) concatenate model_id, ts, parameters, and outputs
    final = pd.concat(
        [
            filtered[["model_id", "ts"]],
            params_df,
            outputs_df
        ],
        axis=1
    )

    return final
