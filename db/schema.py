import sqlite3
from pathlib import Path

def init_db(db_path="mcp.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulations (
                   id          TEXT PRIMARY KEY,
                   name        TEXT,
                   metadata    TEXT,
                   script_path TEXT
               )
    """)

    conn.commit()
    conn.close()
