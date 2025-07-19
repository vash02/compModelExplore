# core/agent_loop.py
# ─────────────────────────────────────────────────────────────────────
"""
Framework‑free reasoning agent over the results DataFrame.

The model never sees the raw data.  Instead it issues code snippets
via the `python_exec` tool; we run them on `df` internally, then feed
back results until it returns a final JSON answer.

Supports two backends:
  • "local": LocalLLM(deepseek-r1:14b)
  • "openai": OpenAI GPT-4 via the new openai.chat.completions.create API
"""

import io
import json
import logging
import os
import sqlite3
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import openai

from llm.local_llm import LocalLLM
from llm.prompt_templates import SYSTEM_PROMPT_TEMPLATE
from db.results_api import load_results
from db.store import get_model_metadata, get_simulation_script_code
from core.tools import python_exec  # your in‑process DataFrame executor

# ───────────────────────── Logging setup ─────────────────────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
log = logging.getLogger("agent_loop")

# ───────────────────────── Paths & constants ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "mcp.db"
MAX_STEPS    = 20

# Ensure your OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY", openai.api_key or "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")

# Track any plots existing before the very first call
seen_before = {p.name for p in Path.cwd().glob("*.png")}


def _store_report(
    model_id: str,
    question: str,
    answer: str,
    image_paths: List[str]
) -> None:
    """
    Insert a reasoning report into the `reasoning_agent` table.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_agent (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id  TEXT    NOT NULL,
            question  TEXT    NOT NULL,
            answer    TEXT    NOT NULL,
            images    TEXT,   -- JSON array of file names
            ts        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        INSERT INTO reasoning_agent (model_id, question, answer, images)
        VALUES (?, ?, ?, ?)
    """, (
        model_id,
        question,
        answer,
        json.dumps(image_paths, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()
    log.info("Stored reasoning report for model %s", model_id)


def ask(
    model_id: str,
    question: str,
    backend: str = "local"
) -> Dict[str, Any]:
    """
    Run a reasoning agent for `model_id` with follow-up `question`.

    backend:
      - "local": uses LocalLLM(deepseek-r1:14b)
      - "openai": uses OpenAI GPT-4 via openai.chat.completions.create

    Returns:
      {
        "answer": "...final explanation...",
        "images": ["plot_<uuid>.png", ...]
      }
    """
    log.info("=== Starting analysis for model_id=%s (backend=%s) ===",
             model_id, backend)

    # 1) load the data, metadata, and simulation code
    df       = load_results(db_path=str(DB_PATH), model_id=model_id)
    meta     = get_model_metadata(model_id, db_path=str(DB_PATH))
    sim_code = get_simulation_script_code(model_id, db_path=str(DB_PATH))
    log.info("Loaded %d rows for model_id=%s", len(df), model_id)

    # 2) build a minimal system prompt (no raw data)
    schema = list(df.columns)
    params = [{"name": k, "description": v}
              for k, v in meta.get("parameters", {}).items()]

    system_prompt = f"""
    You are a scientific reasoning assistant. You have access to the simulation model implementation below,
    but you do *not* see the raw DataFrame directly. When you need to operate on the DataFrame `df`,
    you must issue exactly:

      {{\"function_call\": {{\"name\": \"python_exec\", \"arguments\": {{\"code\": \"<python code>\"}}}}}}

    You will then receive a JSON with keys `ok`, `stdout`, `stderr`, and `images` that you can use to refine your reasoning.

    ──────────── SIMULATION CODE ────────────
    ```python
    {sim_code}
    ```
    ──────────── DATA SCHEMA ────────────
    Columns in df:
    {schema}

    ───────── PARAMETERS (name → description) ─────────
    {params}

    ──────── USAGE ────────

    Issue one or more `python_exec` calls to filter/aggregate/plot.
    Use the returned JSON to inform your next step.
    When ready, return exactly:

      {{\"answer\": \"<your detailed explanation, conclusions, and any recommendations>\"}}

    — Only one valid JSON object per assistant message, either a function_call or an answer.
    """

    # 3) helper to run python_exec tool and pick up new images
    def _run_python(code: str) -> Dict[str, Any]:
        log.info("Tool executing code snippet (%.50s...)", code.strip())
        result = python_exec(code=code, df=df)
        # update our seen_before set so we don't pick these again
        for img in result.get("images", []):
            seen_before.add(img)
        log.debug("Tool returned: %s", result)
        return result

    # 4) prepare conversation history
    history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    # 5) declare our python_exec function for OpenAI
    functions = [{
        "name": "python_exec",
        "description": "Run Python code on `df`, assign result to `_`, return ok/stdout/stderr/images.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python snippet using df; assign your final output to '_', can also save plots."
                }
            },
            "required": ["code"]
        }
    }]

    # accumulate all images we see
    all_images: List[str] = []

    # 6) agent loop
    for step in range(1, MAX_STEPS + 1):
        log.info("=== STEP %d/%d ===", step, MAX_STEPS)

        # 6a) call the chosen LLM
        if backend == "openai":
            resp = openai.chat.completions.create(
                model="gpt-4.1",
                messages=history,
                functions=functions,
                function_call="auto",
                temperature=0.0,
            )
            msg = resp.choices[0].message
            assistant_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if getattr(msg, "function_call", None):
                assistant_msg["function_call"] = {
                    "name": msg.function_call.name,
                    "arguments": msg.function_call.arguments
                }
        else:
            llm = LocalLLM(model="deepseek-r1:14b")
            if hasattr(llm, "chat"):
                reply = llm.chat(history)
            else:
                prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
                reply = llm.generate(prompt)
            assistant_msg = {"role": "assistant", "content": reply}

        history.append(assistant_msg)

        # 6b) handle function_call
        if assistant_msg.get("function_call"):
            args = json.loads(assistant_msg["function_call"]["arguments"] or "{}")
            code = args.get("code", "")
            log.info("Invoking python_exec with code:\n%s", code)
            tool_res = _run_python(code)
            # collect any new images
            all_images.extend(tool_res.get("images", []))
            history.append({
                "role": "function",
                "name": "python_exec",
                "content": json.dumps(tool_res, ensure_ascii=False)
            })
            continue

        # 6c) check for final answer
        try:
            payload = json.loads(assistant_msg["content"])
            if "answer" in payload:
                answer = payload["answer"]
                log.info("Agent provided final answer.")
                _store_report(model_id, question, answer, all_images)
                return {"answer": answer, "images": all_images}
        except json.JSONDecodeError:
            pass

        # 6d) otherwise repeat protocol
        history.append({
            "role": "user",
            "content": "Please respond with either a python_exec function call or a final answer in JSON."
        })

    log.error("Agent loop exited without an answer")
    return {"answer": "(no answer)", "images": all_images}
