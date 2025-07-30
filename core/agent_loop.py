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
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional
import openai
from utils.config import settings

openai.api_key = settings.openai_api_key

from core.utils import extract_code_map
from llm.local_llm import LocalLLM
from db.results_api import load_results
from db.store import get_model_metadata, get_simulation_script_code
from core.tools import PythonExecTool

# ───────────────────────── Logging setup ─────────────────────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
log = logging.getLogger("agent_loop")

# ───────────────────────── Paths & constants ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "mcp.db"
MAX_STEPS    = 40
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
    backend: str = "local",
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Run a reasoning agent for `model_id` with follow-up `question`.

    backend:
      - "local": uses LocalLLM(deepseek-r1:14b)
      - "openai": uses OpenAI GPT-4 via openai.chat.completions.create

    Returns:
      {
        "history": [...],
        "code_map": {step_index: code, …},
        "answer": "...final explanation...",
        "images": ["plot_<uuid>.png", ...]
      }
    """
    log.info("=== Starting analysis for model_id=%s (backend=%s) ===", model_id, backend)

    # 1) load the data, metadata, and simulation code
    df       = load_results(db_path=str(DB_PATH), model_id=model_id)
    meta     = get_model_metadata(model_id, db_path=str(DB_PATH))
    sim_code = get_simulation_script_code(model_id, db_path=str(DB_PATH))
    log.info("Loaded %d rows for model_id=%s", len(df), model_id)

    # 2) build the system prompt
    schema = list(df.columns)
    print("============schema:", schema)
    params = [{"name": k, "description": v} for k, v in meta.get("parameters", {}).items()]
    system_prompt = f"""
    You are a scientific reasoning assistant. You have access to the simulation model implementation below,
    but you do *not* see the raw DataFrame directly. When you need to operate on the DataFrame `df`,
    you must issue exactly:

      {{\"function_call\": {{\"name\": \"python_exec\", \"arguments\": {{\"code\": \"<python code>\"}}}}}}

    You will then receive a JSON with keys `ok`, `stdout`, `stderr`, and `images`.

    ──────────── SIMULATION CODE ────────────
    ```python
    {sim_code}
    ```
    ──────────── DATA SCHEMA ────────────
    Columns in df: {schema}

    ───────── PARAMETERS (name → description) ─────────
    {params}

    Issue `python_exec` calls to filter/aggregate/plot, then return a JSON answer.
    — Only one valid JSON object per assistant message.
    
    - *If no python_exec calls are needed to answer the question, you must still output exactly this JSON and nothing else:*
    ```json
    {{
      "answer":  "<your concise interpretation>",
      "values":  [ … ],        # include only if numeric values were requested
      "images":  [ …, … ]   # include only if plots were generated
    }}
    ```
    
    - If you do not need any code to answer, still wrap your final interpretation in the exact JSON format shown below—do not emit plain text alone.
    """

    # prepare conversation
    history = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]
    functions = [{
        "name": "python_exec",
        "description": "Run Python code on `df`, return ok/stdout/stderr/images.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"}
            },
            "required": ["code"]
        }
    }]

    all_images: List[str] = []

    # agent loop
    for step in range(1, MAX_STEPS + 1):
        log.info("=== STEP %d/%d ===", step, MAX_STEPS)

        # early stop
        if stop_flag and stop_flag():
            log.info("Stop requested at step %d", step)
            return {"history": history, "code_map": {}, "answer": "(stopped)", "images": all_images}

        # call LLM
        if backend == "openai":
            system_msg = history[0]
            recent_msgs = history[-2:]
            messages_for_model = [system_msg] + recent_msgs
            resp = openai.chat.completions.create(
                model="gpt-4.1",
                messages=messages_for_model,
                functions=functions,
                function_call="auto",
                # temperature=0.0,
            )
            msg = resp.choices[0].message
            assistant_msg = {"role": msg.role, "content": msg.content or ""}
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
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history)
                reply = llm.generate(prompt)
            assistant_msg = {"role": "assistant", "content": reply}

        history.append(assistant_msg)

        # handle python_exec
        if assistant_msg.get("function_call"):
            args = json.loads(assistant_msg["function_call"]["arguments"] or "{}")
            code = args.get("code", "")
            tool = PythonExecTool(code)
            tool_res = tool.run_python(code, df)
            new_imgs = tool_res.get("images", [])
            all_images.extend(new_imgs)
            log.info("Captured images this step: %s", new_imgs)
            history.append({
                "role": "function",
                "name": "python_exec",
                "content": json.dumps(tool_res, ensure_ascii=False)
            })
            continue

        # check for final answer
        if not assistant_msg.get("function_call"):
            try:
                payload = json.loads(assistant_msg["content"])
                if "answer" in payload:
                    answer = payload["answer"]
                    log.info("Agent provided final answer.")
                    log.info("All images collected: %s", all_images)
                    _store_report(model_id, question, answer, all_images)
                    return {
                        "history":  history,
                        "code_map": extract_code_map(history),
                        "answer":   answer,
                        "images":   all_images
                    }
            except json.JSONDecodeError:
                answer = assistant_msg["content"].strip()
                log.info("Treating plain text as final answer.")
                _store_report(model_id, question, answer, all_images)
                return {
                    "history": history,
                    "code_map": extract_code_map(history),
                    "answer": answer,
                    "images": all_images
                }

        # enforce protocol
        history.append({
            "role": "user",
            "content": "Please respond with either a python_exec call or a JSON answer."
        })

    # no answer case
    log.error("Agent loop exhausted without answer")
    log.info("All images collected: %s", all_images)
    return {
        "history":  history,
        "code_map": extract_code_map(history),
        "answer":   "(no answer)",
        "images":   all_images
    }



# def ask(
#     model_id: str,
#     question: str,
#     backend: str = "local",
#     stop_flag: Optional[Callable[[], bool]] = None,
#     chat_history: Optional[List[Dict[str, Any]]] = None,   # ← new
# ) -> Dict[str, Any]:
#     df = load_results(db_path=DB_PATH, model_id=model_id)
#     meta = get_model_metadata(model_id, db_path=DB_PATH)
#     sim_code = get_simulation_script_code(model_id, db_path=DB_PATH)
#
#     schema = df.columns.tolist()
#     params = [{"name": k, "description": v}
#               for k, v in meta.get("parameters", {}).items()]
#
#     agent = make_agent(df, sim_code, params, schema, backend=backend)
#     # Attach the stop‐flag callback if provided
#     callbacks = [StopFlagCallback(stop_flag)] if stop_flag else None
#
#     # Build the inputs dict that matches your chain’s input schema
#     inputs = {
#             "input": question,
#             "chat_history": chat_history or []}
#     try:
#         raw = agent.invoke(inputs, callbacks=callbacks)
#         if isinstance(raw, str):
#             return {"answer": raw, "history": chat_history, "images": all_images}
#             # if dict, but has no "answer" key:
#         if "answer" not in raw and "output" in raw:
#             raw["answer"] = raw["output"]
#         return raw
#     except StopRequested:
#         return {"answer": "(stopped)"}
#     # LangChain will have internally performed all tool‐calls for you.
#     return result