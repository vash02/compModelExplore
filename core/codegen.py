# core/codegen.py
# ---------------------------------------------------------------------
# Generate a Python simulate.py from metadata (≤4 retries).
# Now installs any third-party REQUIREMENTS before running smoke tests.
# Uses OpenAI Chat API instead of LocalLLM.
# ---------------------------------------------------------------------

from __future__ import annotations
import os
import ast
import json
import re
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Dict, List

import openai
from slugify import slugify

from db.store import store_simulation_script
# from llm.local_llm import LocalLLM   # ← commented out for now
from core.utils import (
    strip_trailing_extras,
    sanitize_simulation_code,
    validate_simulation_code,
)
from core.smoke_tests import _runtime_smoke_test, _dedent_if_needed
from llm.prompt_templates import codegen_prompt_template

BLACK_CMD = ["black", "-q", "-"]

# ────────────────── helpers: Black / error context ────────────────────
def _beautify(code: str) -> str:
    try:
        return subprocess.run(
            BLACK_CMD, input=code, text=True, capture_output=True, check=True
        ).stdout
    except Exception:
        return code


def _syntax_context(code: str, err: SyntaxError, around: int = 2) -> str:
    lines = code.splitlines()
    lineno = err.lineno or 0
    first = max(1, lineno - around)
    last = min(len(lines), lineno + around)
    return "\n".join(
        f"{'→' if i == lineno else ' '} {i:>4}: {lines[i-1]}"
        for i in range(first, last + 1)
    )


def _runtime_context(trace: str, limit: int = 25) -> str:
    return "\n".join(trace.splitlines()[-limit:]) or "<no traceback captured>"


# ───────────────────── helpers: requirements handling ──────────────────
def _extract_requirements(script: str) -> List[str]:
    """
    • Parse `REQUIREMENTS = ["pkg1", "pkg2"]`
    • Fallback: any `import X` / `from X import …`
    """
    pkgs: List[str] = []

    m = re.search(r"REQUIREMENTS\s*=\s*\[(.*?)\]", script, re.S)
    if m:
        pkgs.extend(re.findall(r"[\"']([^\"']+)[\"']", m.group(1)))

    for line in script.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        m1 = re.match(r"import\s+([\w_]+)", line)
        m2 = re.match(r"from\s+([\w_]+)", line)
        name = m1.group(1) if m1 else m2.group(1) if m2 else None
        if name and name not in {"__future__", "typing"}:
            pkgs.append(name)

    return sorted(set(pkgs))


def _install_requirements(pkgs: List[str]) -> None:
    """Install any package that is *not* already importable."""
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            print(f"📦 Installing '{pkg}' …")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--no-cache-dir",
                        pkg,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                print(f"⚠️  pip install failed for '{pkg}': {e}")


# ─────────────────────────── generate_code ─────────────────────────────

MODELS_ROOT = Path("models")  # <── permanent home

# Replace LocalLLM with OpenAI Chat API
def generate_code(
    metadata: Dict,
    max_attempts: int = 4,
    temperature: float = 0.15,
) -> str:
    """
    • Prompt OpenAI Chat API → sanitize → static‐check (AST) → runtime smoke‐test
    • On success: prettify with Black, save to models/<slug>/simulate.py,
      store path in DB, and return model_id.
    • Retries up to `max_attempts` times.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    for attempt in range(1, max_attempts + 1):
        prompt_text = codegen_prompt_template.format(
            metadata_json=json.dumps(metadata, indent=2)
        )

        # Call OpenAI ChatCompletion
        response = openai.chat.completions.create(
            model="gpt-4-0613",
            messages=[{"role": "system", "content": prompt_text}],
            temperature=temperature,
        )
        raw = response.choices[0].message.content or ""
        cleaned = sanitize_simulation_code(raw)

        # ── 1. static validation ───────────────────────────────────────
        try:
            validate_simulation_code(cleaned)
        except SyntaxError as syn:
            ctx = _syntax_context(cleaned, syn)
            print(f"[✗] attempt {attempt}/{max_attempts} – SyntaxError: {syn.msg}\n → {ctx}")
            continue
        except ValueError as e:
            print(f"[✗] attempt {attempt}/{max_attempts} – {e}")
            continue

        # ── 2. runtime smoke‐test ───────────────────────────────────────
        ok, log = _runtime_smoke_test(cleaned, timeout=30)
        if not ok:
            dedented = _dedent_if_needed(cleaned)
            if dedented != cleaned:
                ok, log = _runtime_smoke_test(dedented, timeout=30)
                if ok:
                    cleaned = dedented
        if not ok:
            last_line = log.strip().splitlines()[-1] if log else ""
            print(f"[✗] attempt {attempt}/{max_attempts} – runtime error:\n    {last_line}")
            continue

        # ── 3. PASS → save & persist ────────────────────────────────────
        model_slug = slugify(metadata.get("model_name", "unnamed_model"))
        model_dir = MODELS_ROOT / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (model_dir / "simulate.py").write_text(_beautify(cleaned))

        model_id = store_simulation_script(
            model_name=model_slug,
            metadata=metadata,
            script_path=str(model_dir / "simulate.py"),
        )
        print(f"[✓] stored model_id = {model_id}  dir = {model_dir}")
        return model_id

    raise RuntimeError("Exceeded attempts without valid code")
