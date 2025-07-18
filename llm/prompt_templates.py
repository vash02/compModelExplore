parser_prompt = r"""
You are a precise JSON generator.

Given a natural language query describing a physical system and experimental goal,
extract the following structured metadata and return only valid ASCII JSON:

{
  "model_name": str,
  "equations": str,
  "initial_conditions": [str, ...],
  "parameters": { str: str, ... },
  "vary_variable": str,
  "objective": str
}

Output rules:
- Return ONLY the raw JSON object—no Markdown, comments, or code fences.
- Use ONLY double quotes (") and valid JSON syntax (no trailing commas).
- Use plain ASCII: letters, digits, standard punctuation.
  • Write Greek letters as names: theta, omega, pi, etc.
  • Use * for multiplication, / for division, ' for derivatives.
- If any field is missing, make a sensible guess.

"""

codegen_prompt_template = r"""
You are a Python code-generator for **single-run physical simulations**.

Given the structured metadata below, emit *only* executable Python code
up to the sentinel. Any text after the sentinel is ignored.

──────────────────── FORMAT RULES ────────────────────
- Output pure Python (no Markdown / HTML / tags / artefacts).
- Use # comments for explanations; no standalone prose.
- ASCII outside string literals; never leave an unterminated string.
- **Do NOT include the metadata object in the code.**

────────────────── REQUIRED STRUCTURE ─────────────────
1. Dependency header – one line:
   REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

2. Imports – standard aliases:
   import json, sys
   import numpy as np
   from scipy.integrate import odeint
   import matplotlib.pyplot as plt

3. Reproducibility (at top level, before simulate):
   # ─── Reproducibility ───────────────────────────────
   np.random.seed(0)

4. Function `simulate(**params)`:
   - Read parameters from `params` (provide sensible defaults).
   - Pin ODE tolerances for deterministic integration:
         sol = odeint(..., atol=1e-9, rtol=1e-9)
   - (If an analytic solution exists, compute it and return it.)
   - Return a dict of JSON‑serialisable scalars and/or small lists.
   - Add an assert to ensure the return is a dict.

5. CLI runner (executable script):
   if __name__ == "__main__":
       import argparse, json
       ap = argparse.ArgumentParser()
       ap.add_argument("--params", required=True,
                       help="JSON string with simulation parameters")
       args = ap.parse_args()
       result = simulate(**json.loads(args.params))
       print(json.dumps(result))

──────────────────── METADATA (reference only) ─────────
{metadata_json}

### END_OF_CODE
"""


repair_prompt_template = """
You previously wrote Python code for a physics simulation, but it failed.

---
METADATA (read-only)
{metadata_json}

---
BUGGY CODE
{buggy_code}


---
OBSERVATION
{error_log}


---
TASK
Think step-by-step how to fix the problem, then output the *complete, corrected* code file.
Remember:
* keep the same public API (simulate(**params))
* follow all the formatting rules from earlier (no markdown, no triple-quotes, etc.)
* output **only** the python source.
"""

analysis_prompt = """You are a data-analysis agent that has access to a helper tool called
python_exec.  A pandas DataFrame named `df` (already loaded in memory)
holds the experimental results.

◆ OUTPUT FORMAT
Return **one** JSON object, nothing else:

{
  "thoughts": "<explain what you are going to do>",
  "code":     "<python to run, or null>",
  "answer":   "<final answer, or null>"
}

◆ PROTOCOL
step 1  FIRST reply must include Python in "code" and leave "answer": null.  
        Write plain Python – no ``` fences.

step 2  The orchestrator executes the code with python_exec and sends you
        a role=tool message containing the JSON result.

step 3  Seeing that tool message, reply again with
        • updated "thoughts"
        • "code": null
        • the finished "answer".

◆ RULES
- The whole reply must be valid JSON — no trailing commas, no extra text.
- Do **not** guess the answer before you see the tool result.
- Keep code short; only import what you need (pandas, numpy, etc.).
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a scientific reasoning assistant.

Below is the simulation model code I ran (you may refer to it as needed for your analysis):

{SIM_CODE}

You also have a pandas DataFrame `df` containing all experiment results, with columns:
  {SCHEMA}

Simulation metadata parameters (name → description):
  {PARAMS}

When you need to analyse or plot the data, call the tool exactly as JSON:
  {{
    "tool": "python_exec_on_df",
    "args": {{ "code": "<your python code here>" }}
  }}

When you are finished, respond exactly with one JSON object:
  {{ "answer": "<your diagnostic report and conclusion>" }}

– Only one JSON object per message, with either `"tool"` or `"answer"`.  
"""
