# main.py – End-to-end driver
# 1) Parse NL query ➜ metadata
# 2) Generate & store simulation code ➜ model_id
# 3) Derive parameter ranges / grid
# 4) Pull script from DB and run 1000 experiments ➜ results CSV
# 5) Ask the reasoning agent a follow-up question over those results

from pathlib import Path

from core.parser import parse_nl_input
from core.codegen import generate_code
from mcp_sim_tool.core.param_utils import (
    extract_ranges_from_prompt,
    prompt_for_ranges,
    generate_param_grid,
)
from mcp_sim_tool.core.runner import run_batch
from db.schema import init_db
from core.agent_loop import ask

# 0. Init DB (creates mcp.db + tables if missing)
init_db()

# 1. Natural-language query
query = """
I want to experiment with a simple pendulum using Newton’s laws.
Use θ'' + (g/L)*sin(θ) = 0.
Initial conditions: θ(0)=0.1 rad, ω(0)=0.
Please vary the length L from 0.1 m to 1.0 m and tell me how the period depends on L.
"""

print("== Parsing Natural Language Input ==")
metadata = parse_nl_input(query, retries=4, temperature=0.3)
print(metadata)

# 2. Generate & store simulation code ➜ model_id
print("\n== Generating Simulation Code ==")
model_id = generate_code(metadata)
print(f"[✓] Stored model with ID: {model_id}")

# 3. Determine parameter ranges
print("\n== Determining Parameter Ranges ==")
param_ranges = extract_ranges_from_prompt(metadata)
missing = [p for p in metadata.get("parameters", {}) if p not in param_ranges]
if missing:
    param_ranges.update(prompt_for_ranges(missing))
print("Parameter Ranges:", param_ranges)

# 4. Build parameter grid (≈1000 combos)
print("\n== Generating Parameter Grid ==")
grid = generate_param_grid(param_ranges, total=1000)
print(f"Generated {len(grid)} parameter sets.")

# 5. Run experiments (script auto-pulled from DB inside run_batch)
print("\n== Running Experiments ==")
out_csv = Path(f"experiments/{model_id}_results.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
run_batch(
    model_id=model_id,
    param_grid=grid,
    output_csv=str(out_csv)
)
print(f"[✓] All done – results saved to {out_csv}")

# 6. Ask the reasoning agent your follow-up question
followup = (
    "For which L does the pendulum’s period stop increasing? "
    "And if the data are insufficient, suggest new L values to test."
)
print("\n== Reasoning Agent’s Answer ==")
print(ask(followup, ))
