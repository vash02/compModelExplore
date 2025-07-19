# quick_sweep.py  (runs 1 000 experiments even if some blow up)

import json, itertools, pathlib, uuid
from runner import run_simulation   # your new robust helper
# from utils.logger import setup_logging

# logger = setup_logging("quick_sweep")

SIMULATE_PATH = "../../experiments/simple-pendulum/simulate.py"   # adjust as needed
LIB_PATH       = "../../experiments/simple-pendulum"              # where REQUIREMENTS were installed

# --- build a param grid -----------------------------------------------------
L_values = [0.5 + 0.05*i for i in range(20)]       # 0.5 … 1.45
g_values = [9.8, 3.7, 1.62]                        # Earth, Mars, Moon
grid      = list(itertools.product(L_values, g_values))
assert len(grid) >= 1_000   # or expand your grid

results, failures = [], []

for L, g in grid[:1000]:
    params = {"L": L, "g": g}
    res = run_simulation(SIMULATE_PATH, LIB_PATH, params)
    if "error" in res:
        failures.append(res)
    else:
        results.append(res)

# --- persist ---------------------------------------------------------------
pathlib.Path("sweeps").mkdir(exist_ok=True)
with open("sweeps/run_{}.json".format(uuid.uuid4().hex[:8]), "w") as f:
    json.dump({"results": results, "failures": failures}, f, indent=2)

logger.info(f"✅ {len(results)} runs succeeded, ❌ {len(failures)} failed")
