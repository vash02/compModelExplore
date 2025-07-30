from importlib import util as iu
from pathlib     import Path
from typing      import Dict, List
import json, subprocess, sys, uuid, re

import pandas as pd
from tqdm import tqdm

from db.store import get_simulation_path, store_simulation_results  # returns models/<id>/simulate.py

# ───────────── helpers ─────────────
def extract_requirements(script: str) -> List[str]:
    req = re.findall(r"REQUIREMENTS\s*=\s*\[(.*?)\]", script, re.S)
    if req:
        return sorted(set(re.findall(r"[\"']([^\"']+)[\"']", req[0])))
    return []

def install_requirements(pkgs: List[str], lib_dir: Path):
    if not pkgs:
        return
    lib_dir.mkdir(exist_ok=True)
    for p in pkgs:
        if iu.find_spec(p):       # already importable
            continue
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "--no-cache-dir", "--target", str(lib_dir), p],
            check=True
        )

def import_simulate(script_path: Path):
    name = f"simulate_{uuid.uuid4().hex}"
    spec = iu.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    mod  = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "simulate"), "simulate() missing"
    return mod.simulate

# ───────────── single run ─────────────
def run_simulation(script_path: Path, params: Dict) -> Dict:
    print("========================== params", params)
    try:
        simulate = import_simulate(script_path)
        # print("========================== params", params)
        res = simulate(**params)
        if not isinstance(res, dict):
            raise TypeError("simulate() must return dict")
        return {**params, **res}
    except Exception as e:
        return {**params, "error": str(e)}

# ───────────── batch runner ─────────────
def run_batch(model_id: str,
              param_grid: List[Dict],
              output_csv: str = "results.csv",
              db_path: str = "mcp.db") -> None:

    script_path = Path(get_simulation_path(model_id, db_path=db_path))
    model_dir   = script_path.parent
    lib_dir     = model_dir / "lib"

    # install deps once (if not already there)
    install_requirements(
        extract_requirements(script_path.read_text()),
        lib_dir
    )
    sys.path.insert(0, str(lib_dir))         # make packages importable
    sys.path.insert(0, str(model_dir))       # so `import simulate` would work
    # print("----------------param_grid-----------------", param_grid)
    rows = [run_simulation(script_path, p)
            for p in tqdm(param_grid, desc=f"Running {model_id}")]

    df = pd.DataFrame(rows)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[✓] {len(df)} rows → {out}")

    # ★ NEW: persist to DB -----------------------------------------
    store_simulation_results(model_id=model_id,
                             rows=rows,
                             param_keys=list(param_grid[0].keys()),
                             db_path=db_path)
    print(f"[✓] Stored {len(rows)} rows in DB {db_path}")
