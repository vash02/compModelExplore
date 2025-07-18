# --- add near the top of codegen.py ----------------------------------
# core/codegen.py  – helpers
import subprocess, sys, tempfile, importlib.util, textwrap, traceback, json
from pathlib import Path
from typing import Tuple

##############################################################
# 1) 30-second smoke-test with automatic de-indent            #
##############################################################
def _runtime_smoke_test(py_code: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Write <py_code> to simulate.py, import it in a fresh runner process,
    call simulate(), dump JSON to stdout.  Returns (success, combined-output).
    """
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)

        # ---------- simulate.py (dedented & left-stripped) ----------
        sim_path = wd / "simulate.py"
        sim_path.write_text(textwrap.dedent(py_code).lstrip())

        # ---------- runner.py ---------------------------------------
        runner_src = textwrap.dedent(f"""
            import importlib.util, json, sys, traceback
            spec = importlib.util.spec_from_file_location("sim", r"{sim_path}")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            try:
                result = mod.simulate()
                json.dump(result, sys.stdout)
            except Exception:
                traceback.print_exc()
                sys.exit(1)
        """).lstrip()

        runner_path = wd / "runner.py"
        runner_path.write_text(runner_src)

        # ---------- execute -----------------------------------------
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stderr + proc.stdout


##############################################################
# 2) one-shot auto-repair for indentation                     #
##############################################################
def _dedent_if_needed(code: str) -> str:
    """
    If the first non-blank line is indented, dedent & lstrip everything.
    """
    first_real = next((l for l in code.splitlines() if l.strip()), "")
    if first_real.startswith((" ", "\t")):
        return textwrap.dedent(code).lstrip()
    return code

