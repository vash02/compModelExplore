import contextlib
import io
import traceback
import textwrap
import uuid
import os
import matplotlib
matplotlib.use("Agg")          # headless matplotlib
import pandas as pd, json

from pandas import DataFrame

def python_exec(code: str, df: DataFrame) -> dict:
    """
    Execute `code` on `df`.  Any new `*.png` files that appear
    during execution are renamed to unique names and returned.
    """
    # 1) snapshot existing pngs
    before = {f for f in os.listdir() if f.lower().endswith(".png")}

    # 2) run user code
    local_ns = {"df": df}
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(textwrap.dedent(code), local_ns)
    except Exception:
        stderr_buf.write(traceback.format_exc())
        ok = False

    # 3) find new pngs
    after = {f for f in os.listdir() if f.lower().endswith(".png")}
    new = after - before
    unique_images = []
    for old in new:
        new_name = f"plot_{uuid.uuid4().hex}.png"
        try:
            os.rename(old, new_name)
            unique_images.append(new_name)
        except Exception:
            # if rename fails, still include the old
            unique_images.append(old)

    return {
        "ok": ok,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "images": unique_images
    }
