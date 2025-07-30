import contextlib
import io
import traceback
import textwrap
import uuid
import os
from typing import Dict

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")          # headless matplotlib
import pandas as pd, json

from pandas import DataFrame


# tools.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class PythonExecArgs(BaseModel):
    code: str

class PythonExecTool(BaseTool):
    name: str = Field("python_exec")
    description: str = (
        "Execute Python code against the DataFrame `df`. "
        "Return a dict with keys: ok, stdout, stderr, images."
    )
    _df: DataFrame = PrivateAttr()

    def __init__(self, df: DataFrame):
        super().__init__()  # ensure BaseModel init runs
        self._df = df  # store

    def _run(self, args: PythonExecArgs) -> dict:
        # your existing python_exec wrapper
        return self.run_python(code=args.code, df=self._df)

    def run_python(self, code: str, df: pd.DataFrame) -> Dict:
        """
        Execute `code` on `df`.  Any plt.show() calls auto‐save to unique PNGs
        and we return *only* the images created by this call.
        """

        # 1) snapshot existing plots on disk
        before = {f for f in os.listdir() if f.lower().endswith(".png")}

        # 2) override plt.show() to save & print
        old_show = plt.show

        def _save_and_print(*args, **kwargs):
            fname = f"plots/plot_{uuid.uuid4().hex}.png"
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            print(fname)  # optional, for stdout scanning
            return fname

        plt.show = _save_and_print

        # 3) execute the code
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        ok = True
        # prepare a namespace with df, plt, pd, np
        local_ns = {"df": df, "plt": plt, "pd": pd, "np": __import__("numpy")}
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(textwrap.dedent(code), local_ns)
        except Exception:
            stderr_buf.write(traceback.format_exc())
            ok = False

        # 4) restore plt.show
        plt.show = old_show

        # 5) snapshot again & compute only the new files
        after = {f for f in os.listdir() if f.lower().endswith(".png")}
        new_images = sorted(after - before)

        # 6) return only those new images
        return {
            "ok": ok,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "images": new_images
        }