import re
from fractions import Fraction
from itertools import product
from typing import Dict, Tuple, List, Any, Union, Optional
import numpy as np

# NUMBER_PAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

import re
from typing import Any, Dict, Tuple

NUMBER_PAT    = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RANGE_PATS    = [
    rf"from\s*({NUMBER_PAT})\s*to\s*({NUMBER_PAT})",
    rf"({NUMBER_PAT})\s*[-–]\s*({NUMBER_PAT})"
]

def _parse_number_or_fraction(text: str) -> Optional[float]:
    text = text.strip()
    # 1) Try plain float
    try:
        return float(text)
    except ValueError:
        pass

    # 2) Try a slash‑fraction, even if the parts are floats
    if "/" in text:
        num, denom = text.split("/", 1)
        try:
            return float(num) / float(denom)
        except Exception:
            return None

    # 3) Fallback to integer-only Fraction
    try:
        return float(Fraction(text))
    except Exception:
        return None

def extract_param_settings(
    metadata: Dict[str, Any]
) -> Dict[str, Union[float, Tuple[float, float], List[str]]]:
    """
    Extracts:
      - Fixed parameters as single floats.
      - Sweep variables with (min, max) tuples or () if unknown.
      - A "vary_variable" list of names to sweep.

    Returns a dict:
      {
        "sigma": 10.0,
        "rho": (20.0, 40.0),
        "beta": 2.666…,
        ...,
        "vary_variable": ["rho", "L", ...]
      }
    """
    params = metadata.get("parameters", {})

    # Normalize vary_variable into a dict {name: raw}
    raw_vary_data = metadata.get("vary_variable")
    if isinstance(raw_vary_data, dict):
        raw_vary = raw_vary_data
    elif isinstance(raw_vary_data, (list, tuple)):
        raw_vary = {str(k): () for k in raw_vary_data}
    elif isinstance(raw_vary_data, str):
        raw_vary = {raw_vary_data: ()}
    else:
        raw_vary = {}

    settings: Dict[str, Union[float, Tuple[float, float], List[str]]] = {}
    vary_vars = list(raw_vary.keys())

    # 1) Process sweep variables
    for name in vary_vars:
        raw = raw_vary.get(name, ())
        rng: Tuple[float, float] = ()

        # If explicit tuple provided
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                rng = (float(raw[0]), float(raw[1]))
            except Exception:
                rng = ()

        # If string, try to parse as range or single number/fraction
        elif isinstance(raw, str):
            text = raw.strip()
            for pat in RANGE_PATS:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    rng = (float(m.group(1)), float(m.group(2)))
                    break
            if rng == ():
                num = _parse_number_or_fraction(text)
                if num is not None:
                    rng = (num, num)

        settings[name] = rng

    # 2) Process fixed parameters
    for name, desc in params.items():
        if name in settings:
            continue
        val: Union[float, None] = None
        if isinstance(desc, (int, float)):
            val = float(desc)
        elif isinstance(desc, str):
            val = _parse_number_or_fraction(desc)
        elif isinstance(desc, dict) and "start" in desc and "end" not in desc:
            val = _parse_number_or_fraction(str(desc.get("start")))

        if val is not None:
            settings[name] = val
    print("vary_variable", vary_vars)
    # 3) Include the list of sweep variables
    # settings["vary_variable"] = vary_vars

    return settings


def prompt_for_ranges(param_names: list) -> dict:
    """
    Ask user interactively to enter parameter ranges.
    """
    print("\nNo ranges found for some parameters.")
    ranges = {}
    for param in param_names:
        min_val = float(input(f"Enter minimum value for '{param}': "))
        max_val = float(input(f"Enter maximum value for '{param}': "))
        ranges[param] = (min_val, max_val)
    return ranges

def generate_param_grid(
    ranges: Dict[str, Tuple[float, float]],
    total: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate a parameter grid of up to `total` points, but only bin
    those parameters whose end > start. Constant parameters remain fixed.

    Args:
        ranges: dict of param_name -> (min, max)
        total:  approximate maximum number of combinations

    Returns:
        List of dicts: [{param1: val1, param2: val2, ...}, ...]
    """
    # Identify which parameters actually vary
    vary_keys = [k for k, (lo, hi) in ranges.items() if hi > lo]
    n_vary = len(vary_keys)

    # Determine number of steps per varying parameter
    if n_vary > 0:
        steps_per_param = int(total ** (1 / n_vary)) + 1
    else:
        steps_per_param = 1

    # Build axes: linspace for varying, single-value list for constant
    grid_axes: Dict[str, List[float]] = {}
    for name, (lo, hi) in ranges.items():
        if hi > lo:
            grid_axes[name] = list(np.linspace(lo, hi, steps_per_param))
        else:
            grid_axes[name] = [lo]

    # Preserve original parameter order
    param_names = list(ranges.keys())
    axes_list = [grid_axes[name] for name in param_names]

    # Generate the Cartesian product, then truncate to `total`
    all_combos = list(product(*axes_list))
    grid = [dict(zip(param_names, combo)) for combo in all_combos[:total]]
    return grid
