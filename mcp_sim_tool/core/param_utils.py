import re
from itertools import product
from typing import Dict, Tuple, List
import numpy as np


import re
from typing import Dict, Tuple, Any

def extract_ranges_from_prompt(metadata: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    Look for “from X to Y” patterns in each parameter’s description.
    Returns a dict mapping param-name → (min, max).
    """
    param_ranges: Dict[str, Tuple[float, float]] = {}
    patterns = [
        r"from\s*([\d\.]+)\s*to\s*([\d\.]+)",
        r"([\d\.]+)\s*-\s*([\d\.]+)",
        # add any other patterns you like…
    ]

    for name, desc in metadata.get("parameters", {}).items():
        desc_str = desc if isinstance(desc, str) else str(desc)
        for pat in patterns:
            m = re.search(pat, desc_str)
            if m:
                low, high = float(m.group(1)), float(m.group(2))
                param_ranges[name] = (low, high)
                break

    return param_ranges

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

def generate_param_grid(ranges: Dict[str, Tuple[float, float]], total: int = 1000) -> List[Dict]:
    """
    Generate a parameter grid of ~`total` points based on given param ranges.

    Args:
        ranges: dict of param_name -> (min, max)
        total: total number of combinations to generate

    Returns:
        List of dicts: [{param1: val1, param2: val2, ...}, ...]
    """
    param_names = list(ranges.keys())
    num_params = len(param_names)

    # Decide how many steps per parameter
    steps_per_param = int(total ** (1 / num_params)) + 1

    grid_axes = {
        k: np.linspace(v[0], v[1], steps_per_param)
        for k, v in ranges.items()
    }

    all_combinations = list(product(*grid_axes.values()))
    param_grid = [
        dict(zip(param_names, combo))
        for combo in all_combinations[:total]
    ]
    return param_grid
