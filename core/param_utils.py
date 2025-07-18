# before:
# def generate_param_grid(param_ranges: Dict[str, Tuple[float, float]], total: int = 1000) -> List[Dict]:
#     …
#     steps_per_param = int(total ** (1 / num_params)) + 1

# after:
import numpy as np
from typing import Dict, List, Tuple

def generate_param_grid(
    param_ranges: Dict[str, Tuple[float, float]],
    *,
    total: int | None = None,
    samples: int | None = None
) -> List[Dict[str, float]]:
    """
    Build a full-factorial grid over param_ranges.
    Either:
      • Specify `samples` = steps per parameter, or
      • Specify `total` = approximate total points (we’ll infer samples)
    """
    keys = list(param_ranges.keys())
    num_params = len(keys)

    if samples is not None:
        steps_per_param = samples
    elif total is not None:
        # old behavior: infer roughly equal sampling along each axis
        steps_per_param = int(total ** (1 / num_params)) + 1
    else:
        raise ValueError("Must specify either total or samples")

    # build the grid
    axes = [
        np.linspace(rng[0], rng[1], steps_per_param)
        for rng in param_ranges.values()
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    flat_axes = [m.flatten() for m in mesh]

    grid = [
        dict(zip(keys, vals))
        for vals in zip(*flat_axes)
    ]
    return grid


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
