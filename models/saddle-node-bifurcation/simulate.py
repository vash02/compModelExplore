REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Read parameters from `params` (provide sensible defaults).
    mu = params.get("mu", 0.0)
    x0 = params.get("x0", 0.0)

    # Define the system of equations
    def system(x, t):
        dxdt = x[1]
        dvdt = mu - x[0] ** 2
        return [dxdt, dvdt]

    # Time vector
    t = np.linspace(0, 10, 1000)

    # Solve the ODE system
    sol = odeint(system, [x0, 0], t, atol=1e-9, rtol=1e-9)

    # Find fixed points
    fixed_points = []
    for i in range(1, len(sol[:, 0]) - 1):
        if (
            sol[i - 1, 0] < sol[i, 0] > sol[i + 1, 0]
            or sol[i - 1, 0] > sol[i, 0] < sol[i + 1, 0]
        ):
            fixed_points.append((t[i], sol[i, 0]))

    # Return a dict of JSON‑serialisable scalars and/or small lists.
    result = {"fixed_points": fixed_points, "trajectory": sol[:, 0].tolist()}

    # Ensure the return is a dict.
    assert isinstance(result, dict)
    return result


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
