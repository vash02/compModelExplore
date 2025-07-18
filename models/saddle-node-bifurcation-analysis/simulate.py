REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Read parameters from params
    mu = params.get("mu", 0)

    # Define the system of equations
    def system(x, t):
        dxdt = -x + mu - 2 * x
        return dxdt

    # Time vector
    t = np.linspace(0, 10, 1000)

    # Initial condition
    x0 = 0

    # Solve ODE
    sol = odeint(system, x0, t, atol=1e-9, rtol=1e-9)

    # Eigenvalue analysis
    lambda_ = -1 - 2  # eigenvalue
    stability = "stable" if lambda_ < 0 else "unstable"

    # Return results
    result = {"mu": mu, "stability": stability, "solution": sol.tolist()}

    # Ensure the return is a dict
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
