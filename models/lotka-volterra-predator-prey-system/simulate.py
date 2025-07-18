REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Read parameters from params
    alpha = float(params.get("alpha", 1.0))
    beta = float(params.get("beta", 0.1))
    delta = float(params.get("delta", 0.075))
    gamma = float(params.get("gamma", 1.5))
    x0 = float(params.get("x0", 40))
    y0 = float(params.get("y0", 9))
    t = np.linspace(0, 50, 1000)  # Time array

    # Define the system of ODEs
    def system(state, t):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return dxdt, dydt

    # Solve the system of ODEs
    sol = odeint(system, (x0, y0), t, atol=1e-9, rtol=1e-9)

    # Return the solution as a dict
    result = {"t": t.tolist(), "x": sol[:, 0].tolist(), "y": sol[:, 1].tolist()}

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
