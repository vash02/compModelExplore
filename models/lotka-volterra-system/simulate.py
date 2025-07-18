REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Set default parameters if not provided
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 0.1)
    gamma = params.get("gamma", 1.5)
    delta = params.get("delta", 0.075)
    theta = params.get("theta", 1.0)  # Assuming theta = 1 if not provided
    x0 = params.get("x0", 40)  # Initial condition for x

    # Define the system of equations
    def system(state, t):
        x, y = state
        dxdt = -beta * x / theta * y + alpha - delta * y
        dydt = (delta * x * y) / theta - gamma * y
        return dxdt, dydt

    # Time points
    t = np.linspace(0, 100, 1000)

    # Solve ODE
    sol = odeint(system, [x0, 0], t, atol=1e-9, rtol=1e-9)

    # Return results
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
