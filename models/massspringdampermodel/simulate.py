REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Read parameters from params
    m = params.get("m", 1.0)  # mass of the system (kg)
    c = params.get("c", 1.0)  # damping coefficient (N*s/m)
    k = params.get("k", 1.0)  # spring constant (N/m)
    theta = params.get("theta", np.pi)  # angle in radians

    # Define the system of equations
    def system(y, t):
        x, v = y
        dxdt = v
        dvdt = (np.square(theta) - c * v - k * x) / m
        return [dxdt, dvdt]

    # Initial conditions
    y0 = [1.0, 0.0]  # x(0)=1, x'(0)=0

    # Time array for solution
    t = np.linspace(0, 10, 1000)

    # Solve the system of equations
    sol = odeint(system, y0, t, atol=1e-9, rtol=1e-9)

    # Compute damping ratio
    damping_ratio = c / (2 * np.sqrt(m * k))

    # Return results
    result = {
        "time": t.tolist(),
        "solution": sol.tolist(),
        "damping_ratio": damping_ratio,
    }

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
