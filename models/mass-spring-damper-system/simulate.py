REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─── Reproducibility ───────────────────────────────
np.random.seed(0)


def simulate(**params):
    # Read parameters from `params` (provide sensible defaults).
    m = params.get("m", 1.0)  # mass
    c = params.get("c", 1.0)  # damping coefficient
    k = params.get("k", 50.0)  # spring constant

    # Define the system of ODEs
    def system(y, t):
        x, v = y
        dxdt = v
        dvdt = -(c / m) * v - (k / m) * x
        return [dxdt, dvdt]

    # Initial conditions
    y0 = [1.0, 0.0]  # initial displacement and velocity

    # Time array for solution
    t = np.linspace(0, 10, 1000)

    # Solve the ODE system
    sol = odeint(system, y0, t, atol=1e-9, rtol=1e-9)

    # Calculate damping ratio
    damping_ratio = c / (2 * np.sqrt(k * m))

    # Return results
    result = {
        "time": t.tolist(),
        "displacement": sol[:, 0].tolist(),
        "velocity": sol[:, 1].tolist(),
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
