REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def simulate(**params):
    alpha = params.get("alpha", 0.15)  # Default alpha value

    def dxdt(x, t):
        return [alpha - x**2]

    # Set up the time span for integration
    t_span = (0, 10)

    # Initial condition
    x0 = [0]

    # Solve the differential equation
    sol = integrate.solve_ivp(dxdt, t_span, x0, method="RK45")

    return {"time": sol.t.tolist(), "x": sol.y[0].tolist()}


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
