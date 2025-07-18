REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def simulate(**params):
    # Default parameters
    alpha_default = 0.15
    alpha = params.get("alpha", alpha_default)
    x0 = [0]  # Initial condition for the system

    # Define the differential equation
    def dxdt(x, t, alpha):
        return -alpha * x - x**2

    # Time span for integration
    t_span = (0, 10)

    # Solve the differential equation
    sol = integrate.solve_ivp(dxdt, t_span, x0, args=(alpha,), dense_output=True)

    # Evaluate solution at evenly spaced time points
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    y_eval = sol.sol(t_eval)[0]

    return {"equilibrium_points": list(y_eval)}


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
