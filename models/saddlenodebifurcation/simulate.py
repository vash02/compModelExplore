REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def simulate(**params):
    alpha = params.get("alpha", 0)
    step_size = float(params.get("step_size", 0.1))

    def system(x, t, alpha):
        return [alpha - x**2]

    alphas = np.arange(-2, 2, step_size)
    equilibria = []

    for a in alphas:
        sol = integrate.solve_ivp(system, (0, 10), [0], args=(a,), method="RK45")
        if len(sol.y[0]) > 0 and sol.y[0][-1] == 0:
            equilibria.append((a, 0))

    return {"equilibria": equilibria}


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
