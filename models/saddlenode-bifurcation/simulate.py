REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def simulate(**params):
    alpha_values = [param for param in np.arange(0, 0.41, 0.05) if param != 0]
    equilibrium_points = []

    for alpha in alpha_values:

        def system(t, x):
            return [(alpha - x[0] ** 2)]

        sol = integrate.solve_ivp(system, [0, 10], [0], method="RK45")
        equilibrium_points.append((sol.y[0][-1], alpha))

    return {"equilibrium_points": [(x, a) for x, a in equilibrium_points]}


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
