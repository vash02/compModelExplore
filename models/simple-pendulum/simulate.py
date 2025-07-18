REQUIREMENTS = ["numpy", "scipy", "matplotlib"]

import json, sys
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def simulate(**params):
    L = params.get("L", 1.0)  # Default length of pendulum
    g = params.get("g", 9.81)  # Standard gravity acceleration

    def pend(y, t, b, c):
        theta, omega = y
        dydt = [omega, -(g / L) * np.sin(theta)]
        return dydt

    theta0 = np.pi / 2  # Initial angle of the pendulum
    omega0 = 0  # Initial angular velocity
    y0 = [theta0, omega0]

    t_max = 10  # Maximum simulation time
    b = 0.25  # Damping coefficient
    c = 5.0  # Spring constant

    sol = integrate.odeint(pend, y0, np.linspace(0, t_max, 100), args=(b, c))
    theta, omega = sol[:, 0], sol[:, 1]

    period = 2 * np.pi * np.sqrt(L / g)  # Theoretical period for small oscillations

    return {"period": float(period)}


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="JSON string with simulation parameters"
    )
    args = ap.parse_args()
    result = simulate(**json.loads(args.params))
    print(json.dumps(result))
