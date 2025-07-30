import json
import sqlite3

import numpy as np
import matplotlib.pyplot as plt
from db.results_api import load_results
DB_PATH = '../../mcp.db'
def run():
    df = load_results(db_path='../../mcp.db',model_id='simple-pendulum')
    # Use only the first model/row for clarity (multiple models can clutter the plot)
    row = df.iloc[5]
    t = np.array(row['t'])
    theta = np.array(row['theta'])

    # Remove NaN values from theta and the corresponding t
    mask = ~np.isnan(theta)
    t = t[mask]
    theta = theta[mask]
    print("================",t[:10])
    print("================",theta[:10])
    # print(zip(t, theta))
    plt.figure(figsize=(8,4))
    plt.plot(t, theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')

    # Autoscale y-axis based on range, using a 10% margin
    if len(theta) > 0:
        min_theta = np.min(theta)
        max_theta = np.max(theta)
        margin = 0.1 * (max_theta - min_theta)
        plt.ylim([min_theta - margin, max_theta + margin])

    # No legend as requested
    plt.title('Pendulum Angle vs Time')
    plt.tight_layout()
    plt.show()

def metrics():
    import pandas as pd
    import numpy as np

    # Load every result row in the DB
    sql = """
          SELECT model_id, outputs
          FROM results
          -- WHERE LOWER(model_id) LIKE '%lorenz%' \
          """

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql).fetchall()

    # 2) convert to a tally of ok / total runs per model_id
    records = []
    for r in rows:
        outputs = json.loads(r["outputs"] or "{}")
        error_present = "error" in outputs
        records.append({"model_id": r["model_id"], "error": error_present})

    df_all = pd.DataFrame(records)

    # Grab only the model_ids that mention “lorenz” (case‑insensitive)
    summary = (
        df_all.groupby("model_id")["error"]
        .agg(total_runs="count", error_runs="sum")
        .reset_index()
    )

    summary["ok_runs"] = summary["total_runs"] - summary["error_runs"]
    summary["success_rate"] = summary["ok_runs"] / summary["total_runs"]

    print(summary.to_markdown(index=False))


if __name__ == '__main__':
    # run()
    metrics()