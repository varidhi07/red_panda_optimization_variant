"""
run_engineering.py
==================
Run all 5 engineering problems with all 9 algorithms.
Usage:
    python run_engineering.py
    python run_engineering.py --runs 50 --fes 60000
    python run_engineering.py --quick
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")

from algorithms import ALGORITHMS
from engineering_problems import ENGINEERING_PROBLEMS


def run_engineering(n_runs, max_fes, pop_size, quick, out_dir):
    if quick:
        n_runs, max_fes = 5, 1000

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for (prob_name, func, lb, ub, dim) in ENGINEERING_PROBLEMS:
        print(f"\n{'─'*50}")
        print(f"Problem: {prob_name}  (dim={dim})")
        algo_results = {}

        for algo_name, algo in ALGORITHMS.items():
            vals = []
            t0   = time.time()
            for run in range(n_runs):
                np.random.seed(run * 37 + hash(algo_name) % 500)
                res = algo(func, lb, ub, dim, max_fes, pop_size)
                vals.append(res["best_val"])
            elapsed = time.time() - t0
            algo_results[algo_name] = np.array(vals)
            print(f"  {algo_name:<6}  mean={np.mean(vals):+.4e}  std={np.std(vals):.4e}  ({elapsed:.1f}s)")

        # Rank by mean
        means  = {a: np.mean(v) for a, v in algo_results.items()}
        ranked = sorted(means, key=means.get)
        ranks  = {a: ranked.index(a)+1 for a in means}
        mrpo_v = algo_results.get("MRPO", None)

        for algo_name, vals in algo_results.items():
            if mrpo_v is not None and algo_name != "MRPO" and len(vals) >= 5:
                try:   stat, p = wilcoxon(mrpo_v, vals)
                except: p = 1.0
            else:
                p = float("nan")
            rows.append({
                "Problem"    : prob_name,
                "Algorithm"  : algo_name,
                "Mean"       : np.mean(vals),
                "Std"        : np.std(vals),
                "Min"        : np.min(vals),
                "Max"        : np.max(vals),
                "Rank"       : ranks[algo_name],
                "Wilcoxon_p" : p,
                "Significant": "+" if (not np.isnan(p) and p < 0.05) else ("=" if np.isnan(p) else "-"),
            })

    df = pd.DataFrame(rows)
    xlsx = os.path.join(out_dir, "engineering_results.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Full", index=False)
        pivot = df.pivot(index="Problem", columns="Algorithm", values="Mean")
        pivot.to_excel(writer, sheet_name="Mean_Pivot")
        pivot_r = df.pivot(index="Problem", columns="Algorithm", values="Rank")
        pivot_r.to_excel(writer, sheet_name="Rank_Pivot")
    print(f"\n  Saved: {xlsx}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   type=int, default=50)
    parser.add_argument("--fes",    type=int, default=60000)
    parser.add_argument("--pop",    type=int, default=50)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()
    run_engineering(args.runs, args.fes, args.pop, args.quick, args.outdir)
    print("\nEngineering benchmark done!")


if __name__ == "__main__":
    main()
