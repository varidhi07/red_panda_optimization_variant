"""
benchmark.py
============
Runs MRPO vs RPO vs 7 competitors on CEC-2014, 2017, 2020, 2022.

Usage:
    python benchmark.py                    # full run (slow — ~hours)
    python benchmark.py --suite cec2014    # single suite
    python benchmark.py --dim 10 --runs 5 # quick test
    python benchmark.py --quick            # 5 runs, 1000 FEs (smoke test)
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import product as iproduct

warnings.filterwarnings("ignore")

from algorithms import ALGORITHMS

# ── CEC suite loader ────────────────────────────────────────────────────────────

def load_suite(suite_name, dim):
    """
    Returns list of (name, func, lb, ub) for a CEC suite + dimension.
    Matches the (fname, func, lb, ub) unpacking in run_experiment.
    """
    import opfunu
    import inspect
    
    year = suite_name.replace("cec", "")
    benchmark_set = []
    
    try:
        # Try standard opfunu API
        funcs_dict = opfunu.get_functions_by_year(year)
        for f_name, func_cls in sorted(funcs_dict.items(), key=lambda x: int(x[0][1:])):
            obj = func_cls(ndim=dim)
            # Only return 4 values to match the loop: name, func, lb, ub
            benchmark_set.append((f_name, obj.evaluate, obj.lb, obj.ub))
            
    except (AttributeError, ValueError):
        # Fallback for CEC2014 specific structure
        try:
            import opfunu.cec_based.cec2014 as cb14
            for name, obj in inspect.getmembers(cb14, inspect.isclass):
                if name.startswith("F") and name[1:].isdigit():
                    f_instance = obj(ndim=dim)
                    benchmark_set.append((name, f_instance.evaluate, f_instance.lb, f_instance.ub))
            benchmark_set.sort(key=lambda x: int(x[0][1:]))
        except Exception as e:
            print(f"Critical error loading {suite_name}: {e}")

    return benchmark_set


# ── single experiment ────────────────────────────────────────────────────────────

def run_experiment(suite_name, dim, n_runs, max_fes, pop_size, quick):
    """Run all algos on all functions in a suite. Returns raw results dict."""
    if quick:
        n_runs  = 5
        max_fes = 1000

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name.upper()}  |  Dim: {dim}D  |  Runs: {n_runs}  |  FEs: {max_fes}")
    print(f"{'='*60}")

    funcs   = load_suite(suite_name, dim)
    algos   = list(ALGORITHMS.keys())
    results = {}   # results[func_name][algo_name] = list of best_val over runs

    for fname, func, lb, ub in funcs:
        results[fname] = {a: [] for a in algos}
        for algo_name in algos:
            algo = ALGORITHMS[algo_name]
            t0   = time.time()
            for run in range(n_runs):
                np.random.seed(run * 100 + hash(algo_name + fname) % 1000)
                res = algo(func, lb, ub, dim, max_fes, pop_size)
                results[fname][algo_name].append(res["best_val"])
            elapsed = time.time() - t0
            mean = np.mean(results[fname][algo_name])
            print(f"  {fname:<12} {algo_name:<6}  mean={mean:+.4e}  ({elapsed:.1f}s)")

    return results, funcs


# ── statistics ───────────────────────────────────────────────────────────────────

def compute_stats(results):
    """
    Returns a DataFrame with columns:
    Function, Algo, Mean, Std, Rank, Wilcoxon_p (vs MRPO), Significant
    """
    rows = []
    for fname, algo_dict in results.items():
        vals = {a: np.array(v) for a, v in algo_dict.items()}
        # rank by mean
        means  = {a: np.mean(v) for a, v in vals.items()}
        ranked = sorted(means, key=means.get)
        ranks  = {a: ranked.index(a)+1 for a in means}

        mrpo_vals = vals.get("MRPO", None)

        for algo, v in vals.items():
            if mrpo_vals is not None and algo != "MRPO" and len(v) >= 5:
                try:
                    stat, p = wilcoxon(mrpo_vals, v)
                except Exception:
                    p = 1.0
            else:
                p = np.nan

            rows.append({
                "Function"    : fname,
                "Algorithm"   : algo,
                "Mean"        : np.mean(v),
                "Std"         : np.std(v),
                "Rank"        : ranks[algo],
                "Wilcoxon_p"  : p,
                "Significant" : "+" if (not np.isnan(p) and p < 0.05) else ("=" if np.isnan(p) else "-"),
            })
    return pd.DataFrame(rows)


def compute_convergence(suite_name, dim, max_fes, pop_size, quick):
    """Run one run per algo per function and collect history for convergence curves."""
    if quick:
        max_fes = 1000
    funcs  = load_suite(suite_name, dim)
    conv   = {}
    for fname, func, lb, ub in funcs:
        conv[fname] = {}
        for algo_name, algo in ALGORITHMS.items():
            np.random.seed(42)
            res = algo(func, lb, ub, dim, max_fes, pop_size)
            conv[fname][algo_name] = res["history"]
    return conv, funcs


# ── save results ─────────────────────────────────────────────────────────────────

def save_excel(df, conv, suite_name, dim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, f"{suite_name}_{dim}D_results.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # ── Sheet 1: Full stats
        df.reset_index(drop=True).to_excel(writer, sheet_name="Full_Stats", index=False)

        # ── Sheet 2: Mean/Std pivot table (professor-friendly)
        # Use pivot_table (handles duplicates gracefully via aggfunc)
        pivot_mean = df.pivot_table(index="Function", columns="Algorithm",
                                    values="Mean", aggfunc="mean")
        pivot_std  = df.pivot_table(index="Function", columns="Algorithm",
                                    values="Std",  aggfunc="mean")
        pivot_rank = df.pivot_table(index="Function", columns="Algorithm",
                                    values="Rank", aggfunc="min")
        # Significance: take first value per (Function, Algorithm) pair
        pivot_sig  = df.groupby(["Function", "Algorithm"])["Significant"] \
                       .first().unstack("Algorithm")

        # Combine mean ± std into one readable cell
        combined = pivot_mean.copy().astype(object)
        for col in pivot_mean.columns:
            combined[col] = (pivot_mean[col].apply(lambda x: f"{x:.4e}") +
                             " ± " +
                             pivot_std[col].apply(lambda x: f"{x:.4e}"))

        combined.to_excel(writer, sheet_name="Mean_Std_Table")
        pivot_rank.to_excel(writer, sheet_name="Rank_Table")
        pivot_sig.to_excel(writer, sheet_name="Wilcoxon_Significance")

        # ── Sheet 3: Average rank summary
        avg_rank = df.groupby("Algorithm")["Rank"].mean().reset_index()
        avg_rank.columns = ["Algorithm", "Avg_Rank"]
        avg_rank = avg_rank.sort_values("Avg_Rank")
        avg_rank.to_excel(writer, sheet_name="Avg_Rank_Summary", index=False)

    print(f"\n  Saved: {xlsx_path}")
    return xlsx_path


def save_convergence_plots(conv, suite_name, dim, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "convergence_plots", f"{suite_name}_{dim}D")
    os.makedirs(plot_dir, exist_ok=True)

    colors = {
        "MRPO": "#D85A30", "RPO": "#185FA5", "RIME": "#1D9E75",
        "GO"  : "#BA7517", "CPO": "#3C3489", "HO"  : "#993556",
        "WaOA": "#639922", "FOX": "#888780", "GJO" : "#4A1B0C",
    }
    styles = {
        "MRPO": "-",  "RPO": "--", "RIME": "-.", "GO" : ":",
        "CPO" : "-",  "HO" : "--", "WaOA": "-.", "FOX": ":", "GJO": "-",
    }

    for fname, algo_hist in conv.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for algo, hist in algo_hist.items():
            if not hist: continue
            xs = [h[0] for h in hist]
            ys = [h[1] for h in hist]
            ax.plot(xs, ys, label=algo, color=colors.get(algo, "gray"),
                    linestyle=styles.get(algo, "-"), linewidth=1.5)
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (log scale)", fontsize=11)
        ax.set_title(f"Convergence — {suite_name.upper()} {fname} ({dim}D)", fontsize=12)
        ax.set_yscale("symlog")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{fname}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"  Convergence plots saved to: {plot_dir}")
    return plot_dir


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",   default="all",
                        help="cec2014 | cec2017 | cec2020 | cec2022 | all")
    parser.add_argument("--dim",     type=int, default=10)
    parser.add_argument("--runs",    type=int, default=50)
    parser.add_argument("--fes",     type=int, default=60000)
    parser.add_argument("--pop",     type=int, default=50)
    parser.add_argument("--outdir",  default="results")
    parser.add_argument("--quick",   action="store_true",
                        help="Quick smoke test (5 runs, 1000 FEs)")
    args = parser.parse_args()

    suites = ["cec2014", "cec2017", "cec2020", "cec2022"] \
             if args.suite == "all" else [args.suite]

    for suite in suites:
        print(f"\nRunning {suite.upper()}...")
        # Main experiments
        raw, funcs = run_experiment(
            suite, args.dim, args.runs, args.fes, args.pop, args.quick)
        df = compute_stats(raw)

        # Convergence (1 run per algo/func)
        print(f"\n  Computing convergence curves for {suite}...")
        conv, _ = compute_convergence(
            suite, args.dim, args.fes if not args.quick else 1000, args.pop, args.quick)

        # Save
        save_excel(df, conv, suite, args.dim, args.outdir)
        save_convergence_plots(conv, suite, args.dim, args.outdir)

    print("\n\nAll done! Check the 'results/' folder.")


if __name__ == "__main__":
    main()