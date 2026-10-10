"""Full benchmark execution runner across Hindi, English, and Combined tables.
Executes 5-repeat 5-fold nested cross-validation across all candidate models,
generates detailed performance tables, and saves summary artifacts to reports/.
"""

from pathlib import Path
import pandas as pd
from models.experiments import run_benchmark_suite

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


import argparse


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark across datasets.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of outer CV repeats (default 1).")
    args = parser.parse_args()

    datasets = ["combined", "hindi", "english"]
    all_summaries = []

    for ds in datasets:
        print(f"\n=======================================================", flush=True)
        print(f"BENCHMARKING DATASET: {ds.upper()}", flush=True)
        print(f"=======================================================", flush=True)
        summary_df = run_benchmark_suite(
            dataset_alias=ds,
            n_repeats=args.repeats,
            save_reports=True,
        )
        all_summaries.append(summary_df)

    unified_df = pd.concat(all_summaries, ignore_index=True)
    unified_csv = WORKSPACE_ROOT / "reports" / "benchmark_all_datasets.csv"
    unified_df.to_csv(unified_csv, index=False)
    print(f"\n[DONE] Unified benchmark report saved to: {unified_csv}", flush=True)


if __name__ == "__main__":
    main()
