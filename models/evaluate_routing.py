"""Evaluation Suite for Task-Aware Dynamic Routing vs Flat Imputation.

Compares:
1. Flat Median Imputation Baseline (Old Stage 15/16 Workstream A).
2. Task-Aware Dynamic Routing Engine (Stage 17).

Evaluates across:
- All School B students (Grades 3-8, N=100)
- Matched Grades (Grades 4-7, N=73)
- No-Dictation Grade 3 (N=16)
- Partial-Dictation Grade 8 (N=11)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from models.task_router import FULL_PROTOCOL_FEATURES, TaskAwareRouter, TaskRoute

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
STUDENT_DATASET_PATH = WORKSPACE_ROOT / "data" / "datasets" / "dataset_student_level.csv"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_flat_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Train single Full Protocol model on School A and test on School B with flat median imputation."""
    valid_cols = [c for c in FULL_PROTOCOL_FEATURES if c in df_train.columns]
    meds = df_train[valid_cols].median()

    X_train = df_train[valid_cols].fillna(meds).values
    y_train = df_train["label"].values.astype(int)

    X_test = df_test[valid_cols].fillna(meds).values
    y_test = df_test["label"].values.astype(int)

    clf = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    # Metrics on subsets
    def get_metrics(sub_mask: np.ndarray) -> Dict[str, Any]:
        y_sub = y_test[sub_mask]
        sc_sub = scores[sub_mask]
        n_pos = int(np.sum(y_sub == 1))
        n_neg = int(np.sum(y_sub == 0))
        auc = float(roc_auc_score(y_sub, sc_sub)) if n_pos > 0 and n_neg > 0 else None
        auprc = float(average_precision_score(y_sub, sc_sub)) if n_pos > 0 and n_neg > 0 else None
        return {
            "n": int(np.sum(sub_mask)),
            "pos": n_pos,
            "neg": n_neg,
            "auroc": auc,
            "auprc": auprc,
        }

    all_mask = np.ones(len(df_test), dtype=bool)
    g47_mask = df_test["grade"].isin([4, 5, 6, 7]).values
    g3_mask = (df_test["grade"] == 3).values
    g8_mask = (df_test["grade"] == 8).values

    return {
        "all_b": get_metrics(all_mask),
        "g47": get_metrics(g47_mask),
        "g3": get_metrics(g3_mask),
        "g8": get_metrics(g8_mask),
        "scores": scores,
    }


def run_task_routing_evaluation(dataset_path: Path = STUDENT_DATASET_PATH) -> pd.DataFrame:
    """Run comparative evaluation of Task-Aware Routing vs Flat Imputation Baseline."""
    df = pd.read_csv(dataset_path)
    df_a = df[df["school"] == "school_a"].copy().reset_index(drop=True)
    df_b = df[df["school"] == "school_b"].copy().reset_index(drop=True)

    print("=================================================================")
    print("TASK-AWARE ROUTING vs FLAT IMPUTATION EVALUATION")
    print(f"School A (Train): {len(df_a)} students")
    print(f"School B (Test):  {len(df_b)} students")
    print("=================================================================")

    # 1. Flat Baseline
    flat_results = evaluate_flat_baseline(df_a, df_b)

    # 2. Task-Aware Router
    router = TaskAwareRouter(random_state=42)
    router.fit(df_a)
    routed_eval = router.evaluate(df_b)
    routed_scores = router.predict_proba(df_b)
    y_b = df_b["label"].values.astype(int)

    # Helper for subset metrics
    def get_routed_sub(mask: np.ndarray) -> Dict[str, Any]:
        y_sub = y_b[mask]
        sc_sub = routed_scores[mask]
        n_pos = int(np.sum(y_sub == 1))
        n_neg = int(np.sum(y_sub == 0))
        auc = float(roc_auc_score(y_sub, sc_sub)) if n_pos > 0 and n_neg > 0 else None
        auprc = float(average_precision_score(y_sub, sc_sub)) if n_pos > 0 and n_neg > 0 else None
        return {
            "n": int(np.sum(mask)),
            "pos": n_pos,
            "neg": n_neg,
            "auroc": auc,
            "auprc": auprc,
        }

    g47_mask = df_b["grade"].isin([4, 5, 6, 7]).values
    g3_mask = (df_b["grade"] == 3).values
    g8_mask = (df_b["grade"] == 8).values

    routed_sub_metrics = {
        "all_b": get_routed_sub(np.ones(len(df_b), dtype=bool)),
        "g47": get_routed_sub(g47_mask),
        "g3": get_routed_sub(g3_mask),
        "g8": get_routed_sub(g8_mask),
    }

    # Build comparison rows
    comparison_rows = [
        {
            "cohort_slice": "All School B (Grades 3-8)",
            "n_students": 100,
            "positives": 16,
            "controls": 84,
            "flat_baseline_auroc": flat_results["all_b"]["auroc"],
            "routed_auroc": routed_sub_metrics["all_b"]["auroc"],
            "auroc_delta": (routed_sub_metrics["all_b"]["auroc"] or 0) - (flat_results["all_b"]["auroc"] or 0),
            "flat_baseline_auprc": flat_results["all_b"]["auprc"],
            "routed_auprc": routed_sub_metrics["all_b"]["auprc"],
        },
        {
            "cohort_slice": "Matched Grades 4-7 (Full Protocol)",
            "n_students": 73,
            "positives": 13,
            "controls": 60,
            "flat_baseline_auroc": flat_results["g47"]["auroc"],
            "routed_auroc": routed_sub_metrics["g47"]["auroc"],
            "auroc_delta": (routed_sub_metrics["g47"]["auroc"] or 0) - (flat_results["g47"]["auroc"] or 0),
            "flat_baseline_auprc": flat_results["g47"]["auprc"],
            "routed_auprc": routed_sub_metrics["g47"]["auprc"],
        },
        {
            "cohort_slice": "Grade 3 (Zero Dictation Administered)",
            "n_students": 16,
            "positives": 2,
            "controls": 14,
            "flat_baseline_auroc": flat_results["g3"]["auroc"],
            "routed_auroc": routed_sub_metrics["g3"]["auroc"],
            "auroc_delta": (routed_sub_metrics["g3"]["auroc"] or 0) - (flat_results["g3"]["auroc"] or 0),
            "flat_baseline_auprc": flat_results["g3"]["auprc"],
            "routed_auprc": routed_sub_metrics["g3"]["auprc"],
        },
        {
            "cohort_slice": "Grade 8 (Partial Dictation Administered)",
            "n_students": 11,
            "positives": 1,
            "controls": 10,
            "flat_baseline_auroc": flat_results["g8"]["auroc"],
            "routed_auroc": routed_sub_metrics["g8"]["auroc"],
            "auroc_delta": (routed_sub_metrics["g8"]["auroc"] or 0) - (flat_results["g8"]["auroc"] or 0),
            "flat_baseline_auprc": flat_results["g8"]["auprc"],
            "routed_auprc": routed_sub_metrics["g8"]["auprc"],
        },
    ]

    res_df = pd.DataFrame(comparison_rows)
    out_csv = REPORTS_DIR / "task_routing_evaluation.csv"
    res_df.to_csv(out_csv, index=False)

    out_json = REPORTS_DIR / "task_routing_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "comparison_table": comparison_rows,
            "route_breakdown": routed_eval["route_breakdown"],
            "grade_breakdown": routed_eval["grade_breakdown"],
        }, f, indent=2)

    print("\n=== Evaluation Results ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved structured reports to:\n- {out_csv}\n- {out_json}")
    return res_df


if __name__ == "__main__":
    run_task_routing_evaluation()
