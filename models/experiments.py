"""Nested cross-validation experiment runner for Phase 4.
Runs 5-repeat 5-fold student-stratified nested CV, evaluates candidate models,
computes cluster bootstrap confidence intervals, and exports benchmark reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from models.candidates import get_candidate_registry
from models.cv import (
    compute_sample_weights,
    get_feature_columns,
    get_inner_cv_splits,
    get_outer_fold_split,
    impute_fold_features,
    load_dataset,
)
from models.evaluator import (
    aggregate_student_predictions,
    cluster_bootstrap_ci,
    compute_student_metrics,
    find_operational_threshold,
    fit_platt_calibrator,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def evaluate_inner_config(
    model_builder: Any,
    params: Dict[str, Any],
    train_df: pd.DataFrame,
    precomputed_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[float, np.ndarray]:
    """Evaluate a hyperparameter configuration across precomputed inner CV splits.

    Returns the mean student AUROC and the full out-of-fold predicted scores.
    """
    oof_scores = np.zeros(len(train_df), dtype=np.float64)

    for in_val_idx, X_in_tr, X_in_val, y_in_tr, w_in_tr in precomputed_splits:
        model = model_builder(**params)
        try:
            model.fit(X_in_tr, y_in_tr, sample_weight=w_in_tr)
        except TypeError:
            model.fit(X_in_tr, y_in_tr)

        if hasattr(model, "predict_proba"):
            p1 = model.predict_proba(X_in_val)[:, 1]
        elif hasattr(model, "decision_function"):
            df_val = model.decision_function(X_in_val)
            p1 = 1.0 / (1.0 + np.exp(-df_val))
        else:
            p1 = model.predict(X_in_val).astype(float)

        oof_scores[in_val_idx] = p1

    # Measure inner student-level AUROC
    inner_student_df = aggregate_student_predictions(train_df, oof_scores)
    metrics = compute_student_metrics(inner_student_df)
    return metrics["auroc"], oof_scores


def run_nested_cv_for_model(
    df: pd.DataFrame,
    model_key: str,
    n_repeats: int = 5,
    n_outer_folds: int = 5,
    n_inner_splits: int = 3,
    exclude_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete nested CV benchmark for a single candidate model."""
    registry = get_candidate_registry()
    if model_key not in registry:
        raise KeyError(f"Model key '{model_key}' not in candidate registry.")

    model_info = registry[model_key]
    feature_type = model_info["feature_type"]
    builder = model_info["builder"]
    param_grid = model_info["param_grid"]

    # Select features according to model preference
    if feature_type == "none":
        feature_cols: List[str] = []
    elif feature_type == "grade_only":
        feature_cols = ["grade"]
    elif feature_type == "z_score":
        feature_cols = get_feature_columns(df, feature_type="z_score", exclude_columns=exclude_columns)
    else:  # raw
        feature_cols = get_feature_columns(df, feature_type="raw", exclude_columns=exclude_columns)

    sample_weights = compute_sample_weights(df, balance_classes=True)

    all_student_test_records: List[pd.DataFrame] = []
    selected_thresholds: List[float] = []

    start_time = time.time()

    for r in range(n_repeats):
        for k in range(n_outer_folds):
            train_df, test_df = get_outer_fold_split(df, repeat_idx=r, fold_idx=k)
            test_mask = df[f"repeat_{r}_fold"] == k
            train_mask = ~test_mask

            w_train = sample_weights[train_mask]
            w_test = sample_weights[test_mask]

            if len(feature_cols) == 0:
                X_train = np.ones((len(train_df), 1))
                X_test = np.ones((len(test_df), 1))
            else:
                X_train, X_test, _ = impute_fold_features(train_df, test_df, feature_cols)

            y_train = train_df["label"].values.astype(int)
            y_test = test_df["label"].values.astype(int)

            # Inner CV for tuning and calibration
            seed = random_state + r * 100 + k * 10
            inner_splits = get_inner_cv_splits(train_df, n_splits=n_inner_splits, random_state=seed)

            # Precompute inner split feature matrices once per fold
            precomputed_splits = []
            for in_tr_idx, in_val_idx in inner_splits:
                in_tr_df = train_df.iloc[in_tr_idx]
                in_val_df = train_df.iloc[in_val_idx]
                if len(feature_cols) == 0:
                    X_in_tr = np.ones((len(in_tr_df), 1))
                    X_in_val = np.ones((len(in_val_df), 1))
                else:
                    X_in_tr, X_in_val, _ = impute_fold_features(in_tr_df, in_val_df, feature_cols)
                y_in_tr = in_tr_df["label"].values.astype(int)
                w_in_tr = w_train[in_tr_idx]
                precomputed_splits.append((in_val_idx, X_in_tr, X_in_val, y_in_tr, w_in_tr))

            best_params = param_grid[0]
            best_score = -1.0
            best_oof_scores = np.zeros(len(train_df))

            for params in param_grid:
                score, oof_scores = evaluate_inner_config(
                    model_builder=builder,
                    params=params,
                    train_df=train_df,
                    precomputed_splits=precomputed_splits,
                )
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_oof_scores = oof_scores

            # Determine operational threshold targeting >= 90% specificity on inner validation
            inner_student_df = aggregate_student_predictions(train_df, best_oof_scores)
            op_threshold = find_operational_threshold(inner_student_df, min_specificity=0.90)
            selected_thresholds.append(op_threshold)

            # Fit model with best_params on full outer training split
            final_estimator = builder(**best_params)
            try:
                final_estimator.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                final_estimator.fit(X_train, y_train)

            # Platt calibration fit on inner out-of-fold validation scores
            calibrated_predictor = fit_platt_calibrator(
                base_estimator=final_estimator,
                X_val=X_train,
                y_val=y_train,
                sample_weight_val=w_train,
            )

            # Predict calibrated sentence probabilities on outer test set
            test_probs = calibrated_predictor.predict_proba(X_test)[:, 1]

            # Pool to student level
            student_fold_df = aggregate_student_predictions(test_df, test_probs)
            student_fold_df["repeat"] = r
            student_fold_df["fold"] = k
            student_fold_df["threshold"] = op_threshold
            all_student_test_records.append(student_fold_df)

    elapsed_time = time.time() - start_time
    pooled_student_df = pd.concat(all_student_test_records, ignore_index=True)

    # Average student predictions across the repeats (each student tested once per repeat)
    student_summary_df = pooled_student_df.groupby("student_id").agg(
        label=("label", "first"),
        pred_prob=("pred_prob", "mean"),
        threshold=("threshold", "mean"),
    ).reset_index()

    median_threshold = float(np.median(selected_thresholds))

    # Compute aggregate point metrics
    point_metrics = compute_student_metrics(student_summary_df, threshold=median_threshold)

    # Compute 95% Cluster Bootstrap Confidence Intervals
    bootstrap_ci = cluster_bootstrap_ci(
        student_summary_df,
        threshold=median_threshold,
        n_bootstraps=1000,
        ci=0.95,
        random_state=random_state,
    )

    return {
        "model_key": model_key,
        "model_name": model_info["name"],
        "feature_type": feature_type,
        "num_features": len(feature_cols),
        "n_repeats": n_repeats,
        "n_outer_folds": n_outer_folds,
        "elapsed_seconds": round(elapsed_time, 2),
        "median_threshold": round(median_threshold, 4),
        "point_metrics": point_metrics,
        "bootstrap_ci": bootstrap_ci,
        "student_predictions": student_summary_df.to_dict(orient="records"),
    }


def run_benchmark_suite(
    dataset_alias: str = "combined",
    model_keys: Optional[List[str]] = None,
    n_repeats: int = 5,
    save_reports: bool = True,
) -> pd.DataFrame:
    """Run benchmark across candidate models on a specified dataset table.

    Parameters
    ----------
    dataset_alias : str
        'hindi', 'english', or 'combined'.
    model_keys : list of str, optional
        Subset of models to run. Defaults to all candidate models.
    n_repeats : int
        Number of repeats (default 5).
    save_reports : bool
        Whether to export CSV and JSON report artifacts.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table with point estimates and 95% CIs.
    """
    df = load_dataset(dataset_alias)
    registry = get_candidate_registry()

    if model_keys is None:
        model_keys = ["majority", "grade_only", "elastic_net", "svm_rbf", "random_forest", "gradient_boosting"]

    results_list: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []

    print(f"\n=======================================================", flush=True)
    print(f"Starting Phase 4 Benchmark: {dataset_alias.upper()} (N={len(df)} sentences)", flush=True)
    print(f"Repeats: {n_repeats}, Models: {', '.join(model_keys)}", flush=True)
    print(f"=======================================================\n", flush=True)

    for key in model_keys:
        print(f"--> Running {key} ({registry[key]['name']})...", flush=True)
        res = run_nested_cv_for_model(
            df=df,
            model_key=key,
            n_repeats=n_repeats,
            random_state=42,
        )
        results_list.append(res)

        pm = res["point_metrics"]
        ci = res["bootstrap_ci"]

        row = {
            "dataset": dataset_alias,
            "model_key": key,
            "model_name": res["model_name"],
            "features": res["num_features"],
            "auroc": pm["auroc"],
            "auroc_ci": f"{ci['auroc']['ci_low']:.3f}–{ci['auroc']['ci_high']:.3f}",
            "auprc": pm["auprc"],
            "auprc_ci": f"{ci['auprc']['ci_low']:.3f}–{ci['auprc']['ci_high']:.3f}",
            "brier": pm["brier"],
            "sens_at_90_spec": pm["sensitivity_at_90_spec"],
            "sens_at_90_spec_ci": f"{ci['sensitivity_at_90_spec']['ci_low']:.3f}–{ci['sensitivity_at_90_spec']['ci_high']:.3f}",
            "sens_at_95_spec": pm["sensitivity_at_95_spec"],
            "operating_threshold": res["median_threshold"],
            "balanced_acc": pm["balanced_accuracy"],
            "sensitivity": pm["sensitivity"],
            "specificity": pm["specificity"],
            "f1": pm["f1"],
            "time_sec": res["elapsed_seconds"],
        }
        table_rows.append(row)
        print(
            f"    Done in {res['elapsed_seconds']}s | "
            f"AUROC: {pm['auroc']:.3f} (95% CI: {row['auroc_ci']}) | "
            f"AUPRC: {pm['auprc']:.3f} | Brier: {pm['brier']:.3f}",
            flush=True,
        )

    summary_df = pd.DataFrame(table_rows)

    if save_reports:
        reports_dir = WORKSPACE_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        csv_path = reports_dir / f"benchmark_{dataset_alias}.csv"
        json_path = reports_dir / f"benchmark_{dataset_alias}.json"

        summary_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2)

        print(f"\n[Saved Reports] CSV: {csv_path} | JSON: {json_path}")

    return summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 4 nested CV benchmark.")
    parser.add_argument("--dataset", choices=["hindi", "english", "combined"], default="combined")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    run_benchmark_suite(dataset_alias=args.dataset, n_repeats=args.repeats)
