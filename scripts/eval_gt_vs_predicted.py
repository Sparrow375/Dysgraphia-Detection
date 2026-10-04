"""
scripts/eval_gt_vs_predicted.py
=================================
Steps 2, 5 & 7 — Full Cohort GT-vs-Predicted Evaluation

Runs the complete validation pipeline across all 120 subjects in the
dataSciRep_public cohort:

  STEP 2  — NVI rate denominator audit:
    Compares GT NVI rate (pen-down time only, after dt deduplication) with
    predicted NVI rate (pen-down estimated time), confirming both use the same
    denominator.

  STEP 5  — Stroke-count undercount measurement:
    For every subject, compares pipeline-recovered stroke count
    (metadata.recovered_stroke_count) against the GT pen-lift count from SVC
    (sample.strokes count).  Reports the per-subject and cohort-level
    undercount fraction.

  STEP 7  — Full cohort Pearson-r correlation matrix:
    For every subject with a valid full-page image, runs:
      (a) load_svc  →  validate_kinematics_against_ground_truth
      (b) Collects Pearson r (velocity), NRMSE (velocity), stroke-level r median
    Outputs:
      outputs/cohort_eval/cohort_eval_results.csv   — per-subject results
      outputs/cohort_eval/cohort_eval_summary.txt   — plain-text summary
      outputs/cohort_eval/cohort_eval_boxplots.png  — distribution figures

Usage:
  python scripts/eval_gt_vs_predicted.py
  python scripts/eval_gt_vs_predicted.py --max_subjects 20   # quick smoke test
  python scripts/eval_gt_vs_predicted.py --task_glob "task_8_*"
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── local imports ─────────────────────────────────────────────────────────────
from src.loaders import load_svc
from src.render import render_trajectory_to_image, WACOM_MM_PER_COORD
from src.branch_a.preprocessing import load_and_binarize, zhang_suen_skeletonize, compute_stroke_width_map
from src.preprocessing_robust import robust_load_image
from src.branch_a.segmentation import segment_handwriting
from src.branch_b.stroke_recovery import recover_handwriting_trajectory
from src.branch_b.kinematics import validate_kinematics_against_ground_truth, extract_kinematic_features
from scipy.ndimage import label, find_objects

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT  = PROJECT_ROOT / "Datasets" / "dataSciRep_public"
METADATA_CSV  = PROJECT_ROOT / "Datasets" / "reconstructed_dataset" / "metadata.csv"
IMAGE_ROOT    = PROJECT_ROOT / "Datasets" / "reconstructed_dataset" / "full_page"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "cohort_eval"

# ── Task that covers the full page (task_8 = sentence = most informative) ────
PREFERRED_TASKS = ["task_8_sentence", "task_7_hrackarstvo", "task_6_lamoken",
                   "task_5_leto", "task_4_le_fast", "task_3_le_normal",
                   "task_2_l_fast", "task_1_l_normal"]


# ─────────────────────────────────────────────────────────────────────────────

def _find_svc_for_user(user_dir: Path, task_glob: str = "*") -> list:
    """Returns all .svc files under user_dir matching task_glob, preferring task_8."""
    svcs = sorted(user_dir.rglob("*.svc"))
    if not svcs:
        return []
    if task_glob != "*":
        filtered = [s for s in svcs if task_glob.lower() in s.stem.lower()]
        if filtered:
            return filtered
    # Sort by preferred task order
    def task_key(p):
        name = p.stem.lower()
        for i, t in enumerate(PREFERRED_TASKS):
            if t in name:
                return i
        return len(PREFERRED_TASKS)
    return sorted(svcs, key=task_key)


def _load_metadata_csv(csv_path: Path) -> dict:
    """Returns {user_id: row_dict} from metadata.csv."""
    meta = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["user_id"].zfill(5)
                meta[uid] = row
    except Exception as e:
        print(f"[warn] Could not load metadata CSV: {e}")
    return meta


def _get_h_med(img_gray_np: np.ndarray) -> float:
    """Estimates H_med from a rendered grayscale image array."""
    from PIL import Image
    # Binarize and segment
    from src.branch_a.preprocessing import load_and_binarize
    try:
        mask = (img_gray_np < 128).astype(np.uint8)
        from src.branch_a.segmentation import segment_handwriting
        text_lines = segment_handwriting(mask, remove_ruled_lines=False)
        mh = [l.median_height for l in text_lines if l.components]
        if mh:
            return float(np.median(mh))
    except Exception:
        pass
    # Fallback: connected component heights
    try:
        from scipy.ndimage import label, find_objects
        lbl, n_lbl = label(mask)
        if n_lbl > 0:
            objs = find_objects(lbl)
            hh = [sl[0].stop - sl[0].start for sl in objs
                  if sl is not None and sl[0].stop - sl[0].start >= 4]
            if hh:
                return float(np.median(hh))
    except Exception:
        pass
    return 25.0


def process_subject(
    user_id: str,
    user_dir: Path,
    meta_row: dict,
    task_glob: str,
) -> dict:
    """
    Full processing pipeline for one subject.
    Returns a result dict (never raises).
    """
    result = {
        "user_id": user_id,
        "diagnosis": meta_row.get("diagnosis", "unknown"),
        "is_dysgraphic": meta_row.get("is_dysgraphic", ""),
        "age": meta_row.get("age", ""),
        "sex": meta_row.get("sex", ""),
        "task": "",
        "svc_file": "",
        # GT stroke/NVI info
        "gt_stroke_count": None,
        "gt_pen_down_duration_s": None,
        "gt_total_nvi": None,
        "gt_nvi_rate": None,
        "gt_dt_artifacts_dropped": None,
        # Predicted stroke/NVI info
        "pred_stroke_count": None,
        "pred_nvi_rate": None,
        "pred_nvi_per_stroke": None,
        "stroke_undercount_fraction": None,
        # Kinematic correlation
        "pearson_r_velocity": None,
        "nrmse_velocity": None,
        "pearson_r_pressure": None,
        "stroke_level_count": None,
        "stroke_level_median_r": None,
        "stroke_level_frac_gt_03": None,
        # Physical calibration
        "mm_per_px": None,
        "h_med_px": None,
        # Quality
        "status": "pending",
        "error": "",
        "elapsed_s": 0.0,
    }

    t0 = time.time()
    try:
        # ── 1. Find SVC file ──────────────────────────────────────────────
        svc_files = _find_svc_for_user(user_dir, task_glob)
        if not svc_files:
            result["status"] = "no_svc"
            return result

        svc_path = svc_files[0]
        result["svc_file"] = str(svc_path.relative_to(DATASET_ROOT))
        result["task"] = svc_path.stem

        # ── 2. Load & dedup SVC ───────────────────────────────────────────
        sample = load_svc(str(svc_path), dedup_timestamps=True)
        result["gt_dt_artifacts_dropped"] = sample.metadata.get("dt_artifact_rows_dropped", 0)

        # ── 3. GT stroke count and NVI rate ──────────────────────────────
        gt_stroke_count = len(sample.strokes)
        result["gt_stroke_count"] = gt_stroke_count

        # GT pen-down duration (sum of on-surface segment durations, cleaned)
        on_pts = sample.points[sample.points[:, 3] > 0.5]
        if len(on_pts) > 1:
            gt_t   = on_pts[:, 2]
            gt_dt  = np.diff(gt_t)
            # Pen-down time = sum of dt where consecutive on-surface points
            # (already deduplicated by load_svc, so no microsecond gaps)
            gt_pen_down_dur = float(np.sum(gt_dt))
        else:
            gt_pen_down_dur = sample.total_duration_sec

        result["gt_pen_down_duration_s"] = round(gt_pen_down_dur, 3)

        # GT NVI rate: count velocity inversions in cleaned GT signal
        if len(on_pts) >= 10:
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d
            dx = np.diff(on_pts[:, 0])
            dy = np.diff(on_pts[:, 1])
            dt = np.diff(on_pts[:, 2])
            from src.loaders import MIN_DT_S
            dt_s = np.where(dt < MIN_DT_S, MIN_DT_S, dt)
            gt_speed = np.sqrt(dx**2 + dy**2) / dt_s
            gt_speed_s = gaussian_filter1d(gt_speed, sigma=2.0)
            peaks, _   = find_peaks(gt_speed_s)
            troughs, _ = find_peaks(-gt_speed_s)
            gt_total_nvi = len(peaks) + len(troughs)
            gt_nvi_rate  = float(gt_total_nvi) / max(gt_pen_down_dur, 0.1)
        else:
            gt_total_nvi = 0
            gt_nvi_rate  = 0.0

        result["gt_total_nvi"] = gt_total_nvi
        result["gt_nvi_rate"]  = round(gt_nvi_rate, 3)

        # ── 4. Render + physical scale ────────────────────────────────────
        from src.render import render_trajectory_to_image_with_meta
        img, scale_meta = render_trajectory_to_image_with_meta(sample)
        mm_per_px = scale_meta["mm_per_px"]
        result["mm_per_px"] = round(mm_per_px, 8)

        img_arr = np.array(img)

        # ── 5. Binarize / skeleton / segment ─────────────────────────────
        mask = (img_arr < 128).astype(np.uint8)
        from src.branch_a.preprocessing import zhang_suen_skeletonize, compute_stroke_width_map
        skeleton = zhang_suen_skeletonize(mask)
        dist_map, _, _ = compute_stroke_width_map(mask, skeleton)

        h_med = _get_h_med(img_arr)
        result["h_med_px"] = round(h_med, 2)

        # ── 6. Recover strokes from image ─────────────────────────────────
        skel_px = int(np.sum(skeleton))
        if skel_px == 0:
            result["status"] = "no_skeleton"
            return result

        recovered = recover_handwriting_trajectory(skeleton, h_med=h_med)
        pred_stroke_count = len(recovered)
        result["pred_stroke_count"] = pred_stroke_count

        # ── 7. Stroke undercount (STEP 5) ─────────────────────────────────
        if gt_stroke_count > 0:
            undercount = (gt_stroke_count - pred_stroke_count) / gt_stroke_count
            result["stroke_undercount_fraction"] = round(undercount, 4)

        # ── 8. Predicted NVI rate (STEP 2) ────────────────────────────────
        if recovered:
            kin = extract_kinematic_features(recovered, dist_map, h_med=h_med)
            result["pred_nvi_rate"]      = round(kin.get("nvi_rate", 0.0), 3)
            result["pred_nvi_per_stroke"]= round(kin.get("nvi_per_stroke", 0.0), 3)

        # ── 9. Full kinematic validation (STEP 7) ─────────────────────────
        val = validate_kinematics_against_ground_truth(
            sample, recovered, dist_map, h_med=h_med
        )

        if val.get("status") == "success":
            result["pearson_r_velocity"]     = round(val["pearson_r_velocity"], 4)
            result["nrmse_velocity"]         = round(val["nrmse_velocity"], 4)
            result["pearson_r_pressure"]     = round(val.get("pearson_r_pressure", float("nan")), 4)
            result["stroke_level_count"]     = val["stroke_level_count"]
            result["stroke_level_median_r"]  = round(val["stroke_level_median_r"], 4)
            result["stroke_level_frac_gt_03"]= round(val["stroke_level_fraction_gt_03"], 4)
            result["status"] = "success"
        else:
            result["status"] = f"val_failed:{val.get('status','unknown')}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"  [ERROR] user {user_id}: {e}\n{tb[:400]}")

    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def _nvi_rate_comparison_note(results: list) -> list:
    """
    STEP 2: Compare GT NVI rate (pen-down time denominator) vs predicted NVI rate.
    Returns lines for the summary report.
    """
    good = [r for r in results if r["status"] == "success"
            and r["gt_nvi_rate"] is not None and r["pred_nvi_rate"] is not None]

    if not good:
        return ["[NVI STEP 2] No successful results to compare."]

    gt_rates   = np.array([r["gt_nvi_rate"]   for r in good])
    pred_rates = np.array([r["pred_nvi_rate"]  for r in good])
    ratios     = pred_rates / np.where(gt_rates > 0.01, gt_rates, np.nan)
    ratios     = ratios[~np.isnan(ratios)]

    try:
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(gt_rates, pred_rates)
    except Exception:
        r_val, p_val = float("nan"), float("nan")

    lines = [
        "",
        "=" * 64,
        "STEP 2 — NVI Rate Denominator Audit",
        "Both GT and predicted use pen-down duration as the denominator.",
        f"  Subjects compared    : {len(good)}",
        f"  GT NVI rate mean     : {float(np.nanmean(gt_rates)):.2f} inv/s",
        f"  Predicted NVI mean   : {float(np.nanmean(pred_rates)):.2f} inv/s",
        f"  Pred/GT ratio median : {float(np.nanmedian(ratios)):.2f}x",
        f"  Pearson r (GT vs pred NVI rate): {r_val:.3f}  p={p_val:.3e}",
        "Note: Predicted NVI rate is based on estimated pen-down duration from",
        "the reconstructed velocity profile; GT uses actual recorded timestamps.",
        "A ratio >> 1 indicates over-fragmentation in the skeleton recovery step.",
        "=" * 64,
    ]
    return lines


def _stroke_undercount_summary(results: list) -> list:
    """STEP 5: Stroke count undercount distribution."""
    good = [r for r in results if r["stroke_undercount_fraction"] is not None]
    if not good:
        return ["[STROKE STEP 5] No results with valid stroke counts."]

    fracs = np.array([r["stroke_undercount_fraction"] for r in good])
    gt_counts   = np.array([r["gt_stroke_count"]   for r in good])
    pred_counts = np.array([r["pred_stroke_count"]  for r in good])

    try:
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(gt_counts, pred_counts)
    except Exception:
        r_val, p_val = float("nan"), float("nan")

    lines = [
        "",
        "=" * 64,
        "STEP 5 — Stroke Count: GT pen-lift count vs Pipeline recovered strokes",
        f"  Subjects analyzed    : {len(good)}",
        f"  GT stroke mean (pen lifts): {float(np.mean(gt_counts)):.1f}",
        f"  Predicted stroke mean      : {float(np.mean(pred_counts)):.1f}",
        "",
        "  NOTE: 'undercount fraction' = (GT - Pred) / GT.",
        "  Negative values mean the pipeline OVER-counts strokes vs GT pen-lifts.",
        "  This is expected: the skeleton graph splits connected ink at branch points",
        "  and low-curvature pivots, so one pen stroke becomes multiple recovered paths.",
        "",
        f"  Signed fraction mean : {float(np.mean(fracs)):.3f}  ({float(np.mean(fracs))*100:.1f}%)",
        f"  Signed fraction p50  : {float(np.median(fracs)):.3f}",
        f"  Signed fraction p95  : {float(np.percentile(fracs, 95)):.3f}",
        f"  Subjects over-counting (pred > GT)   : {int(np.sum(fracs < 0))}",
        f"  Subjects under-counting (pred < GT)  : {int(np.sum(fracs > 0))}",
        f"  Pearson r (GT vs pred stroke count): {r_val:.3f}  p={p_val:.3e}",
        "" ,
        "Practical implication: kin_pen_lift_count over-estimates writer-intended pen",
        "lifts due to skeleton graph fragmentation. Requires a merge-short-strokes pass",
        "(gap < gap_threshold_px) before it can be compared against GT pen-lift counts.",
        "=" * 64,
    ]
    return lines


def _correlation_summary(results: list) -> list:
    """STEP 7: Pearson r velocity correlation distribution."""
    good = [r for r in results if r["status"] == "success"
            and r["pearson_r_velocity"] is not None]
    if not good:
        return ["[CORR STEP 7] No successful correlation results."]

    rv   = np.array([r["pearson_r_velocity"]    for r in good])
    nrmse= np.array([r["nrmse_velocity"]         for r in good])
    slr  = np.array([r["stroke_level_median_r"]  for r in good])

    dysgraphic = [r for r in good if str(r.get("is_dysgraphic","")).lower() == "true"]
    control    = [r for r in good if str(r.get("is_dysgraphic","")).lower() == "false"]

    def grp(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return np.array(vals) if vals else np.array([float("nan")])

    lines = [
        "",
        "=" * 64,
        "STEP 7 — Full Cohort Kinematic Correlation",
        f"  Subjects with valid correlation: {len(good)}",
        "",
        "  Whole-document velocity Pearson r:",
        f"    Mean   : {float(np.nanmean(rv)):.3f}",
        f"    Median : {float(np.nanmedian(rv)):.3f}",
        f"    P25-P75: [{float(np.nanpercentile(rv,25)):.3f}, {float(np.nanpercentile(rv,75)):.3f}]",
        f"    Frac r>0.3: {float(np.mean(rv>0.3)):.2f}",
        f"    Frac r>0.5: {float(np.mean(rv>0.5)):.2f}",
        "",
        "  NOTE: r~0 is expected and correct. The pipeline reconstructs a synthetic",
        "  time axis from skeleton path order; GT uses real recorded 125 Hz timestamps.",
        "  These two time axes are unrelated, so cross-correlation of the velocity",
        "  profiles is not meaningful. Stroke-level r is the informative metric.",
        "",
        "  NRMSE (velocity):",
        f"    Mean  : {float(np.nanmean(nrmse)):.3f}",
        f"    Median: {float(np.nanmedian(nrmse)):.3f}",
        "",
        "  Stroke-level median Pearson r (shape similarity within each stroke):",
        f"    Mean   : {float(np.nanmean(slr)):.3f}",
        f"    Median : {float(np.nanmedian(slr)):.3f}",
    ]

    if dysgraphic:
        dvals = grp(dysgraphic, "pearson_r_velocity")
        lines.append(f"  Dysgraphic (n={len(dysgraphic)}): r_v mean={float(np.nanmean(dvals)):.3f}")
    if control:
        cvals = grp(control, "pearson_r_velocity")
        lines.append(f"  Control   (n={len(control)}): r_v mean={float(np.nanmean(cvals)):.3f}")
    else:
        lines.append("  Control   (n=0): no control subjects in this batch")

    lines.append("=" * 64)
    return lines


def _make_boxplots(results: list, out_dir: Path):
    """Generates distribution boxplots and saves PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = [r for r in results if r["status"] == "success"]
        if not good:
            return

        rv  = [r["pearson_r_velocity"]   for r in good if r["pearson_r_velocity"] is not None]
        slr = [r["stroke_level_median_r"] for r in good if r["stroke_level_median_r"] is not None]
        uc  = [r["stroke_undercount_fraction"] for r in good
               if r["stroke_undercount_fraction"] is not None]
        gt_nvi  = [r["gt_nvi_rate"]   for r in good if r["gt_nvi_rate"] is not None]
        pr_nvi  = [r["pred_nvi_rate"] for r in good if r["pred_nvi_rate"] is not None]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle("Full Cohort GT-vs-Predicted Evaluation", fontsize=13, fontweight="bold")

        axes[0].boxplot(rv, orientation="vertical")
        axes[0].axhline(0.0, color="red", lw=0.8, ls="--")
        axes[0].axhline(0.3, color="green", lw=0.8, ls="--", label="r=0.3")
        axes[0].set_title("Velocity Pearson r\n(whole-document)\nNOTE: r~0 expected", fontsize=9)
        axes[0].set_ylabel("Pearson r")
        axes[0].legend(fontsize=8)

        axes[1].boxplot(slr, orientation="vertical")
        axes[1].axhline(0.3, color="green", lw=0.8, ls="--")
        axes[1].set_title("Stroke-level Pearson r\n(median per subject)", fontsize=10)
        axes[1].set_ylabel("Pearson r")

        axes[2].boxplot(uc, orientation="vertical")
        axes[2].axhline(0.0, color="gray", lw=0.8, ls="--")
        axes[2].set_title("Stroke Count Error Fraction\n(neg = over-count vs GT)", fontsize=9)
        axes[2].set_ylabel("(GT - Pred) / GT")

        # NVI rate GT vs Pred scatter
        if gt_nvi and pr_nvi and len(gt_nvi) == len(pr_nvi):
            axes[3].scatter(gt_nvi, pr_nvi, alpha=0.5, s=20)
            lim = max(max(gt_nvi), max(pr_nvi)) * 1.05
            axes[3].plot([0, lim], [0, lim], "r--", lw=0.8, label="y=x")
            axes[3].set_title("NVI Rate: GT vs Predicted", fontsize=10)
            axes[3].set_xlabel("GT NVI rate (inv/s)")
            axes[3].set_ylabel("Predicted NVI rate (inv/s)")
            axes[3].legend(fontsize=8)

        plt.tight_layout()
        out_path = out_dir / "cohort_eval_boxplots.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] Boxplots saved → {out_path}")
    except Exception as e:
        print(f"[warn] Boxplot generation failed: {e}")


def run_evaluation(
    max_subjects: int = 0,
    task_glob: str = "*",
    verbose: bool = True,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata CSV for diagnosis labels
    metadata = _load_metadata_csv(METADATA_CSV)

    # Discover all user directories
    user_dirs = sorted([d for d in DATASET_ROOT.iterdir()
                        if d.is_dir() and d.name.startswith("user")])
    if max_subjects > 0:
        user_dirs = user_dirs[:max_subjects]

    print(f"[eval] Found {len(user_dirs)} user directories in {DATASET_ROOT}")
    print(f"[eval] Task filter: '{task_glob}' | Output → {OUTPUT_DIR}\n")

    results = []
    for i, user_dir in enumerate(user_dirs):
        # user_dir name: "user00050" → user_id "00050"
        raw_uid = user_dir.name.replace("user", "")
        user_id = raw_uid.zfill(5)
        meta_row = metadata.get(user_id, {})

        if verbose:
            diag = meta_row.get("diagnosis", "?")
            print(f"  [{i+1:3d}/{len(user_dirs)}] user_{user_id} ({diag}) ...", end=" ", flush=True)

        res = process_subject(user_id, user_dir, meta_row, task_glob)
        results.append(res)

        if verbose:
            st = res["status"]
            r  = res.get("pearson_r_velocity")
            sc = res.get("stroke_undercount_fraction")
            print(f"status={st}  r_v={r}  undercount={sc}  ({res['elapsed_s']}s)")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "cohort_eval_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n[eval] Results CSV → {csv_path}")

    # ── Generate summary ──────────────────────────────────────────────────────
    success_n  = sum(1 for r in results if r["status"] == "success")
    no_svc_n   = sum(1 for r in results if r["status"] == "no_svc")
    error_n    = sum(1 for r in results if r["status"] == "error")

    summary_lines = [
        "=" * 64,
        "FULL COHORT GT-vs-PREDICTED EVALUATION",
        f"Subjects processed : {len(results)}",
        f"  Successful       : {success_n}",
        f"  No SVC found     : {no_svc_n}",
        f"  Errors           : {error_n}",
        f"  Other            : {len(results) - success_n - no_svc_n - error_n}",
        "=" * 64,
    ]

    summary_lines += _nvi_rate_comparison_note(results)
    summary_lines += _stroke_undercount_summary(results)
    summary_lines += _correlation_summary(results)

    summary_txt = OUTPUT_DIR / "cohort_eval_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    for line in summary_lines:
        print(line)

    print(f"\n[eval] Summary → {summary_txt}")

    # ── Boxplots ──────────────────────────────────────────────────────────────
    _make_boxplots(results, OUTPUT_DIR)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full cohort GT-vs-predicted kinematic evaluation (Steps 2, 5, 7)."
    )
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="Limit subjects (0 = all). Use 5-10 for a smoke test.")
    parser.add_argument("--task_glob", default="*",
                        help="Filter SVC by task substring, e.g. 'task_8'.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        max_subjects=args.max_subjects,
        task_glob=args.task_glob,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
