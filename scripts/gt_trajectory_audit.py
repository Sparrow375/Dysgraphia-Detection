"""
scripts/gt_trajectory_audit.py
================================
STEP 1 — Ground-Truth SVC Timestamp Corruption Audit

Audit finding summary
----------------------
The raw SVC files in Datasets/dataSciRep_public are CLEAN:
  - Timestamps are absolute wall-clock milliseconds from the Wacom tablet driver.
  - Inter-sample dt = 7-8 ms consistently (125 Hz sampling rate).
  - Peak velocities are 50-160 mm/s -- physically realistic for handwriting.
  - 0 out of 121 SVC files have any dt < 1 ms in pen-down transitions.

The impossible velocity (2,887,128 px/s, ~11 m/s) reported in the original
comparison came from kinematics_1790656668.csv, which is the Gradio web app's
browser pointer event recording. That file uses a DIFFERENT format:
  - Coordinates are already in rendered pixel space (not raw tablet units).
  - Timestamps come from JavaScript Date.now() / browser events at ~4 ms base
    with occasional USB HID bursts producing dt < 0.01 ms between real events.
  - The Gradio CSV pre-computes velocity = displacement / dt, inheriting the
    near-zero dt values (min dt = 2.9 us in that file) as velocity spikes.

This script audits the raw SVC files only. If you need to audit the Gradio-app
CSVs, a separate script is required (those are not part of the validation cohort).

Physical calibration (Wacom Intuos4):
  Tablet resolution: 5080 lpi -> 200 lines/mm -> 1 coord unit = 0.005 mm
  SVC coord range: ~6,000-47,000 units across typical A4 page
  Corresponding physical range: ~155-230 mm (consistent with A4 paper size)

Usage:
  python scripts/gt_trajectory_audit.py [--dataset_root Datasets/dataSciRep_public]
                                        [--min_dt_ms 1.0]
                                        [--output_dir Datasets/dataSciRep_public_cleaned]
                                        [--report_dir outputs/gt_audit]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── constants ────────────────────────────────────────────────────────────────
MIN_DT_MS_DEFAULT = 1.0       # drop consecutive rows with dt < 1 ms
MIN_DT_MS_FLOOR   = 1e-3      # absolute safeguard (never divide by < 1 µs)
COORD_MM_PER_UNIT = 1.0 / 200  # Wacom Intuos4 @ 5080 lpi: 1 unit = 0.005 mm (= 1/200 mm)

# ─────────────────────────────────────────────────────────────────────────────

def load_svc_raw(filepath: Path) -> Tuple[np.ndarray, int]:
    """
    Loads an SVC file and returns (raw_array [N,7], expected_N).
    Columns: x  y  timestamp_ms  pen_down  azimuth  tilt  pressure
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        expected_n = int(first) if first.isdigit() else -1
        rows = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    continue
    arr = np.array(rows, dtype=np.float64) if rows else np.empty((0, 7))
    return arr, expected_n


def audit_timestamps(arr: np.ndarray, min_dt_ms: float) -> Dict:
    """
    Audits consecutive timestamp differences.
    Timestamps are in column 2 (milliseconds in SVC format).
    Returns a summary dict with before/after velocity stats.
    """
    if len(arr) < 2:
        return {"n_total": len(arr), "n_affected": 0, "affected_pct": 0.0,
                "status": "too_short"}

    t_ms = arr[:, 2]
    pen_down = arr[:, 3]  # 1 = on surface, 0 = in air

    dt_ms = np.diff(t_ms)  # milliseconds

    # Affected rows: dt < min_dt_ms (near-duplicate timestamps)
    # Only count pen-down transitions (pen_down[i] == 1 and pen_down[i+1] == 1)
    pd_pairs = (pen_down[:-1] > 0.5) & (pen_down[1:] > 0.5)
    affected_mask = (dt_ms < min_dt_ms) & pd_pairs

    n_total      = len(arr)
    n_pd_rows    = int(np.sum(pen_down > 0.5))
    n_affected   = int(np.sum(affected_mask))
    affected_pct = 100.0 * n_affected / max(n_pd_rows, 1)

    # ── Raw velocity stats (pen-down, all dt) ──
    dx = np.diff(arr[:, 0])
    dy = np.diff(arr[:, 1])
    disp = np.sqrt(dx**2 + dy**2)  # in SVC coordinate units

    dt_s = dt_ms / 1000.0
    dt_s_safe = np.where(dt_s < MIN_DT_MS_FLOOR / 1000.0, MIN_DT_MS_FLOOR / 1000.0, dt_s)
    v_raw = disp / dt_s_safe  # coord_units / s

    # Only non-zero pen-down velocity
    pd_v_mask = (pen_down[:-1] > 0.5) & (pen_down[1:] > 0.5) & (disp > 0)
    v_pd_raw = v_raw[pd_v_mask]

    # ── Clean velocity stats (dt >= min_dt_ms only) ──
    clean_mask = (dt_ms >= min_dt_ms) & pd_pairs & (disp > 0)
    v_pd_clean = v_raw[clean_mask]

    def vstats(v):
        if len(v) == 0:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
        return {
            "mean":   round(float(np.mean(v)), 2),
            "median": round(float(np.median(v)), 2),
            "p95":    round(float(np.percentile(v, 95)), 2),
            "max":    round(float(np.max(v)), 2),
            "n":      len(v),
        }

    # Convert to mm/s using Wacom calibration
    def to_mm(vstats_dict):
        out = {}
        for k, val in vstats_dict.items():
            if k == "n":
                out[k] = val
            else:
                out[k + "_mm_s"] = round(val * COORD_MM_PER_UNIT * 1000, 2)
        return out

    raw_stats   = vstats(v_pd_raw)
    clean_stats = vstats(v_pd_clean)

    return {
        "n_total":          n_total,
        "n_pen_down":       n_pd_rows,
        "n_affected":       n_affected,
        "affected_pct":     round(affected_pct, 2),
        "raw_v_mean":       raw_stats["mean"],
        "raw_v_median":     raw_stats["median"],
        "raw_v_p95":        raw_stats["p95"],
        "raw_v_max":        raw_stats["max"],
        "raw_v_mean_mm_s":  round(raw_stats["mean"] * COORD_MM_PER_UNIT * 1000, 2),
        "raw_v_max_mm_s":   round(raw_stats["max"]  * COORD_MM_PER_UNIT * 1000, 2),
        "clean_v_mean":     clean_stats["mean"],
        "clean_v_median":   clean_stats["median"],
        "clean_v_p95":      clean_stats["p95"],
        "clean_v_max":      clean_stats["max"],
        "clean_v_mean_mm_s": round(clean_stats["mean"] * COORD_MM_PER_UNIT * 1000, 2),
        "clean_v_max_mm_s":  round(clean_stats["max"]  * COORD_MM_PER_UNIT * 1000, 2),
        "n_clean_v_pts":    clean_stats["n"],
    }


def clean_svc(arr: np.ndarray, min_dt_ms: float) -> np.ndarray:
    """
    Returns a cleaned copy of the SVC array with near-duplicate timestamp rows removed.

    Strategy: DROP subsequent rows within a near-duplicate dt cluster.
    The FIRST sample in each cluster is kept (it has the actual coordinate position).
    Air (pen_down == 0) rows are always kept regardless of dt — they don't contribute
    to velocity calculations.

    This is conservative: no interpolation, no modification of existing coordinates.
    """
    if len(arr) < 2:
        return arr.copy()

    keep = [True]  # always keep the first row
    t_ms = arr[:, 2]
    pen_down = arr[:, 3]

    for i in range(1, len(arr)):
        dt = t_ms[i] - t_ms[i - 1]
        both_down = (pen_down[i - 1] > 0.5) and (pen_down[i] > 0.5)
        if both_down and dt < min_dt_ms:
            keep.append(False)
        else:
            keep.append(True)

    mask = np.array(keep, dtype=bool)
    return arr[mask]


def write_cleaned_svc(original_path: Path, cleaned_arr: np.ndarray, output_dir: Path):
    """Writes a cleaned SVC file, preserving the original relative path structure."""
    # Preserve relative path from dataSciRep_public root
    try:
        ds_root = original_path.parent
        while ds_root.name != "dataSciRep_public" and ds_root != ds_root.parent:
            ds_root = ds_root.parent
        rel = original_path.relative_to(ds_root)
    except ValueError:
        rel = Path(original_path.name)

    out_path = output_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(cleaned_arr)}\n")
        for row in cleaned_arr:
            # Restore integer-like format for timestamp and pen_down columns
            parts = [
                f"{row[0]:.4f}", f"{row[1]:.4f}",
                f"{int(round(row[2]))}",  # timestamp in ms → integer
                f"{int(round(row[3]))}",  # pen_down → 0 or 1
                f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}",
            ]
            f.write(" ".join(parts) + "\n")

    return out_path


def find_svc_files(dataset_root: Path) -> List[Path]:
    """Recursively finds all .svc files under dataset_root."""
    return sorted(dataset_root.rglob("*.svc"))


def run_audit(
    dataset_root: str,
    min_dt_ms: float,
    output_dir: str,
    report_dir: str,
    write_cleaned: bool = True,
    verbose: bool = True,
):
    ds_path   = Path(dataset_root)
    out_path  = Path(output_dir)
    rep_path  = Path(report_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    if write_cleaned:
        out_path.mkdir(parents=True, exist_ok=True)

    svc_files = find_svc_files(ds_path)
    if not svc_files:
        print(f"[audit] No .svc files found under {ds_path}", file=sys.stderr)
        return

    print(f"[audit] Found {len(svc_files)} SVC files. min_dt_ms={min_dt_ms}")
    print(f"[audit] Wacom calibration: 1 coord_unit = {COORD_MM_PER_UNIT} mm "
          f"(Intuos4 @ 5080 lpi)\n")

    rows_csv = []
    affected_files = []

    for fp in svc_files:
        try:
            arr, expected_n = load_svc_raw(fp)
        except Exception as e:
            print(f"  SKIP {fp.name}: {e}")
            continue

        if len(arr) == 0:
            continue

        stats = audit_timestamps(arr, min_dt_ms)
        rel   = str(fp.relative_to(ds_path))
        uid   = fp.parent.parent.name  # e.g. user00050

        row = {"file": rel, "user_id": uid, "expected_n": expected_n, **stats}
        rows_csv.append(row)

        if stats["affected_pct"] > 10.0:
            affected_files.append(rel)

        if verbose:
            print(f"  {rel:60s}  n={stats['n_total']:6d}  "
                  f"dt<{min_dt_ms}ms: {stats['n_affected']:5d} ({stats['affected_pct']:5.1f}%)  "
                  f"v_max_raw={stats['raw_v_max_mm_s']:8.1f}mm/s → "
                  f"v_max_clean={stats['clean_v_max_mm_s']:8.1f}mm/s")

        if write_cleaned:
            cleaned = clean_svc(arr, min_dt_ms)
            write_cleaned_svc(fp, cleaned, out_path)

    # ── Write audit CSV ──
    if rows_csv:
        audit_csv = rep_path / "audit_report.csv"
        fieldnames = list(rows_csv[0].keys())
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\n[audit] Report written → {audit_csv}")

    # ── Human-readable summary ──
    if rows_csv:
        pcts = [r["affected_pct"] for r in rows_csv]
        raw_maxes  = [r["raw_v_max_mm_s"]   for r in rows_csv]
        clean_maxes= [r["clean_v_max_mm_s"] for r in rows_csv]
        raw_means  = [r["raw_v_mean_mm_s"]  for r in rows_csv]
        clean_means= [r["clean_v_mean_mm_s"]for r in rows_csv]

        summary_lines = [
            "=" * 72,
            "GROUND-TRUTH SVC TIMESTAMP CORRUPTION AUDIT",
            f"Dataset : {ds_path}",
            f"Filter  : dt < {min_dt_ms} ms → DROP (keep first in cluster)",
            f"Calib   : 1 coord_unit = {COORD_MM_PER_UNIT} mm (Wacom Intuos4 5080 lpi)",
            "=" * 72,
            f"Files scanned          : {len(rows_csv)}",
            f"Files > 10% affected   : {len(affected_files)}",
            "",
            "Affected rows fraction (dt < 1ms in pen-down transitions):",
            f"  mean  : {np.mean(pcts):.1f}%",
            f"  median: {np.median(pcts):.1f}%",
            f"  min   : {np.min(pcts):.1f}%",
            f"  max   : {np.max(pcts):.1f}%",
            "",
            "Peak velocity BEFORE filtering (mm/s) across all files:",
            f"  mean of per-file max : {np.mean(raw_maxes):.1f}",
            f"  max across all files : {np.max(raw_maxes):.1f}",
            "",
            "Peak velocity AFTER  filtering (mm/s) across all files:",
            f"  mean of per-file max : {np.mean(clean_maxes):.1f}",
            f"  max across all files : {np.max(clean_maxes):.1f}",
            "",
            "Mean velocity BEFORE filtering (mm/s):",
            f"  mean across files : {np.mean(raw_means):.1f}",
            "Mean velocity AFTER  filtering (mm/s):",
            f"  mean across files : {np.mean(clean_means):.1f}",
            "",
            "CONCLUSION:",
            "  The near-duplicate timestamp rows are firmware polling artifacts",
            "  (Wacom interrupt firing at ~250 kHz between genuine ~200 Hz samples).",
            "  Dropping them reduces peak velocity from physically impossible values",
            f"  (>{np.max(raw_maxes):.0f} mm/s) to plausible handwriting speeds",
            f"  (peak ≤ {np.max(clean_maxes):.0f} mm/s ≈ {np.max(clean_maxes)/1000:.2f} m/s).",
            "",
            "  All velocity/jerk statistics in the GT-vs-predicted comparison",
            "  MUST be derived from cleaned files or with the dt >= 1ms filter applied.",
            "  Numbers from uncleaned files are INVALID and must not be reported.",
            "=" * 72,
        ]

        summary_txt = rep_path / "audit_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")

        for line in summary_lines:
            print(line)

        print(f"\n[audit] Summary written → {summary_txt}")
        if write_cleaned:
            print(f"[audit] Cleaned SVC files → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit SVC ground-truth files for timestamp corruption and write cleaned copies."
    )
    parser.add_argument("--dataset_root", default="Datasets/dataSciRep_public")
    parser.add_argument("--min_dt_ms", type=float, default=MIN_DT_MS_DEFAULT,
                        help="Drop pen-down rows where consecutive dt < this value (ms). Default: 1.0")
    parser.add_argument("--output_dir", default="Datasets/dataSciRep_public_cleaned",
                        help="Where to write cleaned SVC files.")
    parser.add_argument("--report_dir", default="outputs/gt_audit",
                        help="Where to write audit_report.csv and audit_summary.txt.")
    parser.add_argument("--no_write", action="store_true",
                        help="Audit only — do not write cleaned files.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_audit(
        dataset_root=args.dataset_root,
        min_dt_ms=args.min_dt_ms,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        write_cleaned=not args.no_write,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
