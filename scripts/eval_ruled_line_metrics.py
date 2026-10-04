"""
Quantitative Benchmark: Ruled-Line Removal (Step 4)
Pixel-level precision/recall/F1 against synthetic ground-truth masks.

Design requirements:
- Statistically valid: 50 Monte Carlo synthetic samples per paper type.
- Hundreds of crossing intersection points per run (not 5-12).
- Crossing preservation rate reported as the PRIMARY metric for Case C.
- Per-sample statistics reported (mean ± std), not just aggregates.
"""

import os
import sys
import random
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.branch_a.line_removal import detect_and_remove_ruled_lines


def _draw_ruled_lines(mask, y_positions, x0=50, x1=950):
    """Draw 2-pixel-tall horizontal ruled lines."""
    for ry in y_positions:
        ry = int(ry)
        if 0 <= ry < mask.shape[0] - 1:
            mask[ry:ry + 2, x0:x1] = 1


def _draw_grid(mask, row_step, col_step, h=600, w=1000, margin=50):
    """Draw 1px grid lines (horizontal + vertical)."""
    for ry in range(margin, h - margin, row_step):
        mask[ry, margin:w - margin] = 1
    for cx in range(margin, w - margin, col_step):
        mask[margin:h - margin, cx] = 1


def _draw_random_strokes(mask, rng, n_strokes=6, h=600, w=1000):
    """
    Draw synthetic handwriting strokes as thick diagonal/curved lines
    so they cross multiple ruled lines for a realistic crossing count.
    Each stroke is a thick polyline with 2-3 segments.
    """
    for _ in range(n_strokes):
        x0 = rng.randint(80, 300)
        y0 = rng.randint(60, h - 80)
        length = rng.randint(120, 350)
        angle_rad = rng.uniform(-0.3, 0.3)  # near-horizontal handwriting
        x1 = min(x0 + length, w - 60)
        y1 = int(y0 + length * np.tan(angle_rad))
        y1 = max(30, min(h - 30, y1))

        # Draw thick line (3px wide) via bresenham-style rasterization
        pts = _line_points(y0, x0, y1, x1)
        for py, px in pts:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx_ = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx_ < w:
                        mask[ny, nx_] = 1

        # Optional: add a short descender that crosses more lines
        if rng.random() < 0.5:
            xm = (x0 + x1) // 2
            ym = (y0 + y1) // 2
            yd = min(h - 30, ym + rng.randint(40, 90))
            for py, px in _line_points(ym, xm, yd, xm + rng.randint(-10, 10)):
                for dy in range(-1, 2):
                    ny = py + dy
                    if 0 <= ny < h and 0 <= xm < w:
                        mask[ny, px] = 1


def _line_points(y0, x0, y1, x1):
    """Bresenham line rasterization."""
    pts = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    for _ in range(max(dx, dy) + 1):
        pts.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def run_monte_carlo(n_samples=50, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    h, w = 600, 1000

    # ── CASE A: Ruled notebook paper ────────────────────────────────────────
    print("=" * 80)
    print("QUANTITATIVE BENCHMARK: RULED-NOTEBOOK AND GRID LINE REMOVAL (STEP 4)")
    print(f"Monte Carlo N={n_samples} synthetic samples per paper type  |  seed={seed}")
    print("=" * 80)

    case_a_stats = {"tp": [], "fp": [], "fn": [], "prec": [], "rec": [], "f1": [],
                    "cross_total": [], "cross_removed": [], "cross_pres": []}

    for trial in range(n_samples):
        gt_lines = np.zeros((h, w), dtype=np.uint8)
        gt_text = np.zeros((h, w), dtype=np.uint8)

        # Vary line spacing (18-40px) and line count (8-14 lines)
        n_lines = rng.randint(8, 14)
        spacing = h // (n_lines + 1)
        y_starts = [spacing * (i + 1) + rng.randint(-4, 4) for i in range(n_lines)]
        _draw_ruled_lines(gt_lines, y_starts, x0=rng.randint(30, 70), x1=rng.randint(900, 960))

        # Vary stroke count (4-8 strokes), each crossing multiple lines
        _draw_random_strokes(gt_text, rng, n_strokes=rng.randint(4, 8), h=h, w=w)

        composite = (gt_lines | gt_text).astype(np.uint8)
        crossing = (gt_lines & gt_text).astype(np.uint8)
        isolated_lines = (gt_lines & (~gt_text)).astype(np.uint8)

        cleaned, _ = detect_and_remove_ruled_lines(composite)
        removed = ((composite == 1) & (cleaned == 0)).astype(np.uint8)

        tp = int(np.sum(removed & isolated_lines))
        fp = int(np.sum(removed & gt_text))
        fn = int(np.sum((~removed) & isolated_lines))
        cross_total = int(np.sum(crossing))
        cross_removed = int(np.sum(removed & crossing))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cross_pres = (cross_total - cross_removed) / max(cross_total, 1)

        case_a_stats["tp"].append(tp)
        case_a_stats["fp"].append(fp)
        case_a_stats["fn"].append(fn)
        case_a_stats["prec"].append(prec)
        case_a_stats["rec"].append(rec)
        case_a_stats["f1"].append(f1)
        case_a_stats["cross_total"].append(cross_total)
        case_a_stats["cross_removed"].append(cross_removed)
        case_a_stats["cross_pres"].append(cross_pres)

    a_cross_total = int(sum(case_a_stats["cross_total"]))
    a_cross_removed = int(sum(case_a_stats["cross_removed"]))
    a_cross_pres_mean = float(np.mean(case_a_stats["cross_pres"])) * 100
    a_cross_pres_std = float(np.std(case_a_stats["cross_pres"])) * 100

    print(f"\n[Case A: Ruled Notebook Paper — Horizontal Lines Only]  N={n_samples} trials")
    print(f"  Total Crossing Intersection Pixels Evaluated:  {a_cross_total:,}")
    print(f"  Crossing Pixels Incorrectly Removed:           {a_cross_removed}  "
          f"({a_cross_removed / max(a_cross_total, 1) * 100:.2f}%)")
    print(f"  *** Crossing Preservation Rate: {a_cross_pres_mean:.2f}% ± {a_cross_pres_std:.2f}% ***  (PRIMARY METRIC)")
    print(f"  ---")
    print(f"  Mean Precision (isolated line removal):  {np.mean(case_a_stats['prec'])*100:.2f}% ± {np.std(case_a_stats['prec'])*100:.2f}%")
    print(f"  Mean Recall   (isolated line removal):   {np.mean(case_a_stats['rec'])*100:.2f}% ± {np.std(case_a_stats['rec'])*100:.2f}%")
    print(f"  Mean F1-Score (isolated line removal):   {np.mean(case_a_stats['f1'])*100:.2f}% ± {np.std(case_a_stats['f1'])*100:.2f}%")
    print(f"  Mean FP/trial (text pixels removed):     {np.mean(case_a_stats['fp']):.1f} ± {np.std(case_a_stats['fp']):.1f}")

    # ── CASE B: Unlined plain paper ──────────────────────────────────────────
    case_b_fp = []
    for trial in range(n_samples):
        gt_text = np.zeros((h, w), dtype=np.uint8)
        _draw_random_strokes(gt_text, rng, n_strokes=rng.randint(4, 10), h=h, w=w)
        # Add crossbars (horizontal strokes of length 5-15px mimicking letter 't' crossbars)
        for _ in range(rng.randint(3, 8)):
            cx = rng.randint(100, w - 100)
            cy = rng.randint(80, h - 80)
            for px in range(cx - rng.randint(4, 12), cx + rng.randint(4, 12)):
                if 0 <= px < w:
                    gt_text[cy, px] = 1

        cleaned, meta = detect_and_remove_ruled_lines(gt_text)
        removed = ((gt_text == 1) & (cleaned == 0)).astype(np.uint8)
        case_b_fp.append(int(np.sum(removed)))

    print(f"\n[Case B: Unlined Plain Paper — Specificity Test]  N={n_samples} trials")
    print(f"  Mean False Positive Pixels/trial (text removed): {np.mean(case_b_fp):.2f} ± {np.std(case_b_fp):.2f}")
    print(f"  Max FP in any trial:  {max(case_b_fp)}")
    print(f"  Trials with zero FP:  {sum(1 for x in case_b_fp if x == 0)} / {n_samples}")

    # ── CASE C: Grid / Graph Paper ───────────────────────────────────────────
    case_c_stats = {"tp": [], "fp": [], "fn": [], "prec": [], "rec": [], "f1": [],
                    "cross_total": [], "cross_removed": [], "cross_pres": []}

    for trial in range(n_samples):
        gt_grid = np.zeros((h, w), dtype=np.uint8)
        gt_text = np.zeros((h, w), dtype=np.uint8)

        row_step = rng.randint(30, 70)
        col_step = rng.randint(30, 70)
        _draw_grid(gt_grid, row_step=row_step, col_step=col_step, h=h, w=w)
        _draw_random_strokes(gt_text, rng, n_strokes=rng.randint(4, 8), h=h, w=w)

        composite = (gt_grid | gt_text).astype(np.uint8)
        crossing = (gt_grid & gt_text).astype(np.uint8)
        isolated_grid = (gt_grid & (~gt_text)).astype(np.uint8)

        cleaned, _ = detect_and_remove_ruled_lines(composite, detect_grid=True)
        removed = ((composite == 1) & (cleaned == 0)).astype(np.uint8)

        tp = int(np.sum(removed & isolated_grid))
        fp = int(np.sum(removed & gt_text))
        fn = int(np.sum((~removed) & isolated_grid))
        cross_total = int(np.sum(crossing))
        cross_removed = int(np.sum(removed & crossing))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cross_pres = (cross_total - cross_removed) / max(cross_total, 1)

        case_c_stats["tp"].append(tp)
        case_c_stats["fp"].append(fp)
        case_c_stats["fn"].append(fn)
        case_c_stats["prec"].append(prec)
        case_c_stats["rec"].append(rec)
        case_c_stats["f1"].append(f1)
        case_c_stats["cross_total"].append(cross_total)
        case_c_stats["cross_removed"].append(cross_removed)
        case_c_stats["cross_pres"].append(cross_pres)

    c_cross_total = int(sum(case_c_stats["cross_total"]))
    c_cross_removed = int(sum(case_c_stats["cross_removed"]))
    c_cross_pres_mean = float(np.mean(case_c_stats["cross_pres"])) * 100
    c_cross_pres_std = float(np.std(case_c_stats["cross_pres"])) * 100
    c_cross_fail_rate = c_cross_removed / max(c_cross_total, 1) * 100

    print(f"\n[Case C: Grid / Graph Paper — Horizontal + Vertical Rules]  N={n_samples} trials")
    print(f"  Total Crossing Intersection Pixels Evaluated:  {c_cross_total:,}")
    print(f"  >>> {c_cross_removed} of {c_cross_total} grid-line crossings had handwriting ink incorrectly removed")
    print(f"  >>> Crossing Failure Rate: {c_cross_fail_rate:.1f}%  (PRIMARY METRIC)")
    print(f"  *** Crossing Preservation Rate: {c_cross_pres_mean:.2f}% ± {c_cross_pres_std:.2f}% ***")
    print(f"  ---")
    print(f"  Mean Precision (grid pixel removal):     {np.mean(case_c_stats['prec'])*100:.2f}% ± {np.std(case_c_stats['prec'])*100:.2f}%")
    print(f"  Mean Recall   (grid pixel removal):      {np.mean(case_c_stats['rec'])*100:.2f}% ± {np.std(case_c_stats['rec'])*100:.2f}%")
    print(f"  Mean F1-Score (grid pixel removal):      {np.mean(case_c_stats['f1'])*100:.2f}% ± {np.std(case_c_stats['f1'])*100:.2f}%")
    print(f"  Note: Aggregate F1 is background-dominated. Crossing preservation is the")
    print(f"        operationally relevant metric for handwriting ink integrity.")
    print("=" * 80)

    return {
        "case_a": case_a_stats,
        "case_b_fp": case_b_fp,
        "case_c": case_c_stats,
        "totals": {
            "a_cross_total": a_cross_total,
            "a_cross_removed": a_cross_removed,
            "a_cross_pres_mean": a_cross_pres_mean,
            "a_cross_pres_std": a_cross_pres_std,
            "c_cross_total": c_cross_total,
            "c_cross_removed": c_cross_removed,
            "c_cross_pres_mean": c_cross_pres_mean,
            "c_cross_pres_std": c_cross_pres_std,
            "c_cross_fail_rate": c_cross_fail_rate,
        }
    }


if __name__ == "__main__":
    run_monte_carlo(n_samples=50, seed=42)
