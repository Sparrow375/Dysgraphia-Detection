"""
pipeline.py - orchestrates steps 1-14 for one image and for batches.

CLI:
    python -m bhk_features IMG_OR_DIR [IMG_OR_DIR ...] --out results/ [--config cfg.json] [--dpi 300] [--no-debug]

Outputs (in --out):
    bhk_features.csv     one row per image (status, scale info, per-feature status/score/raw values)
    bhk_features.json    one entry per image with full raw dicts, notes and the conversion rule of every feature
    config_used.json     the exact configuration used
    debug/<stem>_0N_*.png  cleaned image, lines, words/components, final measurement overlay
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import BHKConfig
from . import preprocessing as P
from . import segmentation as S
from . import features as F
from . import scoring as SC
from . import visualization as V

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def process_image(path, cfg: BHKConfig, out_dir):
    """Run the full pipeline on one image. Returns the full (nested) record; never raises."""
    path = Path(path)
    out_dir = Path(out_dir)
    dbg = out_dir / "debug"
    if cfg.save_debug_images:
        dbg.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    rec = dict(image=path.name, status="ok", reason="", features={})
    t0 = time.time()
    try:
        # 1. load + validate
        bgr = P.load_and_validate(path, cfg)
        rec["height_px"], rec["width_px"] = bgr.shape[:2]
        # 2. grayscale + denoise (+ illumination flattening)
        gray = P.to_gray_denoised(bgr, cfg)
        norm = P.normalise_illumination(gray, cfg)
        # 3-5. ruling, margin, clean binary
        ink, thr = P.binarise(norm, cfg)
        c1, ruling_mask, ruling_info = P.remove_horizontal_ruling(ink, cfg)
        c2, margin_mask, margin_info = P.remove_vertical_margin(c1, cfg)
        clean = P.clean_binary(c2, ruling_info, cfg)
        rec.update(ink_threshold=thr, n_rulings=ruling_info["n_lines"], ruling_pitch_px=ruling_info["pitch_px"],
                   ruling_thickness_px=ruling_info["thickness_px"], ruling_angle_deg=ruling_info["angle_deg"],
                   n_margin_lines=margin_info["n_lines"])
        # 6-7. lines + words
        seg = S.segment(clean, cfg)
        rec.update(n_components=len(seg.comps), n_lines=len(seg.lines), n_words=len(seg.words),
                   ref_h_px=seg.ref_h, text_skew_deg=seg.skew_deg, segmentation_ok=bool(seg.ok))
        # 8. debug images (always written, also on failure) - BHK features only if segmentation succeeded
        if cfg.save_debug_images:
            V.save_cleaned(dbg / f"{stem}_01_cleaned.png", clean, ruling_mask, margin_mask)
            if seg.lines:
                V.save_lines(dbg / f"{stem}_02_lines.png", clean, seg)
                V.save_words(dbg / f"{stem}_03_words_components.png", clean, seg)
        if not seg.ok:
            rec["status"], rec["reason"] = "segmentation_failed", seg.reason
            return rec
        # 9-11. features + scores
        ctx = F.FeatureContext(clean, norm, ink, seg, ruling_info, margin_info, cfg)
        results = F.extract_all(ctx)
        scores = {fr.name: SC.score_feature(fr, cfg) for fr in results}
        for fr in results:
            rec["features"][fr.name] = dict(status=fr.status, raw=fr.raw, notes=fr.notes, **scores[fr.name])
        # 13. final overlay
        if cfg.save_debug_images:
            V.save_final(dbg / f"{stem}_04_bhk_measurements.png", bgr, seg, results, scores, ctx, ruling_info, margin_info)
    except Exception as e:  # keep the batch going; the reason is recorded in the output
        rec["status"] = "invalid_image" if isinstance(e, ValueError) else "error"
        rec["reason"] = f"{type(e).__name__}: {e}"
        if rec["status"] == "error":
            rec["traceback"] = traceback.format_exc()
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def flatten(rec):
    """One flat CSV row per image."""
    row = {k: v for k, v in rec.items() if k not in ("features", "traceback")}
    for name, f in rec.get("features", {}).items():
        row[f"{name}__status"] = f["status"]
        row[f"{name}__score_status"] = f["score_status"]
        row[f"{name}__score"] = f["score"]
        for k, v in f["raw"].items():
            row[f"{name}__raw__{k}"] = json.dumps(v, cls=_Enc) if isinstance(v, (list, dict, tuple)) else v
    return row


def collect_images(inputs):
    files = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] not found: {p}", file=sys.stderr)
    return files


def run_batch(inputs, out_dir, cfg: BHKConfig):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in collect_images(inputs):
        print(f"[bhk] {f.name} ...", flush=True)
        r = process_image(f, cfg, out_dir)
        print(f"      -> {r['status']} {r['reason']}  ({r.get('seconds', '?')}s)", flush=True)
        recs.append(r)
    rows = [flatten(r) for r in recs]
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(out_dir / "bhk_features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "bhk_features.json", "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=2, cls=_Enc)
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, cls=_Enc)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description="BHK-inspired static handwriting features (classical image processing)")
    ap.add_argument("inputs", nargs="+", help="image files and/or directories")
    ap.add_argument("--out", default="bhk_out")
    ap.add_argument("--config", help="JSON file overriding BHKConfig fields")
    ap.add_argument("--dpi", type=float, help="scan DPI (enables writing size in mm)")
    ap.add_argument("--no-debug", action="store_true")
    a = ap.parse_args(argv)
    cfg = BHKConfig.from_json(a.config) if a.config else BHKConfig()
    if a.dpi:
        cfg.dpi = a.dpi
    if a.no_debug:
        cfg.save_debug_images = False
    recs = run_batch(a.inputs, a.out, cfg)
    bad = [r for r in recs if r["status"] != "ok"]
    print(f"done: {len(recs) - len(bad)} ok, {len(bad)} not processed -> {a.out}")
    return 0 if recs else 1


if __name__ == "__main__":
    sys.exit(main())
