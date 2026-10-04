"""
Batch Feature Extraction CLI — scripts/batch_extract.py

Usage:
    python scripts/batch_extract.py --input_dir <folder> --output_csv <file.csv> [--output_json_dir <dir>]

Takes a folder of handwriting images (JPEG/PNG/BMP/TIFF) and runs
extract_from_image() on each, writing one CSV of feature vectors + quality
flags per image, plus optionally one JSON per sample.

IMPORTANT: This script does not classify, score risk, or output diagnostic
verdicts.  All kinematic outputs are ESTIMATES — see pipeline.py docstring.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import extract_from_image, CANONICAL_FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(input_dir: Path):
    images = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(Path(root) / fname)
    return images


def batch_extract(
    input_dir: str,
    output_csv: str,
    output_json_dir: str = None,
    remove_ruled_lines: bool = True,
    compute_kinematics: bool = True,
    verbose: bool = True,
):
    """
    Process all handwriting images in input_dir and write results to output_csv.
    Returns a list of result dicts (one per image).
    """
    in_path = Path(input_dir)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_json_dir:
        json_dir = Path(output_json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
    else:
        json_dir = None

    images = find_images(in_path)
    if not images:
        print(f"[batch_extract] No images found in {in_path}", file=sys.stderr)
        return []

    if verbose:
        print(f"[batch_extract] Found {len(images)} images in {in_path}")
        print(f"[batch_extract] Feature schema version: {FEATURE_SCHEMA_VERSION}")
        print(f"[batch_extract] Output CSV: {out_csv}")

    all_names = [item["name"] for item in CANONICAL_FEATURE_SCHEMA]
    flag_keys = [
        "low_ink", "ruled_residual", "low_resolution",
        "too_few_strokes", "no_text_lines_found", "abnormal_h_med",
        "line_removal_applied", "unreliable_extraction",
    ]
    conf_keys = [f"conf_{n}" for n in all_names]

    results = []
    t_total = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(
            ["sample_id", "image_path"] +
            all_names +
            flag_keys +
            conf_keys +
            ["h_med_px", "stroke_count", "line_count", "elapsed_s",
             "pipeline_version", "schema_version"]
        )

        for idx, img_path in enumerate(images):
            t0 = time.time()
            sample_id = img_path.stem

            try:
                result = extract_from_image(
                    str(img_path),
                    sample_id=sample_id,
                    remove_ruled_lines=remove_ruled_lines,
                    compute_kinematics=compute_kinematics,
                )
            except Exception as exc:
                if verbose:
                    print(f"  [{idx+1}/{len(images)}] ERROR {img_path.name}: {exc}")
                result = {
                    "feature_vector_20d": [None] * 20,
                    "quality_flags": {"unreliable_extraction": True},
                    "per_feature_confidence": {n: 0.0 for n in all_names},
                    "metadata": {
                        "failure_reason": str(exc),
                        "h_med_px": None, "recovered_stroke_count": 0,
                        "text_line_count": 0, "elapsed_seconds": time.time() - t0,
                        "pipeline_version": "2.0.0",
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    },
                }

            fvec = result.get("feature_vector_20d", [None] * 20)
            if hasattr(fvec, "tolist"):
                fvec = fvec.tolist()
            fvec_safe = [("" if (v is None or (isinstance(v, float) and v != v)) else round(float(v), 6))
                         for v in fvec]

            qf = result.get("quality_flags", {})
            pfc = result.get("per_feature_confidence", {})
            meta = result.get("metadata", {})

            flag_vals = [int(bool(qf.get(k, False))) if qf.get(k) is not None else ""
                         for k in flag_keys]
            conf_vals = [round(float(pfc.get(n, 0.0)), 4) for n in all_names]

            row = (
                [sample_id, str(img_path)] +
                fvec_safe + flag_vals + conf_vals +
                [
                    meta.get("h_med_px", ""),
                    meta.get("recovered_stroke_count", ""),
                    meta.get("text_line_count", ""),
                    round(meta.get("elapsed_seconds", 0.0), 3),
                    meta.get("pipeline_version", "2.0.0"),
                    meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
                ]
            )
            writer.writerow(row)
            csvfile.flush()

            if json_dir:
                payload = result.get("export_payload", {})
                json_path = json_dir / f"{sample_id}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    # NaN-safe JSON serialisation
                    import math
                    def _safe(obj):
                        if isinstance(obj, float) and math.isnan(obj):
                            return None
                        return obj
                    # Use default=str as a last-resort fallback
                    json.dump(payload, jf, indent=2, default=str)

            results.append(result)

            if verbose:
                unreliable = qf.get("unreliable_extraction", True)
                status = "UNRELIABLE" if unreliable else "ok"
                elapsed = round(time.time() - t0, 2)
                strk = meta.get("recovered_stroke_count", "?")
                print(f"  [{idx+1:4d}/{len(images)}] {status} {img_path.name}  "
                      f"strokes={strk}  t={elapsed}s")

    total_t = round(time.time() - t_total, 1)
    n_ok = sum(1 for r in results
               if not r.get("quality_flags", {}).get("unreliable_extraction", True))
    n_fail = len(results) - n_ok

    if verbose:
        print(f"\n[batch_extract] Done. {len(results)} images processed in {total_t}s")
        print(f"  Reliable:   {n_ok} / {len(results)}")
        print(f"  Unreliable: {n_fail} / {len(results)}")
        print(f"  CSV:  {out_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction from handwriting images. "
                    "Outputs a CSV of feature vectors and quality flags. "
                    "No classification or risk scoring is performed."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory of handwriting images (scanned recursively).")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV file path.")
    parser.add_argument("--output_json_dir", default=None,
                        help="Optional: directory to write per-sample JSON exports.")
    parser.add_argument("--no_line_removal", action="store_true",
                        help="Disable ruled-line removal (faster, less robust).")
    parser.add_argument("--no_kinematics", action="store_true",
                        help="Skip kinematic feature estimation.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-image progress output.")
    args = parser.parse_args()

    batch_extract(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_json_dir=args.output_json_dir,
        remove_ruled_lines=not args.no_line_removal,
        compute_kinematics=not args.no_kinematics,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
