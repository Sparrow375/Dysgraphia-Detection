"""
Consolidated Dysgraphia Feature Extraction Pipeline — v3.0.0 (Kinematics Only)
==============================================================================
Pure mathematical and biophysical kinematic feature extraction only.
NOTE: Branch A (BHK static handwriting-scale features) and OCR are owned by
separate teammates. This module is kinematics-only.

IMPORTANT LIMITATION (documented as required):
  A static image contains NO timing information.
  All kinematic outputs (stroke order, direction, pen lifts, relative speed
  profile) are ESTIMATES produced by the Two-Thirds Power Law + Plamondon
  Sigma-Lognormal model applied to the recovered skeleton trajectory.
  They are NOT measurements of actual pen speed, acceleration or timing.
  Every kinematic field carries a ``confidence`` score in [0, 1] that
  reflects geometric quality of the recovered skeleton path, NOT ground-truth
  agreement. Treat magnitudes as relative proxies only.

Strictly does NOT classify, score risk, or output any diagnostic verdict.
Outputs a clean 13-dimensional kinematic feature vector plus per-stroke kinematic
estimates for downstream ensembling with OCR and supervised classifiers.

SCHEMA_VERSION = "3.0.0"
"""

from __future__ import annotations

from typing import Dict, Any, Union, Optional, List, Tuple
from pathlib import Path
import time
import json
import math
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects

from src.core.preprocessing import load_and_binarize, zhang_suen_skeletonize, compute_stroke_width_map
from src.preprocessing_robust import robust_load_image
from src.core.line_removal import detect_and_remove_ruled_lines
from src.core.segmentation import segment_handwriting
from src.branch_b.stroke_recovery import recover_handwriting_trajectory
from src.branch_b.kinematics import extract_kinematic_features, estimate_stroke_velocity
from src.branch_b.calibrated_kinematics import CalibratedKinematicsReconstructor

_CALIBRATOR_INSTANCE: Optional[CalibratedKinematicsReconstructor] = None

def _get_calibrator() -> CalibratedKinematicsReconstructor:
    global _CALIBRATOR_INSTANCE
    if _CALIBRATOR_INSTANCE is None:
        _CALIBRATOR_INSTANCE = CalibratedKinematicsReconstructor()
    return _CALIBRATOR_INSTANCE

FEATURE_SCHEMA_VERSION = "3.0.0"

CANONICAL_FEATURE_SCHEMA: list = [
    {"index": 0, "name": "kin_mean_velocity", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Two-Thirds Power Law", "units": "H_med/s (ESTIMATED)",
     "stability_label": "unstable_low_resolution",
     "instability_note": "Degrades significantly below 300px image height."},
    {"index": 1, "name": "kin_peak_velocity", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Plamondon Sigma-Lognormal Theory", "units": "H_med/s (ESTIMATED)",
     "stability_label": "unstable_low_resolution"},
    {"index": 2, "name": "kin_velocity_skewness", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Plamondon Asymmetric Impulse Response", "units": "dimensionless (ESTIMATED)",
     "stability_label": "unstable_fragmentation",
     "instability_note": "Highly sensitive to stroke fragmentation. See CORRECTIONS.md Issue 1."},
    {"index": 3, "name": "kin_nvi_rate", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Drotar et al. 2016", "units": "inversions/s (ESTIMATED)",
     "stability_label": "unstable_fragmentation",
     "instability_note": "Inversely proportional to mean pixel stroke length. See CORRECTIONS.md Issue 3."},
    {"index": 4, "name": "kin_nvi_per_stroke", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Drotar et al. 2016", "units": "inversions/stroke (ESTIMATED)",
     "stability_label": "unstable_fragmentation"},
    {"index": 5, "name": "kin_nvi_per_h_med", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Normalized Trajectory Smoothness", "units": "1/H_med (ESTIMATED)",
     "stability_label": "unstable_fragmentation"},
    {"index": 6, "name": "kin_jerk_metric", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Teulings et al.", "units": "H_med^2/s^5 (ESTIMATED)",
     "stability_label": "unstable_fragmentation"},
    {"index": 7, "name": "kin_dimensionless_jerk", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Flash & Hogan (1985)", "units": "dimensionless (ESTIMATED)",
     "stability_label": "unstable_fragmentation"},
    {"index": 8, "name": "kin_spatial_roughness_4_8hz", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Deuschl et al. 1998 / Welch PSD on estimated velocity profile",
     "units": "ratio [0,1] (ESTIMATED)",
     "stability_label": "unstable_low_stroke_count",
     "instability_note": "Requires >=10 recovered strokes for PSD stability. NOT a direct tremor measurement."},
    {"index": 9, "name": "kin_pen_lift_count", "category": "Kinematics Continuity",
     "provenance": "Motor Program Continuity & Pen Lifts", "units": "integer count (ESTIMATED)",
     "stability_label": "unstable_fragmentation"},
    {"index": 10, "name": "kin_mean_stroke_length", "category": "Neuromotor Kinematics (ESTIMATED)",
     "provenance": "Motor Coordination Standards", "units": "H_med units (ESTIMATED)",
     "stability_label": "stable"},
    {"index": 11, "name": "kin_ink_width_ratio_mean", "category": "Biophysical Pressure (ESTIMATED)",
     "provenance": "Optical Stroke Width (2*EDT) / H_med", "units": "W/H_med (ESTIMATED)",
     "stability_label": "unstable_under_pen_type",
     "instability_note": "Optical proxy for stylus down-force; sensitive to pen nib and contact response."},
    {"index": 12, "name": "kin_ink_width_ratio_std", "category": "Biophysical Pressure (ESTIMATED)",
     "provenance": "Optical Down-force Variability along Trajectory", "units": "W/H_med (ESTIMATED)",
     "stability_label": "unstable_under_pen_type",
     "instability_note": "Variability in optical stylus down-force proxy along continuous strokes."},
]

QUALITY_FLAG_DESCRIPTIONS: dict = {
    "low_ink": "Fewer than 500 ink pixels or ink density <0.1%. Feature vector unreliable.",
    "ruled_residual": "Ruled or grid paper detected. Line removal applied; residual may remain.",
    "low_resolution": "Shortest dim <400px or total pixels <300k. Resolution is low.",
    "blur": "Reserved — not yet implemented.",
    "skew_corrected": "Reserved — deskew not yet implemented.",
    "multi_line_aggregated": "Features aggregated across multiple text lines using median/IQR.",
    "too_few_strokes": "Fewer than 5 strokes recovered. Kinematic features are unreliable.",
    "unreliable_extraction": "One or more critical quality flags set. Do not use feature vector.",
    "no_text_lines_found": "Segmentation found zero text lines. H_med fallback was used.",
    "abnormal_h_med": "H_med <5px or >60% image height. Character height estimate suspect.",
    "line_removal_applied": "Ruled/grid lines detected and removed before feature extraction.",
}


def _stroke_confidence(stroke: np.ndarray, h_med: float) -> float:
    """Geometric path quality score [0-1]. NOT ground-truth velocity accuracy."""
    n = len(stroke)
    if n < 3:
        return 0.05
    diffs = np.diff(stroke, axis=0)
    seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    arc_h_med = float(np.sum(seg_lens)) / max(float(h_med), 1.0)
    len_conf = min(arc_h_med / 3.0, 1.0)
    if n >= 5:
        dx = np.gradient(stroke[:, 0])
        dy = np.gradient(stroke[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        speed_sq = dx ** 2 + dy ** 2
        denom = np.maximum(speed_sq ** 1.5, 1e-6)
        kappa = np.abs(dx * ddy - dy * ddx) / denom
        smooth_conf = float(np.exp(-0.5 * float(np.mean(kappa)) * h_med))
    else:
        smooth_conf = 0.3
    return float(np.clip(0.5 * len_conf + 0.5 * smooth_conf, 0.05, 1.0))


def _stroke_direction_deg(stroke: np.ndarray) -> float:
    if len(stroke) < 2:
        return 0.0
    dx = float(stroke[-1, 0] - stroke[0, 0])
    dy = float(stroke[-1, 1] - stroke[0, 1])
    return float(math.degrees(math.atan2(dy, dx)))


def _per_stroke_kinematics(stroke: np.ndarray, dist_map: np.ndarray, h_med: float, idx: int) -> dict:
    conf = _stroke_confidence(stroke, h_med)
    n = len(stroke)
    diffs = np.diff(stroke, axis=0)
    arc_px = float(np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)))
    if n >= 3:
        s, v_est, kappa_est = estimate_stroke_velocity(stroke, h_med=h_med, return_resampled=False)
        mean_v = float(np.mean(v_est))
        peak_v = float(np.max(v_est))
        mean_k = float(np.mean(kappa_est)) if len(kappa_est) == n else 0.0
    else:
        mean_v = peak_v = mean_k = 0.0

    # Optical stroke width and pressure proxy along this stroke
    widths = []
    pressures = []
    if len(stroke) > 0 and dist_map is not None:
        h_dm, w_dm = dist_map.shape
        for pt in stroke:
            ix = int(np.clip(np.round(pt[0]), 0, w_dm - 1))
            iy = int(np.clip(np.round(pt[1]), 0, h_dm - 1))
            w_px = float(2.0 * dist_map[iy, ix])
            widths.append(w_px)
            pressures.append(w_px / max(h_med, 1.0))
    mean_w = float(np.mean(widths)) if widths else 0.0
    min_w = float(np.min(widths)) if widths else 0.0
    max_w = float(np.max(widths)) if widths else 0.0
    mean_p = float(np.mean(pressures)) if pressures else 0.0
    max_p = float(np.max(pressures)) if pressures else 0.0

    return {
        "stroke_index": idx,
        "point_count": n,
        "arc_length_px": round(arc_px, 2),
        "arc_length_h_med": round(arc_px / max(h_med, 1.0), 4),
        "start_xy": [round(float(stroke[0, 0]), 1), round(float(stroke[0, 1]), 1)],
        "end_xy": [round(float(stroke[-1, 0]), 1), round(float(stroke[-1, 1]), 1)],
        "direction_deg": round(_stroke_direction_deg(stroke), 2),
        "mean_ink_width_px": round(mean_w, 2),
        "min_ink_width_px": round(min_w, 2),
        "max_ink_width_px": round(max_w, 2),
        "mean_ink_width_ratio": round(mean_p, 4),
        "max_ink_width_ratio": round(max_p, 4),
        "estimated_mean_velocity_h_med_per_s": round(mean_v, 4),
        "estimated_peak_velocity_h_med_per_s": round(peak_v, 4),
        "estimated_mean_curvature": round(mean_k, 6),
        "confidence": round(conf, 4),
        "confidence_note": (
            "Geometric path quality [0-1]. NOT ground-truth velocity accuracy. "
            "Based on stroke length/H_med and skeleton path smoothness."
        ),
    }


def _per_feature_confidence(qf: dict, n_strokes: int, h_med: float, h_img: int,
                             ink_pixels: int, lines_detected: int) -> dict:
    low_ink = qf.get("low_ink", False)
    low_res = qf.get("low_resolution", False)
    too_few = qf.get("too_few_strokes", False)

    base_kin = 1.0
    if low_ink or too_few:
        base_kin *= 0.05
    if low_res:
        base_kin *= 0.5
    base_kin *= max(min(n_strokes / 10.0, 1.0), 0.05)

    confs = {}
    for item in CANONICAL_FEATURE_SCHEMA:
        name = item["name"]
        stab = item.get("stability_label", "stable")
        c = base_kin
        if stab == "unstable_under_pen_type":
            c *= 0.7
        if stab in ("unstable_fragmentation", "unstable_low_resolution"):
            c *= 0.6
        if stab == "unstable_low_stroke_count" and n_strokes < 10:
            c *= max(n_strokes / 10.0, 0.1)
        confs[name] = round(float(np.clip(c, 0.0, 1.0)), 4)
    return confs


def extract_from_image(
    image,
    sample_id=None,
    remove_ruled_lines: bool = True,
    compute_kinematics: bool = True,
    use_calibration: bool = True,
    task: Optional[str] = None,
) -> dict:
    """
    Single-image entry point. Takes ANY photo or scan of handwriting.

    Returns a dict with keys:
      feature_vector_13d, feature_vector, feature_dict, feature_schema_version,
      per_feature_confidence, quality_flags,
      strokes (per-stroke ESTIMATED kinematics + confidence),
      pen_lifts, metadata, export_payload.

    IMPORTANT: All kin_* outputs are ESTIMATES from the Two-Thirds Power Law
    + Plamondon Sigma-Lognormal model. They are NOT measurements of actual
    pen speed, timing or acceleration.

    Failure policy: never returns silent NaN or zero. On failure, returns
    unreliable_extraction=True with an explicit failure_reason and NaN vector.
    """
    t0 = time.time()
    steps: list = []

    if sample_id is None:
        if hasattr(image, "name"):
            sample_id = str(Path(getattr(image, "name")).stem)
        elif isinstance(image, (str, Path)):
            sample_id = str(Path(image).stem)
        else:
            sample_id = "sample_001"

    gray_image = None
    try:
        robust_data = robust_load_image(image, auto_deskew=True)
        raw_mask = robust_data["binary_mask"]
        init_qf = robust_data["quality_flags"]
        gray_image = robust_data.get("gray_image", None)
        steps.append("robust_load_binarize")
        if init_qf.get("skew_corrected"):
            steps.append(f"deskew({init_qf.get('skew_angle_deg')}deg)")
    except Exception as exc:
        try:
            raw_mask = load_and_binarize(image)
            init_qf = {"blur": None, "skew_corrected": False}
            steps.append("binarize_fallback")
        except Exception as exc2:
            return _failure_result(sample_id, f"binarize_failed:{exc2}", t0)

    h_img, w_img = raw_mask.shape
    ink_px = int(np.sum(raw_mask))

    # Line removal
    line_removed = False
    lines_det = 0
    if remove_ruled_lines:
        try:
            clean_mask, line_meta = detect_and_remove_ruled_lines(raw_mask)
            lines_det = (line_meta.get("horizontal_lines_count", 0) +
                         line_meta.get("vertical_lines_count", 0))
            if lines_det > 0:
                line_removed = True
                steps.append(f"line_removal({lines_det}_lines)")
        except Exception as exc:
            clean_mask = raw_mask.copy()
            line_meta = {"ruled_paper_detected": False, "grid_paper_detected": False}
            steps.append(f"line_removal_skipped:{exc}")
    else:
        clean_mask = raw_mask.copy()
        line_meta = {"ruled_paper_detected": False, "grid_paper_detected": False}

    # Skeletonize
    try:
        skeleton = zhang_suen_skeletonize(clean_mask)
        steps.append("skeletonize")
    except Exception as exc:
        return _failure_result(sample_id, f"skeletonize_failed:{exc}", t0)

    dist_map, skel_widths, _ = compute_stroke_width_map(clean_mask, skeleton)
    skel_px = int(np.sum(skeleton))

    # Segmentation / H_med (required for scale-normalizing kinematics)
    try:
        text_lines = segment_handwriting(clean_mask, remove_ruled_lines=False)
        steps.append("segment")
    except Exception as exc:
        text_lines = []
        steps.append(f"segment_failed:{exc}")

    no_text_lines = len(text_lines) == 0
    mh = [l.median_height for l in text_lines if l.components]
    if mh:
        h_med = float(np.median(mh))
    else:
        lbl, n_lbl = label(clean_mask)
        if n_lbl > 0:
            objs = find_objects(lbl)
            hh = [sl[0].stop - sl[0].start for sl in objs
                  if sl is not None and (sl[0].stop - sl[0].start) >= 4]
            h_med = float(np.median(hh)) if hh else 25.0
        else:
            h_med = 25.0
        steps.append("h_med_fallback")
    h_med = max(h_med, 1.0)

    # Stroke recovery
    if skel_px > 0:
        try:
            recovered = recover_handwriting_trajectory(skeleton, h_med=h_med)
            steps.append(f"stroke_recovery({len(recovered)}_strokes)")
        except Exception as exc:
            recovered = []
            steps.append(f"stroke_recovery_failed:{exc}")
    else:
        recovered = []
        steps.append("no_skeleton")

    n_strokes = len(recovered)

    # Quality flags
    low_ink = (ink_px < 500) or (ink_px / max(h_img * w_img, 1) < 0.001)
    low_res = min(w_img, h_img) < 400 or (w_img * h_img) < 300_000
    too_few = n_strokes < 5
    abn_h = (h_med < 5.0) or (h_med > 0.6 * h_img)
    unreliable = low_ink or (n_strokes == 0) or abn_h

    qf: dict = {
        "low_ink": bool(low_ink or init_qf.get("low_ink", False)),
        "ruled_residual": bool(line_meta.get("ruled_paper_detected", False) or
                               line_meta.get("grid_paper_detected", False)),
        "low_resolution": bool(low_res or init_qf.get("low_resolution", False)),
        "blur": init_qf.get("blur", None),
        "blur_score": init_qf.get("blur_score", None),
        "skew_corrected": bool(init_qf.get("skew_corrected", False)),
        "skew_angle_deg": init_qf.get("skew_angle_deg", 0.0),
        "multi_line_aggregated": bool(len(text_lines) >= 2),
        "too_few_strokes": bool(too_few),
        "no_text_lines_found": bool(no_text_lines),
        "abnormal_h_med": bool(abn_h),
        "line_removal_applied": bool(line_removed),
        "unreliable_extraction": bool(unreliable),
    }

    # Kinematic features (ESTIMATED & CALIBRATED)
    kin_res = {}
    if compute_kinematics and recovered:
        # Determine task if unspecified
        task_id = task
        if task_id is None:
            s_name = str(sample_id or "")
            task_id = "S2" if ("_S2" in s_name or "_s2" in s_name) else "S1"

        try:
            if use_calibration:
                reconstructor = _get_calibrator()
                kin_res = reconstructor.extract_calibrated_features(
                    recovered, dist_map, gray_image=gray_image, h_med=h_med, task=task_id
                )
                steps.append("kinematics_calibrated")
            else:
                kin_res = extract_kinematic_features(recovered, dist_map, h_med=h_med)
                steps.append("kinematics_estimated")
        except Exception as exc:
            steps.append(f"kinematics_failed:{exc}")

    kin_names = [item["name"] for item in CANONICAL_FEATURE_SCHEMA]

    if kin_res:
        kin_vec = np.array([
            kin_res.get("mean_velocity", 0.0),
            kin_res.get("peak_velocity", 0.0),
            kin_res.get("velocity_skewness", 0.0),
            kin_res.get("nvi_rate", 0.0),
            kin_res.get("nvi_per_stroke", 0.0),
            kin_res.get("nvi_per_h_med", 0.0),
            kin_res.get("jerk_metric", 0.0),
            kin_res.get("dimensionless_jerk", 0.0),
            kin_res.get("spatial_roughness_4_8hz", 0.0),
            float(kin_res.get("pen_lift_count", n_strokes)),
            kin_res.get("mean_stroke_length", 0.0),
            kin_res.get("ink_width_ratio_mean", 0.0),
            kin_res.get("ink_width_ratio_std", 0.0),
        ], dtype=np.float64)
        kin_metrics = {k: v for k, v in kin_res.items()
                       if k not in ("reconstructed_v_full", "reconstructed_p_full")}
    else:
        nan_kin = float("nan")
        kin_vec = np.full(len(kin_names), nan_kin, dtype=np.float64)
        reason = ("no_strokes" if not recovered else
                  "not_requested" if not compute_kinematics else "failed")
        kin_metrics = {n: nan_kin for n in kin_names}
        kin_metrics["_reason"] = reason
        qf["unreliable_extraction"] = True

    combined = kin_vec
    all_names = kin_names
    feat_dict = {n: float(v) for n, v in zip(all_names, combined)}

    # Per-stroke kinematics
    strokes_out = [_per_stroke_kinematics(stk, dist_map, h_med, i)
                   for i, stk in enumerate(recovered)]

    # Pen-lifts
    if recovered:
        orient_ok = sum(1 for s in recovered
                        if -90 <= _stroke_direction_deg(s) <= 90)
        so_conf = round(orient_ok / max(len(recovered), 1), 3)
    else:
        so_conf = 0.0

    pen_lifts = {
        "count": n_strokes,
        "stroke_order_confidence": so_conf,
        "confidence_note": (
            "Fraction of strokes oriented left-right/top-down per reading-direction prior. "
            "NOT a measure of true pen-lift accuracy."
        ),
    }

    pfc = _per_feature_confidence(qf, n_strokes, h_med, h_img, ink_px, lines_det)

    elapsed = round(time.time() - t0, 3)
    metadata = {
        "sample_id": sample_id,
        "image_size_px": [w_img, h_img],
        "h_med_px": round(h_med, 2),
        "ink_pixels": ink_px,
        "skeleton_pixels": skel_px,
        "text_line_count": len(text_lines),
        "character_component_count": sum(len(l.components) for l in text_lines),
        "lines_detected": lines_det,
        "recovered_stroke_count": n_strokes,
        "processing_steps": steps,
        "elapsed_seconds": elapsed,
        "pipeline_version": DysgraphiaFeaturePipeline.PIPELINE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "kinematic_note": (
            "All kin_* outputs are ESTIMATES from the Two-Thirds Power Law + "
            "Plamondon Sigma-Lognormal model. NOT measurements of actual pen timing."
        ),
    }

    def _js(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, np.floating):
            f = float(v)
            return None if math.isnan(f) else f
        if isinstance(v, np.integer):
            return int(v)
        return v

    export_payload = {
        "sample_id": sample_id,
        "pipeline_version": DysgraphiaFeaturePipeline.PIPELINE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_vector": [_js(x) for x in combined],
        "feature_names": all_names,
        "feature_dict": {k: _js(v) for k, v in feat_dict.items()},
        "per_feature_confidence": pfc,
        "quality_flags": qf,
        "metadata": metadata,
        "null_means_nan": True,
    }

    return {
        "feature_vector_13d": combined,
        "feature_vector": combined,
        "feature_names": all_names,
        "feature_dict": feat_dict,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "per_feature_confidence": pfc,
        "quality_flags": qf,
        "recovered_strokes": recovered,
        "strokes": strokes_out,
        "pen_lifts": pen_lifts,
        "metadata": metadata,
        "export_payload": export_payload,
        "kinematic_vector": kin_vec,
        "kinematic_feature_names": all_names,
        "kinematic_metrics": kin_metrics,
        "combined_vector": combined,
        "combined_feature_names": all_names,
        "image_scale_metadata": {
            "image_width_px": w_img, "image_height_px": h_img,
            "h_med_px": round(h_med, 2),
            "total_ink_pixels": ink_px,
            "ink_density": round(ink_px / max(h_img * w_img, 1), 5),
            "text_line_count": len(text_lines),
            "character_component_count": sum(len(l.components) for l in text_lines),
        },
        "visual_artifacts": {
            "binary_mask": clean_mask,
            "raw_binary_mask": raw_mask,
            "skeleton": skeleton,
            "dist_map": dist_map,
            "skel_widths": skel_widths,
            "text_lines": text_lines,
            "recovered_strokes": recovered,
            "line_removal_meta": line_meta,
            "velocity_profile": kin_res.get("reconstructed_v_full", np.array([])),
            "pressure_profile": kin_res.get("reconstructed_p_full", np.array([])),
        },
    }


def _failure_result(sample_id: str, reason: str, t0: float) -> dict:
    nan13 = np.full(13, float("nan"), dtype=np.float64)
    all_names = [item["name"] for item in CANONICAL_FEATURE_SCHEMA]
    return {
        "feature_vector_13d": nan13,
        "feature_vector": nan13,
        "feature_names": all_names,
        "feature_dict": {n: float("nan") for n in all_names},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "per_feature_confidence": {n: 0.0 for n in all_names},
        "quality_flags": {
            "low_ink": False, "ruled_residual": False, "low_resolution": False,
            "blur": None, "skew_corrected": False, "multi_line_aggregated": False,
            "too_few_strokes": True, "no_text_lines_found": True,
            "abnormal_h_med": True, "line_removal_applied": False,
            "unreliable_extraction": True,
        },
        "strokes": [],
        "pen_lifts": {"count": 0, "stroke_order_confidence": 0.0,
                      "confidence_note": "extraction_failed"},
        "metadata": {
            "sample_id": sample_id, "failure_reason": reason,
            "elapsed_seconds": round(time.time() - t0, 3),
            "pipeline_version": DysgraphiaFeaturePipeline.PIPELINE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        "export_payload": {
            "sample_id": sample_id, "failure_reason": reason,
            "feature_vector": [None] * 13, "quality_flags": {"unreliable_extraction": True},
        },
        "kinematic_vector": nan13,
        "kinematic_feature_names": all_names,
        "kinematic_metrics": {"failure_reason": reason},
        "combined_vector": nan13,
        "combined_feature_names": all_names,
        "image_scale_metadata": {},
        "visual_artifacts": {},
    }


class DysgraphiaFeaturePipeline:
    """
    Unified kinematic feature extractor for dysgraphia handwriting analysis.
    The canonical entry point is the module-level extract_from_image() function.
    This class wraps it for callers holding a pipeline instance.
    """
    PIPELINE_VERSION = "3.0.0"
    CANONICAL_FEATURE_NAMES = [item["name"] for item in CANONICAL_FEATURE_SCHEMA]
    KINEMATIC_FEATURE_NAMES = CANONICAL_FEATURE_NAMES

    def __init__(self, compute_kinematics: bool = True):
        self.compute_kinematics = compute_kinematics

    def extract(self, image_input, compute_kinematics=None, remove_ruled_lines=True, sample_id=None):
        do_kin = self.compute_kinematics if compute_kinematics is None else compute_kinematics
        return extract_from_image(image_input, sample_id=sample_id,
                                  remove_ruled_lines=remove_ruled_lines,
                                  compute_kinematics=do_kin)

    def export_sample(self, image_input, sample_id=None, output_path=None):
        results = self.extract(image_input, sample_id=sample_id)
        payload = results["export_payload"]
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            elif out_p.suffix.lower() == ".csv":
                import csv
                with open(out_p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    fnames = payload.get("feature_names", self.CANONICAL_FEATURE_NAMES)
                    flag_keys = list(payload.get("quality_flags", {}).keys())
                    w.writerow(["sample_id"] + fnames + flag_keys)
                    fvec = ["" if v is None else v
                            for v in payload.get("feature_vector", [None] * 13)]
                    flags = list(payload.get("quality_flags", {}).values())
                    w.writerow([payload.get("sample_id", "unknown")] + fvec + flags)
        return payload

    run_pipeline = extract
