"""
Data loaders for online handwriting / tablet kinematic datasets.
Standardizes data from dataSciRep_public (.svc) and DiaGraMo (.json)
into a unified SampleData representation.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SampleData:
    dataset_name: str
    sample_id: str
    task_name: str
    points: np.ndarray  # Shape (N, 7): [x, y, t, pen_status, azimuth, tilt, pressure]
    strokes: List[np.ndarray] = field(default_factory=list)  # List of (M, 7) arrays for on-surface strokes
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling_rate_hz: float = 0.0
    total_duration_sec: float = 0.0
    in_air_ratio: float = 0.0

    @property
    def x(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.points[:, 1]

    @property
    def t(self) -> np.ndarray:
        return self.points[:, 2]

    @property
    def pen_status(self) -> np.ndarray:
        return self.points[:, 3]

    @property
    def azimuth(self) -> np.ndarray:
        return self.points[:, 4]

    @property
    def tilt(self) -> np.ndarray:
        return self.points[:, 5]

    @property
    def pressure(self) -> np.ndarray:
        return self.points[:, 6]


def _extract_strokes(points: np.ndarray) -> List[np.ndarray]:
    """
    Extracts contiguous on-surface strokes from points array.
    A stroke is a sequence of consecutive points where pen_status == 1
    and pressure > 0.
    """
    on_surface = (points[:, 3] > 0.5) & (points[:, 6] > 0)
    strokes = []
    current_stroke = []

    for i in range(len(points)):
        if on_surface[i]:
            current_stroke.append(points[i])
        else:
            if len(current_stroke) >= 2:
                strokes.append(np.array(current_stroke, dtype=np.float64))
            current_stroke = []

    if len(current_stroke) >= 2:
        strokes.append(np.array(current_stroke, dtype=np.float64))

    return strokes


def _compute_kinematic_meta(points: np.ndarray) -> Dict[str, float]:
    """Computes basic timing and sampling rate metrics."""
    if len(points) < 2:
        return {"sampling_rate_hz": 0.0, "total_duration_sec": 0.0, "in_air_ratio": 0.0}

    dt = np.diff(points[:, 2])
    positive_dt = dt[dt > 0]
    median_dt = np.median(positive_dt) if len(positive_dt) > 0 else 0.0
    sampling_rate = 1.0 / median_dt if median_dt > 0 else 0.0
    total_dur = float(points[-1, 2] - points[0, 2])
    in_air_count = np.sum((points[:, 3] <= 0.5) | (points[:, 6] <= 0))
    in_air_ratio = float(in_air_count) / float(len(points))

    return {
        "sampling_rate_hz": float(sampling_rate),
        "total_duration_sec": float(total_dur),
        "in_air_ratio": float(in_air_ratio),
    }


def load_svc(filepath: str) -> SampleData:
    """
    Loads an .svc file from dataSciRep_public.
    Format:
      Line 1: N (number of points)
      Lines 2..N+1: x y timestamp pen_down azimuth tilt pressure
    """
    filename = os.path.basename(filepath)
    sample_id = os.path.splitext(filename)[0]

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        expected_n = int(first_line) if first_line.isdigit() else None
        
        raw_rows = []
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 7:
                raw_rows.append([float(p) for p in parts[:7]])

    raw_arr = np.array(raw_rows, dtype=np.float64)
    if len(raw_arr) == 0:
        raise ValueError(f"Empty or invalid SVC file: {filepath}")

    # Standardize columns: [x, y, t, pen_status, azimuth, tilt, pressure]
    # Timestamp in SVC is typically in milliseconds
    t_raw = raw_arr[:, 2]
    t_sec = (t_raw - t_raw[0]) / 1000.0

    points = np.column_stack([
        raw_arr[:, 0],      # x
        raw_arr[:, 1],      # y
        t_sec,              # t (seconds)
        raw_arr[:, 3],      # pen_down (1 = down, 0 = up)
        raw_arr[:, 4],      # azimuth
        raw_arr[:, 5],      # tilt
        raw_arr[:, 6]       # pressure
    ])

    strokes = _extract_strokes(points)
    meta_stats = _compute_kinematic_meta(points)

    return SampleData(
        dataset_name="dataSciRep_public",
        sample_id=sample_id,
        task_name="handwriting_hw",
        points=points,
        strokes=strokes,
        metadata={"filepath": filepath, "expected_n": expected_n, "loaded_n": len(points)},
        sampling_rate_hz=meta_stats["sampling_rate_hz"],
        total_duration_sec=meta_stats["total_duration_sec"],
        in_air_ratio=meta_stats["in_air_ratio"],
    )


def load_diagramo_json(filepath: str) -> SampleData:
    """
    Loads a JSON file from DiaGraMo project (e.g. TSK4 or TSK16).
    Schema:
      meta_data: {...}
      data: {x: [...], y: [...], time: [...], pen_status: [...], azimuth: [...], tilt: [...], pressure: [...]}
    """
    filename = os.path.basename(filepath)
    sample_id = os.path.splitext(filename)[0]

    with open(filepath, "r", encoding="utf-8") as f:
        content = json.load(f)

    meta = content.get("meta_data", {})
    data_dict = content.get("data", {})

    required_keys = ["x", "y", "time", "pen_status", "azimuth", "tilt", "pressure"]
    for k in required_keys:
        if k not in data_dict:
            raise KeyError(f"Missing required key '{k}' in DiaGraMo JSON: {filepath}")

    n_points = len(data_dict["x"])
    x_arr = np.array(data_dict["x"], dtype=np.float64)
    y_arr = np.array(data_dict["y"], dtype=np.float64)
    t_arr = np.array(data_dict["time"], dtype=np.float64)
    t_sec = t_arr - t_arr[0]  # Normalize start to 0.0s
    pen_arr = np.array(data_dict["pen_status"], dtype=np.float64)
    az_arr = np.array(data_dict["azimuth"], dtype=np.float64)
    tilt_arr = np.array(data_dict["tilt"], dtype=np.float64)
    p_arr = np.array(data_dict["pressure"], dtype=np.float64)

    points = np.column_stack([x_arr, y_arr, t_sec, pen_arr, az_arr, tilt_arr, p_arr])
    strokes = _extract_strokes(points)
    meta_stats = _compute_kinematic_meta(points)

    # Infer task name from filename
    task_name = "TSK"
    if "TSK4" in sample_id:
        task_name = "TSK4_dictation"
    elif "TSK16" in sample_id:
        task_name = "TSK16_copy_sentence"
    elif "TSK" in sample_id:
        task_name = sample_id.split("_")[1] if len(sample_id.split("_")) > 1 else "TSK"

    return SampleData(
        dataset_name="DiaGraMo",
        sample_id=sample_id,
        task_name=task_name,
        points=points,
        strokes=strokes,
        metadata={"filepath": filepath, "meta_data": meta, "loaded_n": n_points},
        sampling_rate_hz=meta_stats["sampling_rate_hz"],
        total_duration_sec=meta_stats["total_duration_sec"],
        in_air_ratio=meta_stats["in_air_ratio"],
    )
