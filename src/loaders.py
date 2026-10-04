"""
Data loaders for online handwriting / tablet kinematic datasets.
Standardizes data from dataSciRep_public (.svc) and DiaGraMo (.json)
into a unified SampleData representation.

Timestamp deduplication (STEP 1 fix):
  Wacom tablet firmware fires polling interrupts at ~250 kHz between genuine
  ~200 Hz motion samples, producing consecutive rows with dt < 1 ms and
  real displacement.  Computing velocity from these rows yields physically
  impossible values (>2,000 mm/s).  load_svc() and load_diagramo_json() now
  silently drop pen-down rows where consecutive dt < MIN_DT_S (1 ms default)
  and record a 'dt_artifact_rows_dropped' count in sample metadata.
"""

# Minimum legitimate inter-sample interval for pen-down rows.
# Rows with dt < MIN_DT_S between consecutive pen-down samples are treated as
# firmware polling artifacts and dropped before any kinematic computation.
MIN_DT_S: float = 1e-3  # 1 ms

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


def _dedup_timestamps(points: np.ndarray) -> tuple:
    """
    Removes pen-down rows where consecutive dt < MIN_DT_S (firmware polling artifacts).

    Rules:
      - Air rows (pen_status == 0) are always kept — they mark pen-lift boundaries.
      - Only pen-down → pen-down transitions with dt < MIN_DT_S are dropped.
      - The FIRST sample in each near-duplicate cluster is kept (real position);
        subsequent duplicates within the cluster are dropped.
      - No coordinates are modified; this is strictly a row-removal operation.

    Returns (cleaned_points, n_dropped).
    """
    if len(points) < 2:
        return points, 0

    keep = np.ones(len(points), dtype=bool)
    t   = points[:, 2]
    pen = points[:, 3]

    for i in range(1, len(points)):
        dt = t[i] - t[i - 1]
        both_down = (pen[i - 1] > 0.5) and (pen[i] > 0.5)
        if both_down and dt < MIN_DT_S:
            keep[i] = False

    n_dropped = int(np.sum(~keep))
    return points[keep], n_dropped


def load_svc(filepath: str, dedup_timestamps: bool = True) -> SampleData:
    """
    Loads an .svc file from dataSciRep_public.
    Format:
      Line 1: N (number of points)
      Lines 2..N+1: x y timestamp pen_down azimuth tilt pressure

    dedup_timestamps (default True):
      Drops consecutive pen-down rows with dt < MIN_DT_S (1 ms).
      These are firmware polling artifacts that cause physically-impossible
      velocity values when computing dx/dt.  See module docstring.
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
    # Timestamp in SVC is typically in milliseconds → convert to seconds
    t_raw = raw_arr[:, 2]
    t_sec = (t_raw - t_raw[0]) / 1000.0

    points = np.column_stack([
        raw_arr[:, 0],      # x
        raw_arr[:, 1],      # y
        t_sec,              # t (seconds, zero-based)
        raw_arr[:, 3],      # pen_down (1 = on surface, 0 = in air)
        raw_arr[:, 4],      # azimuth
        raw_arr[:, 5],      # tilt
        raw_arr[:, 6],      # pressure
    ])

    # ── Timestamp deduplication (Step 1 fix) ──────────────────────────────
    n_dropped = 0
    if dedup_timestamps:
        points, n_dropped = _dedup_timestamps(points)
    # ──────────────────────────────────────────────────────────────────────

    strokes = _extract_strokes(points)
    meta_stats = _compute_kinematic_meta(points)

    return SampleData(
        dataset_name="dataSciRep_public",
        sample_id=sample_id,
        task_name="handwriting_hw",
        points=points,
        strokes=strokes,
        metadata={
            "filepath": filepath,
            "expected_n": expected_n,
            "loaded_n": len(points),
            "dt_artifact_rows_dropped": n_dropped,
            "min_dt_s_threshold": MIN_DT_S,
        },
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
    if "TSK3" in sample_id:
        task_name = "TSK3_sentence_dictation"
    elif "TSK4" in sample_id:
        task_name = "TSK4_alphabet_dictation"
    elif "TSK15" in sample_id:
        task_name = "TSK15_word_copy"
    elif "TSK16" in sample_id:
        task_name = "TSK16_sentence_copy"
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


def list_diagramo_text_tasks(
    dataset_root: str = "Datasets/DiaGraMo-project"
) -> Dict[str, List[str]]:
    """
    Discovers all available JSON files for the 4 standardized handwriting text tasks:
    TSK3 (Sentence Dictation), TSK4 (Alphabet), TSK15 (Word Copy), TSK16 (Sentence Copy).
    """
    import glob
    tasks = ["TSK3", "TSK4", "TSK15", "TSK16"]
    task_files = {tsk: [] for tsk in tasks}

    for tsk in tasks:
        pattern = os.path.join(dataset_root, "**", f"*{tsk}*.json")
        matches = sorted(glob.glob(pattern, recursive=True))
        task_files[tsk] = matches

    return task_files


def load_diagramo_text_cohort(
    dataset_root: str = "Datasets/DiaGraMo-project",
    tasks: Optional[List[str]] = None,
    max_per_task: int = 5
) -> Dict[str, List[SampleData]]:
    """
    Loads multi-task cohorts across DiaGraMo standardized text tasks.
    """
    if tasks is None:
        tasks = ["TSK3", "TSK4", "TSK15", "TSK16"]

    all_files = list_diagramo_text_tasks(dataset_root)
    cohort: Dict[str, List[SampleData]] = {tsk: [] for tsk in tasks}

    for tsk in tasks:
        files = all_files.get(tsk, [])[:max_per_task]
        for f in files:
            try:
                sample = load_diagramo_json(f)
                cohort[tsk].append(sample)
            except Exception as e:
                print(f"Warning: could not load {f}: {e}")

    return cohort

