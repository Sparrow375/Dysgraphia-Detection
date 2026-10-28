"""
experiments/inksight/01_prepare_data.py
Prepare word-level image crops and GT trajectories for InkSight evaluation.

Reads dataSciRep and DiaGraMo TSK4/TSK16 tablet data.
Renders each word from its GT trajectory using our existing renderer.
Applies 5 image variants (clean, ruled, grid, blur, camera-noise).
Outputs a manifest CSV and saves word crops as PNGs.

CRITERIA.md rules enforced:
  - GT = pen-on-paper strokes only (air strokes excluded: pen_status==1 and pressure>0)
  - Sampling by SUBJECT, SEED=42, N=50 (balanced by label: 25 dysgraphic, 25 control)
  - Test set = 20 subjects (10 per label) held out before any metric computation
  - Strict input parity: 224x224 white-padded crops with recorded (scale, pad_x, pad_y)

Usage:
    C:\\Users\\embar\\miniconda3\\envs\\ai_env\\python.exe experiments/inksight/01_prepare_data.py
"""

import sys
import pathlib
import json
import csv
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.loaders import load_svc, _extract_strokes

DATASETS_ROOT = REPO_ROOT / "Datasets"
OUT_DIR = REPO_ROOT / "experiments" / "inksight" / "data"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
SUBJECT_SPLIT_PATH = OUT_DIR / "subject_split.json"

SEED = 42
N_SUBJECTS = 50       # target (balanced: 25 dysgraphic, 25 control)
N_TEST = 20           # subjects held out (10 per label)
TARGET_SIZE = 224     # InkSight input size
WORDS_PER_SUBJECT = 3 # 3 representative words per subject
MIN_STROKE_PTS = 3    # minimum points per stroke to keep

VARIANTS = ["clean", "ruled", "grid", "blur", "camera"]


# ---------------------------------------------------------------------------
# Coordinate transform (matches 00_roundtrip_test.py)
# ---------------------------------------------------------------------------

def compute_pad_resize_params(orig_w: int, orig_h: int, target: int = TARGET_SIZE):
    """
    Returns (scale, pad_x, pad_y) such that:
        x_224 = x_orig * scale + pad_x
        y_224 = y_orig * scale + pad_y
    """
    scale = min(target / orig_w, target / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    if new_h < target:
        pad_y = (target - new_h) // 2
        pad_x = 0
    else:
        pad_x = (target - new_w) // 2
        pad_y = 0

    return scale, pad_x, pad_y


def render_word_image(strokes, bbox, variant: str, rng: np.random.Generator) -> Image.Image:
    """
    Render strokes directly into a 224x224 PIL Image using mapped coordinates.
    strokes: list of np.ndarray shape (N, 2) — x, y in tablet pixel units
    bbox: (x_min, y_min, x_max, y_max) in tablet pixel units
    variant: one of VARIANTS
    Returns: 224x224 padded PIL RGB Image
    """
    x_min, y_min, x_max, y_max = bbox
    orig_w = max(float(x_max - x_min), 1.0)
    orig_h = max(float(y_max - y_min), 1.0)

    scale, pad_x, pad_y = compute_pad_resize_params(orig_w, orig_h, target=TARGET_SIZE)

    # Base: 224x224 white canvas
    img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Effective word area in 224 space
    x0_word = int(pad_x)
    x1_word = int(pad_x + orig_w * scale)
    y0_word = int(pad_y)
    y1_word = int(pad_y + orig_h * scale)

    # Add background pattern
    if variant == "ruled":
        spacing = max(int(orig_h * scale) // 4, 14)
        for y in range(y0_word + spacing // 2, y1_word, spacing):
            draw.line([(0, y), (TARGET_SIZE, y)], fill=(200, 210, 225), width=1)
    elif variant == "grid":
        spacing = max(min(int(orig_w * scale), int(orig_h * scale)) // 5, 14)
        for y in range(y0_word + spacing // 2, y1_word, spacing):
            draw.line([(0, y), (TARGET_SIZE, y)], fill=(210, 215, 225), width=1)
        for x in range(x0_word + spacing // 2, x1_word, spacing):
            draw.line([(x, 0), (x, TARGET_SIZE)], fill=(210, 215, 225), width=1)

    # Draw pen strokes (black ink, width=2, rounded joint) directly mapped into 224 space
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pts_224 = []
        for p in stroke:
            px = (float(p[0]) - x_min) * scale + pad_x
            py = (float(p[1]) - y_min) * scale + pad_y
            pts_224.append((int(round(px)), int(round(py))))
        if len(pts_224) >= 2:
            draw.line(pts_224, fill=(15, 15, 15), width=2, joint="curve")

    # Post-processing variants directly on 224x224
    if variant == "blur":
        sigma = float(rng.uniform(0.8, 1.2))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif variant == "camera":
        arr = np.array(img, dtype=np.float32)
        noise = rng.normal(0, 5.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


# ---------------------------------------------------------------------------
# Word segmentation algorithm
# ---------------------------------------------------------------------------

def segment_strokes_into_words(strokes_list):
    """
    Given a list of (M, >=2) stroke point arrays (sorted or unsorted),
    segments them into candidate word groups.
    Returns: list of dicts: {'strokes': [np.ndarray (N, 2)], 'bbox': (x0, y0, x1, y1)}
    """
    if not strokes_list:
        return []

    # Filter out tiny noise strokes (< 3 pts)
    valid_strokes = [s for s in strokes_list if len(s) >= MIN_STROKE_PTS]
    if len(valid_strokes) < 3:
        return []

    stroke_metas = []
    for i, s in enumerate(valid_strokes):
        pts = s[:, :2]
        t0 = s[0, 2] if s.shape[1] > 2 else float(i)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        h = max(y_max - y_min, 1.0)
        w = max(x_max - x_min, 1.0)
        stroke_metas.append({
            "idx": i,
            "pts": pts,
            "t0": t0,
            "bbox": (x_min, y_min, x_max, y_max),
            "h": h,
            "w": w,
            "x_center": (x_min + x_max) / 2.0,
            "y_center": (y_min + y_max) / 2.0,
        })

    # Estimate median stroke height
    valid_hs = [m["h"] for m in stroke_metas if m["h"] > 5]
    if not valid_hs:
        return []
    median_h = float(np.median(valid_hs))

    # Sort strokes chronologically
    stroke_metas.sort(key=lambda m: m["t0"])

    # Group into words
    words = []
    curr_word = [stroke_metas[0]]

    for m in stroke_metas[1:]:
        prev = curr_word[-1]
        dy = abs(m["y_center"] - prev["y_center"])
        gap_x = m["bbox"][0] - prev["bbox"][2]

        # Break conditions:
        # 1. Significant horizontal gap between strokes (> 0.8 * median_h)
        # 2. Large vertical shift (> 1.4 * median_h)
        # 3. Carriage return sweep backward (gap_x < -1.8 * median_h)
        is_new_word = (gap_x > 0.8 * median_h) or (dy > 1.4 * median_h) or (gap_x < -1.8 * median_h)

        if is_new_word:
            words.append(curr_word)
            curr_word = [m]
        else:
            curr_word.append(m)

    if curr_word:
        words.append(curr_word)

    # Format words and filter valid ones
    candidate_words = []
    for w in words:
        if len(w) < 3: # Must have at least 3 strokes
            continue
        w_pts = [m["pts"] for m in w]
        all_xy = np.vstack(w_pts)
        x0, y0 = all_xy.min(axis=0)
        x1, y1 = all_xy.max(axis=0)
        w_width = x1 - x0
        w_height = y1 - y0

        # Reject degenerate boxes or tiny clusters
        if w_width < 25 or w_height < 20:
            continue

        # Add modest padding margin (5% margin) around word bbox
        margin_x = max(w_width * 0.05, 4.0)
        margin_y = max(w_height * 0.05, 4.0)
        bbox = (x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y)

        candidate_words.append({
            "strokes": w_pts,
            "bbox": bbox,
            "n_strokes": len(w),
            "width": w_width,
            "height": w_height,
        })

    return candidate_words


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_datascirep_subjects():
    """
    Load dataSciRep subjects with word segmentation.
    Returns: list of dicts with keys: subject_id, dataset, label, words
    """
    xlsx_path = DATASETS_ROOT / "dataSciRep_public" / "data2_SciRep_pub.xlsx"
    if not xlsx_path.exists():
        print(f"WARNING: {xlsx_path} not found")
        return []

    df = pd.read_excel(xlsx_path)
    subjects = []
    n_dys = 0
    n_ctl = 0

    for _, row in df.iterrows():
        user_num = int(row["ID"])
        user_id = f"user{user_num:05d}"
        diag_str = str(row["diag"]).strip()
        label = 1 if diag_str == "DYSGR" else 0

        if label == 1 and n_dys >= 25:
            continue
        if label == 0 and n_ctl >= 25:
            continue
        if n_dys >= 25 and n_ctl >= 25:
            break

        user_dir = DATASETS_ROOT / "dataSciRep_public" / user_id
        svc_files = sorted(list(user_dir.glob("**/*.svc")))
        if not svc_files:
            continue

        all_words = []
        for svc_file in svc_files:
            try:
                sample = load_svc(str(svc_file))
                # sample.strokes are pen-on-paper strokes
                words = segment_strokes_into_words(sample.strokes)
                all_words.extend(words)
            except Exception as e:
                continue

        if not all_words:
            continue

        # Sort candidate words by number of strokes descending and pick best 3
        all_words.sort(key=lambda w: (w["n_strokes"], w["width"]), reverse=True)
        selected_words = []
        for i, w in enumerate(all_words[:WORDS_PER_SUBJECT]):
            selected_words.append({
                "word_id": f"word_{i+1:02d}",
                "strokes": w["strokes"],
                "bbox": w["bbox"],
                "n_strokes": w["n_strokes"],
            })

        subjects.append({
            "subject_id": user_id,
            "dataset": "dataSciRep",
            "label": label,
            "words": selected_words,
        })
        if label == 1:
            n_dys += 1
        else:
            n_ctl += 1
        print(f"  [dataSciRep] {user_id} (label={label}): {len(selected_words)} words | total: dys={n_dys}, ctl={n_ctl}", flush=True)

    return subjects


def load_diagramo_subjects():
    """
    Load DiaGraMo TSK4 / TSK16 subjects with word segmentation.
    Returns: list of dicts with keys: subject_id, dataset, label, words
    """
    csv_path = DATASETS_ROOT / "DiaGraMo-project" / "data" / "processed_data" / "cognitive_test_data" / "cognitiveTestCSV2data_cleaned.csv"
    clean_dir = DATASETS_ROOT / "DiaGraMo-project" / "data" / "clean_data"
    if not csv_path.exists() or not clean_dir.exists():
        print(f"WARNING: DiaGraMo paths not found")
        return []

    df = pd.read_csv(csv_path, sep=";")
    subjects = []
    n_dys = 0
    n_ctl = 0

    for _, row in df.iterrows():
        if pd.isna(row["group"]):
            continue
        pid = str(row["id_participant"]).strip()
        label = int(row["group"])

        if label == 1 and n_dys >= 25:
            continue
        if label == 0 and n_ctl >= 25:
            continue
        if n_dys >= 25 and n_ctl >= 25:
            break

        json_file = clean_dir / f"{pid}_merged.json"
        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception:
            continue

        # Look for TSK4 (dictation) and TSK16 (copy) tasks
        all_words = []
        for key in content.keys():
            if "TSK4" in key or "TSK16" in key:
                task_obj = content[key]
                if not isinstance(task_obj, dict) or "data" not in task_obj:
                    continue
                data_dict = task_obj["data"]
                req = ["x", "y", "time", "pen_status", "pressure"]
                if not all(k in data_dict for k in req):
                    continue

                x_arr = np.array(data_dict["x"], dtype=np.float64)
                y_arr = np.array(data_dict["y"], dtype=np.float64)
                t_arr = np.array(data_dict["time"], dtype=np.float64)
                pen_arr = np.array(data_dict["pen_status"], dtype=np.float64)
                p_arr = np.array(data_dict["pressure"], dtype=np.float64)

                # GT rule: pen_status == 1 and pressure > 0
                points = np.column_stack([x_arr, y_arr, t_arr, pen_arr, np.zeros_like(x_arr), np.zeros_like(x_arr), p_arr])
                strokes = _extract_strokes(points)
                words = segment_strokes_into_words(strokes)
                all_words.extend(words)

        if not all_words:
            continue

        all_words.sort(key=lambda w: (w["n_strokes"], w["width"]), reverse=True)
        selected_words = []
        for i, w in enumerate(all_words[:WORDS_PER_SUBJECT]):
            selected_words.append({
                "word_id": f"word_{i+1:02d}",
                "strokes": w["strokes"],
                "bbox": w["bbox"],
                "n_strokes": w["n_strokes"],
            })

        subjects.append({
            "subject_id": pid,
            "dataset": "DiaGraMo",
            "label": label,
            "words": selected_words,
        })
        if label == 1:
            n_dys += 1
        else:
            n_ctl += 1
        print(f"  [DiaGraMo] {pid} (label={label}): {len(selected_words)} words | total: dys={n_dys}, ctl={n_ctl}", flush=True)

    return subjects


# ---------------------------------------------------------------------------
# Subject sampling (CRITERIA.md §7)
# ---------------------------------------------------------------------------

def sample_subjects(all_subjects, n_total=N_SUBJECTS, n_test=N_TEST, seed=SEED):
    """
    Sample n_total subjects balanced by label.
    Hold out n_test (n_test//2 per label) as test set.
    Returns: (train_subjects, test_subjects)
    """
    rng = random.Random(seed)

    dysgraphic = [s for s in all_subjects if s["label"] == 1]
    control = [s for s in all_subjects if s["label"] == 0]

    n_per_label = n_total // 2
    n_test_per_label = n_test // 2

    # Deterministic sort before shuffle
    dysgraphic.sort(key=lambda s: (s["dataset"], s["subject_id"]))
    control.sort(key=lambda s: (s["dataset"], s["subject_id"]))

    rng.shuffle(dysgraphic)
    rng.shuffle(control)

    selected_dys = dysgraphic[:n_per_label]
    selected_ctl = control[:n_per_label]

    test_dys = selected_dys[:n_test_per_label]
    train_dys = selected_dys[n_test_per_label:]
    test_ctl = selected_ctl[:n_test_per_label]
    train_ctl = selected_ctl[n_test_per_label:]

    train = train_dys + train_ctl
    test = test_dys + test_ctl

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("InkSight Evaluation — Data Preparation (01_prepare_data.py)")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)
    (OUT_DIR / "gt_strokes").mkdir(exist_ok=True)

    # 1. Load subjects from both datasets
    print("\n--- Loading dataSciRep ---")
    scirep = load_datascirep_subjects()
    print(f"Loaded {len(scirep)} valid subjects from dataSciRep")

    print("\n--- Loading DiaGraMo ---")
    diagramo = load_diagramo_subjects()
    print(f"Loaded {len(diagramo)} valid subjects from DiaGraMo")

    all_subjects = scirep + diagramo
    dys_count = sum(1 for s in all_subjects if s["label"] == 1)
    ctl_count = sum(1 for s in all_subjects if s["label"] == 0)
    print(f"\nTotal available subjects: {len(all_subjects)} ({dys_count} dysgraphic, {ctl_count} control)")

    # 2. Sample subjects
    train_subjects, test_subjects = sample_subjects(all_subjects, n_total=N_SUBJECTS, n_test=N_TEST, seed=SEED)

    split = {
        "seed": SEED,
        "n_total": len(train_subjects) + len(test_subjects),
        "train": [s["subject_id"] for s in train_subjects],
        "test": [s["test_id"] if "test_id" in s else s["subject_id"] for s in test_subjects],
        "n_train_dysgraphic": sum(1 for s in train_subjects if s["label"] == 1),
        "n_train_control": sum(1 for s in train_subjects if s["label"] == 0),
        "n_test_dysgraphic": sum(1 for s in test_subjects if s["label"] == 1),
        "n_test_control": sum(1 for s in test_subjects if s["label"] == 0),
    }
    with open(SUBJECT_SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSubject split saved: {SUBJECT_SPLIT_PATH}")
    print(f"  Train/Dev: {len(train_subjects)} subjects ({split['n_train_dysgraphic']} dys, {split['n_train_control']} ctl)")
    print(f"  Held-out Test: {len(test_subjects)} subjects ({split['n_test_dysgraphic']} dys, {split['n_test_control']} ctl)")

    # 3. Render word images and save GT strokes
    print("\n--- Rendering word crops & saving GT ---")
    rng = np.random.default_rng(SEED)
    manifest_rows = []

    all_selected = train_subjects + test_subjects
    for s_idx, subj in enumerate(all_selected):
        split_name = "test" if subj["subject_id"] in split["test"] else "train"
        print(f"  Rendering [{s_idx+1}/{len(all_selected)}] {subj['subject_id']} ({len(subj['words'])} words, split={split_name})...", flush=True)

        # Compute subject median stroke height H_med across its words
        subj_stroke_hs = []
        for w in subj["words"]:
            for s in w["strokes"]:
                if len(s) >= 2:
                    h = float(s[:, 1].max() - s[:, 1].min())
                    if h > 2:
                        subj_stroke_hs.append(h)
        h_med_subj = float(np.median(subj_stroke_hs)) if subj_stroke_hs else 50.0

        for word_data in subj["words"]:
            strokes = word_data["strokes"]
            bbox = word_data["bbox"]
            word_id = word_data["word_id"]

            orig_w = max(int(round(bbox[2] - bbox[0])), 1)
            orig_h = max(int(round(bbox[3] - bbox[1])), 1)
            scale, pad_x, pad_y = compute_pad_resize_params(orig_w, orig_h)

            # Save GT strokes as numpy npz
            gt_path = OUT_DIR / "gt_strokes" / f"{subj['subject_id']}_{word_id}.npz"
            np.savez(str(gt_path),
                     strokes=np.array([np.array(s, dtype=np.float32) for s in strokes], dtype=object),
                     bbox=np.array(bbox, dtype=np.float32),
                     h_med=h_med_subj,
                     scale=scale,
                     pad_x=pad_x,
                     pad_y=pad_y)

            # Render 5 variants
            for variant in VARIANTS:
                img = render_word_image(strokes, bbox, variant, rng)
                img_path = OUT_DIR / "images" / f"{subj['subject_id']}_{word_id}_{variant}.png"
                img.save(str(img_path))

                manifest_rows.append({
                    "subject_id": subj["subject_id"],
                    "dataset": subj["dataset"],
                    "label": subj["label"],
                    "split": split_name,
                    "word_id": word_id,
                    "variant": variant,
                    "image_path": str(img_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "gt_strokes_path": str(gt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "word_w_px": orig_w,
                    "word_h_px": orig_h,
                    "h_med_subj": h_med_subj,
                    "scale": scale,
                    "pad_x": pad_x,
                    "pad_y": pad_y,
                    "n_gt_strokes": len(strokes),
                    "bbox_json": json.dumps([round(float(v), 2) for v in bbox]),
                })

    # Write manifest
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest successfully written: {MANIFEST_PATH}")
    print(f"Total word crops rendered: {len(manifest_rows)}")
    print(f"  Train/Dev samples: {sum(1 for r in manifest_rows if r['split'] == 'train')}")
    print(f"  Test samples:      {sum(1 for r in manifest_rows if r['split'] == 'test')}")


if __name__ == "__main__":
    main()
