"""
Decompose multi-line sentence crops into clean, single-line image strips.
Saves to data/line_crops_dataset/ with lines_index.csv.
Leaves data/sentence_crops_dataset/ completely untouched for easy backtracking.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def detect_split_valleys(img):
    """Detect horizontal valleys in an image for line splitting."""
    h, w = img.shape[:2]
    if h < 180:
        return []
    
    # Binarize
    ink = (img < 210).astype(np.float32)
    margin = int(w * 0.05)
    ink_core = ink[:, margin:w-margin]
    proj = np.sum(ink_core, axis=1)
    if np.max(proj) == 0:
        return []
        
    smooth = gaussian_filter1d(proj / np.max(proj), sigma=5)
    peaks, _ = find_peaks(smooth, distance=50, prominence=0.15, height=0.20)
    if len(peaks) <= 1:
        return []
        
    valleys = []
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i+1]
        v_idx = p1 + np.argmin(smooth[p1:p2])
        if smooth[v_idx] < 0.65 * min(smooth[p1], smooth[p2]):
            valleys.append(int(v_idx))
            
    return valleys

def main():
    src_csv = REPO_ROOT / "data" / "sentence_crops_dataset" / "sentences_index.csv"
    src_dir = REPO_ROOT / "data" / "sentence_crops_dataset"
    out_dir = REPO_ROOT / "data" / "line_crops_dataset"
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} sentence crops from {src_csv}")
    
    line_records = []
    split_count = 0
    total_lines = 0
    
    for idx, row in df.iterrows():
        tuid = str(row["task_uid"])
        rel = str(row["rel_path"])
        img_path = src_dir / rel
        
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        valleys = detect_split_valleys(gray)
        
        if not valleys:
            # Single line crop
            line_uid = f"{tuid}_line_01"
            out_filename = f"{line_uid}.jpg"
            out_path = out_img_dir / out_filename
            cv2.imwrite(str(out_path), img)
            
            rec = dict(row)
            rec["line_uid"] = line_uid
            rec["line_index"] = 1
            rec["total_lines_in_task"] = 1
            rec["line_rel_path"] = f"images/{out_filename}"
            rec["line_width"] = w
            rec["line_height"] = h
            line_records.append(rec)
            total_lines += 1
        else:
            # Multi-line crop: split into sub-lines
            split_count += 1
            y_bounds = [0] + valleys + [h]
            n_sub = len(y_bounds) - 1
            
            for l_idx in range(n_sub):
                y_start = max(0, y_bounds[l_idx] - 8)
                y_end = min(h, y_bounds[l_idx+1] + 8)
                sub_img = img[y_start:y_end, :]
                sub_h, sub_w = sub_img.shape[:2]
                
                line_uid = f"{tuid}_line_{l_idx+1:02d}"
                out_filename = f"{line_uid}.jpg"
                out_path = out_img_dir / out_filename
                cv2.imwrite(str(out_path), sub_img)
                
                rec = dict(row)
                rec["line_uid"] = line_uid
                rec["line_index"] = l_idx + 1
                rec["total_lines_in_task"] = n_sub
                rec["line_rel_path"] = f"images/{out_filename}"
                rec["line_width"] = sub_w
                rec["line_height"] = sub_h
                line_records.append(rec)
                total_lines += 1

    df_lines = pd.DataFrame(line_records)
    out_csv = out_dir / "lines_index.csv"
    df_lines.to_csv(out_csv, index=False)
    
    print(f"\nDecomposition Complete:")
    print(f"  Multi-line crops split: {split_count} / {len(df)}")
    print(f"  Total single-line crops generated: {total_lines}")
    print(f"  Saved metadata to: {out_csv}")

if __name__ == "__main__":
    main()

