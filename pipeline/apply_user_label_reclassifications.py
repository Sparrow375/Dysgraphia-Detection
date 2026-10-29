"""
Apply user re-classifications to dataset ground truth:
1. G5_A_Roll07 (School A, Grade 5): Positive (1) -> Negative (0)
2. SCHB_G3_IMG_20261007_101635 (School B, Grade 3): Positive (1) -> Negative (0)
3. SCHB_G4_IMG_20261007_100350 (School B, Grade 4): Negative (0) -> Positive (1)
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd

UPDATES = {
    "G5_A_Roll07": 0,
    "SCHB_G3_IMG_20261007_101635": 0,
    "SCHB_G4_IMG_20261007_100350": 1,
}

def main():
    print("Applying user re-classifications...")
    
    # 1. Update data/manifest.csv
    manifest_path = REPO_ROOT / "data" / "manifest.csv"
    if manifest_path.exists():
        df_m = pd.read_csv(manifest_path)
        for sid, new_lbl in UPDATES.items():
            df_m.loc[df_m["student_id"] == sid, "label"] = float(new_lbl)
            df_m.loc[df_m["student_id"] == sid, "label_source"] = "expert_manual_review"
        df_m.to_csv(manifest_path, index=False)
        print(f"Updated data/manifest.csv")

    # 2. Update processed student review_status.json
    for sid, new_lbl in UPDATES.items():
        # Find school
        sch = "school_b" if "SCHB" in sid else "school_a"
        s_dir = REPO_ROOT / "data" / "processed" / sch / sid
        status_file = s_dir / "review_status.json"
        st = {"verified": True, "label": int(new_lbl), "notes": f"Reclassified by expert review to {new_lbl}"}
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                st["label"] = int(new_lbl)
                st["verified"] = True
                st["notes"] = f"Reclassified by expert review to {new_lbl}"
            except Exception:
                pass
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        print(f"Updated review status for {sid} -> label {new_lbl}")

    # 3. Update data/datasets/dataset_student_level.csv
    ds_path = REPO_ROOT / "data" / "datasets" / "dataset_student_level.csv"
    if ds_path.exists():
        df_s = pd.read_csv(ds_path)
        for sid, new_lbl in UPDATES.items():
            df_s.loc[df_s["student_id"] == sid, "label"] = int(new_lbl)
        df_s.to_csv(ds_path, index=False)
        print(f"Updated data/datasets/dataset_student_level.csv")

    # 4. Update data/sentence_crops_dataset/sentences_index.csv
    sent_path = REPO_ROOT / "data" / "sentence_crops_dataset" / "sentences_index.csv"
    if sent_path.exists():
        df_sent = pd.read_csv(sent_path)
        for sid, new_lbl in UPDATES.items():
            df_sent.loc[df_sent["student_id"] == sid, "label"] = int(new_lbl)
        df_sent.to_csv(sent_path, index=False)
        print(f"Updated data/sentence_crops_dataset/sentences_index.csv")

    # 5. Update exp7_all_models_oof_probs.npz y_true
    oof_path = REPO_ROOT / "data" / "features" / "exp7_all_models_oof_probs.npz"
    if oof_path.exists():
        oof_dict = dict(np.load(oof_path))
        sids = list(oof_dict["student_ids"])
        y = oof_dict["y_true"].copy()
        for sid, new_lbl in UPDATES.items():
            if sid in sids:
                idx = sids.index(sid)
                y[idx] = int(new_lbl)
        oof_dict["y_true"] = y
        np.savez(oof_path, **oof_dict)
        print(f"Updated exp7_all_models_oof_probs.npz y_true")

    print("All label files synchronized successfully.")

if __name__ == "__main__":
    main()
