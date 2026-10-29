"""
Export student-level multimodal benchmark scores and error-analysis classifications.
Saves to data/features/exp7_student_model_scores.json for instant UI consumption in QA App and reports.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from baseline.evaluate_bilingual_ocr_fusion import run_5x5_cv_probe

def main():
    oof_data = np.load("data/features/exp7_all_models_oof_probs.npz")
    student_ids = list(oof_data["student_ids"])
    y = oof_data["y_true"]

    p_dino = oof_data["dinov2_joint"]
    p_clip = oof_data["clip_blend"]
    p_dual = oof_data["dual_vit_50_50_rank"]
    p_tab = oof_data["tabular_top30"]

    ocr_npz = np.load("data/features/exp7_bilingual_ocr_student_features.npz")
    fold_df = pd.DataFrame({"student_id": student_ids})
    for r in range(5):
        fold_df[f"repeat_{r}_fold"] = ocr_npz[f"repeat_{r}_fold"]

    p_en_ocr = run_5x5_cv_probe(ocr_npz["X_english_ocr"], y, fold_df, n_components=12, C=0.1)

    r_dino = rankdata(p_dino) / len(p_dino)
    r_clip = rankdata(p_clip) / len(p_clip)
    r_dual = rankdata(p_dual) / len(p_dual)
    r_en_ocr = rankdata(p_en_ocr) / len(p_en_ocr)
    r_tab = rankdata(p_tab) / len(p_tab)

    p_tri = 0.84 * r_dual + 0.08 * r_en_ocr + 0.08 * r_tab
    r_tri = rankdata(p_tri) / len(p_tri)

    th80_tri = float(np.percentile(p_tri[y == 0], 80.0))
    th90_tri = float(np.percentile(p_tri[y == 0], 90.0))

    df_manifest = pd.read_csv('data/manifest.csv')
    manifest_dict = df_manifest.set_index('student_id').to_dict('index')

    df_student = pd.read_csv('data/datasets/dataset_student_level.csv')
    tool_dict = df_student.set_index('student_id')['detected_tool'].to_dict() if 'detected_tool' in df_student.columns else {}

    df_sentences = pd.read_csv('data/sentence_crops_dataset/sentences_index.csv')

    student_dict = {}
    for i, sid in enumerate(student_ids):
        lbl = int(y[i])
        m_info = manifest_dict.get(sid, {})
        s_crops = df_sentences[df_sentences['student_id'] == sid]
        tasks = sorted(s_crops['task_name'].dropna().unique().tolist())
        
        score_tri = float(p_tri[i])
        rank_tri = float(r_tri[i])
        caught80 = score_tri >= th80_tri
        caught90 = score_tri >= th90_tri

        if lbl == 1:
            class90 = "CAUGHT" if caught90 else "MISSED"
            class80 = "CAUGHT" if caught80 else "MISSED"
            is_missed = not caught90
            is_false_alarm = False
        else:
            class90 = "FALSE_ALARM" if caught90 else "CORRECT_CONTROL"
            class80 = "FALSE_ALARM" if caught80 else "CORRECT_CONTROL"
            is_missed = False
            is_false_alarm = caught90

        student_dict[sid] = {
            "student_id": sid,
            "label": lbl,
            "school": m_info.get("school", ""),
            "grade": int(m_info.get("grade", 0)) if pd.notna(m_info.get("grade")) else None,
            "roll_number": int(m_info.get("roll_number", 0)) if pd.notna(m_info.get("roll_number")) else None,
            "tool": tool_dict.get(sid, "unknown"),
            "tri_score": round(score_tri, 4),
            "tri_rank": round(rank_tri, 4),
            "dual_score": round(float(p_dual[i]), 4),
            "dual_rank": round(float(r_dual[i]), 4),
            "dino_rank": round(float(r_dino[i]), 4),
            "clip_rank": round(float(r_clip[i]), 4),
            "ocr_score": round(float(p_en_ocr[i]), 4),
            "ocr_rank": round(float(r_en_ocr[i]), 4),
            "tab_score": round(float(p_tab[i]), 4),
            "tab_rank": round(float(r_tab[i]), 4),
            "num_crops": len(s_crops),
            "tasks_present": tasks,
            "caught_spec80": bool(caught80),
            "caught_spec90": bool(caught90),
            "classification_90": class90,
            "classification_80": class80,
            "is_missed_positive": is_missed,
            "is_false_alarm": is_false_alarm,
        }

    out_file = REPO_ROOT / "data" / "features" / "exp7_student_model_scores.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "threshold_spec80": round(th80_tri, 4),
            "threshold_spec90": round(th90_tri, 4),
            "students": student_dict,
        }, f, indent=2)

    print(f"Exported student scores for {len(student_dict)} students to {out_file}")

if __name__ == "__main__":
    main()
