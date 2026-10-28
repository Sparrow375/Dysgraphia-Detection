"""
Experiment 7, Candidate 4: VLM Rubric Scorer
Evaluates pediatric handwriting quality using a Vision-Language Model (Qwen2-VL-2B-Instruct).
Scores each sentence crop across 7 standardized clinical dimensions:
1. legibility: 1 (indecipherable) to 5 (perfect clarity)
2. letter_formation: 1 (severe deformation/broken loops) to 5 (canonical letters)
3. spacing_regularity: 1 (erratic crowding/gaps) to 5 (rhythmic, uniform spacing)
4. size_consistency: 1 (wildly fluctuating heights) to 5 (steady x-height)
5. baseline_adherence: 1 (severe downhill drift/wobble) to 5 (straight alignment)
6. stroke_fluidity: 1 (hesitation, tremor, shaky lines) to 5 (smooth ballistic strokes)
7. pressure_uniformity: 1 (patchy vanishing lines/gouges) to 5 (even optical density)

Results are cached to disk incrementally and aggregated to student level for 5x5 CV probing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("exp7_vlm_rubric")

DATA_DIR = REPO_ROOT / "data"
SENTENCE_DATASET_DIR = DATA_DIR / "sentence_crops_dataset"
INDEX_CSV = SENTENCE_DATASET_DIR / "sentences_index.csv"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = DATA_DIR / "reports"
RUBRIC_CACHE_FILE = FEATURES_DIR / "vlm_rubric_cache.json"

RUBRIC_CRITERIA = [
    "legibility",
    "letter_formation",
    "spacing_regularity",
    "size_consistency",
    "baseline_adherence",
    "stroke_fluidity",
    "pressure_uniformity",
]

PROMPT_TEXT = """You are an expert pediatric occupational therapist assessing children's handwriting for dysgraphia screening.
Analyze this handwriting crop and assign a continuous score from 1.0 to 5.0 (in increments of 0.5: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0) for each of these 7 clinical criteria:
1. legibility: 1.0 (illegible/severely impaired) to 5.0 (effortless clarity)
2. letter_formation: 1.0 (severely deformed/broken loops) to 5.0 (canonical formation)
3. spacing_regularity: 1.0 (erratic inter-word/inter-letter gaps) to 5.0 (uniform spacing)
4. size_consistency: 1.0 (wildly fluctuating letter heights) to 5.0 (steady, uniform x-height)
5. baseline_adherence: 1.0 (downhill drift or severe waviness) to 5.0 (straight horizontal alignment)
6. stroke_fluidity: 1.0 (tremor, hesitation, shaky strokes) to 5.0 (smooth ballistic strokes)
7. pressure_uniformity: 1.0 (patchy faint strokes and heavy gouges) to 5.0 (even density)

Return strictly valid JSON with no markdown and no explanation:
{
  "legibility": 3.5,
  "letter_formation": 3.0,
  "spacing_regularity": 4.0,
  "size_consistency": 3.0,
  "baseline_adherence": 3.5,
  "stroke_fluidity": 2.5,
  "pressure_uniformity": 4.0
}"""


def parse_vlm_json_response(raw_text: str) -> Optional[Dict[str, float]]:
    """Parse JSON string from model generation, handling markdown blocks."""
    # Strip markdown code blocks if present
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        cleaned = {}
        for k in RUBRIC_CRITERIA:
            if k in data:
                val = float(data[k])
                cleaned[k] = max(1.0, min(5.0, val))
            else:
                return None
        return cleaned
    except Exception as e:
        logger.debug("Failed parsing JSON: %s (raw: %s)", e, raw_text)
        return None


def load_rubric_cache() -> Dict[str, Dict[str, float]]:
    """Load existing rubric cache from disk."""
    if RUBRIC_CACHE_FILE.exists():
        try:
            with open(RUBRIC_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read cache: %s. Starting fresh.", e)
    return {}


def save_rubric_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """Save rubric cache atomically to disk."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = RUBRIC_CACHE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp_path.replace(RUBRIC_CACHE_FILE)


def score_sentences_with_vlm(
    limit: Optional[int] = None,
    batch_size: int = 1,
    device_str: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run Qwen2-VL-2B inference over all sentence crops with resume support."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    cache = load_rubric_cache()
    logger.info("Loaded %d existing scored sentences from cache.", len(cache))

    df_index = pd.read_csv(INDEX_CSV)
    if limit is not None and limit > 0:
        unique_students = df_index["student_id"].unique()[:limit]
        df_index = df_index[df_index["student_id"].isin(unique_students)].copy()
        logger.info("Limited scoring to %d students (%d tasks)", limit, len(df_index))

    # Filter out tasks already cached
    pending_tasks = [
        row for _, row in df_index.iterrows() if str(row["task_uid"]) not in cache
    ]
    logger.info("Pending tasks to score: %d / %d", len(pending_tasks), len(df_index))

    if not pending_tasks:
        logger.info("All tasks already cached! Skipping VLM inference.")
        return cache

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    logger.info("Loading %s on device: %s...", model_id, device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    logger.info("Beginning VLM scoring loop...")
    start_time = time.time()
    n_scored = 0

    for idx, row in enumerate(pending_tasks):
        tuid = str(row["task_uid"])
        rel_path = str(row["rel_path"])
        img_path = SENTENCE_DATASET_DIR / rel_path

        if not img_path.exists():
            logger.warning("Image missing: %s", img_path)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            # If extremely wide, resize height to 224 maintaining aspect ratio
            w, h = image.size
            if h != 224:
                new_w = max(224, int(w * (224 / h)))
                image = image.resize((new_w, 224), Image.Resampling.BILINEAR)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = [image], None
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # Temperature 0.0 greedy
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            scores = parse_vlm_json_response(output_text)
            if scores is not None:
                cache[tuid] = scores
                n_scored += 1
            else:
                # Fallback: parse whatever floats exist in output
                logger.warning("Parse retry for %s: '%s'", tuid, output_text.strip())
                # Default middle score if complete parse failure
                cache[tuid] = {k: 3.0 for k in RUBRIC_CRITERIA}

            # Periodic save every 20 sentences
            if n_scored % 20 == 0:
                save_rubric_cache(cache)
                logger.info(
                    "Scored %d / %d tasks (%.1fs elapsed, %.2fs/task)",
                    n_scored,
                    len(pending_tasks),
                    time.time() - start_time,
                    (time.time() - start_time) / max(1, n_scored),
                )

        except Exception as e:
            logger.error("Error processing %s: %s", tuid, e)
            continue

    save_rubric_cache(cache)
    logger.info("VLM scoring complete: %d total sentences cached.", len(cache))
    return cache


def build_student_rubric_features(
    cache: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Aggregate sentence-level rubric scores to student-level feature matrix."""
    df_index = pd.read_csv(INDEX_CSV)

    student_groups = df_index.groupby("student_id")
    unique_students = list(student_groups.groups.keys())
    unique_students.sort()

    student_features_list = []
    student_labels = []
    feature_names = []

    # Features: for each of the 7 criteria -> [mean, std, min]
    for c in RUBRIC_CRITERIA:
        feature_names.extend([f"vlm_{c}_mean", f"vlm_{c}_std", f"vlm_{c}_min"])

    for sid in unique_students:
        s_rows = df_index[df_index["student_id"] == sid]
        label = int(s_rows["label"].iloc[0])
        student_labels.append(label)

        s_scores: Dict[str, List[float]] = {c: [] for c in RUBRIC_CRITERIA}
        for _, row in s_rows.iterrows():
            tuid = str(row["task_uid"])
            if tuid in cache:
                task_s = cache[tuid]
                for c in RUBRIC_CRITERIA:
                    s_scores[c].append(task_s[c])

        # Compute aggregates
        feats = []
        for c in RUBRIC_CRITERIA:
            vals = s_scores[c] if s_scores[c] else [3.0]
            feats.append(float(np.mean(vals)))
            feats.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            feats.append(float(np.min(vals)))

        student_features_list.append(feats)

    X = np.array(student_features_list, dtype=np.float32)
    y = np.array(student_labels, dtype=np.int64)
    sids = np.array(unique_students)
    return X, y, sids, feature_names


def evaluate_vlm_rubric_cv(
    X: np.ndarray,
    y: np.ndarray,
    sids: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Evaluate 5x5 repeated CV probe on VLM rubric features."""
    df_index = pd.read_csv(INDEX_CSV)
    student_folds = df_index.drop_duplicates(subset=["student_id"]).set_index("student_id")
    student_folds = student_folds.reindex(sids)

    n_students = len(sids)
    oof_probs = np.zeros((5, n_students), dtype=np.float32)

    for r in range(5):
        fold_col = student_folds[f"repeat_{r}_fold"].values
        for f in range(5):
            test_mask = (fold_col == f)
            train_mask = (fold_col != f)

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te = X[test_mask]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            # Inner CV tuning
            best_c = 0.1
            best_score = -1.0
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + f)
            for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
                val_aucs = []
                for sub_tr, sub_val in skf.split(X_tr_sc, y_tr):
                    clf = LogisticRegression(C=c_cand, class_weight="balanced", max_iter=1000, random_state=42)
                    clf.fit(X_tr_sc[sub_tr], y_tr[sub_tr])
                    p_val = clf.predict_proba(X_tr_sc[sub_val])[:, 1]
                    val_aucs.append(roc_auc_score(y_tr[sub_val], p_val))
                if np.mean(val_aucs) > best_score:
                    best_score = np.mean(val_aucs)
                    best_c = c_cand

            final_clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
            final_clf.fit(X_tr_sc, y_tr)
            oof_probs[r, test_mask] = final_clf.predict_proba(X_te_sc)[:, 1]

    mean_oof = np.mean(oof_probs, axis=0)
    auc = float(roc_auc_score(y, mean_oof))
    ap = float(average_precision_score(y, mean_oof))
    brier = float(brier_score_loss(y, mean_oof))

    neg_scores = mean_oof[y == 0]
    pos_scores = mean_oof[y == 1]
    th80 = float(np.percentile(neg_scores, 80.0))
    det80 = int(np.sum(pos_scores >= th80))
    th90 = float(np.percentile(neg_scores, 90.0))
    det90 = int(np.sum(pos_scores >= th90))

    results = {
        "candidate": "Candidate 4: VLM Clinical Rubric Scorer",
        "auroc": auc,
        "auprc": ap,
        "brier": brier,
        "sens_at_80_spec": det80 / len(pos_scores),
        "det_at_80_spec": det80,
        "sens_at_90_spec": det90 / len(pos_scores),
        "det_at_90_spec": det90,
    }

    logger.info("\n" + "=" * 80)
    logger.info("CANDIDATE 4 VLM RUBRIC SCOREBOARD (5x5 Repeated CV)")
    logger.info("=" * 80)
    logger.info(
        f"AUROC: {auc:.4f} | AUPRC: {ap:.4f} | "
        f"Sens@80%: {results['sens_at_80_spec']*100:.1f}% ({det80}/40) | "
        f"Sens@90%: {results['sens_at_90_spec']*100:.1f}% ({det90}/40)"
    )
    logger.info("=" * 80)

    # Save to JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "exp7_vlm_rubric_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Candidate 4: VLM Rubric Scorer")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N students for fast pilot")
    args = parser.parse_args()

    cache = score_sentences_with_vlm(limit=args.limit)
    X, y, sids, feat_names = build_student_rubric_features(cache)
    evaluate_vlm_rubric_cv(X, y, sids, feat_names)


if __name__ == "__main__":
    main()

