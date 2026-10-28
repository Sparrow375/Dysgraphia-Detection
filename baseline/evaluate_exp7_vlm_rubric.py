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
REPORTS_DIR = REPO_ROOT / "reports"
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

PROMPT_TEXT = """You are an expert pediatric handwriting clinician assessing this handwriting sample.
Carefully inspect the image strokes and score this handwriting from 1.0 (severely impaired) to 5.0 (flawless) on:
- legibility: readability of words
- letter_formation: correct letter shapes and closed loops
- spacing_regularity: consistency of spaces between words and letters
- size_consistency: steady uniform letter height
- baseline_adherence: straight horizontal baseline alignment without downhill drift
- stroke_fluidity: smooth continuous lines without tremor or shaky hesitation
- pressure_uniformity: even ink darkness without faint gaps or gouges

Output strictly a JSON object with your numeric evaluations, formatted as:
{
  "legibility": <float 1.0-5.0>,
  "letter_formation": <float 1.0-5.0>,
  "spacing_regularity": <float 1.0-5.0>,
  "size_consistency": <float 1.0-5.0>,
  "baseline_adherence": <float 1.0-5.0>,
  "stroke_fluidity": <float 1.0-5.0>,
  "pressure_uniformity": <float 1.0-5.0>
}"""


def parse_vlm_json_response(raw_text: str) -> Optional[Dict[str, float]]:
    """Parse JSON string from model generation, handling markdown blocks."""
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
    batch_size: int = 4,
    device_str: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run Qwen2-VL-2B inference over all sentence crops with resume support and batching."""
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

    logger.info("Beginning VLM scoring loop (batch_size=%d)...", batch_size)
    start_time = time.time()
    n_scored = 0

    for b_start in range(0, len(pending_tasks), batch_size):
        b_rows = pending_tasks[b_start : b_start + batch_size]
        batch_images = []
        batch_tuids = []

        for row in b_rows:
            tuid = str(row["task_uid"])
            rel_path = str(row["rel_path"])
            img_path = SENTENCE_DATASET_DIR / rel_path
            if not img_path.exists():
                logger.warning("Image missing: %s", img_path)
                continue
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            if h != 224:
                new_w = max(224, int(w * (224 / h)))
                image = image.resize((new_w, 224), Image.Resampling.BILINEAR)
            batch_images.append(image)
            batch_tuids.append(tuid)

        if not batch_images:
            continue

        try:
            messages_list = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": PROMPT_TEXT},
                        ],
                    }
                ]
                for img in batch_images
            ]
            text_prompts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            inputs = processor(
                text=text_prompts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                )
                trimmed = generated_ids[:, len(inputs.input_ids[0]) :]
                output_texts = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            for tuid, out_text in zip(batch_tuids, output_texts):
                scores = parse_vlm_json_response(out_text)
                if scores is not None:
                    cache[tuid] = scores
                else:
                    logger.warning("Parse retry for %s: '%s'", tuid, out_text.strip())
                    cache[tuid] = {k: 3.0 for k in RUBRIC_CRITERIA}
                n_scored += 1

            if n_scored % 20 < batch_size or (b_start + batch_size >= len(pending_tasks)):
                save_rubric_cache(cache)
                logger.info(
                    "Scored %d / %d tasks (%.1fs elapsed, %.2fs/task)",
                    n_scored,
                    len(pending_tasks),
                    time.time() - start_time,
                    (time.time() - start_time) / max(1, n_scored),
                )

        except Exception as e:
            logger.error("Error processing batch starting at %d: %s", b_start, e)
            continue

    save_rubric_cache(cache)
    logger.info("VLM scoring complete: %d total sentences cached.", len(cache))
    return cache


def build_student_rubric_features(
    cache: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Aggregate sentence-level rubric scores to student-level feature matrix for scored students."""
    df_index = pd.read_csv(INDEX_CSV)

    cached_tuids = set(cache.keys())
    df_cached = df_index[df_index["task_uid"].astype(str).isin(cached_tuids)].copy()
    unique_students = sorted(df_cached["student_id"].unique())

    student_features_list = []
    student_labels = []
    feature_names = []

    for c in RUBRIC_CRITERIA:
        feature_names.extend([f"vlm_{c}_mean", f"vlm_{c}_std", f"vlm_{c}_min"])

    for sid in unique_students:
        s_rows = df_cached[df_cached["student_id"] == sid]
        label = int(s_rows["label"].iloc[0])
        student_labels.append(label)

        s_scores: Dict[str, List[float]] = {c: [] for c in RUBRIC_CRITERIA}
        for _, row in s_rows.iterrows():
            tuid = str(row["task_uid"])
            if tuid in cache:
                task_s = cache[tuid]
                for c in RUBRIC_CRITERIA:
                    s_scores[c].append(task_s[c])

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


def compute_bhk_correlations(
    X: np.ndarray,
    sids: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute Pearson and Spearman correlations between VLM rubric features and tabular BHK features."""
    student_csv = DATA_DIR / "datasets" / "dataset_student_level.csv"
    if not student_csv.exists():
        logger.warning("Tabular dataset not found at %s. Skipping BHK correlation.", student_csv)
        return {}

    df_tab = pd.read_csv(student_csv).set_index("student_id")
    common_sids = [s for s in sids if s in df_tab.index]
    if len(common_sids) < 10:
        logger.warning("Too few overlapping students (%d) for BHK correlation.", len(common_sids))
        return {}

    sub_tab = df_tab.loc[common_sids]
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    sub_X = X[[sid_to_idx[s] for s in common_sids]]

    pairs = [
        ("vlm_spacing_regularity_mean", "gap_mean_median"),
        ("vlm_baseline_adherence_mean", "baseline_rmse_norm_median"),
        ("vlm_stroke_fluidity_mean", "jerk_proxy_median"),
        ("vlm_letter_formation_mean", "components_per_unit_width_median"),
        ("vlm_legibility_mean", "components_per_unit_width_median"),
    ]

    corr_results = {}
    for vlm_feat, tab_feat in pairs:
        if vlm_feat in feature_names and tab_feat in sub_tab.columns:
            v_idx = feature_names.index(vlm_feat)
            v_vals = sub_X[:, v_idx]
            t_vals = sub_tab[tab_feat].values
            valid_mask = ~np.isnan(t_vals) & ~np.isnan(v_vals)
            if np.sum(valid_mask) > 5:
                r_p, p_p = pearsonr(v_vals[valid_mask], t_vals[valid_mask])
                r_s, p_s = spearmanr(v_vals[valid_mask], t_vals[valid_mask])
                corr_results[f"{vlm_feat}__vs__{tab_feat}"] = {
                    "pearson_r": float(r_p),
                    "pearson_p": float(p_p),
                    "spearman_rho": float(r_s),
                    "spearman_p": float(p_s),
                }
                logger.info(
                    "Correlation %s vs %s: Pearson r=%.3f (p=%.3e), Spearman rho=%.3f (p=%.3e)",
                    vlm_feat, tab_feat, r_p, p_p, r_s, p_s
                )

    return corr_results


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

    # BHK correlations
    bhk_corrs = compute_bhk_correlations(X, sids, feature_names)

    results = {
        "candidate": "Candidate 4: VLM Clinical Rubric Scorer",
        "n_students": n_students,
        "n_positives": int(np.sum(y == 1)),
        "n_controls": int(np.sum(y == 0)),
        "auroc": auc,
        "auprc": ap,
        "brier": brier,
        "sens_at_80_spec": det80 / max(1, len(pos_scores)),
        "det_at_80_spec": det80,
        "sens_at_90_spec": det90 / max(1, len(pos_scores)),
        "det_at_90_spec": det90,
        "bhk_correlations": bhk_corrs,
    }

    logger.info("\n" + "=" * 80)
    logger.info("CANDIDATE 4 VLM RUBRIC SCOREBOARD (5x5 Repeated CV)")
    logger.info("=" * 80)
    logger.info(
        f"Students: {n_students} ({results['n_positives']} pos, {results['n_controls']} ctl) | "
        f"AUROC: {auc:.4f} | AUPRC: {ap:.4f} | "
        f"Sens@80%: {results['sens_at_80_spec']*100:.1f}% ({det80}/{results['n_positives']}) | "
        f"Sens@90%: {results['sens_at_90_spec']*100:.1f}% ({det90}/{results['n_positives']})"
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for VLM inference")
    args = parser.parse_args()

    cache = score_sentences_with_vlm(limit=args.limit, batch_size=args.batch_size)
    X, y, sids, feat_names = build_student_rubric_features(cache)
    evaluate_vlm_rubric_cv(X, y, sids, feat_names)


if __name__ == "__main__":
    main()
