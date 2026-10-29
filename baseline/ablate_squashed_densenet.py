"""
Clean Ablation: Frozen DenseNet201 on Squashed Sentences (Zero Tiling)
Evaluates whether tiling alone drives the performance gain by testing:
Frozen DenseNet201 -> Squashed Sentence (224x224) -> 5x5 CV Ridge Probe
vs.
Frozen DenseNet201 -> Native Tiling (224x224 patches) -> 5x5 CV Ridge Probe
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as T

DATA_DIR = "data/sentence_crops_dataset"
INDEX_CSV = f"{DATA_DIR}/sentences_index.csv"

CACHE_NPZ = "data/features/exp7_densenet201_squashed.npz"

def extract_squashed_densenet():
    if Path(CACHE_NPZ).exists():
        d = np.load(CACHE_NPZ, allow_pickle=True)
        return d["X_student"], d["y_student"], d["student_ids"]

    df = pd.read_csv(INDEX_CSV)
    print(f"Extracting squashed features for {len(df)} sentences...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    # Remove classifier
    encoder = model.features.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    t0 = time.time()
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_path = f"{DATA_DIR}/{row['rel_path']}"
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            # encoder output: [1, 1920, 7, 7]
            out = encoder(tensor)
            # Global Average Pooling + Max Pooling -> 3840 dim
            gap = torch.mean(out, dim=[2, 3]).squeeze(0)
            gmp = torch.amax(out, dim=[2, 3]).squeeze(0)
            feat = torch.cat([gap, gmp]).cpu().numpy()
            feats.append(feat)

            if (idx + 1) % 200 == 0:
                print(f"Processed {idx + 1} / {len(df)} in {time.time() - t0:.1f}s")

    X_sentences = np.array(feats, dtype=np.float32)
    df["feat_idx"] = np.arange(len(df))

    # Student-level pooling
    student_ids = np.sort(df["student_id"].unique())

    student_feats = []
    labels = []
    for sid in student_ids:
        s_rows = df[df["student_id"] == sid]
        labels.append(int(s_rows["label"].iloc[0]))
        s_feats = X_sentences[s_rows["feat_idx"].values]
        # mean + std pooling across student's sentences
        mu = np.mean(s_feats, axis=0)
        std = np.std(s_feats, axis=0) if len(s_rows) > 1 else np.zeros_like(mu)
        student_feats.append(np.concatenate([mu, std]))

    X_student = np.array(student_feats, dtype=np.float32)
    y_student = np.array(labels, dtype=np.int64)

    np.savez_compressed(CACHE_NPZ, X_student=X_student, y_student=y_student, student_ids=student_ids)
    print(f"Extraction complete and cached to {CACHE_NPZ}! Student feature shape: {X_student.shape}")
    return X_student, y_student, student_ids

def run_probe(X, y, student_ids):
    df_index = pd.read_csv(INDEX_CSV)
    df_folds = df_index.drop_duplicates(subset=["student_id"]).set_index("student_id").reindex(student_ids)

    n_students = len(y)
    oof_probs = np.zeros((5, n_students), dtype=np.float32)

    for r in range(5):
        fold_col = df_folds[f"repeat_{r}_fold"].values
        for f in range(5):
            te_mask = fold_col == f
            tr_mask = fold_col != f

            X_tr, y_tr = X[tr_mask], y[tr_mask]
            X_te = X[te_mask]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            pca = PCA(n_components=24, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr_sc)
            X_te_pca = pca.transform(X_te_sc)

            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr_pca, y_tr)
            oof_probs[r, te_mask] = clf.predict_proba(X_te_pca)[:, 1]

    mean_oof = np.mean(oof_probs, axis=0)
    auc = roc_auc_score(y, mean_oof)
    ap = average_precision_score(y, mean_oof)

    neg = mean_oof[y == 0]
    pos = mean_oof[y == 1]
    det80 = int(np.sum(pos >= np.percentile(neg, 80.0)))
    det90 = int(np.sum(pos >= np.percentile(neg, 90.0)))

    print("\n" + "=" * 60)
    print("SQUASHED DENSENET201 ABLATION RESULTS (5x5 Repeated CV)")
    print("=" * 60)
    print(f"AUROC: {auc:.4f} | AUPRC: {ap:.4f}")
    print(f"Sens@80% Spec: {det80/len(pos)*100:.1f}% ({det80}/40)")
    print(f"Sens@90% Spec: {det90/len(pos)*100:.1f}% ({det90}/40)")
    print("=" * 60)

if __name__ == "__main__":
    X, y, sids = extract_squashed_densenet()
    run_probe(X, y, sids)
