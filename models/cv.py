"""Cross-validation and feature management utilities for Phase 4 modeling.
Enforces zero student leakage across outer and inner folds, computes student-balanced
sample weights, and performs leakage-free fold imputation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def load_dataset(dataset_name_or_path: Union[str, Path]) -> pd.DataFrame:
    """Load a dataset table by alias or absolute/relative path.

    Recognized aliases: 'hindi', 'english', 'combined'.
    """
    aliases = {
        "hindi": WORKSPACE_ROOT / "data" / "datasets" / "dataset_hindi.csv",
        "english": WORKSPACE_ROOT / "data" / "datasets" / "dataset_english.csv",
        "combined": WORKSPACE_ROOT / "data" / "datasets" / "dataset_combined.csv",
    }
    
    path_key = str(dataset_name_or_path).lower().strip()
    if path_key in aliases:
        file_path = aliases[path_key]
    else:
        file_path = Path(dataset_name_or_path)
        if not file_path.is_absolute():
            file_path = WORKSPACE_ROOT / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    if "label" not in df.columns:
        raise ValueError(f"Dataset {file_path.name} missing 'label' column.")
    if "student_id" not in df.columns:
        raise ValueError(f"Dataset {file_path.name} missing 'student_id' column.")

    return df


def get_feature_columns(
    df: pd.DataFrame,
    feature_type: str = "raw",
    include_task_flags: bool = True,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """Extract ordered list of feature column names from dataset dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    feature_type : str
        'raw' for scale-normalized features (prefixed with 'raw_')
        'z_score' for robust z-scores (prefixed with 'z_raw_')
    include_task_flags : bool
        Whether to include one-hot task columns ('task_copy', 'task_dictated', 'task_own').
    exclude_columns : list of str, optional
        Additional columns to explicitly exclude (used in ablation studies).

    Returns
    -------
    list of str
        Clean list of feature column names.
    """
    prefix = "raw_" if feature_type == "raw" else "z_raw_"
    feature_cols = [c for c in df.columns if c.startswith(prefix)]

    if include_task_flags:
        task_cols = [c for c in ["task_copy", "task_dictated", "task_own"] if c in df.columns]
        feature_cols.extend(task_cols)

    if exclude_columns:
        feature_cols = [c for c in feature_cols if c not in exclude_columns]

    return feature_cols


def compute_sample_weights(df: pd.DataFrame, balance_classes: bool = True) -> np.ndarray:
    """Compute per-sentence sample weights balancing student contribution and class imbalance.

    Weight formula:
        w_i = (1 / n_{sentences, s}) * w_{class, y}
    where:
        n_{sentences, s} is total sentences written by student s
        w_{class, y} is inverse student-level class frequency

    Normalized so mean weight equals 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe containing 'student_id' and 'label'.
    balance_classes : bool
        Whether to balance positive and negative classes at the student level.

    Returns
    -------
    np.ndarray
        1D array of sample weights, length == len(df).
    """
    student_counts = df["student_id"].value_counts().to_dict()
    student_inv_freq = df["student_id"].map(lambda sid: 1.0 / student_counts.get(sid, 1.0)).values

    if not balance_classes:
        weights = student_inv_freq
    else:
        # Compute student-level class balance
        student_level = df.drop_duplicates(subset=["student_id"])[["student_id", "label"]]
        n_total_students = len(student_level)
        n_pos_students = (student_level["label"] == 1.0).sum()
        n_neg_students = n_total_students - n_pos_students

        w_pos = (n_total_students / (2.0 * max(n_pos_students, 1)))
        w_neg = (n_total_students / (2.0 * max(n_neg_students, 1)))

        class_weights = np.where(df["label"].values == 1.0, w_pos, w_neg)
        weights = student_inv_freq * class_weights

    # Normalize weights so that mean weight is 1.0
    weights = weights / (np.mean(weights) + 1e-12)
    return weights.astype(np.float64)


def impute_fold_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Impute NaNs inside a cross-validation fold using strictly training-set medians.

    Guarantees zero test-set data leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split dataframe.
    test_df : pd.DataFrame
        Test/validation split dataframe.
    feature_cols : list of str
        Features to extract and impute.

    Returns
    -------
    tuple of (X_train, X_test, medians_dict)
        Numpy arrays of imputed feature matrices and fitted medians.
    """
    train_sub = train_df[feature_cols].copy()
    test_sub = test_df[feature_cols].copy()

    train_medians = train_sub.median(numeric_only=True).to_dict()
    medians_dict: Dict[str, float] = {}

    for col in feature_cols:
        med = train_medians.get(col, 0.0)
        if pd.isna(med):
            med = 0.0
        medians_dict[col] = float(med)
        train_sub[col] = train_sub[col].fillna(med)
        test_sub[col] = test_sub[col].fillna(med)

    X_train = train_sub.values.astype(np.float64)
    X_test = test_sub.values.astype(np.float64)

    return X_train, X_test, medians_dict


def get_outer_fold_split(
    df: pd.DataFrame,
    repeat_idx: int = 0,
    fold_idx: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve train and test splits for a given outer CV repeat and fold.

    Enforces that train and test students are mutually exclusive.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset dataframe with fold columns 'repeat_{r}_fold'.
    repeat_idx : int
        Repeat index (0 to 4).
    fold_idx : int
        Fold index (0 to 4).

    Returns
    -------
    tuple of (train_df, test_df)
    """
    fold_col = f"repeat_{repeat_idx}_fold"
    if fold_col not in df.columns:
        raise KeyError(f"Fold column '{fold_col}' not found in dataframe.")

    test_mask = df[fold_col] == fold_idx
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Integrity verification
    train_students = set(train_df["student_id"])
    test_students = set(test_df["student_id"])
    overlap = train_students.intersection(test_students)
    if len(overlap) > 0:
        raise AssertionError(f"Student leakage detected in repeat {repeat_idx} fold {fold_idx}: {overlap}")

    return train_df, test_df


def get_inner_cv_splits(
    train_df: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate inner cross-validation folds grouped strictly by student_id and stratified by label.

    Parameters
    ----------
    train_df : pd.DataFrame
        Outer training split.
    n_splits : int
        Number of inner validation splits (default 3).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    list of (inner_train_idx, inner_val_idx)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    # We group by student_id and stratify by label
    for inner_tr_idx, inner_val_idx in sgkf.split(
        train_df,
        y=train_df["label"].values.astype(int),
        groups=train_df["student_id"].values,
    ):
        inner_tr_students = set(train_df.iloc[inner_tr_idx]["student_id"])
        inner_val_students = set(train_df.iloc[inner_val_idx]["student_id"])
        assert len(inner_tr_students.intersection(inner_val_students)) == 0, "Student overlap in inner CV!"
        splits.append((inner_tr_idx, inner_val_idx))

    return splits
