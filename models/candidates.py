import warnings
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Majority class baseline predicting the empirical positive base rate."""

    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.prior_pos_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if sample_weight is not None:
            self.prior_pos_ = float(np.average(y == 1, weights=sample_weight))
        else:
            self.prior_pos_ = float(np.mean(y == 1))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        p1 = np.full(n_samples, self.prior_pos_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)


def generate_elastic_net_grid() -> List[Dict[str, Any]]:
    """Generate 20 regularized configurations for Elastic-Net Logistic Regression."""
    c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    configs = []
    for c in c_values:
        for l1 in l1_ratios:
            configs.append({"C": c, "l1_ratio": l1})
    # Subsample / select structured 20 configs
    selected_indices = np.linspace(0, len(configs) - 1, 20, dtype=int)
    return [configs[i] for i in selected_indices]


def generate_svm_grid() -> List[Dict[str, Any]]:
    """Generate 20 configurations for SVM-RBF."""
    c_values = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    gamma_values = ["scale", "auto", 0.01, 0.05, 0.1]
    configs = []
    for c in c_values:
        for g in gamma_values:
            configs.append({"C": c, "gamma": g})
    selected_indices = np.linspace(0, len(configs) - 1, 20, dtype=int)
    return [configs[i] for i in selected_indices]


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def generate_elastic_net_grid() -> List[Dict[str, Any]]:
    """Generate focused regularized configurations for Elastic-Net Logistic Regression."""
    return [
        {"C": 0.05, "l1_ratio": 0.2},
        {"C": 0.05, "l1_ratio": 0.7},
        {"C": 0.1, "l1_ratio": 0.3},
        {"C": 0.1, "l1_ratio": 0.7},
        {"C": 0.5, "l1_ratio": 0.3},
        {"C": 0.5, "l1_ratio": 0.7},
        {"C": 1.0, "l1_ratio": 0.5},
        {"C": 5.0, "l1_ratio": 0.5},
    ]


def generate_svm_grid() -> List[Dict[str, Any]]:
    """Generate focused configurations for SVM-RBF."""
    return [
        {"C": 0.1, "gamma": "scale"},
        {"C": 0.5, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 5.0, "gamma": "scale"},
        {"C": 0.5, "gamma": 0.05},
        {"C": 1.0, "gamma": 0.05},
        {"C": 5.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.01},
    ]


def generate_rf_grid() -> List[Dict[str, Any]]:
    """Generate focused configurations for Random Forest."""
    return [
        {"max_depth": 3, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"max_depth": 3, "min_samples_leaf": 5, "max_features": 0.5},
        {"max_depth": 4, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5},
        {"max_depth": 5, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"max_depth": 5, "min_samples_leaf": 5, "max_features": "log2"},
        {"max_depth": 6, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"max_depth": 6, "min_samples_leaf": 5, "max_features": 0.5},
    ]


def generate_gb_grid() -> List[Dict[str, Any]]:
    """Generate focused configurations for GradientBoostingClassifier."""
    return [
        {"learning_rate": 0.05, "max_depth": 2, "subsample": 0.8},
        {"learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
        {"learning_rate": 0.1, "max_depth": 2, "subsample": 0.8},
        {"learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
        {"learning_rate": 0.05, "max_depth": 2, "subsample": 1.0},
        {"learning_rate": 0.1, "max_depth": 2, "subsample": 1.0},
    ]


def get_candidate_registry() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of candidate models, builders, parameter grids, and feature preferences.

    Returns
    -------
    dict
        Registry of candidate models.
    """
    return {
        "majority": {
            "name": "Majority Class",
            "feature_type": "none",
            "builder": lambda **params: MajorityClassifier(),
            "param_grid": [{}],
        },
        "grade_only": {
            "name": "Grade-Only Logistic",
            "feature_type": "grade_only",
            "builder": lambda **params: LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42, **params),
            "param_grid": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        },
        "elastic_net": {
            "name": "Elastic-Net Logistic Regression",
            "feature_type": "z_score",
            "builder": lambda **params: LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=2500,
                random_state=42,
                **params,
            ),
            "param_grid": generate_elastic_net_grid(),
        },
        "svm_rbf": {
            "name": "SVM (RBF Kernel)",
            "feature_type": "z_score",
            "builder": lambda **params: SVC(
                kernel="rbf",
                probability=False,
                random_state=42,
                **params,
            ),
            "param_grid": generate_svm_grid(),
        },
        "random_forest": {
            "name": "Random Forest",
            "feature_type": "raw",
            "builder": lambda **params: RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=42,
                **params,
            ),
            "param_grid": generate_rf_grid(),
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "feature_type": "raw",
            "builder": lambda **params: GradientBoostingClassifier(
                n_estimators=30,
                random_state=42,
                **params,
            ),
            "param_grid": generate_gb_grid(),
        },
    }
