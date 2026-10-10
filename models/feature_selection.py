"""Stability Feature Selection for Small-Cohort Dysgraphia Screening.

Implements Randomized Lasso / Subsampled Stability Selection (Meinshausen & Bühlmann, 2010).
Subsamples the training cohort and fits regularized L1 Logistic Regression with random
feature scaling to identify the most robust, non-spurious feature subset.
Guarantees zero test-leakage when fit strictly on training splits.
"""

from typing import List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import warnings


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Stability Feature Selector using Randomized Subsampled L1 Logistic Regression.

    Parameters
    ----------
    n_subsamples : int, default=100
        Number of subsampled bootstrap iterations.
    subsample_fraction : float, default=0.75
        Fraction of samples drawn per iteration (stratified by class).
    c_values : sequence of float, default=(0.05, 0.1, 0.2, 0.5)
        Regularization strengths evaluated per subsample.
    top_k : int or None, default=10
        Number of top stable features to select. If None, uses threshold.
    threshold : float, default=0.30
        Minimum stability selection frequency if top_k is None.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_subsamples: int = 100,
        subsample_fraction: float = 0.75,
        c_values: Sequence[float] = (0.05, 0.1, 0.2, 0.5),
        top_k: Optional[int] = 10,
        threshold: float = 0.30,
        random_state: int = 42,
    ):
        self.n_subsamples = n_subsamples
        self.subsample_fraction = subsample_fraction
        self.c_values = list(c_values)
        self.top_k = top_k
        self.threshold = threshold
        self.random_state = random_state

        self.stability_scores_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Fit stability selection across subsampled iterations on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary class labels (0 or 1).

        Returns
        -------
        self : StabilitySelector
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = [f"f_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)
        feature_counts = np.zeros(n_features, dtype=np.float64)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]

        n_pos = max(1, int(len(pos_idx) * self.subsample_fraction))
        n_neg = max(1, int(len(neg_idx) * self.subsample_fraction))

        # Suppress scikit-learn warnings during iterative fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for _ in range(self.n_subsamples):
                sub_pos = rng.choice(pos_idx, size=n_pos, replace=False)
                sub_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                sub_idx = np.concatenate([sub_pos, sub_neg])

                # Randomized feature scaling: perturb weights in [0.5, 1.0]
                weights = rng.uniform(0.5, 1.0, size=n_features)
                X_sub = X_arr[sub_idx] * weights
                y_sub = y_arr[sub_idx]

                for c in self.c_values:
                    clf = LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=c,
                        class_weight="balanced",
                        random_state=self.random_state,
                    )
                    clf.fit(X_sub, y_sub)
                    selected = np.abs(clf.coef_[0]) > 1e-5
                    feature_counts += selected.astype(np.float64)

        total_trials = self.n_subsamples * len(self.c_values)
        self.stability_scores_ = feature_counts / max(total_trials, 1)

        sorted_indices = np.argsort(self.stability_scores_)[::-1]

        if self.top_k is not None:
            k = min(self.top_k, n_features)
            self.selected_indices_ = sorted_indices[:k]
        else:
            self.selected_indices_ = np.where(self.stability_scores_ >= self.threshold)[0]
            if len(self.selected_indices_) == 0:
                # Fallback to at least top 3 features if threshold is too strict
                self.selected_indices_ = sorted_indices[:min(3, n_features)]

        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[self.selected_indices_] = True

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Filter feature matrix to selected stable features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Subset of features with shape (n_samples, n_selected).
        """
        if self.support_ is None:
            raise RuntimeError("StabilitySelector has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        return X_arr[:, self.support_]

    def get_support(self) -> np.ndarray:
        """Return boolean mask of selected features."""
        if self.support_ is None:
            raise RuntimeError("StabilitySelector has not been fitted yet.")
        return self.support_.copy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        """Return list of selected feature names."""
        if self.support_ is None:
            raise RuntimeError("StabilitySelector has not been fitted yet.")

        names = input_features or self.feature_names_in_ or [f"f_{i}" for i in range(len(self.support_))]
        return [str(names[i]) for i in range(len(self.support_)) if self.support_[i]]
