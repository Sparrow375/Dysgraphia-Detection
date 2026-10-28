"""
Static Ink Execution Timing and Calibration Engine (Solution 1 & 3)
Extracts static hesitation features (curvature entropy, tortuosity, hook density)
and fits cross-validated calibrators on the train split to predict true execution duration
and hardware pressure weights.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import RidgeCV


def extract_static_hesitation_features(
    strokes: List[np.ndarray],
    h_med: float = 25.0,
    binary_mask: np.ndarray = None,
) -> Dict[str, float]:
    """
    Extracts geometric hesitation and complexity features from static strokes.
    """
    if not strokes:
        return {
            "total_arc_px": 0.0,
            "total_arc_norm": 0.0,
            "n_strokes": 0,
            "mean_tortuosity": 1.0,
            "curvature_entropy": 0.0,
            "terminal_hook_ratio": 0.0,
            "h_med": float(h_med),
        }

    h_med = max(float(h_med), 1.0)
    arc_lens = []
    tortuosities = []
    all_kappas = []
    hook_count = 0

    for stk in strokes:
        if len(stk) < 2:
            continue
        diffs = np.diff(stk, axis=0)
        seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
        arc = float(np.sum(seg_lens))
        arc_lens.append(arc)

        # Tortuosity = arc_length / chord_length
        chord = float(np.linalg.norm(stk[-1] - stk[0]))
        tort = arc / max(chord, 1.0)
        tortuosities.append(min(tort, 10.0))

        # Curvature along stroke
        if len(stk) >= 5:
            dx = np.gradient(stk[:, 0])
            dy = np.gradient(stk[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            speed_sq = dx**2 + dy**2
            denom = np.maximum(speed_sq**1.5, 1e-4)
            kappa = np.abs(dx * ddy - dy * ddx) / denom
            all_kappas.extend(kappa)

            # Terminal hook check (sharp turn in last 15% of points)
            k_end = max(int(0.15 * len(stk)), 2)
            end_dx = stk[-1, 0] - stk[-k_end, 0]
            end_dy = stk[-1, 1] - stk[-k_end, 1]
            mid_dx = stk[-k_end, 0] - stk[0, 0]
            mid_dy = stk[-k_end, 1] - stk[0, 1]
            n_end = np.hypot(end_dx, end_dy)
            n_mid = np.hypot(mid_dx, mid_dy)
            if n_end > 2.0 and n_mid > 2.0:
                cos_ang = (end_dx * mid_dx + end_dy * mid_dy) / (n_end * n_mid)
                if cos_ang < 0.2:  # turn > 78 degrees at stroke ending
                    hook_count += 1

    total_arc = float(sum(arc_lens))
    mean_tort = float(np.mean(tortuosities)) if tortuosities else 1.0
    hook_ratio = float(hook_count) / max(len(strokes), 1)

    # Curvature entropy
    if all_kappas:
        k_hist, _ = np.histogram(all_kappas, bins=20, density=True)
        k_hist = k_hist[k_hist > 1e-6]
        curv_entropy = float(-np.sum(k_hist * np.log(k_hist + 1e-6)))
    else:
        curv_entropy = 0.0

    return {
        "total_arc_px": total_arc,
        "total_arc_norm": total_arc / h_med,
        "n_strokes": len(strokes),
        "mean_tortuosity": mean_tort,
        "curvature_entropy": curv_entropy,
        "terminal_hook_ratio": hook_ratio,
        "h_med": float(h_med),
    }


class TimingCalibrator:
    """
    Calibrates true physical execution duration and hardware pressure weights
    strictly using training split samples.
    """

    def __init__(self):
        self.model = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
        self.press_alpha = 0.4
        self.press_beta = 0.6

    def _prepare_X(self, feature_list: List[Dict[str, float]]) -> np.ndarray:
        rows = []
        for f in feature_list:
            rows.append([
                f["total_arc_norm"],
                np.sqrt(max(f["total_arc_norm"], 0.0)),
                float(f["n_strokes"]),
                f["mean_tortuosity"],
                f["curvature_entropy"],
                f["terminal_hook_ratio"],
                f["h_med"]
            ])
        return np.array(rows, dtype=np.float64)

    def fit(self, train_features: List[Dict[str, float]], true_durations: List[float]):
        X = self._prepare_X(train_features)
        y = np.array(true_durations, dtype=np.float64)

        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0

        X_scaled = (X - self.feature_means) / self.feature_stds
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_duration(self, feat_dict: Dict[str, float]) -> float:
        if not self.is_fitted:
            # Fallback heuristic: 0.15s per normalized H_med unit
            return max(feat_dict.get("total_arc_norm", 10.0) * 0.15, 2.0)

        X = self._prepare_X([feat_dict])
        X_scaled = (X - self.feature_means) / self.feature_stds
        pred = float(self.model.predict(X_scaled)[0])
        # Physical lower bound based on minimum handwriting speeds
        min_phys = max(feat_dict.get("total_arc_norm", 10.0) * 0.05, 1.5)
        return max(pred, min_phys)

    def save(self, filepath: str):
        data = {
            "coefs": list(self.model.coef_) if self.is_fitted else [],
            "intercept": float(self.model.intercept_) if self.is_fitted else 0.0,
            "means": list(self.feature_means) if self.feature_means is not None else [],
            "stds": list(self.feature_stds) if self.feature_stds is not None else [],
            "is_fitted": self.is_fitted,
            "press_alpha": self.press_alpha,
            "press_beta": self.press_beta
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        if data.get("is_fitted"):
            self.model.coef_ = np.array(data["coefs"])
            self.model.intercept_ = data["intercept"]
            self.feature_means = np.array(data["means"])
            self.feature_stds = np.array(data["stds"])
            self.press_alpha = data.get("press_alpha", 0.4)
            self.press_beta = data.get("press_beta", 0.6)
            self.is_fitted = True
