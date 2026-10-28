"""
scoring.py - pipeline step 10: raw measurement -> 0..1 PROXY score, with the rule spelled out.

These scores are NOT official BHK points and the anchors in config.BHKConfig.anchors are
placeholders (the BHK source gives no pixel thresholds for an automated system).
score = 0 -> measurement at/below `lo`;  score = 1 -> at/above `hi`  (linear in between).
For a feature with several sub-measurements the score is the MAX of the available sub-scores.
Features with status 'requires_review' or 'insufficient_data' get score = None on purpose.
"""
import numpy as np

SCORE_TYPE = "proxy_0_1_uncalibrated (NOT an official BHK score)"


def _ramp(v, lo, hi, invert=False):
    if invert:                      # smaller value is worse: 0 at v>=lo, 1 at v<=hi (lo > hi)
        return float(np.clip((lo - v) / (lo - hi), 0.0, 1.0))
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _subrules(name, cfg):
    a = cfg.anchors.get(name)
    if a is None:
        return []
    if "keys" in a:
        out = []
        for t in a["keys"]:
            out.append(dict(key=t[0], lo=t[1], hi=t[2], invert=len(t) > 3 and t[3] == "invert"))
        return out
    return [dict(key=a["key"], lo=a["lo"], hi=a["hi"], invert=False)]


def describe_rule(name, status_hint, cfg):
    """Human-readable conversion rule (also stored in the JSON output)."""
    if status_hint == "requires_review":
        return ("No automated score: the BHK item needs character identity / human judgement. Only descriptors and "
                "review candidates are reported.")
    if name == "writing_size":
        a = cfg.anchors[name]
        return (f"v = ref_h / ruling_pitch. score = min(1, d / {a['ramp']}) where d = distance of v outside the "
                f"placeholder band [{a['band_lo']}, {a['band_hi']}] (0 inside the band). Too small AND too large both score >0.")
    parts = []
    for r in _subrules(name, cfg):
        if r["invert"]:
            parts.append(f"{r['key']}: 0 at >= {r['lo']}, 1 at <= {r['hi']} (smaller is worse)")
        else:
            parts.append(f"{r['key']}: 0 at <= {r['lo']}, 1 at >= {r['hi']}")
    return "score = max over [" + "; ".join(parts) + "], linear ramp in between. Anchors are placeholders."


def score_feature(fr, cfg):
    """Return dict(score, score_status, rule, sub_scores, score_type)."""
    rule = describe_rule(fr.name, fr.status, cfg)
    base = dict(rule=rule, score_type=SCORE_TYPE, sub_scores={})
    if fr.status == "requires_review":
        return dict(base, score=None, score_status="requires_review")
    if fr.status != "ok":
        return dict(base, score=None, score_status="insufficient_data")
    if fr.name == "writing_size":
        a = cfg.anchors["writing_size"]
        v = fr.raw.get("size_norm")
        if v is None:
            return dict(base, score=None, score_status="insufficient_data")
        d = max(a["band_lo"] - v, v - a["band_hi"], 0.0)
        s = float(min(1.0, d / a["ramp"]))
        return dict(base, score=s, score_status="scored", sub_scores={"size_norm": s})
    subs = {}
    for r in _subrules(fr.name, cfg):
        v = fr.raw.get(r["key"])
        if v is not None and np.isfinite(v):
            subs[r["key"]] = _ramp(v, r["lo"], r["hi"], r["invert"])
    if not subs:
        return dict(base, score=None, score_status="insufficient_data")
    return dict(base, score=float(max(subs.values())), score_status="scored", sub_scores=subs)
