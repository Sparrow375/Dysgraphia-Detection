"""
segmentation.py - pipeline steps 6-7:
  6. text lines   : horizontal projection profile (on a virtually de-skewed copy, used ONLY to
                    find the bands; every later measurement uses the original pixels)
  7. words        : connected components + horizontal-gap analysis inside each line

Native resolution is preserved: components are labelled on the original (un-rotated)
clean binary; the skew rotation is applied to *coordinates* only.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import BHKConfig
from .preprocessing import estimate_line_angle


@dataclass
class Comp:
    id: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    main: bool = False
    line: int = -1
    word: int = -1

    @property
    def bbox(self):  # x0, y0, x1, y1 (exclusive)
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Word:
    id: int
    line: int
    comp_ids: List[int]
    bbox: Tuple[int, int, int, int]
    xr: Tuple[float, float]            # extent along the de-skewed x axis


@dataclass
class Line:
    id: int
    comp_ids: List[int] = field(default_factory=list)
    word_ids: List[int] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    band: Tuple[int, int] = (0, 0)     # y-range in the de-skewed frame


@dataclass
class Segmentation:
    ok: bool
    reason: str
    labels: np.ndarray
    comps: List[Comp]
    lines: List[Line]
    words: List[Word]
    skew_deg: float
    M: np.ndarray                      # 2x3 rotation used to de-skew coordinates
    ref_h: float                       # median height of the larger half of components (px)
    profile: Optional[np.ndarray] = None
    bands: List[Tuple[int, int]] = field(default_factory=list)

    def rot(self, pts):
        """Apply the de-skew rotation to Nx2 (x, y) points."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        return pts @ self.M[:, :2].T + self.M[:, 2]


# ----------------------------------------------------------------------------- helpers
def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _split_run(sm, s, e, hmed, cfg):
    """Recursively split an over-tall band at the deepest valley of the smoothed profile."""
    if (e - s) <= cfg.line_split_height_factor * hmed:
        return [(s, e)]
    lo, hi = s + int(0.25 * (e - s)), e - int(0.25 * (e - s))
    if hi <= lo:
        return [(s, e)]
    k = lo + int(np.argmin(sm[lo:hi]))
    if sm[k] < cfg.line_split_valley_frac * sm[s:e].max():
        return _split_run(sm, s, k, hmed, cfg) + _split_run(sm, k, e, hmed, cfg)
    return [(s, e)]


def find_line_bands(deskewed, ref_h, cfg):
    """Horizontal projection profile -> list of (y0, y1) text bands in the de-skewed frame."""
    H = deskewed.shape[0]
    prof = (deskewed > 0).sum(axis=1).astype(float)
    sm = ndi.gaussian_filter1d(prof, max(1.0, cfg.line_profile_sigma_frac_H * H))
    if sm.max() <= 0:
        return [], prof
    runs = _runs(sm > cfg.line_thr_frac * sm.max())
    if not runs:
        return [], prof
    hmed = min(float(np.median([e - s for s, e in runs])), cfg.line_height_hint_factor * ref_h)
    # merge bands separated by tiny gaps (diacritics / ascender bridges)
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] < cfg.line_merge_gap_frac * hmed:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    bands = []
    for s, e in merged:
        bands += _split_run(sm, s, e, hmed, cfg)
    ink = np.array([prof[s:e].sum() for s, e in bands])
    keep = ink >= cfg.line_min_ink_frac * np.median(ink)
    return [b for b, k in zip(bands, keep) if k], prof


# ----------------------------------------------------------------------------- main entry
def segment(clean: np.ndarray, cfg: BHKConfig) -> Segmentation:
    h, w = clean.shape
    n, labels, st, cen = cv2.connectedComponentsWithStats(clean, connectivity=8)
    comps = [Comp(i, int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]),
                  int(st[i, 4]), float(cen[i, 0]), float(cen[i, 1])) for i in range(1, n)]
    I = np.eye(2, 3)
    if len(comps) < cfg.min_words:
        return Segmentation(False, f"only {len(comps)} components after cleaning", labels, comps,
                            [], [], 0.0, I, 0.0)

    areas = np.array([c.area for c in comps])
    big = [c for c in comps if c.area >= np.median(areas)]
    ref_h = float(np.median([c.h for c in big]))                     # letter-height reference
    for c in comps:
        c.main = c.area >= cfg.main_comp_min_area_frac * ref_h ** 2

    # --- global skew of the text via projection-profile sharpness
    skew = estimate_line_angle(clean, vertical=False, search_deg=cfg.skew_search_deg)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew, 1.0)
    deskewed = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_NEAREST)
    bands, prof = find_line_bands(deskewed, ref_h, cfg)
    seg = Segmentation(False, "", labels, comps, [], [], skew, M, ref_h, prof, bands)
    if len(bands) < cfg.min_lines:
        seg.reason = f"{len(bands)} text lines found (need >= {cfg.min_lines})"
        return seg

    # --- assign every component to the band holding its (de-skewed) centroid, else nearest band
    cpts = seg.rot([[c.cx, c.cy] for c in comps])
    centres = np.array([(s + e) / 2.0 for s, e in bands])
    lines = [Line(i, band=(int(b[0]), int(b[1]))) for i, b in enumerate(bands)]
    for c, p in zip(comps, cpts):
        inside = [i for i, (s, e) in enumerate(bands) if s <= p[1] < e]
        li = inside[0] if inside else int(np.argmin(np.abs(centres - p[1])))
        c.line = li
        lines[li].comp_ids.append(c.id)
    lines = [l for l in lines if l.comp_ids]
    for new_i, l in enumerate(lines):          # re-index after dropping empty bands
        l.id = new_i
        for cid in l.comp_ids:
            comps[cid - 1].line = new_i

    # --- words: merge overlapping / near x-intervals (de-skewed x) inside each line
    thr = cfg.word_gap_frac * ref_h
    words: List[Word] = []
    for l in lines:
        iv = []
        for cid in l.comp_ids:
            c = comps[cid - 1]
            x0, y0, x1, y1 = c.bbox
            corners = seg.rot([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            iv.append((corners[:, 0].min(), corners[:, 0].max(), cid))
        iv.sort()
        groups = [[iv[0]]]
        for item in iv[1:]:
            if item[0] - max(g[1] for g in groups[-1]) < thr:
                groups[-1].append(item)
            else:
                groups.append([item])
        for g in groups:
            ids = [it[2] for it in g]
            bx = [comps[i - 1].bbox for i in ids]
            bbox = (min(b[0] for b in bx), min(b[1] for b in bx),
                    max(b[2] for b in bx), max(b[3] for b in bx))
            wd = Word(len(words), l.id, ids, bbox,
                      (min(it[0] for it in g), max(it[1] for it in g)))
            words.append(wd)
            l.word_ids.append(wd.id)
            for i in ids:
                comps[i - 1].word = wd.id
        bx = [comps[i - 1].bbox for i in l.comp_ids]
        l.bbox = (min(b[0] for b in bx), min(b[1] for b in bx),
                  max(b[2] for b in bx), max(b[3] for b in bx))

    seg.lines, seg.words = lines, words
    if len(words) < cfg.min_words:
        seg.reason = f"{len(words)} words found (need >= {cfg.min_words})"
        return seg
    seg.ok, seg.reason = True, "ok"
    return seg
