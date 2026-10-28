"""
features.py - pipeline step 9: the 13 BHK-inspired STATIC, image-derived measurements.

Each extractor returns a FeatureResult holding
    raw     : explicit numerical measurements (dimensionless / normalised unless *_px)
    status  : 'ok'               automated image measurement (still only a PROXY of the BHK item)
              'requires_review'  the BHK item needs character identity / human judgement;
                                 only descriptors + review candidates are produced, no score
              'insufficient_data' not enough lines/words/strokes to measure
    overlay : geometry used (drawn by visualization.py)

Conversion raw -> 0..1 proxy score lives in scoring.py (kept separate on purpose).
Scale handling: spatial values are divided by ref_h (median height of the larger half of the
connected components = a letter-height reference measured in the image itself) or by the ruling
pitch, so results do not depend on scan resolution.  Nothing is resized.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .config import BHKConfig
from .segmentation import Segmentation


@dataclass
class FeatureResult:
    name: str
    status: str
    raw: dict
    notes: str = ""
    overlay: list = field(default_factory=list)   # [(kind, payload)]


@dataclass
class WordZone:
    word_id: int
    line: int
    script: str            # 'latin' (x-height model valid) | 'headline' (Devanagari-like) | 'short'
    x_h: float = np.nan    # core-zone height (px)
    asc: float = np.nan    # ascender extension above core zone (px)
    desc: float = np.nan   # descender extension below core zone (px)
    ref_pt: Optional[tuple] = None    # (x, y) baseline (latin) / headline (headline) point, original coords
    zone_geom: Optional[tuple] = None  # (x_left, x_right, y_top_at_xc, y_bot_at_xc, xc) original coords


# ============================================================================ context
class FeatureContext:
    """Holds the shared inputs and lazily-computed (cached) heavy intermediates."""

    def __init__(self, clean, norm, ink_raw, seg: Segmentation, ruling_info, margin_info, cfg: BHKConfig):
        self.clean, self.norm, self.ink_raw = clean, norm, ink_raw
        self.seg, self.cfg = seg, cfg
        self.ruling_info, self.margin_info = ruling_info, margin_info
        self.H, self.W = clean.shape
        self.ref_h = seg.ref_h
        self.comps = seg.comps
        self.main = [c for c in seg.comps if c.main]
        self.pitch = ruling_info.get("pitch_px")
        self._cache = {}

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ---- skeleton / distance transform ----
    @property
    def skeleton(self):
        return self.cached("skel", lambda: skeletonize(self.clean > 0))

    @property
    def dt(self):
        return self.cached("dt", lambda: cv2.distanceTransform(self.clean, cv2.DIST_L2, 5))

    @property
    def skel_nb(self):
        """Neighbour count of every skeleton pixel (8-neighbourhood)."""
        def f():
            sk = self.skeleton.astype(np.uint8)
            nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
            return nb * sk
        return self.cached("nb", f)

    @property
    def paths(self):
        return self.cached("paths", lambda: skeleton_paths(self))

    @property
    def zones(self) -> List[WordZone]:
        return self.cached("zones", lambda: compute_word_zones(self))


# ============================================================================ small helpers
def _cv(x):
    x = np.asarray(x, float)
    return float(x.std() / x.mean()) if len(x) >= 2 and x.mean() > 0 else None


def _walk(pts):
    """Order the pixels of one simple skeleton branch by walking from an endpoint."""
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(pts)}
    nbrs = []
    for (y, x) in pts:
        nbrs.append([idx[(y + dy, x + dx)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dy or dx) and (y + dy, x + dx) in idx])
    ends = [i for i, l in enumerate(nbrs) if len(l) == 1]
    cur = ends[0] if ends else 0
    seen, order = {cur}, [cur]
    while True:
        nxt = [j for j in nbrs[cur] if j not in seen]
        if not nxt:
            break
        nxt.sort(key=lambda j: abs(pts[j][0] - pts[cur][0]) + abs(pts[j][1] - pts[cur][1]))
        cur = nxt[0]
        seen.add(cur)
        order.append(cur)
    return pts[order]


def skeleton_paths(ctx: FeatureContext):
    """
    Split the skeleton at junctions (junction pixel + 3x3 neighbourhood removed) into simple
    branches and order each one.  Returns [{'comp': id, 'xy': Nx2 float (x, y)}].
    """
    sk = ctx.skeleton.astype(np.uint8)
    junction = ((sk == 1) & (ctx.skel_nb >= 3)).astype(np.uint8)
    jd = cv2.dilate(junction, np.ones((3, 3), np.uint8))
    sk2 = sk * (1 - jd)
    n, lab = cv2.connectedComponents(sk2, connectivity=8)
    ys, xs = np.nonzero(sk2)
    labs = lab[ys, xs]
    order = np.argsort(labs, kind="stable")
    ys, xs, labs = ys[order], xs[order], labs[order]
    cuts = np.where(np.diff(labs) != 0)[0] + 1
    out = []
    for gy, gx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        if len(gy) < 3:
            continue
        o = _walk(np.stack([gy, gx], 1))
        out.append(dict(comp=int(ctx.seg.labels[o[0, 0], o[0, 1]]),
                        xy=np.stack([o[:, 1], o[:, 0]], 1).astype(float)))
    return out


def corners_on_path(xy, k, thr_deg, smooth=3):
    """
    Sharp-angle detector on one ordered skeleton branch.
    turn(i) = angle between incoming vector P[i]-P[i-k] and outgoing vector P[i+k]-P[i]
    (coordinates first smoothed by a moving average).  Runs of consecutive indices with
    turn >= thr_deg count as ONE corner (placed at the maximum).
    """
    n = len(xy)
    if n < 2 * k + 3:
        return []
    P = ndi.uniform_filter1d(xy, size=max(1, smooth), axis=0, mode="nearest")
    a = P[k:n - k] - P[0:n - 2 * k]
    b = P[2 * k:n] - P[k:n - k]
    ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                                       (a * b).sum(1))))
    hit = ang >= thr_deg
    pts = []
    m = np.concatenate([[0], hit.astype(np.int8), [0]])
    d = np.diff(m)
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        j = s + int(np.argmax(ang[s:e]))
        pts.append(tuple(P[k + j]))
    return pts


# ============================================================================ word zones
def compute_word_zones(ctx: FeatureContext) -> List[WordZone]:
    """
    Per-word vertical ink profile (rows de-skewed by shifting y by -(x-xc)*tan(skew), i.e. no
    resampling).
    * 'headline' word (Devanagari-like): some row is covered by >= headline_cov of the word
      width and lies in the upper part -> reference point = that row.
    * otherwise 'latin' model: core zone = contiguous rows around the profile peak whose ink >=
      core_frac * peak.  x_h = core height, asc = rows above core, desc = rows below core.
    """
    cfg, seg = ctx.cfg, ctx.seg
    t = np.tan(np.deg2rad(seg.skew_deg))
    zones = []
    for wd in seg.words:
        x0, y0, x1, y1 = wd.bbox
        sub = seg.labels[y0:y1, x0:x1]
        main_ids = [i for i in wd.comp_ids if ctx.comps[i - 1].main]   # specks / ruling dust must not shape the profile
        ys, xs = np.nonzero(np.isin(sub, main_ids))
        if len(ys) == 0:
            continue
        xc = (xs.min() + xs.max()) / 2.0
        yc = ys - (xs - xc) * t
        ymin = yc.min()
        yi = np.round(yc - ymin).astype(int)
        prof = np.bincount(yi).astype(float)
        prof_raw = prof
        prof = ndi.gaussian_filter1d(prof, max(1.0, cfg.zone_smooth_frac * ctx.ref_h))   # smooth: peak AREA, not 1-px spikes
        width = xs.max() - xs.min() + 1
        if len(prof) < cfg.min_zone_word_height_frac * ctx.ref_h:
            zones.append(WordZone(wd.id, wd.line, "short"))
            continue
        cov = prof_raw / width
        k = int(np.argmax(prof))
        if cov.max() >= cfg.headline_cov and k <= 0.45 * len(prof):
            zones.append(WordZone(wd.id, wd.line, "headline",
                                  ref_pt=(x0 + xc, y0 + ymin + k)))
            continue
        reg = prof >= cfg.core_frac * prof.max()
        i0 = i1 = k
        while i0 > 0 and reg[i0 - 1]:
            i0 -= 1
        while i1 < len(prof) - 1 and reg[i1 + 1]:
            i1 += 1
        z = WordZone(wd.id, wd.line, "latin", x_h=float(i1 - i0 + 1), asc=float(i0),
                     desc=float(len(prof) - 1 - i1),
                     ref_pt=(x0 + xc, y0 + ymin + i1),
                     zone_geom=(x0 + xs.min(), x0 + xs.max(), y0 + ymin + i0, y0 + ymin + i1, x0 + xc))
        zones.append(z)
    return zones


# ============================================================================ 1. writing size
def f_writing_size(ctx):
    cfg = ctx.cfg
    hs = [c.h for c in ctx.main]
    if not hs:
        return FeatureResult("writing_size", "insufficient_data", {}, "no main components")
    if ctx.pitch:
        ref_name, ref_px = "ruling_pitch", ctx.pitch
    else:
        cs = sorted((b[0] + b[1]) / 2.0 for b in ctx.seg.bands)
        ref_name, ref_px = ("text_line_pitch", float(np.median(np.diff(cs)))) if len(cs) >= 2 else (None, None)
    lat = [z.x_h for z in ctx.zones if z.script == "latin"]
    raw = dict(
        ref_h_px=float(ctx.ref_h),
        median_main_comp_height_px=float(np.median(hs)),
        norm_reference=ref_name, norm_reference_px=ref_px,
        size_norm=(ctx.ref_h / ref_px) if ref_px else None,        # letter height / ruling pitch
        size_mm=(ctx.ref_h / cfg.dpi * 25.4) if cfg.dpi > 0 else None,
        latin_xheight_px=float(np.median(lat)) if lat else None,
        latin_xheight_norm=(float(np.median(lat)) / ref_px) if (lat and ref_px) else None,
    )
    return FeatureResult("writing_size", "ok" if ref_px else "insufficient_data", raw,
                         "size_norm = ref_h / ruling pitch (falls back to text-line pitch). Absolute mm only if cfg.dpi is set.")


# ============================================================================ 2. left margin
def f_left_margin(ctx):
    seg = ctx.seg
    lefts, pts = [], []
    for l in seg.lines:
        ms = [ctx.comps[i - 1] for i in l.comp_ids if ctx.comps[i - 1].main]
        if not ms:
            continue
        p = seg.rot([[c.x, c.y + c.h / 2.0] for c in ms])[:, 0]
        j = int(np.argmin(p))
        lefts.append(p[j])
        pts.append((ms[j].x, ms[j].y + ms[j].h / 2.0))
    if len(lefts) < 2:
        return FeatureResult("non_aligned_left_margin", "insufficient_data", dict(n_lines=len(lefts)))
    lefts = np.array(lefts)
    idx = np.arange(len(lefts))
    slope = float(np.polyfit(idx, lefts, 1)[0]) if len(lefts) >= 3 else float(lefts[-1] - lefts[0])
    mcs = ctx.margin_info.get("centres") or []
    dist = (float(np.mean([p[0] for p in pts])) - max(mcs)) / ctx.ref_h if mcs else None
    raw = dict(n_lines=len(lefts), left_x_deskewed_px=[round(float(v), 1) for v in lefts],
               std_px=float(lefts.std()), std_norm=float(lefts.std() / ctx.ref_h),
               range_norm=float((lefts.max() - lefts.min()) / ctx.ref_h),
               drift_per_line_norm=slope / ctx.ref_h,
               total_drift_norm=slope * (len(lefts) - 1) / ctx.ref_h,
               mean_distance_to_page_margin_norm=dist)
    fit = np.polyfit(idx, lefts, 1) if len(lefts) >= 3 else None
    return FeatureResult("non_aligned_left_margin", "ok", raw,
                         "Paragraph indents count as misalignment.", [("points", pts, "left_edge")])


# ============================================================================ 3. alignment / skew
def f_alignment(ctx):
    cfg, seg = ctx.cfg, ctx.seg
    ref_angle = ctx.ruling_info.get("angle_deg") if ctx.ruling_info.get("n_lines", 0) >= 2 else None
    angles, rels, rms, segs, allpts = [], [], [], [], []
    for l in seg.lines:
        zs = [z for z in ctx.zones if z.line == l.id and z.ref_pt is not None]
        grp = [z for z in zs if z.script == "latin"]
        grp2 = [z for z in zs if z.script == "headline"]
        grp = grp if len(grp) >= len(grp2) else grp2
        if len(grp) < cfg.min_words_baseline:
            continue
        P = np.array([z.ref_pt for z in grp], np.float32)
        vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()]
        if abs(vx) < 1e-6:
            continue
        ang = float(np.degrees(np.arctan2(vy, vx)))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
        res = P[:, 1] - (y0 + (P[:, 0] - x0) * vy / vx)
        angles.append(ang)
        rels.append(ang - (ref_angle or 0.0))
        rms.append(float(np.sqrt(np.mean(res ** 2)) / ctx.ref_h))
        xa, xb = float(P[:, 0].min()), float(P[:, 0].max())
        segs.append((xa, y0 + (xa - x0) * vy / vx, xb, y0 + (xb - x0) * vy / vx))
        allpts += [tuple(p) for p in P]
    if not angles:
        return FeatureResult("skewed_alignment", "insufficient_data", dict(n_lines_fitted=0))
    raw = dict(n_lines_fitted=len(angles), reference="ruling" if ref_angle is not None else "image_axis",
               ruling_angle_deg=ref_angle, line_angles_deg=[round(a, 2) for a in angles],
               rel_angles_deg=[round(a, 2) for a in rels],
               mean_abs_rel_angle_deg=float(np.mean(np.abs(rels))), std_angle_deg=float(np.std(angles)),
               global_text_skew_deg=float(seg.skew_deg),
               mean_rms_resid_norm=float(np.mean(rms)))
    return FeatureResult("skewed_alignment", "ok", raw,
                         "Baseline = robust (Huber) line through word baseline points (Latin) or headline points (Devanagari). "
                         "A single global ruling angle is the reference, so perspective differences between rulings add error.",
                         [("segments", segs, "baseline"), ("points", allpts, "baseline_pt")])


# ============================================================================ 4. word spacing
def f_word_spacing(ctx):
    cfg, seg = ctx.cfg, ctx.seg
    gaps, rects = [], []
    for l in seg.lines:
        ws = sorted((seg.words[i] for i in l.word_ids), key=lambda w: w.xr[0])
        for a, b in zip(ws[:-1], ws[1:]):
            gaps.append(b.xr[0] - a.xr[1])
            rects.append((a.bbox[2], l.bbox[1], b.bbox[0], l.bbox[3], gaps[-1] / ctx.ref_h))
    if len(gaps) < 2:
        return FeatureResult("insufficient_word_spacing", "insufficient_data", dict(n_gaps=len(gaps)))
    g = np.array(gaps) / ctx.ref_h
    raw = dict(n_gaps=len(g), median_gap_norm=float(np.median(g)), mean_gap_norm=float(g.mean()),
               min_gap_norm=float(g.min()), gap_cv=_cv(g),
               frac_gaps_narrow=float((g < cfg.narrow_gap_frac).mean()),
               narrow_threshold_norm=cfg.narrow_gap_frac, word_split_threshold_norm=cfg.word_gap_frac)
    return FeatureResult("insufficient_word_spacing", "ok", raw,
                         "Words closer than word_gap_frac*ref_h are merged by the segmenter, so extremely tight spacing "
                         "shows up as merged words (fewer gaps) rather than as small gaps.",
                         [("gaps", rects, cfg.narrow_gap_frac)])


# ============================================================================ 5. sharp angles
def f_sharp_angles(ctx):
    cfg = ctx.cfg
    k = max(3, int(round(cfg.curv_step_frac * ctx.ref_h)))
    min_len = cfg.min_path_len_frac * ctx.ref_h
    pts, n_paths = [], 0
    for p in ctx.paths:
        if len(p["xy"]) < min_len:
            continue
        n_paths += 1
        pts += corners_on_path(p["xy"], k, cfg.sharp_turn_deg, cfg.curv_smooth_px)
    length_norm = float(ctx.skeleton.sum() / ctx.ref_h)
    if length_norm <= 0 or not n_paths:
        return FeatureResult("sharp_angles", "insufficient_data", {})
    raw = dict(n_sharp=len(pts), n_paths=n_paths, arm_px=k, skeleton_length_norm=length_norm,
               sharp_per_10_refh=len(pts) / length_norm * 10.0,
               sharp_per_main_comp=len(pts) / max(1, len(ctx.main)))
    return FeatureResult("sharp_angles", "ok", raw,
                         "Corners at skeleton junctions are not counted; pen-lift stroke ends are not corners. "
                         "Angular print-style letters (e.g. 'A', 'E', 'K', Devanagari matras) legitimately produce sharp angles.",
                         [("points", pts, "sharp")])


# ============================================================================ 6. broken links
def _nested(a, b):
    """component a lies (almost) inside the bbox of b."""
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    return ax0 >= bx0 - 2 and ay0 >= by0 - 2 and ax1 <= bx1 + 2 and ay1 <= by1 + 2


def f_broken_links(ctx):
    seg, comps = ctx.seg, ctx.comps
    breaks, gaps, rects, used = 0, [], [], 0
    wb, n_main = 0, 0
    for wd in seg.words:
        ms = [comps[i - 1] for i in wd.comp_ids if comps[i - 1].main]
        ms = [a for a in ms if not any((b is not a) and _nested(a, b) for b in ms)]
        if not ms:
            continue
        used += 1
        n_main += len(ms)
        ms.sort(key=lambda c: c.x)
        nb = 0
        for a, b in zip(ms[:-1], ms[1:]):
            gap = max(0, b.x - (a.x + a.w))
            gaps.append(gap / ctx.ref_h)
            nb += 1
            rects.append((a.x + a.w, min(a.y, b.y), max(a.x + a.w + 1, b.x), max(a.y + a.h, b.y + b.h)))
        breaks += nb
        wb += nb > 0
    if used == 0:
        return FeatureResult("broken_links", "insufficient_data", {})
    raw = dict(n_words_used=used, n_main_pieces=n_main, n_breaks=breaks, breaks_per_word=breaks / used,
               frac_words_with_breaks=wb / used, breaks_per_main_comp=breaks / max(1, n_main),
               median_break_gap_norm=float(np.median(gaps)) if gaps else None)
    return FeatureResult("broken_links", "ok", raw,
                         "Every extra connected piece inside a word counts as a break. BHK assumes cursive; print-style "
                         "writing breaks by design. Ruling removal can also create artificial breaks where strokes sat on the line.",
                         [("rects", rects, "break")])


# ============================================================================ 7. collisions
def _inter_frac(a, b):
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / float(min(a.w * a.h, b.w * b.h))


def f_collisions(ctx):
    cfg, seg, comps = ctx.cfg, ctx.seg, ctx.comps
    pairs, rects = 0, []

    def check(a, b):
        small, big = (a, b) if a.area < b.area else (b, a)
        if small.area < cfg.collision_diacritic_area_ratio * big.area:
            return False
        f = _inter_frac(a, b)
        return cfg.collision_overlap_frac <= f < 0.9    # >=0.9 = one inside the other (counter/loop), skip

    within = inter = 0
    for wd in seg.words:
        ms = [comps[i - 1] for i in wd.comp_ids if comps[i - 1].main]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if check(ms[i], ms[j]):
                    within += 1
                    rects.append((min(ms[i].x, ms[j].x), min(ms[i].y, ms[j].y),
                                  max(ms[i].x + ms[i].w, ms[j].x + ms[j].w), max(ms[i].y + ms[i].h, ms[j].y + ms[j].h)))
    for l1, l2 in zip(seg.lines[:-1], seg.lines[1:]):
        m1 = [comps[i - 1] for i in l1.comp_ids if comps[i - 1].main]
        m2 = [comps[i - 1] for i in l2.comp_ids if comps[i - 1].main]
        for a in m1:
            for b in m2:
                if check(a, b):
                    inter += 1
                    rects.append((min(a.x, b.x), min(a.y, b.y), max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)))
    n = max(1, len(ctx.main))
    raw = dict(n_main_comps=len(ctx.main), n_overlaps_within_words=within, n_overlaps_between_lines=inter,
               collisions_per_main_comp=(within + inter) / n)
    return FeatureResult("collisions", "ok", raw,
                         "Proxy only: bounding-box overlap of distinct connected components. Letters that merge into ONE "
                         "component (touching) cannot be separated without character identity -> manual review recommended.",
                         [("rects", rects, "collision")])


# ============================================================================ 8. irregular size
def f_irregular_size(ctx):
    per_line, allh = [], []
    for l in ctx.seg.lines:
        h = np.array([ctx.comps[i - 1].h for i in l.comp_ids if ctx.comps[i - 1].main], float)
        if len(h) >= 3:
            q1, q3 = np.percentile(h, [25, 75])
            per_line.append(float((q3 - q1) / np.median(h)))
        allh += list(h)
    if not per_line:
        return FeatureResult("irregular_letter_size", "insufficient_data", dict(n_lines_used=0))
    raw = dict(n_lines_used=len(per_line), per_line_iqr_over_median=[round(v, 3) for v in per_line],
               mean_line_iqr_over_median=float(np.mean(per_line)), page_height_cv=_cv(allh))
    return FeatureResult("irregular_letter_size", "ok", raw,
                         "IQR/median of main-component heights per line (robust to a few ascenders). Cursive words are single "
                         "components, so this is a word-level, not letter-level, size spread.")


# ============================================================================ 9. extension heights
def f_extension_heights(ctx):
    cfg = ctx.cfg
    lat = [z for z in ctx.zones if z.script == "latin"]
    head = sum(z.script == "headline" for z in ctx.zones)
    if len(lat) < 3:
        return FeatureResult("inconsistent_height_extensions", "insufficient_data",
                             dict(n_latin_words=len(lat), n_headline_words=head),
                             "x-height model is only valid for Latin-like words; Devanagari headline words are excluded.")
    thr = cfg.ext_min_ratio
    noext = [z for z in lat if z.asc < thr * z.x_h and z.desc < thr * z.x_h]
    asc = [z for z in lat if z.asc >= thr * z.x_h]
    desc = [z for z in lat if z.desc >= thr * z.x_h]
    raw = dict(n_latin_words=len(lat), n_headline_words_excluded=head, n_no_ext=len(noext), n_asc=len(asc), n_desc=len(desc),
               xh_cv_no_ext=_cv([z.x_h for z in noext]),
               asc_ext_cv=_cv([z.asc for z in asc]), desc_ext_cv=_cv([z.desc for z in desc]),
               xh_median_px=float(np.median([z.x_h for z in lat])),
               asc_to_xh_median=float(np.median([z.asc / z.x_h for z in asc])) if asc else None,
               desc_to_xh_median=float(np.median([z.desc / z.x_h for z in desc])) if desc else None)
    geoms = [z.zone_geom for z in lat if z.zone_geom]
    return FeatureResult("inconsistent_height_extensions", "ok", raw,
                         "Zones per word from the row profile; ascender/descender heights are compared across words.",
                         [("zones", geoms, np.tan(np.deg2rad(ctx.seg.skew_deg)))])


# ============================================================================ 10/11. review features
def comp_descriptors(ctx):
    def f():
        sk, nb, labels = ctx.skeleton, ctx.skel_nb, ctx.seg.labels
        out = {}
        for c in ctx.comps:
            x0, y0, x1, y1 = c.bbox
            m = (labels[y0:y1, x0:x1] == c.id).astype(np.uint8)
            cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            holes = int(sum(1 for h in hier[0] if h[3] != -1))
            big = max(cnts, key=cv2.contourArea)
            ha = cv2.contourArea(cv2.convexHull(big))
            ss = sk[y0:y1, x0:x1] & (m > 0)
            nbs = nb[y0:y1, x0:x1]
            out[c.id] = dict(aspect=c.w / float(c.h), fill=c.area / float(c.w * c.h),
                             solidity=c.area / ha if ha > 0 else 1.0, holes=holes,
                             endpoints=int((ss & (nbs == 1)).sum()), junction_px=int((ss & (nbs >= 3)).sum()))
        return out
    return ctx.cached("desc", f)


def f_atypical(ctx):
    cfg = ctx.cfg
    D = comp_descriptors(ctx)
    ids = [c.id for c in ctx.main if c.id in D]
    if len(ids) < 8:
        return FeatureResult("atypical_letter_forms", "requires_review", dict(n_main_comps=len(ids)),
                             "Too few components for outlier statistics. Needs character identity / manual review.")
    keys = ["aspect", "fill", "solidity", "holes", "endpoints"]
    M = np.array([[np.log(D[i]["aspect"]) if k == "aspect" else D[i][k] for k in keys] for i in ids], float)
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826
    z = np.zeros_like(M)
    ok = mad > 1e-9
    z[:, ok] = (M[:, ok] - med[ok]) / mad[ok]
    out_ids = [i for i, zr in zip(ids, z) if np.any(np.abs(zr) > cfg.outlier_z)]
    raw = dict(n_main_comps=len(ids), n_shape_outliers=len(out_ids), outlier_frac=len(out_ids) / len(ids),
               outlier_comp_ids=out_ids, descriptor_medians=dict(zip(keys, [float(v) for v in med])))
    rects = [ctx.comps[i - 1].bbox for i in out_ids]
    return FeatureResult("atypical_letter_forms", "requires_review", raw,
                         "REQUIRES REVIEW: deciding whether a form is atypical needs character identity. Shape-descriptor "
                         "outliers (robust MAD z-score) are listed only as candidates for a human reviewer; no score is given.",
                         [("rects", rects, "atypical")])


def f_ambiguous(ctx):
    cfg = ctx.cfg
    D = comp_descriptors(ctx)
    ext = [c.id for c in ctx.main if c.id in D and (D[c.id]["aspect"] > cfg.aspect_extreme or D[c.id]["aspect"] < 1 / cfg.aspect_extreme)]
    frag = [c.id for c in ctx.comps if (not c.main) and c.h >= cfg.fragment_min_h_frac * ctx.ref_h]
    raw = dict(n_extreme_aspect=len(ext), extreme_aspect_ids=ext, n_stray_fragments=len(frag), stray_fragment_ids=frag,
               candidate_frac=(len(ext) + len(frag)) / max(1, len(ctx.comps)))
    rects = [ctx.comps[i - 1].bbox for i in ext + frag]
    return FeatureResult("ambiguous_letter_forms", "requires_review", raw,
                         "REQUIRES REVIEW: legibility/ambiguity of a letter cannot be decided without recognising it (and "
                         "OCR is excluded by design). Components with extreme proportions and stray fragments are only candidates.",
                         [("rects", rects, "ambiguous")])


# ============================================================================ 12. traced / corrected
def f_traced(ctx):
    cfg = ctx.cfg
    sk, dt = ctx.skeleton, ctx.dt
    if sk.sum() == 0:
        return FeatureResult("traced_corrected_letters", "insufficient_data", {})
    w0 = float(np.median(2 * dt[sk]))                                   # median stroke width (px)
    blob_mask = (dt >= cfg.thick_factor * w0 / 2.0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(blob_mask, connectivity=8)
    blobs = [tuple(cen[i]) for i in range(1, n) if st[i, 4] >= cfg.thick_blob_min_area_frac * w0 ** 2]
    cnts, hier = cv2.findContours(ctx.clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for c, h in zip(cnts, hier[0]):
        if h[3] != -1:
            a = cv2.contourArea(c)
            if cfg.tiny_hole_min_area_px <= a <= cfg.tiny_hole_area_frac * ctx.ref_h ** 2:
                m = c.reshape(-1, 2).mean(0)
                holes.append((float(m[0]), float(m[1])))
    # smudge: non-paper gray far from any ink / ruling / margin pixel, inside the text area
    near = cv2.dilate(ctx.ink_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.smudge_halo_px + 1,) * 2))
    x0 = min(l.bbox[0] for l in ctx.seg.lines); y0 = min(l.bbox[1] for l in ctx.seg.lines)
    x1 = max(l.bbox[2] for l in ctx.seg.lines); y1 = max(l.bbox[3] for l in ctx.seg.lines)
    sm = (ctx.norm < cfg.smudge_gray_level) & (near == 0)
    area = max(1, (x1 - x0) * (y1 - y0))
    smf = float(sm[y0:y1, x0:x1].sum() / area)
    nm = max(1, len(ctx.main))
    raw = dict(median_stroke_width_px=w0, median_stroke_width_norm=w0 / ctx.ref_h,
               n_thick_blobs=len(blobs), thick_blobs_per_main_comp=len(blobs) / nm,
               n_tiny_holes=len(holes), tiny_holes_per_main_comp=len(holes) / nm, smudge_fraction=smf)
    return FeatureResult("traced_corrected_letters", "ok", raw,
                         "Proxy: over-written strokes -> locally very thick ink; ink-filled loops -> tiny holes; erasures -> grey "
                         "smudge away from ink. Heavy pen pressure or JPEG blur can mimic these.",
                         [("points", blobs, "thick"), ("points", holes, "hole"), ("mask_box", (x0, y0, x1, y1, sm[y0:y1, x0:x1]), "smudge")])


# ============================================================================ 13. unstable trace
def f_unstable(ctx):
    cfg = ctx.cfg
    dt = ctx.dt
    step = max(2, int(round(cfg.jitter_step_frac * ctx.ref_h)))
    min_len = cfg.min_path_len_frac * ctx.ref_h
    cvs, ws, zz, total_len, pts = [], [], 0, 0.0, []
    for p in ctx.paths:
        xy = p["xy"]
        n = len(xy)
        if n < min_len:
            continue
        total_len += n
        t = int(n * cfg.width_trim_frac)
        wd = 2 * dt[xy[t:n - t, 1].astype(int), xy[t:n - t, 0].astype(int)]
        if len(wd) >= 5 and wd.mean() > 0:
            cvs.append(wd.std() / wd.mean())
            ws.append(len(wd))
        Q = ndi.uniform_filter1d(xy, size=cfg.jitter_smooth_px, axis=0, mode="nearest")[::step]
        if len(Q) < 4:
            continue
        d = np.diff(Q, axis=0)
        hd = np.arctan2(d[:, 1], d[:, 0])
        dth = np.degrees((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi)
        sig = np.abs(dth) >= cfg.jitter_min_turn_deg
        sgn = np.sign(dth)
        for i in range(len(dth) - 1):
            if sig[i] and sig[i + 1] and sgn[i] != sgn[i + 1]:
                zz += 1
                pts.append(tuple(Q[i + 1]))
    if not cvs or total_len == 0:
        return FeatureResult("unstable_trace", "insufficient_data", {})
    raw = dict(n_paths_used=len(cvs), width_cv=float(np.average(cvs, weights=ws)),
               n_zigzags=zz, zigzag_per_refh=zz / (total_len / ctx.ref_h), sample_step_px=step)
    return FeatureResult("unstable_trace", "ok", raw,
                         "Stroke-width coefficient of variation along skeleton branches + count of alternating heading changes "
                         "(tremor proxy). Static image only: no pen kinematics are available or used.",
                         [("points", pts, "wobble")])


# ============================================================================ driver
EXTRACTORS = [f_writing_size, f_left_margin, f_alignment, f_word_spacing, f_sharp_angles, f_broken_links,
              f_collisions, f_irregular_size, f_extension_heights, f_atypical, f_ambiguous, f_traced, f_unstable]


def extract_all(ctx: FeatureContext) -> List[FeatureResult]:
    """Run all 13 extractors (caller must have checked that segmentation succeeded)."""
    if not ctx.seg.ok:
        raise RuntimeError("segmentation failed - BHK features must not be computed")
    return [fn(ctx) for fn in EXTRACTORS]
