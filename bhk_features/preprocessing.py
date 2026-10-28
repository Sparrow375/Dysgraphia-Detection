"""
preprocessing.py - pipeline steps 1-5:
  1. load + validate            4. detect/remove the vertical margin
  2. grayscale + denoise        5. clean binary handwriting image
  3. detect/remove horizontal ruling

Nothing here resizes the image: every output has the input's pixel grid.
Ink is 255, background 0 in all binary images.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import BHKConfig


# ----------------------------------------------------------------------------- 1. load / validate
def load_and_validate(path, cfg: BHKConfig) -> np.ndarray:
    """Read an image (BGR) and raise ValueError with a clear reason if unusable."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)  # honours EXIF rotation; drops alpha
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {p}")
    h, w = bgr.shape[:2]
    if min(h, w) < cfg.min_image_side_px:
        raise ValueError(f"Image too small ({w}x{h}); min side = {cfg.min_image_side_px}px")
    if max(h, w) > cfg.max_image_side_px:
        raise ValueError(f"Image too large ({w}x{h}); max side = {cfg.max_image_side_px}px")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < cfg.min_gray_std:
        raise ValueError(f"Image looks blank / has almost no contrast (gray std={gray.std():.1f})")
    return bgr


# ----------------------------------------------------------------------------- 2. gray + denoise
def to_gray_denoised(bgr: np.ndarray, cfg: BHKConfig) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = cfg.denoise_method
    if m == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=cfg.nlm_h,
                                        templateWindowSize=5, searchWindowSize=15)
    if m == "median":
        return cv2.medianBlur(gray, cfg.median_ksize)
    if m == "gaussian":
        return cv2.GaussianBlur(gray, (0, 0), cfg.gaussian_sigma)
    return gray


def normalise_illumination(gray: np.ndarray, cfg: BHKConfig) -> np.ndarray:
    """
    Flatten uneven lighting:  norm = gray / background * 255.
    background = grey-closing (removes thin dark strokes) followed by a blur.
    Output is uint8, paper ~255, ink dark.
    """
    mn = min(gray.shape[:2])
    k = max(3, int(cfg.bg_kernel_frac * mn) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    kb = max(3, int(cfg.bg_blur_frac * mn) | 1)
    bg = cv2.GaussianBlur(bg, (kb, kb), 0).astype(np.float32)
    norm = gray.astype(np.float32) * 255.0 / np.maximum(bg, 1.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarise(norm: np.ndarray, cfg: BHKConfig):
    """Ink mask (255 = ink). T = min(Otsu + offset, ink_max_level). Returns (mask, T)."""
    otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(otsu + cfg.ink_otsu_offset, cfg.ink_max_level)
    return ((norm < t).astype(np.uint8) * 255), float(t)


# ----------------------------------------------------------------------------- helpers
def estimate_line_angle(binary: np.ndarray, vertical: bool, search_deg: float) -> float:
    """
    Rotation angle `a` (deg, cv2 convention: +a = counter-clockwise on screen) that makes
    the dominant line direction exactly horizontal (vertical=False) or vertical (True).
    Method: rotate, take the row (or column) ink profile, maximise sum(profile^2) - sharp,
    tall peaks mean the lines are level.  Coarse 0.5 deg then fine 0.1 deg search.
    A text line descending to the right by s degrees gives a = +s; the same `a` convention
    is used for near-vertical lines (see line_kernel).
    """
    h, w = binary.shape
    c = (w / 2.0, h / 2.0)
    b = (binary > 0).astype(np.uint8)

    def score(a):
        m = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1 if not vertical else 0).astype(np.float64)
        return float((prof ** 2).sum())

    coarse = np.arange(-search_deg, search_deg + 1e-6, 0.5)
    a0 = coarse[int(np.argmax([score(a) for a in coarse]))]
    fine = np.arange(a0 - 0.5, a0 + 0.5 + 1e-6, 0.1)
    return float(fine[int(np.argmax([score(a) for a in fine]))])


def line_kernel(length: int, angle_deg: float, vertical: bool) -> np.ndarray:
    """
    1-px-thick straight line as a morphological structuring element, so opening
    follows a slightly slanted ruling/margin WITHOUT resampling the image.
    Horizontal: direction (1, tan a). Vertical: direction (-tan a, 1)   (a = estimate_line_angle).
    """
    a = np.deg2rad(angle_deg)
    if not vertical:
        dx = length - 1.0
        dy = dx * np.tan(a)
    else:
        dy = length - 1.0
        dx = -dy * np.tan(a)
    kw, kh = int(round(abs(dx))) + 1, int(round(abs(dy))) + 1
    k = np.zeros((kh, kw), np.uint8)
    sx = 0 if dx >= 0 else kw - 1
    sy = 0 if dy >= 0 else kh - 1
    cv2.line(k, (sx, sy), (kw - 1 - sx, kh - 1 - sy), 1, 1)
    return k


def _local_centre(coords_major, coords_minor, n_major, smooth):
    """Per-major-axis median of the minor coordinate (e.g. per-column median y of a ruling),
    gap-interpolated and median-smoothed -> local centre of a possibly curved line."""
    centre = np.full(n_major, np.nan)
    order = np.argsort(coords_major)
    cm, cn = coords_major[order], coords_minor[order]
    uniq, start = np.unique(cm, return_index=True)
    for u, seg in zip(uniq, np.split(cn, start[1:])):
        centre[u] = np.median(seg)
    idx = np.arange(n_major)
    ok = ~np.isnan(centre)
    centre = np.interp(idx, idx[ok], centre[ok])
    return ndi.median_filter(centre, size=max(3, smooth | 1), mode="nearest")


def _remove_lines(ink, vertical, cfg):
    """
    Generic line remover for ruling (vertical=False) or margin (vertical=True).
    Returns (cleaned, removed_mask, info).
    Steps: (a) estimate slope  (b) morphological opening with a long slanted line kernel
    (c) keep only pieces that, after bridging, span a large part of the page
    (d) restrict to a thin band around each line's LOCAL centre (so a headline / stroke that
        merely touches the line is not eaten)  (e) stroke protection: a pixel is removed
    only if the ink run PERPENDICULAR to the line is no longer than tolerance * line thickness;
    where a pen stroke crosses the line the run is longer and the pixel is kept.
    """
    h, w = ink.shape
    search = cfg.margin_angle_search_deg if vertical else cfg.ruling_angle_search_deg
    angle = estimate_line_angle(ink, vertical, search)
    if vertical:
        L = max(15, int(cfg.margin_kernel_frac_height * h))
        Lc = max(15, int(cfg.margin_close_frac_height * h))
        min_span, tol, pad, smooth = (cfg.margin_min_span_frac * h, cfg.margin_run_tolerance,
                                      cfg.margin_band_pad_px, cfg.margin_smooth_rows)
        n_major = h
    else:
        L = max(15, int(cfg.ruling_kernel_frac_width * w))
        Lc = max(15, int(cfg.ruling_close_frac_width * w))
        min_span, tol, pad, smooth = (cfg.ruling_min_span_frac * w, cfg.ruling_run_tolerance,
                                      cfg.ruling_band_pad_px, cfg.ruling_smooth_cols)
        n_major = w

    # Ruled lines in photographed/scanned notebooks are rarely parallel to each other
    # (perspective, page curl), so a single slope is not enough: take the UNION of openings
    # over a grid of slopes (+-search, step cfg.line_angle_step_deg).
    opened = np.zeros_like(ink)
    for a in np.arange(-search, search + 1e-6, cfg.line_angle_step_deg):
        opened |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, line_kernel(L, float(a), vertical))
    # bridge small gaps along the line (box kernel tolerant to a few px of wobble)
    bridge = (7, Lc) if vertical else (Lc, 7)
    merged = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, bridge))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    span_idx = cv2.CC_STAT_HEIGHT if vertical else cv2.CC_STAT_WIDTH
    keep = [i for i in range(1, n) if stats[i, span_idx] >= min_span]
    info = dict(angle_deg=angle, n_lines=0, thickness_px=None, centres=[])
    if not keep:
        return ink.copy(), np.zeros_like(ink), info

    # run lengths perpendicular to the line direction
    struct = (np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) if vertical
              else np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))
    rl_lab, _ = ndi.label(ink > 0, structure=struct)
    run_len = np.bincount(rl_lab.ravel())[rl_lab]

    band_mask = np.zeros(ink.shape, bool)
    for i in keep:
        ys, xs = np.nonzero((lab == i) & (opened > 0))
        if len(ys) == 0:
            continue
        major, minor = (ys, xs) if vertical else (xs, ys)
        centre = _local_centre(major, minor, n_major, smooth)
        info["centres"].append(float(np.median(centre[np.unique(major)])))
        info["n_lines"] += 1
        # placeholder band half-width until thickness is known; refined below
        band_mask[ys, xs] = True
        info.setdefault("_pix", []).append((major, minor, centre, int(major.min()), int(major.max())))

    thick = float(np.median(run_len[band_mask])) if band_mask.any() else 1.0
    info["thickness_px"] = thick
    half = 0.5 * thick + pad
    mask = np.zeros(ink.shape, bool)
    pix = info.pop("_pix")
    if vertical:
        # The two strokes of a double margin often touch / converge, so a single 'local centre'
        # is meaningless here: use the detected line pixels themselves, grown by 1 px.
        sel = np.isin(lab, keep) & (opened > 0)
        pix = []
        mask = cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for major, minor, centre, u0, u1 in pix:
        # band of +-half around this line's local centre (per column / per row)
        lo =np.clip(np.floor(centre - half).astype(int), 0, (w if vertical else h) - 1)
        hi = np.clip(np.ceil(centre + half).astype(int), 0, (w if vertical else h) - 1)
        for u in range(u0, u1 + 1):
            if vertical:
                mask[u, lo[u]:hi[u] + 1] = True
            else:
                mask[lo[u]:hi[u] + 1, u] = True
    # every ink pixel inside the band is a candidate (not only the 'opened' ones), so faint /
    # broken parts of the ruling are caught too; the stroke test below protects handwriting
    mask &= (ink > 0)
    # stroke protection
    max_run = tol * thick + 1.0
    removed = mask & (run_len <= max_run)
    cleaned = ink.copy()
    cleaned[removed] = 0
    info["centres"] = sorted(info["centres"])
    return cleaned, (removed.astype(np.uint8) * 255), info


# ----------------------------------------------------------------------------- 3 / 4
def remove_horizontal_ruling(ink, cfg):
    """Step 3: remove notebook ruling. Returns (cleaned, ruling_mask, info incl. pitch_px)."""
    cleaned, mask, info = _remove_lines(ink, vertical=False, cfg=cfg)
    ys = info["centres"]
    pitch = None
    if len(ys) >= cfg.ruling_min_lines_for_pitch:
        d = np.diff(ys)
        # drop spurious near-duplicate detections (< 40% of the median spacing)
        d = d[d > 0.4 * np.median(d)]
        if len(d):
            pitch = float(np.median(d))
    info["pitch_px"] = pitch
    return cleaned, mask, info


def remove_vertical_margin(ink, cfg):
    """Step 4: remove the vertical margin line(s). Returns (cleaned, margin_mask, info)."""
    return _remove_lines(ink, vertical=True, cfg=cfg)


# ----------------------------------------------------------------------------- 5. clean binary
def clean_binary(ink, ruling_info, cfg):
    """
    Final tidy-up: drop tiny specks, and thin horizontal leftovers of the ruling that are
    far wider than they are tall (never real handwriting strokes).
    """
    h, w = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    t = ruling_info.get("thickness_px") or 6.0
    drop = np.zeros(n, bool)
    area = st[:, cv2.CC_STAT_AREA]
    drop[area < cfg.min_cc_area_px] = True
    wid, hei = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    wide = wid >= cfg.residual_line_min_width_frac * w
    thin = hei <= cfg.residual_line_max_height_factor * t
    tall = hei >= cfg.residual_line_min_width_frac * h
    narrow = wid <= cfg.residual_line_max_height_factor * t
    # fill-ratio test so a genuine long, thin diagonal pen stroke is not dropped
    flat = area >= 0.6 * wid * hei
    drop |= (wide & thin & flat) | (tall & narrow & flat)
    drop[0] = False
    out = ink.copy()
    out[drop[lab]] = 0
    return out
