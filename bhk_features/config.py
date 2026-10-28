"""
config.py - the SINGLE place where every parameter / threshold lives.

IMPORTANT (read before interpreting any number this package produces)
---------------------------------------------------------------------
* Everything computed here is a *BHK-inspired computational proxy* derived
  from a scanned image with classical image processing (OpenCV / SciPy /
  scikit-image).  None of it is an official BHK (Concise Evaluation Scale for
  Children's Handwriting) clinical score, and none of it should be used as a
  diagnosis.
* The BHK source does not give pixel-level thresholds for an automated
  system, so the ``anchors`` below are PLACEHOLDERS chosen only so that the
  0-1 "proxy score" is well-defined.  Calibrate them on your own labelled
  data (e.g. teacher/clinician ratings) before drawing any conclusion.
* "Normalised" spatial values are divided by a scale reference measured in
  the image itself (letter height ``ref_h`` or the ruling pitch), so the
  numbers do not depend on scan resolution.  Images are NEVER resized before
  measurement.
"""
from dataclasses import dataclass, field, asdict
import json
from typing import Dict


@dataclass
class BHKConfig:
    # ------------------------------------------------------------------ 1. input validation
    min_image_side_px: int = 200        # reject images smaller than this (either side)
    max_image_side_px: int = 20000      # reject absurdly large images
    min_gray_std: float = 8.0           # reject (near-)blank images: grayscale std below this
    dpi: float = 0.0                    # scan DPI if known (>0 enables size in mm); 0 = unknown

    # ------------------------------------------------------------------ 2. gray + denoise
    denoise_method: str = "nlm"         # 'nlm' | 'median' | 'gaussian' | 'none'
    nlm_h: float = 6.0                  # fastNlMeans filter strength (higher = smoother)
    median_ksize: int = 3               # median blur kernel (odd)
    gaussian_sigma: float = 1.0         # gaussian sigma (px)
    bg_kernel_frac: float = 0.05        # background-estimation closing kernel = frac * min(H, W)
    bg_blur_frac: float = 0.05          # blur applied to the background estimate (frac * min(H, W))

    # ------------------------------------------------------------------ 3-5. binarisation
    ink_otsu_offset: float = 0.0        # added to Otsu threshold (on illumination-normalised image)
    ink_max_level: float = 215.0        # threshold is capped at this gray level (0-255)

    # ---- horizontal ruling
    ruling_angle_search_deg: float = 5.0   # ruling slopes tried: +-this many degrees
    line_angle_step_deg: float = 0.75      # step of the slope grid for ruling / margin opening
    ruling_kernel_frac_width: float = 0.06 # morphological line kernel length = frac * image width
    ruling_close_frac_width: float = 0.10  # bridge gaps in ruling pieces before the span test
    ruling_min_span_frac: float = 0.40     # a ruling must span >= this fraction of image width
    ruling_run_tolerance: float = 1.25     # a ruling pixel is removed only if the local vertical
                                           # ink run <= tolerance * ruling thickness (else a pen
                                           # stroke crosses there and the pixel is kept)
    ruling_band_pad_px: int = 2            # +- band around the locally-fitted ruling centre (px)
    ruling_smooth_cols: int = 101          # median-filter width (px) for the local ruling centre
    # ---- vertical margin
    margin_angle_search_deg: float = 5.0
    margin_kernel_frac_height: float = 0.12
    margin_close_frac_height: float = 0.10
    margin_min_span_frac: float = 0.40
    margin_run_tolerance: float = 1.25
    margin_band_pad_px: int = 2
    margin_smooth_rows: int = 101
    # ---- cleanup
    min_cc_area_px: int = 15               # connected components smaller than this = noise
    residual_line_min_width_frac: float = 0.015  # thin leftover horizontal fragments wider than this...
    residual_line_max_height_factor: float = 1.6  # ...and thinner than factor*ruling thickness are removed

    # ------------------------------------------------------------------ 6-7. segmentation
    skew_search_deg: float = 6.0           # search range for global text skew (projection method)
    line_profile_sigma_frac_H: float = 0.004  # gaussian smoothing of row profile = frac * image height
    line_thr_frac: float = 0.04            # rows with profile > frac * max are 'text rows'
    line_merge_gap_frac: float = 0.15      # merge bands separated by < frac * median band height
    line_split_height_factor: float = 1.9  # bands taller than factor * median are split at valleys
    line_split_valley_frac: float = 0.35   # ... if the valley is below frac * band peak
    line_height_hint_factor: float = 2.5   # sanity hint: typical line height = factor * ref_h
    line_min_ink_frac: float = 0.10        # drop bands with < frac * median band ink
    word_gap_frac: float = 0.50            # horizontal gap >= frac * ref_h separates words
    main_comp_min_area_frac: float = 0.03  # 'main' component: area >= frac * ref_h^2
    min_lines: int = 1                     # segmentation fails below this many lines
    min_words: int = 3                     # ... or below this many words

    # ------------------------------------------------------------------ 9. feature measurements
    # -- writing size
    ruling_min_lines_for_pitch: int = 3    # need >= this many rulings to trust the ruling pitch
    # -- zones (x-height model, Latin-like words only)
    headline_cov: float = 0.55             # row ink coverage >= this => Devanagari headline word
    zone_smooth_frac: float = 0.05         # gaussian sigma of the per-word row profile = frac * ref_h
    core_frac: float = 0.25                # core zone = rows with profile >= frac * max
    ext_min_ratio: float = 0.35            # ascender/descender counts if extension >= ratio * x-height
    min_zone_word_height_frac: float = 0.30  # skip words shorter than frac * ref_h
    # -- alignment
    min_words_baseline: int = 3            # baseline fit needs this many words per line
    # -- word spacing
    narrow_gap_frac: float = 0.90          # word gap < frac * ref_h counted as 'insufficient'
    # -- sharp angles
    curv_step_frac: float = 0.12           # turning-angle arm length = frac * ref_h
    sharp_turn_deg: float = 75.0           # turn >= this (deg) counts as a sharp angle (interior angle <= 105 deg)
    curv_smooth_px: int = 3                # moving-average window on skeleton coordinates before angle measurement
    min_path_len_frac: float = 0.30        # skeleton paths shorter than frac * ref_h ignored
    # -- collisions
    collision_overlap_frac: float = 0.25   # bbox-intersection / smaller-bbox area
    collision_diacritic_area_ratio: float = 0.10  # ignore pairs where smaller area < ratio * larger
    # -- traced / corrected
    thick_factor: float = 2.0              # blob = stroke width >= factor * median stroke width
    thick_blob_min_area_frac: float = 2.0  # blob area >= frac * (median stroke width)^2
    tiny_hole_area_frac: float = 0.01      # hole area <= frac * ref_h^2 = ink-filled loop
    tiny_hole_min_area_px: int = 4         # ignore 1-3 px pinholes (binarisation noise)
    smudge_gray_level: float = 200.0       # normalised gray below this = non-paper (smudge candidate)
    smudge_halo_px: int = 4                # ignore pixels within this many px of ink/ruling
    # -- unstable trace
    jitter_step_frac: float = 0.08         # heading sampled every frac * ref_h px along a stroke
    jitter_smooth_px: int = 3              # moving-average window on skeleton coordinates
    jitter_min_turn_deg: float = 25.0      # heading change below this is ignored (not a wobble)
    width_trim_frac: float = 0.10          # trim this fraction at both stroke ends for width stats
    # -- atypical / ambiguous review candidates
    outlier_z: float = 3.5                 # robust (MAD) z-score for shape outliers
    aspect_extreme: float = 4.0            # w/h > this or < 1/this is flagged for review
    fragment_min_h_frac: float = 0.25      # non-main comp taller than frac * ref_h = stray fragment

    # ------------------------------------------------------------------ 10. score anchors
    # PLACEHOLDERS (NOT official BHK thresholds).  score = linear ramp 0..1
    # between lo and hi of the named raw measurement; 'invert' means smaller is worse.
    anchors: Dict[str, dict] = field(default_factory=lambda: {
        "writing_size":                   dict(band_lo=0.40, band_hi=1.60, ramp=0.50),
        "non_aligned_left_margin":        dict(key="std_norm", lo=0.15, hi=0.75),
        "skewed_alignment":               dict(keys=[("mean_abs_rel_angle_deg", 1.0, 5.0),
                                                     ("mean_rms_resid_norm", 0.08, 0.30)]),
        "insufficient_word_spacing":      dict(keys=[("frac_gaps_narrow", 0.10, 0.50),
                                                     ("median_gap_norm", 1.20, 0.60, "invert")]),
        "sharp_angles":                   dict(key="sharp_per_10_refh", lo=0.5, hi=3.0),
        "broken_links":                   dict(key="breaks_per_word", lo=0.3, hi=2.0),
        "collisions":                     dict(key="collisions_per_main_comp", lo=0.02, hi=0.15),
        "irregular_letter_size":          dict(key="mean_line_iqr_over_median", lo=0.25, hi=0.70),
        "inconsistent_height_extensions": dict(keys=[("xh_cv_no_ext", 0.10, 0.35),
                                                     ("asc_ext_cv", 0.15, 0.45),
                                                     ("desc_ext_cv", 0.15, 0.45)]),
        "traced_corrected_letters":       dict(keys=[("thick_blobs_per_main_comp", 0.02, 0.20),
                                                     ("tiny_holes_per_main_comp", 0.05, 0.40),
                                                     ("smudge_fraction", 0.002, 0.03)]),
        "unstable_trace":                 dict(keys=[("width_cv", 0.25, 0.60),
                                                     ("zigzag_per_refh", 0.3, 2.0)]),
    })

    # ------------------------------------------------------------------ outputs
    save_debug_images: bool = True

    # convenience --------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "BHKConfig":
        """Load overrides from a JSON file (only keys present are replaced)."""
        cfg = BHKConfig()
        with open(path, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if not hasattr(cfg, k):
                    raise KeyError(f"Unknown config key: {k}")
                setattr(cfg, k, v)
        return cfg
