"""
visualization.py - pipeline steps 8 and 13: debug images.
All drawing is done at the image's native resolution (nothing is resized).
Colours are BGR.
"""
import cv2
import numpy as np

RED, GREEN, BLUE, ORANGE = (0, 0, 220), (0, 160, 0), (220, 90, 0), (0, 140, 255)
PURPLE, CYAN, MAGENTA, YELLOW, GRAY = (180, 0, 160), (200, 200, 0), (200, 0, 200), (0, 200, 220), (150, 150, 150)


def _s(img):
    """Drawing scale factor relative to a ~1600 px wide image."""
    return max(1.0, img.shape[1] / 1600.0)


def save_cleaned(path, clean, ruling_mask, margin_mask):
    """Cleaned binary (black ink on white). Removed ruling = light red, removed margin = light blue."""
    img = np.full(clean.shape + (3,), 255, np.uint8)
    img[ruling_mask > 0] = (200, 200, 255)
    img[margin_mask > 0] = (255, 210, 200)
    img[clean > 0] = (0, 0, 0)
    cv2.imwrite(str(path), img)


def line_polygon(seg, line):
    """Tilt-aware polygon (original coords) of a line band: de-skewed rectangle mapped back."""
    xr = [seg.words[i].xr for i in line.word_ids]
    x0, x1 = min(a for a, _ in xr), max(b for _, b in xr)
    y0, y1 = line.band
    c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    inv = cv2.invertAffineTransform(seg.M)
    return (c @ inv[:, :2].T + inv[:, 2]).astype(np.int32)


def save_lines(path, clean, seg):
    img = cv2.cvtColor(255 - clean, cv2.COLOR_GRAY2BGR)
    s = _s(img)
    for l in seg.lines:
        cv2.polylines(img, [line_polygon(seg, l)], True, RED, int(3 * s))
        x0, y0 = line_polygon(seg, l)[0]
        cv2.putText(img, f"L{l.id}", (int(x0) + 4, int(y0) + int(28 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, RED, int(2 * s))
    cv2.imwrite(str(path), img)


def save_words(path, clean, seg):
    img = cv2.cvtColor(255 - clean, cv2.COLOR_GRAY2BGR)
    s = _s(img)
    for c in seg.comps:
        x0, y0, x1, y1 = c.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), BLUE if c.main else GRAY, max(1, int(s)))
    for w in seg.words:
        x0, y0, x1, y1 = w.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, int(3 * s))
        cv2.putText(img, str(w.id), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, GREEN, int(2 * s))
    cv2.imwrite(str(path), img)


def _draw_overlay(img, kind, payload, tag, ctx):
    s = _s(img)
    r = int(5 * s)
    col = dict(left_edge=RED, baseline_pt=ORANGE, baseline=ORANGE, sharp=RED, thick=PURPLE, hole=CYAN,
               wobble=MAGENTA, break_=ORANGE).get(tag, RED)
    if kind == "points":
        for (x, y) in payload:
            cv2.circle(img, (int(x), int(y)), r, col, -1 if tag != "left_edge" else int(2 * s))
    elif kind == "segments":
        for (xa, ya, xb, yb) in payload:
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), ORANGE, int(2 * s))
    elif kind == "rects":
        c = dict(break_=ORANGE, collision=RED, atypical=MAGENTA, ambiguous=YELLOW).get(tag.replace("break", "break_") if tag == "break" else tag, RED)
        for rc in payload:
            cv2.rectangle(img, (int(rc[0]), int(rc[1])), (int(rc[2]), int(rc[3])), c, int(2 * s))
    elif kind == "gaps":
        thr = tag
        ov = img.copy()
        for (x0, y0, x1, y1, g) in payload:
            if x1 > x0:
                cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), RED if g < thr else GREEN, -1)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
    elif kind == "zones":
        t = tag
        for (xl, xr, yt, yb, xc) in payload:
            for yy, c in ((yt, GREEN), (yb, BLUE)):
                cv2.line(img, (int(xl), int(yy + (xl - xc) * t)), (int(xr), int(yy + (xr - xc) * t)), c, max(1, int(1.5 * s)))
    elif kind == "mask_box":
        x0, y0, x1, y1, m = payload
        ov = img.copy()
        ov[y0:y1, x0:x1][m] = PURPLE
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)


def save_final(path, bgr, seg, results, scores, ctx, ruling_info, margin_info):
    """
    Final per-sample debug image: faded original + every region/measurement used by the BHK
    proxies (lines, words, baselines, left-edge points, word gaps, x-height zones, sharp
    corners, breaks, collisions, review candidates, thick blobs/holes/smudge) + a score panel.
    """
    s = _s(bgr)
    img = cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)
    img[ctx.clean > 0] = (60, 60, 60)
    for l in seg.lines:
        cv2.polylines(img, [line_polygon(seg, l)], True, (120, 120, 255), max(1, int(2 * s)))
    for w in seg.words:
        x0, y0, x1, y1 = w.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), (110, 190, 110), 1)
    for fr in results:
        for ov in fr.overlay:
            _draw_overlay(img, ov[0], ov[1], ov[2], ctx)
    # panel
    pw = int(620 * s)
    panel = np.full((img.shape[0], pw, 3), 255, np.uint8)
    fs, y = 0.62 * s, int(34 * s)
    cv2.putText(panel, "BHK-inspired PROXIES", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), int(2 * s)); y += int(26 * s)
    cv2.putText(panel, "(uncalibrated, not clinical scores)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (90, 90, 90), int(1.5 * s)); y += int(34 * s)
    for fr in results:
        sc = scores[fr.name]
        txt = f"{sc['score']:.2f}" if sc["score"] is not None else ("REVIEW" if sc["score_status"] == "requires_review" else "n/a")
        cv2.putText(panel, f"{fr.name[:30]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(1.5 * s))
        cv2.putText(panel, txt, (pw - int(130 * s), y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 200) if txt == "REVIEW" else (0, 100, 0), int(2 * s))
        y += int(36 * s)
    leg = ["Legend:", "red pts: sharp angles / left edge", "orange: baselines, broken-link gaps", "red boxes: collisions",
           "green/blue lines: x-height zone", "red/green tint: narrow/ok word gaps", "magenta/yellow: review candidates",
           "purple: thick blobs / smudge, cyan: tiny holes"]
    y += int(14 * s)
    for t in leg:
        cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (60, 60, 60), int(1.2 * s)); y += int(24 * s)
    cv2.imwrite(str(path), np.hstack([img, panel]))
