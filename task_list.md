# Task List — Phase 1 v3 Segmentation Fix

## Current Sprint: Fix Segmentation & Cropping Quality

### Root Cause Analysis & Resolution
1. **Camera Photos vs DocScanner Scans**:
   - Grade 4 was scanned via DocScanner app (flat white paper, high contrast).
   - Grades 3, 5, 6, 7 were raw phone photos with heavy uneven lighting and shadow gradients.
   - Solution: Added `normalize_background` in `pipeline/preprocess.py` using large-kernel morphological dilation (`41x41`) + median blur (`21`) + background division. This flattens illumination gradients cleanly without degrading fine stroke boundaries.
2. **Inverted / Upside-Down Scans**:
   - Landscape sheets like `G3_A_Roll12` and `G7_A_Roll11` were rotated the wrong direction.
   - Solution: Added ruled-line density comparison between page halves in `auto_orient_portrait` to ensure the header box is always at top and ruled lines are at bottom.
3. **Left Margin Line Artifact**:
   - Red double margin lines at `x \approx 180-220` bridged lines into 1000px+ connected components.
   - Solution: Extended margin crop to `clean_mask[:, :225] = 0` and added height filter `ch < 1.8 * r` and `cw < 1200`.

### Tasks
- [x] 1. Diagnose root cause of segmentation failures (DocScanner vs camera phone lighting, inverted sheets, red double margin line)
- [x] 2. Add illumination normalization (`normalize_background`) in `preprocess.py`
- [x] 3. Fix auto-orientation for landscape/inverted sheets in `preprocess.py` via ruled line density check
- [x] 4. Harden margin cutoff (`clean_mask[:, :225] = 0`) and CC filtering (`ch < 1.8 * r`, `cw < 1200`, aspect ratio > 12.0) in `segment.py`
- [x] 5. Reinstate rule detection & rule masking while preserving natural grayscale in cropped output
- [x] 6. Re-run pipeline on all 115 School A sheets
- [x] 7. Regenerate visual QA review gallery (`qa/review_gallery.html`)
- [x] 8. Update `context.md` with v3 changes
- [x] 9. Git commit and push to `origin/baseline-v1`

### Benchmark Requirements
- `G4_A_Roll01` must remain preserved as the benchmark template
- Clean paper backgrounds without shadow blobs or ruled-line false detections across camera photos
- Maintain test coverage (Phase 0 and Phase 1 pytest suites)
