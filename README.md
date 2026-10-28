# bhk_features - BHK-inspired static handwriting features (classical image processing)

Python package that extracts 13 **BHK-inspired computational proxies** from scanned notebook pages.
No CNN/CRNN/OCR/kinematic models. Images are never resized before measurement.

> These are **proxies, not official BHK scores**, and nothing here is a clinical or diagnostic result.
> The 0-1 scores use placeholder anchors in `config.py` (the BHK source gives no pixel thresholds
> for an automated system). Calibrate them on your own rated data before relying on them.

## Install / run
    pip install opencv-python numpy scipy scikit-image
    python -m bhk_features page1.jpg folder_of_scans/ --out results/ [--dpi 300] [--config my_cfg.json] [--no-debug]

Outputs in `results/`: `bhk_features.csv` (one row/image), `bhk_features.json` (raw values, notes,
conversion rule per feature), `config_used.json`, `debug/<name>_01_cleaned.png`, `_02_lines.png`,
`_03_words_components.png`, `_04_bhk_measurements.png`.

## Modules
| file | role |
|---|---|
| `config.py` | every parameter/threshold (single place), JSON-overridable |
| `preprocessing.py` | load/validate, denoise, illumination flattening, ruling + margin removal, clean binary |
| `segmentation.py` | text lines (projection profile), words/components |
| `features.py` | the 13 measurements (raw values + review candidates) |
| `scoring.py` | raw -> 0..1 proxy score, with the rule text |
| `visualization.py` | debug / final overlay images |
| `pipeline.py` | per-image processing, batch run, CSV/JSON, CLI |

## Feature status
`ok` = automated measurement (still a proxy) - `requires_review` (atypical / ambiguous letter forms:
descriptors + candidates only, score = None) - `insufficient_data`.
If segmentation fails (too few lines/words) the row has `status=segmentation_failed` and **no** features.

## Known limitations
* Strokes lying exactly on a ruled line cannot be separated from it; some baseline strokes are thinned or cut,
  which inflates `broken_links` and `traced_corrected_letters` (tiny holes).
* x-height zones only work for Latin-like words; Devanagari headline words are excluded from that feature.
* Alignment uses one global ruling angle as reference; perspective differences between rulings add error.
* BHK assumes cursive writing; print-style writing scores high on broken links by design.
