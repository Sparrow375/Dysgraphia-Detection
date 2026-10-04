# English Dysgraphia Image Harvesting & Curation Engine ("Jugaad Pipeline")

Part of Workstream H in the Dysgraphia Detection Initiative. This repository contains the automated multi-source image harvesters, perceptual deduplication filters, automated quality screeners, and interactive curation gallery used to assemble the English offline dysgraphia handwriting dataset.

## Architecture & Sources
- **Reddit Harvester (`scrapers/harvest_dysgraphia_images.py`):**
  - Harvests genuine smartphone photos of handwriting from `r/dysgraphia`, `r/Handwriting`, `r/dyslexia`.
  - Converts preview thumbnails (`preview.redd.it/...`) into full uncompressed original phone captures (`i.redd.it/...`).
- **Wikimedia Commons Medical Archive:**
  - Queries clinical scans of diagnosed dysgraphic handwriting (`Dysgraphia.jpg`, `Disgrafija.jpg`, etc.).
- **PubMed Central (PMC) Clinical Figures:**
  - Queries peer-reviewed graphonomics literature using NCBI E-utilities API.
- **Educational Therapy Portals:**
  - Ingests verified before/after handwriting samples from dysgraphia educational therapy archives.
- **Handwriting Disorder Benchmark Corpus:**
  - Ingests benchmark samples with clinical motor/spelling impairments.

## Automated Quality Filtering Pipeline
Every scraped image undergoes automated pre-screening:
1. **Dimension Filter:** Rejects icons, avatars, and tracking pixels ($w < 120\text{px}$ or $h < 120\text{px}$).
2. **Aspect Ratio Filter:** Filters out banner ads and UI ribbons ($\text{aspect} > 16.0$).
3. **Perceptual Deduplication:** 64-bit difference hash (dHash) prevents duplicate downloads.
4. **Foreground Variance Verification:** Filters out blank, low-contrast, or solid-color scans ($\sigma_{gray} \ge 8.0$).

## Interactive Review & Curation Dashboard
- **Web Dashboard:** Open [`scraped_candidates/review_gallery.html`](./scraped_candidates/review_gallery.html) in any browser.
  - Category badges (Reddit, Wikimedia, Portals, Benchmark).
  - Displays original resolutions, file sizes, source links, and predicted AI screening scores.
  - Interactive **Accept** and **Reject** buttons persisted to browser `localStorage`.
  - **Export Accepted Manifest** button generates a clean CSV of verified images for model training.
- **Manifests & Images:**
  - [`scraped_candidates/manifest.csv`](./scraped_candidates/manifest.csv)
  - [`scraped_candidates/metadata.json`](./scraped_candidates/metadata.json)
  - [`scraped_candidates/images/`](./scraped_candidates/images/) (113 validated handwriting images)

## Running the Harvester
```bash
pip install -r requirements.txt
python scrapers/harvest_dysgraphia_images.py
```
