"""
Multi-Source English Dysgraphia Handwriting Harvester & Dataset Builder
Sources:
1. Reddit (r/dysgraphia, r/Handwriting, r/dyslexia across top/new/search)
2. Wikimedia Commons (Medical & Clinical Handwriting Archives)
3. Educational & Clinical Portals (Edublox, Dysgraphia.life)
4. PubMed Central (PMC) Open-Access Clinical Figures
5. English Handwriting Disorder Benchmark Corpus
"""

import os
import sys
import time
import re
import json
import html
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure unbuffered UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

def log(msg):
    # Strip emojis for Windows console safety
    clean_msg = msg.encode("ascii", "replace").decode("ascii")
    print(clean_msg, flush=True)

import requests
import cv2
import numpy as np
import pandas as pd

# Load BHK Feature Extractor & Trained Model Bundle
MODEL_BUNDLE = None
try:
    from src.preprocessing import preprocess_handwriting_image
    from src.bhk_features import extract_bhk_features
    import pickle
    bundle_path = os.path.join(PROJECT_ROOT, "model_bundle.pkl")
    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            MODEL_BUNDLE = pickle.load(f)
        log("[OK] Loaded model_bundle.pkl for automated screening evaluation.")
except Exception as e:
    log(f"[INFO] Model bundle note: {e}")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scraped_candidates")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def compute_dhash(image, hash_size=8):
    """Compute difference hash for fast perceptual deduplication."""
    try:
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

# ==========================================================
# 1. REDDIT HARVESTER
# ==========================================================
def harvest_reddit():
    log("\n[1/5] Harvesting Reddit (r/dysgraphia, r/Handwriting, r/dyslexia)...")
    reddit_feeds = [
        ("r/dysgraphia (top-all)", "https://www.reddit.com/r/dysgraphia/top.rss?t=all&limit=100"),
        ("r/dysgraphia (new)", "https://www.reddit.com/r/dysgraphia/new.rss?limit=100"),
        ("r/Handwriting (dysgraphia)", "https://www.reddit.com/r/Handwriting/search.rss?q=dysgraphia&restrict_sr=1&sort=top"),
        ("r/Handwriting (dysgraphic)", "https://www.reddit.com/r/Handwriting/search.rss?q=dysgraphic&restrict_sr=1&sort=top"),
        ("r/dyslexia (handwriting)", "https://www.reddit.com/r/dyslexia/search.rss?q=handwriting&restrict_sr=1&sort=top"),
    ]
    
    candidates = []
    seen_urls = set()
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for label, feed_url in reddit_feeds:
        log(f"  -> Polling {label}...")
        try:
            resp = requests.get(feed_url, headers=HTTP_HEADERS, timeout=12)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                entries = root.findall('atom:entry', ns)
                found = 0
                for entry in entries:
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text if title_elem is not None else "Reddit post"
                    content_elem = entry.find('atom:content', ns)
                    content = content_elem.text if content_elem is not None else ""
                    link_elem = entry.find('atom:link', ns)
                    post_url = link_elem.get('href') if link_elem is not None else ""
                    
                    # Direct image links
                    img_links = re.findall(r'href="([^"]+)"', content)
                    img_links += re.findall(r'src="([^"]+)"', content)
                    
                    for raw_url in img_links:
                        raw_clean = html.unescape(raw_url).split('?')[0]
                        if any(domain in raw_clean for domain in ["i.redd.it", "preview.redd.it", "imgur.com"]):
                            clean_url = raw_clean.replace("preview.redd.it", "i.redd.it")
                            # If imgur without extension, append .jpg
                            if "imgur.com" in clean_url and not clean_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                                clean_url += ".jpg"
                            if clean_url not in seen_urls:
                                seen_urls.add(clean_url)
                                candidates.append({
                                    "source_platform": "Reddit",
                                    "query_or_context": f"{label} | {title}",
                                    "source_page_url": post_url,
                                    "original_image_url": clean_url,
                                })
                                found += 1
                log(f"     [OK] Found {found} new image candidates")
            elif resp.status_code == 429:
                log("     [RATE-LIMITED] Cooling down for 5s...")
                time.sleep(5)
            else:
                log(f"     [STATUS] {resp.status_code}")
        except Exception as e:
            log(f"     [ERROR] {e}")
        time.sleep(3.5)
        
    log(f"  --> Total Reddit candidate images: {len(candidates)}")
    return candidates

# ==========================================================
# 2. WIKIMEDIA COMMONS MEDICAL ARCHIVES
# ==========================================================
def harvest_wikimedia():
    log("\n[2/5] Harvesting Wikimedia Commons Clinical Archives...")
    candidates = []
    seen_urls = set()
    terms = ["dysgraphia", "dysgraphia handwriting", "dysgraphic handwriting", "dyslexia handwriting"]
    
    for term in terms:
        try:
            search_url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(term)}&srnamespace=6&format=json"
            req = urllib.request.Request(search_url, headers={"User-Agent": "DysgraphiaResearch/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                
            items = data.get("query", {}).get("search", [])
            titles = [it["title"] for it in items if not it["title"].lower().endswith(('.wav', '.ogg', '.pdf', '.svg'))]
            
            if not titles:
                continue
                
            joined_titles = "|".join(titles[:15])
            info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(joined_titles)}&prop=imageinfo&iiprop=url|size&format=json"
            req_info = urllib.request.Request(info_url, headers={"User-Agent": "DysgraphiaResearch/1.0"})
            with urllib.request.urlopen(req_info, timeout=10) as resp2:
                data2 = json.loads(resp2.read().decode())
                
            pages = data2.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                title = pdata.get("title", "")
                infos = pdata.get("imageinfo", [])
                if infos:
                    info = infos[0]
                    img_url = info.get("url")
                    desc_url = info.get("descriptionurl")
                    if img_url and img_url not in seen_urls:
                        seen_urls.add(img_url)
                        candidates.append({
                            "source_platform": "Wikimedia Commons",
                            "query_or_context": f"Clinical Archival Sample: {title}",
                            "source_page_url": desc_url or "https://commons.wikimedia.org",
                            "original_image_url": img_url,
                        })
            log(f"  -> Wikimedia query '{term}': {len(titles)} files found")
        except Exception as e:
            log(f"  [ERROR] Wikimedia query error: {e}")
        time.sleep(1.0)
        
    log(f"  --> Total Wikimedia candidates: {len(candidates)}")
    return candidates

# ==========================================================
# 3. CLINICAL & EDUCATIONAL PORTALS
# ==========================================================
def harvest_educational_portals():
    log("\n[3/5] Harvesting Educational & Clinical Articles (Edublox, Dysgraphia.life)...")
    candidates = []
    seen_urls = set()
    
    pages = [
        ("Edublox Dysgraphia Case Study", "https://www.edubloxtutor.com/help-my-9-year-old-son-has-dysgraphia/"),
        ("Edublox Handwriting Curriculum", "https://www.edubloxtutor.com/dysgraphia/"),
        ("Edublox Dysgraphia Symptoms", "https://www.edubloxtutor.com/dysgraphia-symptoms/"),
    ]
    
    for title, url in pages:
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=12)
            if r.status_code == 200:
                imgs = re.findall(r'https?://[^\s\"\'<>]+\.(?:jpg|jpeg|png)', r.text)
                for im in set(imgs):
                    im_lower = im.lower()
                    if ("dysgraphia" in im_lower or "handwriting" in im_lower or "before-and-after" in im_lower) and "logo" not in im_lower:
                        if im not in seen_urls:
                            seen_urls.add(im)
                            candidates.append({
                                "source_platform": "Educational / Clinical Portal",
                                "query_or_context": f"{title} | {os.path.basename(im)}",
                                "source_page_url": url,
                                "original_image_url": im,
                            })
            time.sleep(1.5)
        except Exception as e:
            log(f"  [ERROR] Portal error on {url}: {e}")
            
    log(f"  --> Total Clinical Portal candidates: {len(candidates)}")
    return candidates

# ==========================================================
# 4. PUBMED CENTRAL (PMC) CLINICAL FIGURE HARVESTER
# ==========================================================
def harvest_pmc():
    log("\n[4/5] Harvesting Clinical Figures from PubMed Central (PMC)...")
    candidates = []
    seen_urls = set()
    queries = ["dysgraphia+handwriting", "developmental+dysgraphia+writing"]
    
    for q in queries:
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={q}&retmode=json&retmax=10"
            req = urllib.request.Request(search_url, headers={"User-Agent": "DysgraphiaDatasetHarvester/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            pmc_ids = data.get("esearchresult", {}).get("idlist", [])
            log(f"  -> PMC search '{q}': found {len(pmc_ids)} papers")
            
            for pmcid in pmc_ids[:6]:
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
                req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "DysgraphiaDatasetHarvester/1.0"})
                try:
                    with urllib.request.urlopen(req2, timeout=10) as resp2:
                        xml_str = resp2.read().decode("utf-8", errors="ignore")
                    
                    title_match = re.search(r'<article-title>(.*?)</article-title>', xml_str, re.DOTALL)
                    art_title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else f"PMC{pmcid}"
                    
                    graphics = re.findall(r'<graphic[^>]+xlink:href="([^"]+)"', xml_str)
                    for g in graphics:
                        fig_ext = os.path.splitext(g)[1]
                        if not fig_ext:
                            fig_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/bin/{g}.jpg"
                        else:
                            fig_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/bin/{g}"
                            
                        if fig_url not in seen_urls:
                            seen_urls.add(fig_url)
                            candidates.append({
                                "source_platform": "PubMed Central (Clinical)",
                                "query_or_context": f"PMC{pmcid}: {art_title[:70]}",
                                "source_page_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                                "original_image_url": fig_url,
                            })
                except Exception:
                    pass
                time.sleep(1.0)
        except Exception as e:
            log(f"  [ERROR] PMC error on '{q}': {e}")
            
    log(f"  --> Total PubMed Central candidates: {len(candidates)}")
    return candidates

# ==========================================================
# 5. DISORDER BENCHMARK REPO SAMPLES
# ==========================================================
def harvest_repo_samples():
    log("\n[5/5] Ingesting English Handwriting Disorder Benchmark Samples...")
    candidates = []
    local_source_dir = os.path.join(PROJECT_ROOT, "scratch_repos", "srummanf_Dyslexia", "data", "samples", "all", "dyslexic")
    if os.path.exists(local_source_dir):
        files = [f for f in os.listdir(local_source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in sorted(files, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x):
            fpath = os.path.join(local_source_dir, f)
            candidates.append({
                "source_platform": "Handwriting Disorder Benchmark",
                "query_or_context": f"Open Research Benchmark Sample #{f}",
                "source_page_url": "https://github.com/srummanf/Dyslexia-Detection-from-Handwriting",
                "original_image_url": fpath,
                "is_local_file": True
            })
        log(f"  --> Ingested {len(candidates)} benchmark samples from local repository")
    return candidates

# ==========================================================
# PROCESS, SCREEN, DEDUPLICATE & SAVE
# ==========================================================
def process_and_save_candidates(candidates):
    log("\n" + "=" * 60)
    log(f"Processing {len(candidates)} Candidates (Download, Deduplicate, AI Screening)...")
    log("=" * 60)
    
    seen_hashes = set()
    manifest_records = []
    saved_count = 0
    discarded_count = 0
    
    for idx, cand in enumerate(candidates):
        try:
            img = None
            raw_url = cand["original_image_url"]
            is_local = cand.get("is_local_file", False)
            
            if is_local:
                img = cv2.imread(raw_url)
            else:
                try:
                    # Clean stateless request without cookies from previous pages
                    r = requests.get(raw_url, headers={"User-Agent": USER_AGENT}, timeout=10)
                    if r.status_code == 200 and len(r.content) > 1024:
                        arr = np.frombuffer(r.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                    
            if img is None:
                discarded_count += 1
                continue
                
            h, w = img.shape[:2]
            
            # Dimension filter
            if w < 120 or h < 120:
                discarded_count += 1
                continue
                
            # Aspect ratio filter
            aspect = max(w / h, h / w)
            if aspect > 16.0:
                discarded_count += 1
                continue
                
            # Perceptual hash deduplication
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dh = compute_dhash(gray)
            if dh is not None:
                if dh in seen_hashes:
                    discarded_count += 1
                    continue
                seen_hashes.add(dh)
                
            # Uniform image check
            if np.std(gray) < 8.0:
                discarded_count += 1
                continue
                
            saved_count += 1
            cand_id = f"ENG_CAND_{saved_count:03d}"
            target_filename = f"{cand_id}.jpg"
            target_path = os.path.join(IMAGES_DIR, target_filename)
            
            cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            filesize_kb = round(os.path.getsize(target_path) / 1024, 1)
            
            # BHK features & model prediction
            pred_score = None
            pred_label = "Pending Screening"
            bhk_dict = {}
            
            if MODEL_BUNDLE is not None:
                try:
                    # Resize large images for fast BHK calculation
                    max_dim = max(h, w)
                    if max_dim > 1600:
                        scale_factor = 1600.0 / max_dim
                        small_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
                    else:
                        small_img = img
                        
                    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    binary_mask, _ = preprocess_handwriting_image(rgb_img)
                    feat_dict, feat_vector = extract_bhk_features(binary_mask)
                    bhk_dict = feat_dict
                    
                    scaler = MODEL_BUNDLE.get("scaler")
                    ensemble = MODEL_BUNDLE.get("ensemble_model")
                    threshold = MODEL_BUNDLE.get("optimal_threshold", 0.45)
                    
                    if scaler is not None and ensemble is not None:
                        X_scaled = scaler.transform(feat_vector.reshape(1, -1))
                        proba = ensemble.predict_proba(X_scaled)[0, 1]
                        pred_score = round(float(proba) * 100, 1)
                        if pred_score >= threshold * 100:
                            pred_label = f"Potential Dysgraphia ({pred_score}%)"
                        else:
                            pred_label = f"Low Potential ({pred_score}%)"
                except Exception:
                    pass
                    
            record = {
                "candidate_id": cand_id,
                "filename": target_filename,
                "relative_path": f"images/{target_filename}",
                "source_platform": cand["source_platform"],
                "context_title": cand["query_or_context"],
                "source_page_url": cand["source_page_url"],
                "original_image_url": raw_url,
                "width": int(w),
                "height": int(h),
                "filesize_kb": filesize_kb,
                "ai_screening_risk": pred_score,
                "ai_preliminary_label": pred_label,
                "verification_status": "PENDING_REVIEW",
                "letter_size_cv": round(float(bhk_dict.get("letter_size_cv", 0)), 3) if bhk_dict else None,
                "baseline_drift": round(float(bhk_dict.get("baseline_drift_slope", 0)), 3) if bhk_dict else None,
            }
            manifest_records.append(record)
            
            if saved_count % 15 == 0 or saved_count == 1:
                log(f"  [SAVED] {saved_count} candidates... (Latest: {cand_id} [{w}x{h}] from {cand['source_platform']}, Risk: {pred_score}%)")
                
        except Exception:
            discarded_count += 1
            continue
            
    log("\n" + "=" * 60)
    log(f"HARVESTING COMPLETE!")
    log(f"   [SAVED]     {saved_count} unique validated candidates")
    log(f"   [DISCARDED] {discarded_count} invalid/duplicate/tiny images")
    log("=" * 60)
    
    manifest_csv_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    metadata_json_path = os.path.join(OUTPUT_DIR, "metadata.json")
    
    df = pd.DataFrame(manifest_records)
    df.to_csv(manifest_csv_path, index=False)
    
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
        
    log(f"Saved manifest CSV:  {manifest_csv_path}")
    log(f"Saved metadata JSON: {metadata_json_path}")
    
    return manifest_records

# ==========================================================
# 6. GENERATE INTERACTIVE HTML REVIEW GALLERY
# ==========================================================
def generate_review_gallery(records):
    log("\nGenerating Interactive Offline Review Gallery...")
    html_path = os.path.join(OUTPUT_DIR, "review_gallery.html")
    records_json = json.dumps(records)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dysgraphia English Handwriting Candidates Review Gallery</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #0b0f19;
    --card-bg: #141c2e;
    --card-hover: #1e293b;
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --radius: 12px;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg);
    color: var(--text-main);
    padding: 24px;
    min-height: 100vh;
}}

header {{
    max-width: 1400px;
    margin: 0 auto 24px auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
}}

h1 {{
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.stats-bar {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}}

.stat-pill {{
    background: #1e293b;
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
}}
.stat-pill strong {{ color: var(--text-main); }}
.stat-pill.accepted strong {{ color: var(--success); }}
.stat-pill.rejected strong {{ color: var(--danger); }}

.toolbar {{
    max-width: 1400px;
    margin: 0 auto 24px auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    background: #131b2e;
    padding: 16px 20px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
}}

.filter-group {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}

.filter-btn {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 8px 16px;
    border-radius: 8px;
    font-family: inherit;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}}
.filter-btn:hover, .filter-btn.active {{
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
}}

.export-btn {{
    background: #059669;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-family: inherit;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.export-btn:hover {{ background: #047857; }}

.grid {{
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
}}

.card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.card:hover {{
    transform: translateY(-3px);
    border-color: #4f46e5;
}}
.card.is-accepted {{
    border-color: var(--success);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
}}
.card.is-rejected {{
    opacity: 0.45;
    border-color: var(--danger);
}}

.img-container {{
    position: relative;
    width: 100%;
    height: 220px;
    background: #000;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}}

.img-container img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    transition: transform 0.25s ease;
}}
.img-container:hover img {{
    transform: scale(1.05);
}}

.badge {{
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.badge.reddit {{ color: #ff5722; }}
.badge.wikimedia {{ color: #eab308; }}
.badge.portal {{ color: #38bdf8; }}
.badge.clinical {{ color: #10b981; }}
.badge.benchmark {{ color: #f59e0b; }}

.risk-badge {{
    position: absolute;
    top: 10px;
    right: 10px;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(4px);
}}
.risk-high {{ color: #f87171; border: 1px solid #ef4444; }}
.risk-low {{ color: #34d399; border: 1px solid #10b981; }}

.card-body {{
    padding: 16px;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    gap: 10px;
}}

.card-title {{
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
    color: var(--text-main);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.meta-details {{
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.card-actions {{
    margin-top: auto;
    display: flex;
    gap: 8px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}}

.btn {{
    flex: 1;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid transparent;
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}}
.btn-accept {{
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.3);
}}
.btn-accept:hover, .btn-accept.active {{
    background: #10b981;
    color: white;
}}
.btn-reject {{
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border-color: rgba(239, 68, 68, 0.3);
}}
.btn-reject:hover, .btn-reject.active {{
    background: #ef4444;
    color: white;
}}
.btn-link {{
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    text-decoration: none;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 10px;
    border-radius: 6px;
}}
.btn-link:hover {{
    color: #fff;
    border-color: #6366f1;
}}

#lightbox {{
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.92);
    z-index: 9999;
    align-items: center;
    justify-content: center;
    padding: 30px;
}}
#lightbox.active {{ display: flex; }}
#lightbox img {{
    max-width: 90vw;
    max-height: 85vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 0 30px rgba(0,0,0,0.8);
}}
#lightbox .close-btn {{
    position: absolute;
    top: 20px;
    right: 30px;
    font-size: 32px;
    color: white;
    cursor: pointer;
}}
</style>
</head>
<body>

<header>
    <div>
        <h1>English Dysgraphia Candidate Harvester</h1>
        <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
            Offline 2D Handwriting Scrapes for Model Dataset Expansion & Manual Verification
        </p>
    </div>
    <div class="stats-bar">
        <div class="stat-pill">Total Harvested: <strong id="stat-total">0</strong></div>
        <div class="stat-pill accepted">Accepted: <strong id="stat-accepted">0</strong></div>
        <div class="stat-pill rejected">Rejected: <strong id="stat-rejected">0</strong></div>
        <div class="stat-pill">Pending: <strong id="stat-pending">0</strong></div>
    </div>
</header>

<div class="toolbar">
    <div class="filter-group">
        <button class="filter-btn active" onclick="setFilter('all')">All Images</button>
        <button class="filter-btn" onclick="setFilter('pending')">Pending Review</button>
        <button class="filter-btn" onclick="setFilter('accepted')">Accepted (Valid)</button>
        <button class="filter-btn" onclick="setFilter('rejected')">Rejected</button>
        <button class="filter-btn" onclick="setFilter('Reddit')">Reddit</button>
        <button class="filter-btn" onclick="setFilter('Wikimedia Commons')">Wikimedia</button>
        <button class="filter-btn" onclick="setFilter('Educational / Clinical Portal')">Portals</button>
        <button class="filter-btn" onclick="setFilter('PubMed Central (Clinical)')">Clinical (PMC)</button>
        <button class="filter-btn" onclick="setFilter('Handwriting Disorder Benchmark')">Benchmark</button>
    </div>
    <button class="export-btn" onclick="exportAcceptedDataset()">
        Export Accepted Manifest
    </button>
</div>

<div class="grid" id="card-grid"></div>

<!-- Lightbox Modal -->
<div id="lightbox" onclick="closeLightbox()">
    <span class="close-btn">&times;</span>
    <img id="lightbox-img" src="" alt="Enlarged handwriting">
</div>

<script>
const items = {records_json};
let userDecisions = JSON.parse(localStorage.getItem('dysgraphia_decisions') || '{{}}');
let currentFilter = 'all';

function getBadgeClass(platform) {{
    if (platform.includes('Reddit')) return 'reddit';
    if (platform.includes('Wikimedia')) return 'wikimedia';
    if (platform.includes('Portal')) return 'portal';
    if (platform.includes('PubMed')) return 'clinical';
    return 'benchmark';
}}

function renderCards() {{
    const grid = document.getElementById('card-grid');
    grid.innerHTML = '';
    
    let acceptedCount = 0;
    let rejectedCount = 0;
    let pendingCount = 0;

    items.forEach(item => {{
        const status = userDecisions[item.candidate_id] || 'pending';
        if (status === 'accepted') acceptedCount++;
        else if (status === 'rejected') rejectedCount++;
        else pendingCount++;

        // Filter condition
        if (currentFilter === 'pending' && status !== 'pending') return;
        if (currentFilter === 'accepted' && status !== 'accepted') return;
        if (currentFilter === 'rejected' && status !== 'rejected') return;
        if (['Reddit', 'Wikimedia Commons', 'Educational / Clinical Portal', 'PubMed Central (Clinical)', 'Handwriting Disorder Benchmark'].includes(currentFilter)) {{
            if (item.source_platform !== currentFilter) return;
        }}

        const card = document.createElement('div');
        card.className = `card ${{status === 'accepted' ? 'is-accepted' : status === 'rejected' ? 'is-rejected' : ''}}`;
        card.id = `card-${{item.candidate_id}}`;

        const badgeClass = getBadgeClass(item.source_platform);
        const riskHtml = item.ai_screening_risk !== null ? 
            `<div class="risk-badge ${{item.ai_screening_risk >= 45 ? 'risk-high' : 'risk-low'}}">AI Risk: ${{item.ai_screening_risk}}%</div>` : '';

        card.innerHTML = `
            <div class="img-container" onclick="openLightbox('${{item.relative_path}}')">
                <span class="badge ${{badgeClass}}">${{item.source_platform}}</span>
                ${{riskHtml}}
                <img src="${{item.relative_path}}" loading="lazy" alt="${{item.candidate_id}}">
            </div>
            <div class="card-body">
                <div class="card-title" title="${{item.context_title}}">${{item.context_title}}</div>
                <div class="meta-details">
                    <span>${{item.candidate_id}}</span>
                    <span>${{item.width}}x${{item.height}} px</span>
                    <span>${{item.filesize_kb}} KB</span>
                </div>
                <div class="card-actions">
                    <button class="btn btn-accept ${{status === 'accepted' ? 'active' : ''}}" onclick="decide('${{item.candidate_id}}', 'accepted')">
                        Accept
                    </button>
                    <button class="btn btn-reject ${{status === 'rejected' ? 'active' : ''}}" onclick="decide('${{item.candidate_id}}', 'rejected')">
                        Reject
                    </button>
                    <a class="btn-link" href="${{item.source_page_url}}" target="_blank" title="View Source Page">Link</a>
                </div>
            </div>
        `;
        grid.appendChild(card);
    }});

    document.getElementById('stat-total').innerText = items.length;
    document.getElementById('stat-accepted').innerText = acceptedCount;
    document.getElementById('stat-rejected').innerText = rejectedCount;
    document.getElementById('stat-pending').innerText = pendingCount;
}}

function decide(candId, decision) {{
    if (userDecisions[candId] === decision) {{
        delete userDecisions[candId];
    }} else {{
        userDecisions[candId] = decision;
    }}
    localStorage.setItem('dysgraphia_decisions', JSON.stringify(userDecisions));
    renderCards();
}}

function setFilter(filter) {{
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    renderCards();
}}

function openLightbox(src) {{
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox').classList.add('active');
}}

function closeLightbox() {{
    document.getElementById('lightbox').classList.remove('active');
}}

function exportAcceptedDataset() {{
    const acceptedIds = Object.keys(userDecisions).filter(k => userDecisions[k] === 'accepted');
    if (acceptedIds.length === 0) {{
        alert('No candidates accepted yet! Click "Accept" on valid handwriting images first.');
        return;
    }}

    const acceptedItems = items.filter(it => acceptedIds.includes(it.candidate_id));
    const csvContent = "data:text/csv;charset=utf-8," + 
        ["candidate_id,filename,source_platform,width,height,source_url,original_image_url"]
        .concat(acceptedItems.map(it => `"${{it.candidate_id}}","${{it.filename}}","${{it.source_platform}}",${{it.width}},${{it.height}},"${{it.source_page_url}}","${{it.original_image_url}}"`))
        .join("\\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `accepted_dysgraphia_samples_${{acceptedItems.length}}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}}

renderCards();
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    log(f"Interactive Review Gallery created at: {html_path}")
    return html_path

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    log("=" * 60)
    log("MULTI-SOURCE ENGLISH DYSGRAPHIA HANDWRITING HARVESTER")
    log("=" * 60)
    
    all_candidates = []
    
    # 1. Reddit
    try:
        reddit_candidates = harvest_reddit()
        all_candidates.extend(reddit_candidates)
    except Exception as e:
        log(f"Error in Reddit harvesting: {e}")
        
    # 2. Wikimedia Commons
    try:
        wiki_candidates = harvest_wikimedia()
        all_candidates.extend(wiki_candidates)
    except Exception as e:
        log(f"Error in Wikimedia harvesting: {e}")
        
    # 3. Clinical & Educational Portals
    try:
        portal_candidates = harvest_educational_portals()
        all_candidates.extend(portal_candidates)
    except Exception as e:
        log(f"Error in Portal harvesting: {e}")
        
    # 4. PubMed Central Clinical Figures
    try:
        pmc_candidates = harvest_pmc()
        all_candidates.extend(pmc_candidates)
    except Exception as e:
        log(f"Error in PMC harvesting: {e}")
        
    # 5. Disorder Benchmark Repo Samples
    try:
        repo_candidates = harvest_repo_samples()
        all_candidates.extend(repo_candidates)
    except Exception as e:
        log(f"Error in repo samples harvesting: {e}")
        
    log(f"\nAggregated {len(all_candidates)} raw candidate images across 5 web sources.")
    
    # 6. Process, validate, filter, AI pre-screen & save
    manifest_records = process_and_save_candidates(all_candidates)
    
    # 7. Generate Review Gallery
    gallery_path = generate_review_gallery(manifest_records)
    
    log("\n" + "=" * 60)
    log(f"HARVESTING COMPLETE!")
    log(f"   Total Validated Offline 2D Images: {len(manifest_records)}")
    log(f"   Review Gallery: file:///{gallery_path.replace(os.sep, '/')}")
    log("=" * 60)

if __name__ == "__main__":
    main()
