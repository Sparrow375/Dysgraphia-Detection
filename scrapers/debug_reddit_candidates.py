import requests
import html
import re
import cv2
import numpy as np

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

feed_url = "https://www.reddit.com/r/dysgraphia/top.rss?t=all&limit=25"
r = requests.get(feed_url, headers=headers)

import xml.etree.ElementTree as ET
root = ET.fromstring(r.content)
ns = {'atom': 'http://www.w3.org/2005/Atom'}
entries = root.findall('atom:entry', ns)

sess = requests.Session()
sess.headers.update(headers)

for entry in entries[:6]:
    title = entry.find('atom:title', ns).text
    content = entry.find('atom:content', ns).text
    print(f"\n--- Entry: {title[:40]} ---")
    
    img_links = re.findall(r'href="(https?://i\.redd\.it/[^\"]+)"', content)
    img_links += re.findall(r'src="(https?://preview\.redd\.it/[^\"]+)"', content)
    img_links += re.findall(r'href="(https?://(?:i\.)?imgur\.com/[^\"]+?\.(?:jpg|jpeg|png|webp))"', content)
    
    print(f"Matched raw links: {img_links}")
    for raw in img_links:
        clean = html.unescape(raw).split('?')[0].replace("preview.redd.it", "i.redd.it")
        print(f"Clean URL: {clean}")
        try:
            resp = sess.get(clean, timeout=10)
            print(f"  HTTP: {resp.status_code}, Bytes: {len(resp.content)}")
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print("  cv2.imdecode FAILED")
            else:
                h, w = img.shape[:2]
                print(f"  Decoded: {w}x{h}")
        except Exception as e:
            print(f"  Fetch exception: {e}")
