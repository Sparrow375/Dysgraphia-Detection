import requests
import xml.etree.ElementTree as ET
import re
import html

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

urls = [
    'https://www.reddit.com/r/dysgraphia/search.rss?q=handwriting&restrict_sr=1&sort=top',
    'https://www.reddit.com/r/dysgraphia/search.rss?q=writing&restrict_sr=1&sort=top',
    'https://www.reddit.com/r/dysgraphia/search.rss?q=sample&restrict_sr=1&sort=top',
    'https://www.reddit.com/r/dysgraphia/top.rss?t=all&limit=100',
    'https://www.reddit.com/r/Handwriting/search.rss?q=dysgraphia&restrict_sr=1&sort=top',
    'https://www.reddit.com/r/dyslexia/search.rss?q=dysgraphia&restrict_sr=1&sort=top'
]

all_images = []

for url in urls:
    try:
        r = requests.get(url, headers=headers, timeout=12)
        print(f"URL: {url} -> Status: {r.status_code}")
        if r.status_code == 200:
            # Parse XML
            root = ET.fromstring(r.content)
            # Find entry elements
            # namespace in atom is http://www.w3.org/2005/Atom
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            print(f"  Found {len(entries)} entries")
            for entry in entries:
                title = entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else ''
                content = entry.find('atom:content', ns).text if entry.find('atom:content', ns) is not None else ''
                link_elem = entry.find('atom:link', ns)
                link = link_elem.get('href') if link_elem is not None else ''
                
                # Search for direct images in content
                if content:
                    img_links = re.findall(r'href="(https?://i\.redd\.it/[^\"]+)"', content)
                    img_links += re.findall(r'src="(https?://preview\.redd\.it/[^\"]+)"', content)
                    img_links += re.findall(r'href="(https?://(?:i\.)?imgur\.com/[^\"]+)"', content)
                    for img in set(img_links):
                        # clean up html entities
                        clean_img = html.unescape(img)
                        all_images.append({
                            'title': title,
                            'post_url': link,
                            'image_url': clean_img,
                            'source': 'reddit'
                        })
    except Exception as e:
        print(f"Error on {url}: {e}")

print(f"\nTotal Reddit images discovered: {len(all_images)}")
# Deduplicate
seen = set()
unique = []
for item in all_images:
    if item['image_url'] not in seen:
        seen.add(item['image_url'])
        unique.append(item)

print(f"Unique images: {len(unique)}")
for i, item in enumerate(unique[:10]):
    print(f"[{i+1}] {item['title'][:60]}")
    print(f"    Post:  {item['post_url']}")
    print(f"    Image: {item['image_url']}")
