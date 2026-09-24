import requests
import xml.etree.ElementTree as ET
import re
import html

url = 'https://www.reddit.com/r/dysgraphia/top.rss?t=all&limit=25'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

r = requests.get(url, headers=headers)
root = ET.fromstring(r.content)
ns = {'atom': 'http://www.w3.org/2005/Atom'}
entries = root.findall('atom:entry', ns)

print(f"Total entries: {len(entries)}")
for i, entry in enumerate(entries[:8]):
    title = entry.find('atom:title', ns).text
    content = entry.find('atom:content', ns).text
    links = re.findall(r'href="([^"]+)"', content)
    redd_links = [l for l in links if 'redd.it' in l or 'imgur' in l]
    print(f"\n[{i+1}] {title}")
    for l in redd_links:
        clean = html.unescape(l)
        print(f"   Link: {clean}")
        try:
            head = requests.head(clean, headers=headers, timeout=5)
            print(f"   Status: {head.status_code}, Content-Type: {head.headers.get('Content-Type')}")
        except Exception as e:
            print(f"   Error: {e}")
