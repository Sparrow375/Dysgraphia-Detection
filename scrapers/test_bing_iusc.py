import requests
import json
import html
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

url = 'https://www.bing.com/images/search?q=dysgraphia+handwriting+sample&first=1'
r = requests.get(url, headers=headers, timeout=15)
soup = BeautifulSoup(r.text, 'html.parser')

items = []
for a in soup.find_all('a', class_='iusc'):
    m_attr = a.get('m')
    if m_attr:
        try:
            data = json.loads(m_attr)
            items.append({
                'title': data.get('t', '') or data.get('desc', ''),
                'murl': data.get('murl'),
                'purl': data.get('purl'),
            })
        except Exception:
            pass

print(f"Extracted {len(items)} items with full metadata!")
for i, item in enumerate(items[:8]):
    print(f"[{i+1}] {item['title'][:60]}")
    print(f"    Page:  {item['purl']}")
    print(f"    Image: {item['murl']}")
