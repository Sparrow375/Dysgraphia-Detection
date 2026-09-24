import requests
import re
import html

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}

url = 'https://www.bing.com/images/search?q=dysgraphia+handwriting+sample&first=1&count=35'
r = requests.get(url, headers=headers, timeout=15)
print('Bing status:', r.status_code)

unescaped = html.unescape(r.text)
murls = re.findall(r'"murl":"(https?://[^"]+)"', unescaped)
print(f'Found {len(murls)} images from Bing')
for i, m in enumerate(murls[:5]):
    print(f'{i+1}: {m}')
