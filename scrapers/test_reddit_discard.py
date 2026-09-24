import requests
import cv2
import numpy as np

urls = [
    'https://i.redd.it/qqxs8dyyfpc61.jpg',
    'https://i.redd.it/jmi7hg4h38m61.png',
    'https://i.imgur.com/8YLigBV.jpg',
    'https://i.redd.it/deuzheb1xx7e1.jpeg',
    'https://i.redd.it/oldi0l3s5xgg1.jpeg',
    'https://i.redd.it/fakxsk6ul2m61.png',
    'https://i.redd.it/gcf1t5vxilm61.png',
    'https://i.redd.it/yddbrscxmgx51.jpg',
    'https://i.redd.it/19k08rvsl8891.jpg',
    'https://i.redd.it/zgfis2fb24ph1.jpeg'
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

sess = requests.Session()
sess.headers.update(headers)

for u in urls:
    try:
        r = sess.get(u, timeout=10)
        print(f"URL: {u} -> Status: {r.status_code}, Length: {len(r.content)}")
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print("  -> imdecode returned None!")
            else:
                h, w = img.shape[:2]
                print(f"  -> Decoded OK: {w}x{h}")
    except Exception as e:
        print(f"URL: {u} -> Exception: {e}")
