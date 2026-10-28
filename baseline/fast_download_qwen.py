"""
High-Speed Resumable Downloader for Qwen2-VL-2B model shard.
Uses HTTP Range requests with 4MB chunks and real-time speed logging.
"""

import os
import sys
import time
import requests
import torch

TARGET_FILE = r"C:\Users\Avaneesh\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\895c3a49bc3fa70a340399125c650a463535e71c\model-00001-of-00002.safetensors"
EXPECTED_SIZE = 3988609112  # Exact bytes
URL = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00001-of-00002.safetensors"

def download():
    current_size = os.path.getsize(TARGET_FILE) if os.path.exists(TARGET_FILE) else 0
    print(f"Current local size: {current_size / (1024**3):.2f} GB / {EXPECTED_SIZE / (1024**3):.2f} GB", flush=True)

    if current_size >= EXPECTED_SIZE:
        print("File already completely downloaded!", flush=True)
        return

    headers = {}
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"Resuming download from byte {current_size} ({current_size / (1024**3):.2f} GB)...", flush=True)

    t0 = time.time()
    last_print_time = t0
    last_print_size = current_size

    with requests.get(URL, headers=headers, stream=True, timeout=30) as r:
        if r.status_code not in (200, 206):
            print(f"HTTP Error: {r.status_code} - {r.text}", flush=True)
            return

        with open(TARGET_FILE, "ab" if current_size > 0 else "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                if chunk:
                    f.write(chunk)
                    current_size += len(chunk)
                    now = time.time()
                    if now - last_print_time >= 5.0:
                        speed_mb = (current_size - last_print_size) / (now - last_print_time) / (1024 * 1024)
                        pct = (current_size / EXPECTED_SIZE) * 100
                        remaining_bytes = max(0, EXPECTED_SIZE - current_size)
                        eta_sec = remaining_bytes / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                        print(
                            f"[{pct:5.1f}%] {current_size / (1024**3):.2f} / {EXPECTED_SIZE / (1024**3):.2f} GB "
                            f"| Speed: {speed_mb:5.2f} MB/s | ETA: {int(eta_sec // 60)}m {int(eta_sec % 60):02d}s",
                            flush=True
                        )
                        last_print_time = now
                        last_print_size = current_size

    print(f"Download completed in {time.time() - t0:.1f}s! Final size: {os.path.getsize(TARGET_FILE)} bytes.", flush=True)

    # Now verify model loading into VRAM
    print("Testing model load into GPU VRAM...", flush=True)
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    t_load = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Model loaded successfully in {time.time() - t_load:.2f}s!", flush=True)
    if torch.cuda.is_available():
        print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
        print(f"Reserved VRAM: {torch.cuda.memory_reserved() / (1024**3):.2f} GB", flush=True)

if __name__ == "__main__":
    download()

