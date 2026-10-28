"""Test loading Qwen2-VL-2B into GPU VRAM."""
import sys
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2-VL-2B-Instruct"
print(f"Downloading and loading {model_id} model weights and processor...", flush=True)

t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
print(f"Processor loaded in {time.time() - t0:.2f}s", flush=True)

t1 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"Qwen2-VL-2B model loaded successfully into GPU VRAM in {time.time() - t1:.2f}s!", flush=True)
if torch.cuda.is_available():
    print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
    print(f"Reserved VRAM: {torch.cuda.memory_reserved() / (1024**3):.2f} GB", flush=True)


