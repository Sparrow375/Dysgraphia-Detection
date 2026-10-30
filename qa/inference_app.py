"""
Interactive Web Application Server for Production Dysgraphia Screening.

Provides:
- Web GUI at http://localhost:8080/ for uploading scanned sheets / photos,
  interactively cropping text via HTML5 canvas, and running multiple vision models.
- REST API at /api/screen for automated inference.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from aiohttp import web

from pipeline.inference_engine import InferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_app")

WEB_DIR = REPO_ROOT / "qa" / "web"
HTML_INDEX = WEB_DIR / "inference.html"


async def handle_index(request: web.Request) -> web.Response:
    """Serve the interactive screening HTML interface."""
    if not HTML_INDEX.exists():
        return web.Response(text="inference.html not found", status=404)
    with open(HTML_INDEX, "r", encoding="utf-8") as f:
        html = f.read()
    return web.Response(text=html, content_type="text/html")


async def handle_health(request: web.Request) -> web.Response:
    """Health and GPU status check."""
    engine = InferenceEngine.get_instance()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    return web.json_response({
        "status": "healthy",
        "device": str(engine.device),
        "gpu_name": gpu_name,
        "cuda_available": torch.cuda.is_available(),
    })


async def handle_screen(request: web.Request) -> web.Response:
    """Run screening on uploaded image / crop."""
    reader = await request.multipart()
    image_bytes = None
    config = "no_guideline"
    model_name = "all"
    crop_info = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "image":
            image_bytes = await part.read()
        elif part.name == "config":
            config = (await part.read()).decode("utf-8")
        elif part.name == "model":
            model_name = (await part.read()).decode("utf-8")
        elif part.name == "crop":
            crop_str = (await part.read()).decode("utf-8")
            try:
                crop_info = json.loads(crop_str)
            except Exception:
                crop_info = None

    if not image_bytes:
        return web.json_response({"success": False, "error": "No image uploaded."}, status=400)

    # Decode image with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return web.json_response({"success": False, "error": "Failed to decode image."}, status=400)

    # Apply crop if coordinates provided
    if crop_info and isinstance(crop_info, dict):
        x = max(0, int(crop_info.get("x", 0)))
        y = max(0, int(crop_info.get("y", 0)))
        w = int(crop_info.get("width", img.shape[1]))
        h = int(crop_info.get("height", img.shape[0]))
        if w > 10 and h > 10:
            img = img[y : y + h, x : x + w]

    try:
        engine = InferenceEngine.get_instance()
        result = engine.infer_crop(img, config=config, model_name=model_name)
        return web.json_response(result)
    except Exception as e:
        logger.exception("Inference error")
        return web.json_response({"success": False, "error": str(e)}, status=500)


def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB max upload
    app.router.add_get("/", handle_index)
    app.router.add_get("/inference.html", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_post("/api/screen", handle_screen)
    app.router.add_static("/static/", WEB_DIR, show_index=False)
    return app


def main():
    parser = argparse.ArgumentParser(description="Dysgraphia Screening Studio Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    args = parser.parse_args()

    app = create_app()
    logger.info("Starting Dysgraphia Screening Studio on http://%s:%d ...", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
