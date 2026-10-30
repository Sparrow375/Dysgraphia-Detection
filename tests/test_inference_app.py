"""
Integration Tests for Dysgraphia Inference Web App & REST API.
"""

import asyncio
import io
import numpy as np
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from qa.inference_app import create_app


def test_health_endpoint():
    """Verify health endpoint returns status healthy."""
    async def _test():
        app = create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.get("/api/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"
            assert "device" in data
            assert "gpu_name" in data
        finally:
            await client.close()

    asyncio.run(_test())


def test_index_page():
    """Verify index HTML page loads with screening studio."""
    async def _test():
        app = create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.get("/")
            assert resp.status == 200
            text = await resp.text()
            assert "Dysgraphia Screening Studio" in text
            assert "crop-canvas" in text
        finally:
            await client.close()

    asyncio.run(_test())


def test_screen_endpoint_with_synthetic_handwriting():
    """Verify /api/screen processes an uploaded image."""
    async def _test():
        app = create_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            # Create synthetic image with dark handwriting strokes
            h, w = 120, 400
            img_arr = np.full((h, w, 3), 245, dtype=np.uint8)
            # Add horizontal handwriting strokes of height 45px
            for x in range(30, 360, 25):
                img_arr[35:80, x:x+15] = [40, 40, 40]

            pil_img = Image.fromarray(img_arr)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            form = FormData()
            form.add_field("image", buf, filename="test_crop.png", content_type="image/png")
            form.add_field("config", "no_guideline")
            form.add_field("model", "all")

            resp = await client.post("/api/screen", data=form)
            assert resp.status == 200
            res = await resp.json()

            assert res["success"] is True
            assert "scores" in res
            assert "primary_score" in res["scores"]
            assert "tier" in res
            assert "preview_tiles" in res
            assert len(res["preview_tiles"]) > 0
        finally:
            await client.close()

    asyncio.run(_test())

