"""
Dysgraphia Screening & Kinematics Studio - Web Application Server
Run with: python app.py
Serves the modern glassmorphic web testing ground on http://127.0.0.1:7860
Allows uploading or pasting live handwriting photos to extract 20D multimodal features,
visualize spatial BHK explainability overlays, and inspect recovered kinematic velocity profiles.
"""

import os
import io
import json
import base64
import glob
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.pipeline import DysgraphiaFeaturePipeline, compute_dysgraphia_screening_verdict

# Global pipeline instance
pipeline = DysgraphiaFeaturePipeline(compute_kinematics=True)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")


def render_overlay_b64(orig_img: Image.Image, results: Dict[str, Any]) -> str:
    """
    Renders the BHK explainability overlay:
    - Green bounding boxes around valid letters
    - Cyan centroids
    - Coral fitted baseline wander line
    """
    img_rgb = orig_img.convert("RGB")
    w, h = img_rgb.size
    draw = ImageDraw.Draw(img_rgb)

    text_lines = results.get("visual_artifacts", {}).get("text_lines", [])

    for line in text_lines:
        line_pts_x = []
        line_bottoms_y = []

        for c in line.components:
            # Draw green bounding box: [x_min, y_min, x_max, y_max]
            draw.rectangle([c.x_min, c.y_min, c.x_max, c.y_max], outline="#10b981", width=2)

            # Centroid
            cx, cy = int(round(c.x_center)), int(round(c.y_center))
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill="#38bdf8", outline="#0284c7")

            line_pts_x.append(cx)
            line_bottoms_y.append(c.y_bottom)

        # Draw line baseline fit
        if len(line_pts_x) >= 3 and (max(line_pts_x) - min(line_pts_x) > 20):
            try:
                poly = np.polyfit(line_pts_x, line_bottoms_y, 1)
                x_start = int(min(line_pts_x))
                x_end = int(max(line_pts_x))
                y_start = int(np.polyval(poly, x_start))
                y_end = int(np.polyval(poly, x_end))
                draw.line([(x_start, y_start), (x_end, y_end)], fill="#f43f5e", width=2)
            except Exception:
                pass

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_ink_and_skeleton_b64(results: Dict[str, Any]) -> str:
    """
    Renders side-by-side view:
    Left: Illumination-normalized binary mask (black background, white ink)
    Right: 1-pixel morphological skeleton spine
    """
    mask = results.get("visual_artifacts", {}).get("binary_mask", np.zeros((100, 100)))
    skeleton = results.get("visual_artifacts", {}).get("skeleton", np.zeros((100, 100)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0b0f19")
    ax1.imshow(mask, cmap="gray")
    ax1.set_title("1. Cleaned Binary Ink Mask", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(skeleton, cmap="gray")
    ax2.set_title("2. Centerline Skeleton Spine (1-px)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def render_kinematics_waveform_b64(results: Dict[str, Any]) -> str:
    """
    Renders the continuous reconstructed velocity curve v(t) with:
    - Speed line in vibrant cyan
    - Local velocity peaks marked in emerald green dots
    - Velocity Inversion (NVI) hesitations marked in orange triangles
    """
    v = results.get("visual_artifacts", {}).get("velocity_profile", np.array([]))
    dt = 0.01

    if len(v) == 0:
        v = np.zeros(50)

    t = np.arange(len(v)) * dt

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, facecolor="#0b0f19")
    for ax in (ax_v, ax_a):
        ax.set_facecolor("#131b2e")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # Plot velocity
    ax_v.plot(t, v, color="#38bdf8", lw=1.6, label="Reconstructed Velocity v(t)")

    # Find and annotate peaks & troughs (NVI)
    if len(v) >= 5:
        peaks, _ = find_peaks(v)
        troughs, _ = find_peaks(-v)
        ax_v.plot(t[peaks], v[peaks], "o", color="#34d399", markersize=4, label=f"Velocity Peaks ({len(peaks)})")
        ax_v.plot(t[troughs], v[troughs], "v", color="#fbbf24", markersize=4, label=f"Velocity Troughs/Hesitations ({len(troughs)})")

    ax_v.set_ylabel("Speed (px/s)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax_v.set_title("Reconstructed Neuromuscular Velocity Profile v(t) & NVI Markers", color="#f8fafc", fontsize=11, fontweight="bold")
    ax_v.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", fontsize=8)

    # Plot acceleration
    acc = np.gradient(v, dt)
    ax_a.plot(t, acc, color="#a78bfa", lw=1.2, label="Acceleration a(t)")
    ax_a.axhline(0, color="#475569", linestyle=":", lw=1)
    ax_a.set_xlabel("Reconstructed Time (seconds)", color="#f8fafc", fontsize=10)
    ax_a.set_ylabel("Accel (px/s²)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax_a.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


class StudioHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Dysgraphia Screening Studio."""

    def log_message(self, format, *args):
        # Concise logging
        print(f"[{self.log_date_time_string()}] {args[0]} {args[1]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        elif path == "/style.css":
            self._serve_file(os.path.join(WEB_DIR, "style.css"), "text/css")
        elif path == "/app.js":
            self._serve_file(os.path.join(WEB_DIR, "app.js"), "application/javascript")
        elif path == "/api/demo":
            self._handle_demo(parsed.query)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze()
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(404, "File Not Found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_demo(self, query_str: str):
        params = urllib.parse.parse_qs(query_str)
        demo_type = params.get("type", ["malay_lpd"])[0]

        file_map = {
            "malay_lpd": "Datasets/DATASET DYSGRAPHIA HANDWRITING/Low Potential Dysgraphia/LPD (10).jpg",
            "malay_pd": "Datasets/DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (10).jpg",
            "drotar_leto_ctrl": "Datasets/reconstructed_dataset/by_task/task_5_leto/control/user_00050_task_5_leto.png",
            "drotar_leto_dys": "Datasets/reconstructed_dataset/by_task/task_5_leto/dysgraphic/user_00006_task_5_leto.png",
            "drotar_sentence": "Datasets/reconstructed_dataset/by_task/task_8_sentence/control/user_00050_task_8_sentence.png",
        }

        rel_path = file_map.get(demo_type)
        if not rel_path or not os.path.exists(rel_path):
            # Fallback to any file
            candidates = glob.glob("Datasets/**/*.jpg", recursive=True) or glob.glob("Datasets/**/*.png", recursive=True)
            if candidates:
                rel_path = candidates[0]
            else:
                self.send_error(404, "No demo file available")
                return

        with open(rel_path, "rb") as f:
            content = f.read()

        content_type = "image/png" if rel_path.endswith(".png") else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_analyze(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json({"error": "Empty request body"}, 400)
                return

            body = self.rfile.read(content_length)

            # Simple multipart/form-data or raw image extraction
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                # Find image boundary
                boundary = content_type.split("boundary=")[1].encode()
                parts = body.split(b"--" + boundary)
                image_bytes = None
                for part in parts:
                    if b"Content-Type: image" in part or b"\r\n\r\n" in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            chunk = part[header_end + 4:].rstrip(b"\r\n")
                            if len(chunk) > 100:
                                image_bytes = chunk
                                break
                if image_bytes is None:
                    image_bytes = body
            else:
                image_bytes = body

            # Load into PIL
            orig_pil = Image.open(io.BytesIO(image_bytes))

            # Run 20D feature extraction pipeline
            results = pipeline.extract(orig_pil)

            # Evaluate clinical risk verdict
            verdict = compute_dysgraphia_screening_verdict(results)

            # Generate visual plots
            overlay_b64 = render_overlay_b64(orig_pil, results)
            ink_skeleton_b64 = render_ink_and_skeleton_b64(results)
            kinematics_plot_b64 = render_kinematics_waveform_b64(results)

            response_payload = {
                "status": "success",
                "verdict": verdict,
                "visuals": {
                    "overlay_b64": overlay_b64,
                    "ink_skeleton_b64": ink_skeleton_b64,
                    "kinematics_plot_b64": kinematics_plot_b64,
                },
                "features": {
                    "bhk": results["bhk_metrics"],
                    "kinematics": results["kinematic_metrics"],
                },
                "metadata": results["metadata"]
            }

            self._send_json(response_payload, 200)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 7860):
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    print("\n" + "=" * 70)
    print(">> Dysgraphia Screening & Kinematics Studio Active!")
    print(f">> Open in your web browser: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server(7860)
