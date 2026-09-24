"""
Trajectory rendering and multimodal visualization module.
Transforms online coordinate time-series into clean static ink images
and generates paired diagnostic plots (spatial ink + temporal pressure/speed).
"""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.loaders import SampleData


def render_trajectory_to_image(
    sample: SampleData,
    target_width: int = 1200,
    padding: int = 40,
    line_width: int = 3,
    invert_y: bool = True
) -> Image.Image:
    """
    Renders on-surface strokes into a clean grayscale static image (black ink on white canvas).
    Automatically scales coordinates while preserving aspect ratio.
    """
    if not sample.strokes:
        # Blank canvas if no on-surface strokes
        return Image.new("L", (target_width, 400), color=255)

    # Collect all on-surface points
    all_x = np.concatenate([s[:, 0] for s in sample.strokes])
    all_y = np.concatenate([s[:, 1] for s in sample.strokes])

    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))

    span_x = max(max_x - min_x, 1e-5)
    span_y = max(max_y - min_y, 1e-5)

    # Calculate target height based on aspect ratio
    drawable_width = target_width - 2 * padding
    scale = drawable_width / span_x
    drawable_height = int(np.round(span_y * scale))
    target_height = max(drawable_height + 2 * padding, 100)

    # Create white canvas
    img = Image.new("L", (target_width, target_height), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in sample.strokes:
        pts = stroke[:, :2]
        if len(pts) < 2:
            continue

        # Map to pixel space
        px = padding + (pts[:, 0] - min_x) * scale
        if invert_y:
            py = padding + (max_y - pts[:, 1]) * scale
        else:
            py = padding + (pts[:, 1] - min_y) * scale

        pixel_coords = list(zip(px, py))
        draw.line(pixel_coords, fill=0, width=line_width, joint="curve")

    return img


def compute_instantaneous_velocity(sample: SampleData) -> Tuple[np.ndarray, np.ndarray]:
    """Computes speed |dr/dt| for each point in sample."""
    pts = sample.points
    if len(pts) < 2:
        return np.zeros(len(pts)), pts[:, 2]

    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    dt = np.diff(pts[:, 2])

    # Avoid division by zero
    dt_safe = np.where(dt <= 1e-6, 1e-6, dt)
    speed = np.sqrt(dx**2 + dy**2) / dt_safe
    speed = np.concatenate([[speed[0]], speed])  # Match length
    return speed, pts[:, 2]


def plot_multimodal_diagnostics(
    sample: SampleData,
    rendered_img: Image.Image,
    output_path: str,
    title: Optional[str] = None
) -> None:
    """
    Creates a 4-panel diagnostic figure comparing static ink with raw sensor time-series:
      1. Rendered Static Handwriting (Ink on Paper)
      2. 2D Trajectory with instantaneous pressure encoding
      3. Stylus Pressure P(t) over elapsed time
      4. Stylus Velocity v(t) over elapsed time
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    speed, t = compute_instantaneous_velocity(sample)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    # Panel 1: Rendered Static Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rendered_img, cmap="gray")
    ax1.set_title("1. Rendered Static Ink Image (Branch A Ground Truth Input)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Spatial Trajectory colored by pressure
    ax2 = fig.add_subplot(gs[0, 1])
    on_mask = (sample.pen_status > 0.5) & (sample.pressure > 0)
    if np.any(on_mask):
        sc = ax2.scatter(
            sample.x[on_mask],
            sample.y[on_mask],
            c=sample.pressure[on_mask],
            cmap="viridis",
            s=4,
            alpha=0.8
        )
        cbar = plt.colorbar(sc, ax=ax2, orientation="vertical", shrink=0.8)
        cbar.set_label("Sensor Pressure", fontsize=9)
    ax2.set_title("2. Spatial 2D Trajectory (Pressure-Colored)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Pressure vs Time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, sample.pressure, color="#2b5c8f", lw=1.2, label="Stylus Pressure")
    # Shade on-paper regions
    ax3.fill_between(t, 0, sample.pressure, where=(sample.pen_status > 0.5), color="#2b5c8f", alpha=0.2, label="On-Paper Contact")
    ax3.set_title("3. Stylus Pressure Profile P(t)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Pressure (raw units)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4: Velocity vs Time
    ax4 = fig.add_subplot(gs[1, 1])
    # Clip extreme outlier speed spikes (e.g. sensor teleport) for clean visualization
    q99 = np.percentile(speed, 99) if len(speed) > 0 else 1.0
    speed_plot = np.clip(speed, 0, max(q99 * 1.5, 10.0))
    ax4.plot(t, speed_plot, color="#d95f02", lw=1.0, label="Speed |dr/dt|")
    ax4.set_title("4. Instantaneous Velocity Profile v(t)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Speed (coords/s)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", fontsize=8)

    header = title or f"{sample.dataset_name} | {sample.sample_id} ({sample.task_name})"
    subtext = f"Points: {len(sample.points):,} | Strokes: {len(sample.strokes)} | Duration: {sample.total_duration_sec:.1f}s | Fs: {sample.sampling_rate_hz:.1f} Hz | In-Air: {sample.in_air_ratio*100:.1f}%"
    fig.suptitle(f"{header}\n{subtext}", fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
