import sys
import time
import math
import pandas as pd
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QTabletEvent, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget


class RealtimeTabletKinematics(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XP-Pen Real-time Kinematics Capture")
        self.resize(1200, 750)
        self.setStyleSheet("background-color: #121212;")

        self.records = []
        self.strokes = []           # List of strokes, where each stroke is a list of (QPointF, pressure)
        self.current_stroke = []
        self.stroke_count = 0
        self.was_down = False
        self.start_time = None

        self.live_data = {
            "x": 0.0,
            "y": 0.0,
            "pressure": 0.0,
            "x_tilt": 0,
            "y_tilt": 0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "jerk": 0.0,
            "hz": 0.0,
            "stroke": 0,
            "samples": 0
        }

    def tabletEvent(self, event: QTabletEvent):
        current_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = current_time

        rel_time = current_time - self.start_time
        pos = event.position()
        x, y = pos.x(), pos.y()
        pressure = event.pressure()
        x_tilt = event.xTilt()
        y_tilt = event.yTilt()
        is_down = pressure > 0.0

        # Handle pen-down and stroke transition
        if is_down and not self.was_down:
            self.stroke_count += 1
            self.current_stroke = []
            self.strokes.append(self.current_stroke)

        dt = 0.0
        ds = 0.0
        velocity = 0.0
        acceleration = 0.0
        jerk = 0.0
        hz = 0.0

        # Only compute continuous kinematics if remaining within the same contact stroke
        if self.records:
            prev = self.records[-1]
            dt = rel_time - prev["time_sec"]

            if dt > 1e-6:
                hz = 1.0 / dt
                if is_down and self.was_down:
                    dx = x - prev["x"]
                    dy = y - prev["y"]
                    ds = math.hypot(dx, dy)
                    velocity = ds / dt
                    acceleration = (velocity - prev["velocity"]) / dt
                    jerk = (acceleration - prev["acceleration"]) / dt

        row = {
            "timestamp": current_time,
            "time_sec": rel_time,
            "stroke_id": self.stroke_count if is_down else 0,
            "dt": dt,
            "x": x,
            "y": y,
            "pressure": pressure,
            "x_tilt": x_tilt,
            "y_tilt": y_tilt,
            "is_down": int(is_down),
            "displacement": ds,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
        }
        self.records.append(row)

        if is_down:
            self.current_stroke.append((pos, pressure))

        self.was_down = is_down

        self.live_data.update({
            "x": x,
            "y": y,
            "pressure": pressure,
            "x_tilt": x_tilt,
            "y_tilt": y_tilt,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "hz": hz,
            "stroke": self.stroke_count,
            "samples": len(self.records)
        })

        self.update()
        event.accept()

    def draw_writing_strokes(self, painter: QPainter):
        """Draws each stroke separately so lifting the pen leaves a clean gap."""
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1, _ = stroke[i - 1]
                p2, pressure = stroke[i]
                width = max(1.5, pressure * 8.0)
                pen = QPen(QColor(0, 200, 255, 220), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.draw_writing_strokes(painter)
        self.render_telemetry_hud(painter)

    def render_telemetry_hud(self, painter: QPainter):
        panel_x = self.width() - 320
        panel_y = 20
        panel_w = 300
        panel_h = 350

        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(QBrush(QColor(20, 20, 20, 225)))
        painter.drawRoundedRect(QRectF(panel_x, panel_y, panel_w, panel_h), 8.0, 8.0)

        painter.setFont(QFont("Consolas", 10))
        start_text_y = panel_y + 30
        line_height = 23

        lines = [
            ("Status", "WRITING" if self.live_data['pressure'] > 0 else "HOVER / UP"),
            ("Stroke Index", f"#{self.live_data['stroke']}"),
            ("Position (X, Y)", f"{self.live_data['x']:.1f}, {self.live_data['y']:.1f}"),
            ("Pressure", f"{self.live_data['pressure']:.4f}"),
            ("Tilt (X, Y)", f"{self.live_data['x_tilt']}°, {self.live_data['y_tilt']}°"),
            ("Velocity", f"{self.live_data['velocity']:.1f} px/s"),
            ("Acceleration", f"{self.live_data['acceleration']:.1f} px/s²"),
            ("Jerk", f"{self.live_data['jerk']:.1f} px/s³"),
            ("Polling Rate", f"{self.live_data['hz']:.0f} Hz"),
            ("Total Samples", f"{self.live_data['samples']}"),
        ]

        for i, (label, val) in enumerate(lines):
            y_pos = start_text_y + (i * line_height)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(panel_x + 16, y_pos, label)
            painter.setPen(QColor(0, 230, 255) if label == "Pressure" else QColor(240, 240, 240))
            painter.drawText(panel_x + 160, y_pos, val)

        bar_x = panel_x + 16
        bar_y = start_text_y + (len(lines) * line_height) - 5
        bar_w = panel_w - 32
        bar_h = 10

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3, 3)

        painter.setBrush(QBrush(QColor(0, 210, 255)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * self.live_data["pressure"], bar_h), 3, 3)

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(130, 130, 130))
        painter.drawText(20, self.height() - 20, "[S] Save CSV & Image   |   [C] Clear Canvas")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.save_session()
        elif event.key() == Qt.Key.Key_C:
            self.clear_data()

    def save_session(self):
        if not self.records:
            print("Buffer empty. Nothing to save.")
            return

        base_id = int(time.time())
        csv_filename = f"kinematics_{base_id}.csv"
        img_filename = f"kinematics_{base_id}.png"

        # 1. Export CSV
        df = pd.DataFrame(self.records)
        df.to_csv(csv_filename, index=False)

        # 2. Render and save clean image (strokes only, transparent background)
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor(18, 18, 18))  # Dark background matching canvas
        img_painter = QPainter(pixmap)
        img_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.draw_writing_strokes(img_painter)
        img_painter.end()
        pixmap.save(img_filename, "PNG")

        print(f"Data saved:\n -> {csv_filename} ({len(df)} points)\n -> {img_filename}")

    def clear_data(self):
        self.records.clear()
        self.strokes.clear()
        self.current_stroke = []
        self.stroke_count = 0
        self.was_down = False
        self.start_time = None
        self.live_data = {k: 0.0 for k in self.live_data}
        self.update()
        print("Canvas and buffers cleared.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealtimeTabletKinematics()
    window.show()
    sys.exit(app.exec())