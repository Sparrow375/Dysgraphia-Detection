import sys
import os
import time
import math
import json
import csv
from pathlib import Path
import pandas as pd

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QTabletEvent,
    QPixmap, QTransform, QIcon, QAction
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QFrame, QButtonGroup,
    QMessageBox, QToolTip, QSizePolicy
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
INDEX_CSV = DATA_DIR / "index.csv"
CONFIG_FILE = PROJECT_ROOT / "collector" / "config.json"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "grades": [3, 4, 5, 6, 7],
        "sections": ["A", "B", "C", "D"],
        "tasks": []
    }


def init_index_csv():
    if not INDEX_CSV.exists():
        header = [
            "timestamp_iso",
            "epoch_ms",
            "student_id",
            "grade",
            "section",
            "roll_number",
            "dominant_hand",
            "device_id",
            "operator",
            "tasks_completed",
            "total_samples",
            "total_strokes",
            "total_duration_sec",
            "json_filename",
            "notes"
        ]
        with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


init_index_csv()


class TabletCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Ruling & Rotation state
        self.ruling_mode = "single"  # "single", "four_line", "blank"
        self.rotation_angle = 0      # 0, 90, 180, 270

        # Recording buffers
        self.records = []
        self.strokes = []            # List of strokes, each is list of (QPointF in paper coords, pressure)
        self.current_stroke = []
        self.stroke_count = 0
        self.was_down = False
        self.start_time = None
        self.show_hud = True

        # Active prompt / stimulus text
        self.active_task_id = "S1"
        self.prompt_label = ""
        self.prompt_text = ""
        self.operator_script = ""

        # Live telemetry metrics
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

    def get_transform(self):
        w = float(self.width())
        h = float(self.height())
        transform = QTransform()

        if self.rotation_angle == 0:
            pass
        elif self.rotation_angle == 90:
            transform.translate(w, 0.0)
            transform.rotate(90.0)
        elif self.rotation_angle == 180:
            transform.translate(w, h)
            transform.rotate(180.0)
        elif self.rotation_angle == 270:
            transform.translate(0.0, h)
            transform.rotate(270.0)

        return transform

    def screen_to_paper(self, pt: QPointF) -> QPointF:
        transform = self.get_transform()
        inv_transform, ok = transform.inverted()
        if ok:
            return inv_transform.map(pt)
        return pt

    def get_paper_size(self):
        if self.rotation_angle in (90, 270):
            return float(self.height()), float(self.width())
        return float(self.width()), float(self.height())

    def tabletEvent(self, event: QTabletEvent):
        current_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = current_time

        rel_time = current_time - self.start_time
        raw_pos = event.position()
        paper_pos = self.screen_to_paper(raw_pos)
        x, y = paper_pos.x(), paper_pos.y()
        pressure = event.pressure()
        x_tilt = event.xTilt()
        y_tilt = event.yTilt()
        is_down = pressure > 0.0

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
            "timestamp": time.time(),
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
            "hz": hz
        }
        self.records.append(row)

        if is_down:
            self.current_stroke.append((paper_pos, pressure))

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

    def draw_ruling_lines(self, painter: QPainter):
        pw, ph = self.get_paper_size()

        # Canvas background
        painter.fillRect(QRectF(0, 0, pw, ph), QColor(19, 27, 46))

        if self.ruling_mode == "blank":
            return

        margin_left = 50.0
        margin_right = pw - 30.0

        if self.ruling_mode == "single":
            line_spacing = 42.0
            start_y = 60.0

            pen = QPen(QColor(56, 189, 248, 60), 1.0)
            painter.setPen(pen)

            y = start_y
            while y < ph - 30.0:
                painter.drawLine(QPointF(margin_left, y), QPointF(margin_right, y))
                y += line_spacing

            # Left vertical red margin line
            painter.setPen(QPen(QColor(244, 63, 94, 80), 1.2))
            painter.drawLine(QPointF(margin_left, 0), QPointF(margin_left, ph))

        elif self.ruling_mode == "four_line":
            group_h = 66.0
            sub_gap = group_h / 3.0
            gap = 35.0
            y = 60.0

            while y + group_h < ph - 20.0:
                # 1. Top Ascender (Red)
                painter.setPen(QPen(QColor(244, 63, 94, 110), 1.2))
                painter.drawLine(QPointF(margin_left, y), QPointF(margin_right, y))

                # 2. Midline (Blue dashed)
                pen_mid = QPen(QColor(56, 189, 248, 90), 1.0, Qt.PenStyle.DashLine)
                painter.setPen(pen_mid)
                painter.drawLine(QPointF(margin_left, y + sub_gap), QPointF(margin_right, y + sub_gap))

                # 3. Baseline (Blue solid)
                painter.setPen(QPen(QColor(56, 189, 248, 140), 1.2))
                painter.drawLine(QPointF(margin_left, y + sub_gap * 2), QPointF(margin_right, y + sub_gap * 2))

                # 4. Descender (Red)
                painter.setPen(QPen(QColor(244, 63, 94, 110), 1.2))
                painter.drawLine(QPointF(margin_left, y + group_h), QPointF(margin_right, y + group_h))

                y += group_h + gap

    def draw_writing_strokes(self, painter: QPainter):
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                p1, _ = stroke[i - 1]
                p2, pressure = stroke[i]
                width = max(1.8, pressure * 8.0)
                pen = QPen(QColor(56, 189, 248, 230), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(p1, p2)

    def draw_graphomotor_templates(self, painter: QPainter):
        """If on task S1, render light gray starter patterns so kids know what to draw."""
        if self.active_task_id != "S1":
            return

        pen = QPen(QColor(245, 158, 11, 100), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 14))

        # Starter loops at (70, 75)
        painter.drawText(QPointF(70, 95), "ℓ ℓ ℓ  (Draw continuous loops)")

        # Starter zigzag at (70, 160)
        painter.drawText(QPointF(70, 180), "⋀ ⋀ ⋀  (Draw sharp zigzags)")

        # Starter spiral dot at (70, 245)
        painter.drawText(QPointF(70, 265), "◎  (Start at center dot, draw spiral outwards)")

    def render_telemetry_hud(self, painter: QPainter):
        if not self.show_hud:
            return

        panel_w = 280
        panel_h = 320
        panel_x = self.width() - panel_w - 20
        panel_y = 20

        painter.setPen(QPen(QColor(51, 65, 85, 200), 1))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(QRectF(panel_x, panel_y, panel_w, panel_h), 10.0, 10.0)

        painter.setFont(QFont("Consolas", 10))
        start_text_y = panel_y + 28
        line_height = 23

        status_text = "WRITING" if self.live_data['pressure'] > 0 else "HOVER / UP"
        status_color = QColor(56, 189, 248) if status_text == "WRITING" else QColor(148, 163, 184)

        lines = [
            ("Status", status_text),
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
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(panel_x + 16, y_pos, label)
            painter.setPen(status_color if label == "Status" else (QColor(56, 189, 248) if label == "Pressure" else QColor(241, 245, 249)))
            painter.drawText(panel_x + 145, y_pos, val)

        # Pressure bar
        bar_x = panel_x + 16
        bar_y = start_text_y + (len(lines) * line_height) - 4
        bar_w = panel_w - 32
        bar_h = 8

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(51, 65, 85)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3, 3)

        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w * self.live_data["pressure"], bar_h), 3, 3)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Transform coordinate space for paper rotation
        transform = self.get_transform()
        painter.setTransform(transform)

        # 2. Draw ruled lines on paper
        self.draw_ruling_lines(painter)

        # 3. Draw starter pattern templates if S1
        self.draw_graphomotor_templates(painter)

        # 4. Draw user's writing strokes
        self.draw_writing_strokes(painter)

        # 5. Reset transform to Screen Coordinates for HUD
        painter.resetTransform()
        self.render_telemetry_hud(painter)

    def clear_canvas(self):
        self.records.clear()
        self.strokes.clear()
        self.current_stroke = []
        self.stroke_count = 0
        self.was_down = False
        self.start_time = None
        self.live_data = {k: 0.0 for k in self.live_data}
        self.update()


class StylusDataCollectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal Stylus Data Collector — Field Station")
        self.resize(1280, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI', Inter; }
            QComboBox, QSpinBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                padding: 4px 8px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #f8fafc;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 14px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #334155; }
            QPushButton:checked {
                background-color: #38bdf8;
                color: #0f172a;
                border-color: #38bdf8;
            }
        """)

        self.config = load_config()
        self.tasks_dict = {t["id"]: t for t in self.config.get("tasks", [])}

        self.current_grade = 5
        self.current_section = "A"
        self.current_roll = 1
        self.dominant_hand = "R"
        self.current_task_id = "S1"

        self.init_ui()
        self.update_prompts()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. TOP OPERATOR ROSTER BAR
        top_bar = QFrame(self)
        top_bar.setStyleSheet("background-color: #1e293b; border-bottom: 1px solid #334155;")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(16, 8, 16, 8)
        top_bar_layout.setSpacing(12)

        # Student Badge
        self.badge_label = QLabel("G5-A-01 (R)", self)
        self.badge_label.setStyleSheet("""
            background-color: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.5);
            color: #38bdf8;
            font-weight: bold;
            font-size: 14px;
            padding: 4px 10px;
            border-radius: 6px;
        """)
        top_bar_layout.addWidget(self.badge_label)

        # Grade Selector
        top_bar_layout.addWidget(QLabel("Grade:", self))
        self.grade_combo = QComboBox(self)
        for g in [3, 4, 5, 6, 7]:
            self.grade_combo.addItem(f"Grade {g}", g)
        self.grade_combo.setCurrentIndex(2)  # Default Grade 5
        self.grade_combo.currentIndexChanged.connect(self.on_grade_changed)
        top_bar_layout.addWidget(self.grade_combo)

        # Section Selector
        top_bar_layout.addWidget(QLabel("Sec:", self))
        self.sec_combo = QComboBox(self)
        for s in ["A", "B", "C", "D"]:
            self.sec_combo.addItem(s, s)
        self.sec_combo.currentIndexChanged.connect(self.on_sec_changed)
        top_bar_layout.addWidget(self.sec_combo)

        # Roll Number Spinbox
        top_bar_layout.addWidget(QLabel("Roll #:", self))
        self.roll_spin = QSpinBox(self)
        self.roll_spin.setRange(1, 99)
        self.roll_spin.setValue(1)
        self.roll_spin.valueChanged.connect(self.on_roll_changed)
        top_bar_layout.addWidget(self.roll_spin)

        # Handedness Toggle
        self.hand_group = QButtonGroup(self)
        self.btn_hand_r = QPushButton("R-Hand", self)
        self.btn_hand_r.setCheckable(True)
        self.btn_hand_r.setChecked(True)
        self.btn_hand_l = QPushButton("L-Hand", self)
        self.btn_hand_l.setCheckable(True)
        self.hand_group.addButton(self.btn_hand_r)
        self.hand_group.addButton(self.btn_hand_l)
        self.hand_group.buttonClicked.connect(self.on_hand_changed)
        top_bar_layout.addWidget(self.btn_hand_r)
        top_bar_layout.addWidget(self.btn_hand_l)

        top_bar_layout.addSpacing(10)

        # Task Selector Buttons
        self.task_btn_group = QButtonGroup(self)
        self.btn_s1 = QPushButton("S1: Patterns", self)
        self.btn_s1.setCheckable(True)
        self.btn_s1.setChecked(True)
        self.btn_s2 = QPushButton("S2: Hindi", self)
        self.btn_s2.setCheckable(True)
        self.btn_s3 = QPushButton("S3: English", self)
        self.btn_s3.setCheckable(True)

        self.task_btn_group.addButton(self.btn_s1)
        self.task_btn_group.addButton(self.btn_s2)
        self.task_btn_group.addButton(self.btn_s3)
        self.task_btn_group.buttonClicked.connect(self.on_task_button_clicked)

        top_bar_layout.addWidget(self.btn_s1)
        top_bar_layout.addWidget(self.btn_s2)
        top_bar_layout.addWidget(self.btn_s3)

        top_bar_layout.addStretch()

        # Ruling Mode Switcher
        self.btn_ruling = QPushButton("📐 Single Ruled", self)
        self.btn_ruling.clicked.connect(self.toggle_ruling)
        top_bar_layout.addWidget(self.btn_ruling)

        # Rotation Switcher
        self.btn_rotate = QPushButton("🔄 Rotate 0°", self)
        self.btn_rotate.clicked.connect(self.toggle_rotation)
        top_bar_layout.addWidget(self.btn_rotate)

        # Save & Next Student Button (High Priority)
        self.btn_next = QPushButton("Save & Next ⏩", self)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: bold;
                border: none;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #059669; }
        """)
        self.btn_next.clicked.connect(self.save_and_next_student)
        top_bar_layout.addWidget(self.btn_next)

        main_layout.addWidget(top_bar)

        # 2. STIMULUS PROMPT BANNER
        self.prompt_banner = QFrame(self)
        self.prompt_banner.setStyleSheet("background-color: #1e293b; border-bottom: 2px solid #334155; padding: 6px 16px;")
        banner_layout = QHBoxLayout(self.prompt_banner)
        banner_layout.setContentsMargins(16, 4, 16, 4)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.prompt_title_label = QLabel("S1: Graphomotor Patterns", self)
        self.prompt_title_label.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold; text-transform: uppercase;")
        self.prompt_text_label = QLabel("ℓℓℓ Continuous Loops   |   ⋀⋀⋀ Zigzag   |   ◎ Outward Spiral", self)
        self.prompt_text_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
        self.operator_script_label = QLabel("लाइन के अंत तक पैटर्न बनाओ। फिर बिंदु से शुरू करके गोल-गोल घुमाते हुए बाहर की ओर बनाओ।", self)
        self.operator_script_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-style: italic;")

        text_layout.addWidget(self.prompt_title_label)
        text_layout.addWidget(self.prompt_text_label)
        text_layout.addWidget(self.operator_script_label)
        banner_layout.addLayout(text_layout)

        banner_layout.addStretch()

        # Child Done Button
        self.btn_done = QPushButton("Done ✓", self)
        self.btn_done.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-weight: bold;
                font-size: 15px;
                padding: 8px 20px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #10b981; }
        """)
        self.btn_done.clicked.connect(self.on_task_done)
        banner_layout.addWidget(self.btn_done)

        main_layout.addWidget(self.prompt_banner)

        # 3. DRAWING CANVAS
        self.canvas = TabletCanvas(self)
        main_layout.addWidget(self.canvas, 1)

        # 4. BOTTOM STATUS BAR
        status_bar = QFrame(self)
        status_bar.setStyleSheet("background-color: #0f172a; border-top: 1px solid #334155; padding: 4px 16px;")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(16, 2, 16, 2)

        self.status_msg = QLabel("Ready for Stylus input. [H] Toggle HUD   [C] Clear Canvas   [S] Save Task", self)
        self.status_msg.setStyleSheet("color: #94a3b8; font-size: 12px;")
        status_layout.addWidget(self.status_msg)

        status_layout.addStretch()

        btn_clear = QPushButton("Clear Canvas", self)
        btn_clear.clicked.connect(self.canvas.clear_canvas)
        status_layout.addWidget(btn_clear)

        btn_hud = QPushButton("HUD", self)
        btn_hud.clicked.connect(self.toggle_hud)
        status_layout.addWidget(btn_hud)

        main_layout.addWidget(status_bar)

    def on_grade_changed(self, idx):
        self.current_grade = self.grade_combo.currentData()
        self.update_badge()
        self.update_prompts()

    def on_sec_changed(self, idx):
        self.current_section = self.sec_combo.currentData()
        self.update_badge()

    def on_roll_changed(self, val):
        self.current_roll = val
        self.update_badge()

    def on_hand_changed(self, btn):
        self.dominant_hand = "R" if btn == self.btn_hand_r else "L"
        self.update_badge()

    def update_badge(self):
        s_id = f"G{self.current_grade}_{self.current_section}_Roll{self.current_roll:02d}"
        self.badge_label.setText(f"{s_id} ({self.dominant_hand})")

    def on_task_button_clicked(self, btn):
        if btn == self.btn_s1:
            self.current_task_id = "S1"
        elif btn == self.btn_s2:
            self.current_task_id = "S2"
        elif btn == self.btn_s3:
            self.current_task_id = "S3"

        self.canvas.active_task_id = self.current_task_id
        self.update_prompts()
        self.canvas.clear_canvas()

    def update_prompts(self):
        task = self.tasks_dict.get(self.current_task_id, {})
        grade_str = str(self.current_grade)

        if self.current_task_id == "S1":
            self.prompt_title_label.setText("S1: Graphomotor Patterns (लूप्स, ज़िग-ज़ैग, स्पाइरल)")
            self.prompt_text_label.setText("ℓℓℓ Continuous Loops   |   ⋀⋀⋀ Zigzag   |   ◎ Outward Spiral")
            self.operator_script_label.setText(task.get("instructions_hi", ""))
        elif self.current_task_id == "S2":
            stimuli = task.get("stimuli_by_grade", {})
            sentence = stimuli.get(grade_str, "हमारे स्कूल में एक सुंदर बगीचा है।")
            self.prompt_title_label.setText(f"S2: Hindi Copy — Grade {self.current_grade} (Comfortable Pace)")
            self.prompt_text_label.setText(sentence)
            self.operator_script_label.setText(task.get("instructions_hi", ""))
        elif self.current_task_id == "S3":
            stimuli = task.get("stimuli_by_grade", {})
            sentence = stimuli.get(grade_str, "The quick brown fox jumps over the lazy dog.")
            self.prompt_title_label.setText(f"S3: English Copy — Grade {self.current_grade} (Comfortable Pace)")
            self.prompt_text_label.setText(sentence)
            self.operator_script_label.setText(task.get("instructions_en", ""))

    def toggle_ruling(self):
        modes = ["single", "four_line", "blank"]
        labels = {
            "single": "📐 Single Ruled",
            "four_line": "📐 4-Line Cursive",
            "blank": "📐 Blank Paper"
        }
        idx = modes.index(self.canvas.ruling_mode)
        next_mode = modes[(idx + 1) % len(modes)]
        self.canvas.ruling_mode = next_mode
        self.btn_ruling.setText(labels[next_mode])
        self.canvas.update()

    def toggle_rotation(self):
        angles = [0, 90, 180, 270]
        idx = angles.index(self.canvas.rotation_angle)
        next_angle = angles[(idx + 1) % len(angles)]
        self.canvas.rotation_angle = next_angle
        self.btn_rotate.setText(f"🔄 Rotate {next_angle}°")
        self.canvas.update()

    def toggle_hud(self):
        self.canvas.show_hud = not self.canvas.show_hud
        self.canvas.update()

    def on_task_done(self):
        self.status_msg.setText(f"Task {self.current_task_id} completed. Click 'Save & Next ⏩' or choose next task.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.save_current_task()
        elif event.key() == Qt.Key.Key_C:
            self.canvas.clear_canvas()
        elif event.key() == Qt.Key.Key_H:
            self.toggle_hud()
        else:
            super().keyPressEvent(event)

    def save_current_task(self):
        if not self.canvas.records:
            self.status_msg.setText("Canvas is empty. Nothing to save.")
            return

        epoch_ms = int(time.time() * 1000)
        timestamp_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        student_id = f"G{self.current_grade}_{self.current_section}_Roll{self.current_roll:02d}"
        prefix = f"{student_id}__{epoch_ms}_{self.current_task_id}"

        # 1. Save CSV
        df = pd.DataFrame(self.canvas.records)
        csv_file = SESSIONS_DIR / f"{prefix}_kinematics.csv"
        df.to_csv(csv_file, index=False)

        # 2. Save Clean PNG Render
        pw, ph = self.canvas.get_paper_size()
        pixmap = QPixmap(int(pw), int(ph))
        pixmap.fill(QColor(19, 27, 46))
        p_painter = QPainter(pixmap)
        p_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.canvas.draw_ruling_lines(p_painter)
        self.canvas.draw_writing_strokes(p_painter)
        p_painter.end()

        png_file = SESSIONS_DIR / f"{prefix}_render.png"
        pixmap.save(str(png_file), "PNG")

        self.status_msg.setText(f"Saved: {csv_file.name} ({len(df)} samples)")
        return str(csv_file.name), str(png_file.name), len(df)

    def save_and_next_student(self):
        if not self.canvas.records:
            QMessageBox.information(self, "No Ink Recorded", "Please have the student write or draw before saving.")
            return

        epoch_ms = int(time.time() * 1000)
        timestamp_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        student_id = f"G{self.current_grade}_{self.current_section}_Roll{self.current_roll:02d}"
        duration_sec = (time.perf_counter() - self.canvas.start_time) if self.canvas.start_time else 0.0

        csv_name, png_name, sample_count = self.save_current_task()

        # Save session JSON
        json_file = SESSIONS_DIR / f"{student_id}__{epoch_ms}_session.json"
        session_data = {
            "student_id": student_id,
            "grade": self.current_grade,
            "section": self.current_section,
            "roll_number": self.current_roll,
            "dominant_hand": self.dominant_hand,
            "device_id": "Desktop-XP-Pen",
            "timestamp_iso": timestamp_iso,
            "epoch_ms": epoch_ms,
            "tasks": [{
                "task_id": self.current_task_id,
                "ruling_mode": self.canvas.ruling_mode,
                "rotation_angle": self.canvas.rotation_angle,
                "csv_file": csv_name,
                "png_file": png_name,
                "samples_count": sample_count,
                "strokes_count": len(self.canvas.strokes)
            }]
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        # Append to Master index.csv
        row = [
            timestamp_iso,
            epoch_ms,
            student_id,
            self.current_grade,
            self.current_section,
            self.current_roll,
            self.dominant_hand,
            "Desktop-XP-Pen",
            "SchoolVisitOp",
            self.current_task_id,
            sample_count,
            len(self.canvas.strokes),
            round(duration_sec, 1),
            json_file.name,
            f"Ruling: {self.canvas.ruling_mode}, Rot: {self.canvas.rotation_angle}°"
        ]

        with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Advance to Next Roll Number
        next_roll = self.current_roll + 1
        self.roll_spin.setValue(next_roll)

        # Reset Canvas & Task
        self.canvas.clear_canvas()
        self.btn_s1.setChecked(True)
        self.current_task_id = "S1"
        self.canvas.active_task_id = "S1"
        self.update_prompts()

        self.status_msg.setText(f"✅ Successfully saved {student_id}! Advanced to Roll #{next_roll}.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StylusDataCollectorApp()
    window.show()
    sys.exit(app.exec())
