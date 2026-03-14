import os
import math
import time
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget, QLabel
)
from PyQt6.QtCore import Qt, QSize, QRectF, pyqtSignal, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QColor
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


class ToggleSwitch(QWidget):
    """iOS-style labeled toggle switch with two text labels."""

    toggled = pyqtSignal(bool)

    def __init__(self, left_label: str, right_label: str, parent=None):
        super().__init__(parent)
        self._left = left_label
        self._right = right_label
        self._checked = False
        self._knob_x = 0.0
        self._signals_blocked = False

        self._anim = QPropertyAnimation(self, b"knobX", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        font = self.font()
        font.setPointSize(font.pointSize() - 1)
        self._label_font = font

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(self._preferredSize())

    def _preferredSize(self) -> QSize:
        fm = self.fontMetrics()
        text_w = max(fm.horizontalAdvance(self._left), fm.horizontalAdvance(self._right))
        w = text_w * 2 + 28
        h = fm.height() + 12
        return QSize(max(w, 80), max(h, 26))

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == checked:
            return
        self._checked = checked
        end = 1.0 if checked else 0.0
        self._anim.stop()
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(end)
        self._anim.start()
        if not self._signals_blocked:
            self.toggled.emit(checked)

    def blockSignals(self, block: bool):
        self._signals_blocked = block

    def _getKnobX(self) -> float:
        return self._knob_x

    def _setKnobX(self, val: float):
        self._knob_x = val
        self.update()

    knobX = pyqtProperty(float, _getKnobX, _setKnobX)

    def mousePressEvent(self, event):
        self.setChecked(not self._checked)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = h / 2

        track_color = QColor(70, 70, 70)
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        knob_w = w / 2
        knob_margin = 2
        knob_h = h - 2 * knob_margin
        knob_r = knob_h / 2
        knob_left = knob_margin + self._knob_x * (w - knob_w - 2 * knob_margin)

        knob_color = QColor(100, 100, 100)
        p.setBrush(knob_color)
        p.drawRoundedRect(QRectF(knob_left, knob_margin, knob_w, knob_h), knob_r, knob_r)

        p.setFont(self._label_font)
        left_rect = QRectF(0, 0, w / 2, h)
        right_rect = QRectF(w / 2, 0, w / 2, h)

        left_on_knob = self._knob_x < 0.5
        right_on_knob = not left_on_knob

        bright = QColor(220, 220, 220)
        dim = QColor(140, 140, 140)

        p.setPen(bright if left_on_knob else dim)
        p.drawText(left_rect, Qt.AlignmentFlag.AlignCenter, self._left)
        p.setPen(bright if right_on_knob else dim)
        p.drawText(right_rect, Qt.AlignmentFlag.AlignCenter, self._right)

        p.end()

# ── TCP offset file helpers (kept for external callers) ──────────────────

OFFSET_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'offset.npz')


def save_tcp_offset(offset: list) -> bool:
    """Save TCP offset to file."""
    try:
        os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
        np.savez_compressed(OFFSET_FILE, offset=np.array(offset, dtype=np.float64))
        return True
    except Exception as e:
        print(f"Warning: Could not save TCP offset: {e}")
        return False


def load_tcp_offset() -> list:
    """Load TCP offset from file. Returns None if missing."""
    if not os.path.exists(OFFSET_FILE):
        return None
    try:
        data = np.load(OFFSET_FILE)
        return list(data['offset'])
    except Exception as e:
        print(f"Warning: Could not load TCP offset: {e}")
        return None


def has_saved_offset() -> bool:
    """Check if a saved offset file exists."""
    return os.path.exists(OFFSET_FILE)


# ── TCP Offset Dialog (subclass of OffsetDialog) ─────────────────────────

from .offset_dialog import OffsetDialog


class TCPOffsetDialog(OffsetDialog):
    """Dialog for setting TCP offset with From Sphere, From Current Pose,
    From Previous, and Apply Manual options."""

    def _dialog_title(self):
        return "Set TCP Offset"

    def _frame_labels(self):
        return ("Flange", "TCP")

    def _save_file_path(self):
        return OFFSET_FILE

    def _get_current_offset(self):
        if self.parent.robot is not None:
            offset = self.parent.robot.tcpOffset
            if offset is not None:
                return offset
        from config import runtime_tcp_offset
        return runtime_tcp_offset.copy()

    def _apply_offset(self, offset):
        frame_label = "child" if self._using_child_frame() else "parent"
        print(f"Setting TCP offset ({frame_label} frame): "
              f"[{', '.join(f'{v:.4f}' for v in offset)}]")

        if self.parent.robot is not None and self.parent.connected:
            self.parent.robot.setTcpOffset(offset)
            time.sleep(0.1)
            save_tcp_offset(offset)
            self.parent.update_tcp_offset_display()
            self.parent.vtk_widget.GetRenderWindow().Render()
        else:
            save_tcp_offset(offset)
            print("  (Not connected - offset saved but not applied)")
            self.parent.update_tcp_offset_display()

    def _create_shortcut_buttons(self, button_row):
        self.from_sphere_btn = QPushButton("From\nSphere")
        self.from_sphere_btn.clicked.connect(self._set_from_sphere)
        self.from_sphere_btn.setEnabled(self.parent.fitted_sphere is not None)
        button_row.addWidget(self.from_sphere_btn)

        self.from_home_btn = QPushButton("From Current\nPose")
        self.from_home_btn.clicked.connect(self._set_from_current_to_home)
        button_row.addWidget(self.from_home_btn)

        return [self.from_sphere_btn, self.from_home_btn]

    # ── TCP-specific shortcut actions ────────────────────────────────────

    def _set_from_sphere(self):
        if self.parent.fitted_sphere is None:
            print("Error: No fitted sphere available.")
            return
        self.parent.set_tcp_offset_from_sphere()
        self.parent.update_tcp_offset_display()
        if self.parent.robot is not None and self.parent.robot.tcpOffset is not None:
            save_tcp_offset(self.parent.robot.tcpOffset)
        self.accept()

    def _set_from_current_to_home(self):
        if self.parent.home_position is None:
            print("Error: No home position set")
            return

        current_tcp_pose = None
        if self.parent.robot is not None:
            current_tcp_pose = self.parent.robot.tcpPose
        if current_tcp_pose is None:
            print("Error: No TCP pose available")
            return

        home_position = self.parent.home_position

        try:
            current_position = np.array(current_tcp_pose[:3])
            home_pos = np.array(home_position[:3])
            offset_base = current_position - home_pos

            current_orientation = current_tcp_pose[3:6]
            R_tcp_to_base = axis_angle_to_rotation_matrix(
                current_orientation[0], current_orientation[1], current_orientation[2]
            )
            offset_tcp = R_tcp_to_base.T @ offset_base

            home_orientation = home_position[3:6]
            R_home_to_base = axis_angle_to_rotation_matrix(
                home_orientation[0], home_orientation[1], home_orientation[2]
            )
            R_offset = R_home_to_base.T @ R_tcp_to_base
            rx_offset, ry_offset, rz_offset = rotation_matrix_to_axis_angle(R_offset)

            tcp_offset = [
                float(offset_tcp[0]), float(offset_tcp[1]), float(offset_tcp[2]),
                float(rx_offset), float(ry_offset), float(rz_offset)
            ]

            print(f"Setting TCP offset from current pose relative to home:")
            print(f"  Current TCP: [{', '.join(f'{v:.4f}' for v in current_tcp_pose)}]")
            print(f"  Home:        [{', '.join(f'{v:.4f}' for v in home_position)}]")
            print(f"  Offset:      [{', '.join(f'{v:.4f}' for v in tcp_offset)}]")

            self._apply_offset(tcp_offset)

        except Exception as e:
            print(f"Error setting TCP offset from current to home: {e}")
            import traceback
            traceback.print_exc()
