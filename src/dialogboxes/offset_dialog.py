"""Base dialog for setting a 6-DOF offset [x, y, z, rx, ry, rz].

Subclasses provide:
  - dialog title and frame-toggle labels
  - how to read/apply the current offset from/to the robot
  - a file path for save/load persistence
  - additional shortcut buttons (e.g. "From Sphere")
"""

import os
import math
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget, QLabel
)
from .tcpoffset import ToggleSwitch
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


# ── generic save / load helpers ──────────────────────────────────────────────

def save_offset(path: str, offset: list) -> bool:
    """Save an offset [x,y,z,rx,ry,rz] to an .npz file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, offset=np.array(offset, dtype=np.float64))
        return True
    except Exception as e:
        print(f"Warning: Could not save offset to {path}: {e}")
        return False


def load_offset(path: str) -> list:
    """Load an offset from an .npz file. Returns None if missing."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            return list(data['offset'])
    except Exception as e:
        print(f"Warning: Could not load offset from {path}: {e}")
        return None


def has_saved(path: str) -> bool:
    return os.path.exists(path)


# ── base dialog ──────────────────────────────────────────────────────────────

class OffsetDialog(QDialog):
    """Base dialog for editing a 6-DOF offset with manual inputs,
    a parent/child frame toggle, deg/rad toggle, and persistence.

    Subclasses must override the abstract hooks listed below.
    """

    # ── abstract hooks (override in subclasses) ──────────────────────────

    def _dialog_title(self) -> str:
        raise NotImplementedError

    def _frame_labels(self) -> tuple:
        """Return (parent_label, child_label) for the frame toggle."""
        raise NotImplementedError

    def _save_file_path(self) -> str:
        """Return the .npz path used for save / load."""
        raise NotImplementedError

    def _get_current_offset(self) -> list:
        """Read the current offset from the robot (or a default)."""
        raise NotImplementedError

    def _apply_offset(self, offset: list) -> None:
        """Apply *offset* to the robot, update displays, and save."""
        raise NotImplementedError

    def _create_shortcut_buttons(self, button_row: QHBoxLayout) -> list:
        """Add dialog-specific shortcut buttons to *button_row*.
        Return the button widgets so the base can size them.
        """
        return []

    # ── constructor ──────────────────────────────────────────────────────

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self._dialog_title())
        self.setModal(True)
        self.setMinimumWidth(600)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- shortcut buttons + From Previous + Apply Manual ---
        button_row = QHBoxLayout()
        button_row.setSpacing(4)

        extra_buttons = self._create_shortcut_buttons(button_row)

        self.from_previous_btn = QPushButton("From\nPrevious")
        self.from_previous_btn.clicked.connect(self._set_from_previous)
        self.from_previous_btn.setEnabled(has_saved(self._save_file_path()))
        button_row.addWidget(self.from_previous_btn)

        self.apply_manual_btn = QPushButton("Apply\nManual")
        self.apply_manual_btn.setCheckable(True)
        self.apply_manual_btn.toggled.connect(self._toggle_manual_inputs)
        button_row.addWidget(self.apply_manual_btn)

        layout.addLayout(button_row)

        # --- options row: frame toggle + unit toggle ---
        parent_label, child_label = self._frame_labels()
        options_row = QHBoxLayout()
        options_row.setSpacing(16)

        self.frame_toggle = ToggleSwitch(parent_label, child_label)
        options_row.addWidget(self.frame_toggle)
        options_row.addStretch()

        self.unit_toggle = ToggleSwitch("deg", "rad")
        options_row.addWidget(self.unit_toggle)

        self.options_widget = QWidget()
        self.options_widget.setLayout(options_row)
        layout.addWidget(self.options_widget)
        self.options_widget.setVisible(False)

        self.frame_toggle.toggled.connect(self._on_frame_toggled)
        self.unit_toggle.toggled.connect(self._on_unit_toggled)

        # --- 6-DOF input fields ---
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        trans_col = QVBoxLayout()
        trans_col.setSpacing(4)
        trans_col.addWidget(QLabel("Translations (m):"))
        self.tx_input = QLineEdit("0.0")
        self.ty_input = QLineEdit("0.0")
        self.tz_input = QLineEdit("0.0")
        for label, inp in [("TX:", self.tx_input), ("TY:", self.ty_input), ("TZ:", self.tz_input)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(inp)
            trans_col.addLayout(row)

        rot_col = QVBoxLayout()
        rot_col.setSpacing(4)
        self.rot_unit_label = QLabel("Rotations (deg):")
        rot_col.addWidget(self.rot_unit_label)
        self.rx_input = QLineEdit("0.0")
        self.ry_input = QLineEdit("0.0")
        self.rz_input = QLineEdit("0.0")
        for label, inp in [("RX:", self.rx_input), ("RY:", self.ry_input), ("RZ:", self.rz_input)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(inp)
            rot_col.addLayout(row)

        input_row.addLayout(trans_col)
        input_row.addLayout(rot_col)

        self.input_row_widget = QWidget()
        self.input_row_widget.setLayout(input_row)
        layout.addWidget(self.input_row_widget)
        self.input_row_widget.setVisible(False)

        # --- size the buttons ---
        estimated_line_edit_height = 40
        button_size = (estimated_line_edit_height * 6 // 3) * 2
        for btn in extra_buttons + [self.from_previous_btn, self.apply_manual_btn]:
            btn.setFixedSize(button_size, button_size)

        # --- Set button ---
        self.set_btn = QPushButton("Set")
        self.set_btn.clicked.connect(self._apply_and_close)
        layout.addWidget(self.set_btn)
        self.set_btn.setVisible(False)

        self.setLayout(layout)

        # --- persisted child-frame delta state ---
        state_key = self._persistence_key()
        state = getattr(parent, state_key, None)
        if state is not None:
            self._base_offset = state['base']
            self._last_delta = state['delta']
        else:
            self._base_offset = None
            self._last_delta = None

    # ── persistence key (unique per subclass) ────────────────────────────

    def _persistence_key(self) -> str:
        return f'_offset_delta_state_{os.path.basename(self._save_file_path())}'

    # ── toggle / display helpers ─────────────────────────────────────────

    def _using_degrees(self) -> bool:
        return not self.unit_toggle.isChecked()

    def _using_child_frame(self) -> bool:
        return self.frame_toggle.isChecked()

    def _toggle_manual_inputs(self, checked):
        self.options_widget.setVisible(checked)
        self.input_row_widget.setVisible(checked)
        self.set_btn.setVisible(checked)
        if checked:
            if self._base_offset is not None and self._last_delta is not None:
                self.frame_toggle.blockSignals(True)
                self.frame_toggle.setChecked(True)
                self.frame_toggle.blockSignals(False)
                self._display_delta(self._last_delta)
            else:
                self._update_inputs_with_current()
        self.adjustSize()

    def _on_unit_toggled(self, _checked):
        use_deg = self._using_degrees()
        self.rot_unit_label.setText(f"Rotations ({'deg' if use_deg else 'rad'}):")
        try:
            for inp in (self.rx_input, self.ry_input, self.rz_input):
                val = float(inp.text())
                inp.setText(f"{(math.degrees(val) if use_deg else math.radians(val)):.6f}")
        except ValueError:
            pass

    def _on_frame_toggled(self, _checked):
        if self._using_child_frame():
            if self._base_offset is not None and self._last_delta is not None:
                self._display_delta(self._last_delta)
            else:
                self._base_offset = self._get_current_offset()
                self._set_fields_to_zero()
                self._persist_state()
        else:
            if self._base_offset is not None:
                try:
                    delta = self._read_fields_raw()
                    self._last_delta = list(delta)
                    self._persist_state()
                    composed = self._compose_delta(self._base_offset, delta)
                    self._display_offset(composed)
                except ValueError:
                    self._update_inputs_with_current()
            else:
                self._update_inputs_with_current()

    # ── field read/write ─────────────────────────────────────────────────

    def _set_fields_to_zero(self):
        for inp in (self.tx_input, self.ty_input, self.tz_input,
                    self.rx_input, self.ry_input, self.rz_input):
            inp.setText("0.000000")

    def _display_offset(self, offset: list):
        """Write a parent-frame offset into the 6 input fields."""
        self.tx_input.setText(f"{offset[0]:.6f}")
        self.ty_input.setText(f"{offset[1]:.6f}")
        self.tz_input.setText(f"{offset[2]:.6f}")
        rx, ry, rz = offset[3], offset[4], offset[5]
        if self._using_degrees():
            rx, ry, rz = math.degrees(rx), math.degrees(ry), math.degrees(rz)
        self.rx_input.setText(f"{rx:.6f}")
        self.ry_input.setText(f"{ry:.6f}")
        self.rz_input.setText(f"{rz:.6f}")

    def _display_delta(self, delta: list):
        """Write a child-frame delta into the 6 input fields."""
        self._display_offset(delta)

    def _read_fields_raw(self) -> list:
        """Read the 6 values from input fields (rotations returned in rad)."""
        tx = float(self.tx_input.text())
        ty = float(self.ty_input.text())
        tz = float(self.tz_input.text())
        rx = float(self.rx_input.text())
        ry = float(self.ry_input.text())
        rz = float(self.rz_input.text())
        if self._using_degrees():
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
        return [tx, ty, tz, rx, ry, rz]

    def _parse_inputs_as_parent_offset(self) -> list:
        """Parse fields, composing with base if in child-frame mode."""
        entered = self._read_fields_raw()
        if self._using_child_frame() and self._base_offset is not None:
            return self._compose_delta(self._base_offset, entered)
        return entered

    def _update_inputs_with_current(self):
        try:
            if self._using_child_frame():
                self._base_offset = self._get_current_offset()
                self._set_fields_to_zero()
            else:
                self._display_offset(self._get_current_offset())
        except Exception as e:
            print(f"Warning: Could not update input fields: {e}")

    # ── delta composition ────────────────────────────────────────────────

    @staticmethod
    def _compose_delta(base: list, delta: list) -> list:
        """Compose base offset with a delta expressed in the child frame.

            T_final = T_base @ T_delta
        """
        R_base = axis_angle_to_rotation_matrix(base[3], base[4], base[5])
        p_base = np.array(base[:3])
        dp = np.array(delta[:3])
        dR = axis_angle_to_rotation_matrix(delta[3], delta[4], delta[5])

        p_final = p_base + R_base @ dp
        R_final = R_base @ dR

        rx, ry, rz = rotation_matrix_to_axis_angle(R_final)
        return [float(p_final[0]), float(p_final[1]), float(p_final[2]),
                float(rx), float(ry), float(rz)]

    # ── state persistence across dialog re-opens ─────────────────────────

    def _persist_state(self):
        key = self._persistence_key()
        if self._base_offset is not None:
            try:
                delta = self._read_fields_raw()
            except ValueError:
                delta = [0.0] * 6
            setattr(self.parent, key, {
                'base': list(self._base_offset),
                'delta': list(delta),
            })
            self._last_delta = list(delta)
        else:
            setattr(self.parent, key, None)
            self._last_delta = None

    # ── button actions (common) ──────────────────────────────────────────

    def _set_from_previous(self):
        saved = load_offset(self._save_file_path())
        if saved is None:
            print("Error: No saved offset found")
            return
        print(f"Loading saved offset: [{', '.join(f'{v:.4f}' for v in saved)}]")
        self._apply_offset(saved)
        self.accept()

    def _apply_and_close(self):
        """Apply manual offset (does not close the dialog)."""
        try:
            offset = self._parse_inputs_as_parent_offset()
            self._apply_offset(offset)
            self._persist_state()
        except ValueError:
            print("Error: Invalid input values.")
        except Exception as e:
            print(f"Error applying offset: {e}")

    def reject(self):
        self._persist_state()
        super().reject()
