"""Dialog for setting reference frame offset relative to TCP."""

import os
import numpy as np
from PyQt6.QtWidgets import QPushButton, QHBoxLayout
from .offset_dialog import OffsetDialog, save_offset
from src.utils import axis_angle_to_rotation_matrix

# Path for storing ref frame offset
REF_FRAME_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'refframe.npz')


def save_ref_frame_offset(offset: list) -> bool:
    """Save ref frame offset to file."""
    return save_offset(REF_FRAME_FILE, offset)


class RefFrameOffsetDialog(OffsetDialog):
    """Dialog for setting the reference frame offset relative to TCP."""

    def _dialog_title(self):
        return "Set Ref Frame Offset"

    def _frame_labels(self):
        return ("TCP", "Ref frame")

    def _save_file_path(self):
        return REF_FRAME_FILE

    def _get_current_offset(self):
        if self.parent.robot is not None:
            offset = self.parent.robot.refFrameOffset
            if offset is not None:
                return offset
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _apply_offset(self, offset):
        print(f"Setting ref frame offset: [{', '.join(f'{v:.4f}' for v in offset)}]")

        if self.parent.robot is not None:
            self.parent.robot.setRefFrameOffset(offset)
            self.parent.update_ref_frame_display()
            self.parent.update_visualization()
        save_ref_frame_offset(offset)

    def _create_shortcut_buttons(self, button_row):
        self.from_sphere_btn = QPushButton("From\nSphere")
        self.from_sphere_btn.clicked.connect(self._set_from_sphere)
        self.from_sphere_btn.setEnabled(self.parent.fitted_sphere is not None)
        button_row.addWidget(self.from_sphere_btn)

        return [self.from_sphere_btn]

    # ── ref-frame-specific shortcut ──────────────────────────────────────

    def _set_from_sphere(self):
        """Set ref frame offset so its origin is at the fitted sphere center (relative to TCP)."""
        if self.parent.fitted_sphere is None:
            print("Error: No fitted sphere available. Please fit a sphere first.")
            return

        fitted_sphere_center = self.parent.fitted_sphere.getCenter()
        if fitted_sphere_center is None or len(fitted_sphere_center) < 3 or not np.all(np.isfinite(fitted_sphere_center[:3])):
            print(f"Error: Invalid sphere center: {fitted_sphere_center}")
            return

        if self.parent.robot is None or not self.parent.robot.isConnected():
            print("Error: Not connected to robot")
            return

        tcp_pose = self.parent.robot.tcpPose
        if tcp_pose is None or len(tcp_pose) < 6:
            print("Error: No valid TCP pose available")
            return

        tcp_pos = np.array(tcp_pose[:3])
        sphere_center = np.array(fitted_sphere_center[:3])

        # Vector from TCP to sphere center in base frame
        offset_base = sphere_center - tcp_pos

        # Transform to TCP frame
        R_tcp = axis_angle_to_rotation_matrix(tcp_pose[3], tcp_pose[4], tcp_pose[5])
        offset_tcp = R_tcp.T @ offset_base

        ref_offset = [float(offset_tcp[0]), float(offset_tcp[1]), float(offset_tcp[2]), 0.0, 0.0, 0.0]

        print(f"Ref frame offset set from sphere center (in TCP frame): "
              f"[{ref_offset[0]:.4f}, {ref_offset[1]:.4f}, {ref_offset[2]:.4f}, 0, 0, 0]")
        self._apply_offset(ref_offset)
        self.accept()
