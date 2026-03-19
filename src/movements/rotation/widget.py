"""Axial Rotation movement widget for embedding in the main window."""

import math
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QButtonGroup, QLineEdit, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt
from scipy.spatial.transform import Rotation

from src.ui import ArrowButton, CircleWidget, CollapsibleGroupBox
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from src.motion_logger import getLogfilePath
from src.parameter_tooltips import PARAMETER_TOOLTIPS as TT

from src.movements.async_motion_runner import AsyncMotionRunner
from src.movements.rotation import direct as rotation_direct
from src.movements.rotation import hybrid as rotation_hybrid
from src.movements.rotation import force as rotation_force
from config import defaults as CONFIG

DIRECTION_MAP = {"left": 1, "right": -1}

METHOD_MODULES = {
    "direct": rotation_direct,
    "hybrid": rotation_hybrid,
    "force": rotation_force,
}


class AxialRotationWidget(QWidget):
    """Widget containing axial rotation movement controls."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.path_visible = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        info_label = QLabel("Rotates TCP around its own z-axis while keeping position fixed.")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        layout.addWidget(info_label)

        # Direction selection
        direction_layout = QHBoxLayout()
        direction_layout.setSpacing(2)

        self.direction_group = QButtonGroup(self)
        self.direction_group.setExclusive(True)

        self.left_arrow = ArrowButton("left", label="LEFT")
        self.left_arrow.setFixedSize(80, 55)
        self.circle = CircleWidget(top_label="FRONT", bottom_label="BACK")
        self.circle.setFixedSize(50, 70)
        self.right_arrow = ArrowButton("right", label="RIGHT")
        self.right_arrow.setFixedSize(80, 55)

        self.direction_group.addButton(self.left_arrow)
        self.direction_group.addButton(self.right_arrow)
        self.direction_group.buttonToggled.connect(self._on_direction_changed)

        direction_layout.addStretch()
        direction_layout.addWidget(self.left_arrow)
        direction_layout.addWidget(self.circle)
        direction_layout.addWidget(self.right_arrow)
        direction_layout.addStretch()

        layout.addLayout(direction_layout)

        # Method selection dropdown
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("Direct (force mode + moveL)", "direct")
        self.method_combo.addItem("Hybrid (force mode + moveL)", "hybrid")
        self.method_combo.addItem("Force (force mode + speedL)", "force")
        self.method_combo.setCurrentIndex(2)  # Force as default
        self.method_combo.setToolTip(
            "Direct: Full Fx/Fy/Fz compliance with moveL trajectory\n"
            "Hybrid: Full Fx/Fy/Fz compliance with moveL trajectory\n"
            "Force: Full Fx/Fy/Fz compliance with real-time speedL control loop"
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # =====================================================
        # MOVEMENT PARAMETERS (shared by all methods)
        # =====================================================
        self.movement_group = CollapsibleGroupBox("Movement Parameters", expanded=False)

        self.angle_input = self._add_spin(
            self.movement_group, "Rotation Angle (deg):", 0.0, 90.0,
            CONFIG.rotation.angle, 1, 5.0, tooltip=TT["angle"]
        )
        self.speed_input = self._add_spin(
            self.movement_group, "Speed (m/s):", 0.001, 1.0,
            CONFIG.rotation.speed, 3, 0.01, tooltip=TT["speed"]
        )
        self.accel_input = self._add_spin(
            self.movement_group, "Acceleration (m/s²):", 0.001, 2.0,
            CONFIG.rotation.acceleration, 3, 0.01, tooltip=TT["acceleration"]
        )
        self.max_moment_input = self._add_spin(
            self.movement_group, "Max Moment (Nm):", 0.1, 50.0,
            CONFIG.rotation.max_moment, 2, 0.1, tooltip=TT["max_moment"]
        )

        layout.addWidget(self.movement_group)

        # =====================================================
        # DIRECT-ONLY: Force Mode Parameters
        # =====================================================
        self.direct_params_group = CollapsibleGroupBox("Force Mode Parameters", expanded=False)

        self.direct_z_limit_group = CollapsibleGroupBox("Z Limit Control", expanded=False)
        self.direct_z_limit_input = self._add_spin(
            self.direct_z_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation.z_limit.force_mode_z_limit, 3, 0.01, tooltip=TT["force_mode_z_limit"]
        )
        self.direct_z_damping_input = self._add_spin(
            self.direct_z_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation.z_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.direct_z_gain_input = self._add_spin(
            self.direct_z_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation.z_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.direct_params_group.addWidget(self.direct_z_limit_group)

        self.direct_xy_limit_group = CollapsibleGroupBox("XY Limit Control", expanded=False)
        self.direct_xy_limit_input = self._add_spin(
            self.direct_xy_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation.xy_limit.force_mode_xy_limit, 3, 0.01, tooltip=TT["force_mode_xy_limit"]
        )
        self.direct_xy_damping_input = self._add_spin(
            self.direct_xy_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation.xy_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.direct_xy_gain_input = self._add_spin(
            self.direct_xy_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation.xy_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.direct_params_group.addWidget(self.direct_xy_limit_group)

        layout.addWidget(self.direct_params_group)

        # =====================================================
        # HYBRID: Collection Method Parameters
        # =====================================================
        self.hybrid_params_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)

        self.hybrid_z_limit_group = CollapsibleGroupBox("Z Limit Control", expanded=False)
        self.hybrid_z_limit_input = self._add_spin(
            self.hybrid_z_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation_hybrid.z_limit.force_mode_z_limit, 3, 0.01, tooltip=TT["force_mode_z_limit"]
        )
        self.hybrid_z_damping_input = self._add_spin(
            self.hybrid_z_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation_hybrid.z_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.hybrid_z_gain_input = self._add_spin(
            self.hybrid_z_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation_hybrid.z_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.hybrid_params_group.addWidget(self.hybrid_z_limit_group)

        self.hybrid_xy_limit_group = CollapsibleGroupBox("XY Limit Control", expanded=False)
        self.hybrid_xy_limit_input = self._add_spin(
            self.hybrid_xy_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation_hybrid.xy_limit.force_mode_xy_limit, 3, 0.01, tooltip=TT["force_mode_xy_limit"]
        )
        self.hybrid_xy_damping_input = self._add_spin(
            self.hybrid_xy_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation_hybrid.xy_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.hybrid_xy_gain_input = self._add_spin(
            self.hybrid_xy_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation_hybrid.xy_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.hybrid_params_group.addWidget(self.hybrid_xy_limit_group)

        layout.addWidget(self.hybrid_params_group)

        # =====================================================
        # FORCE: Collection Method Parameters
        # =====================================================
        self.force_params_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)

        self.force_control_dt_input = self._add_spin(
            self.force_params_group, "Control Loop dt (s):", 0.001, 0.1,
            CONFIG.rotation_force.control_loop_dt, 3, 0.001, tooltip=TT["control_loop_dt"]
        )
        self.force_rot_speed_factor_input = self._add_spin(
            self.force_params_group, "Rotation Speed Factor:", 0.1, 50.0,
            CONFIG.rotation_force.rotation_speed_factor, 1, 0.5, tooltip=TT["rotation_speed_factor"]
        )

        self.force_z_limit_group = CollapsibleGroupBox("Z Limit Control", expanded=False)
        self.force_z_limit_input = self._add_spin(
            self.force_z_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation_force.z_limit.force_mode_z_limit, 3, 0.01, tooltip=TT["force_mode_z_limit"]
        )
        self.force_z_damping_input = self._add_spin(
            self.force_z_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation_force.z_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.force_z_gain_input = self._add_spin(
            self.force_z_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation_force.z_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.force_params_group.addWidget(self.force_z_limit_group)

        self.force_xy_limit_group = CollapsibleGroupBox("XY Limit Control", expanded=False)
        self.force_xy_limit_input = self._add_spin(
            self.force_xy_limit_group, "Limit (m/s):", 0.001, 0.5,
            CONFIG.rotation_force.xy_limit.force_mode_xy_limit, 3, 0.01, tooltip=TT["force_mode_xy_limit"]
        )
        self.force_xy_damping_input = self._add_spin(
            self.force_xy_limit_group, "Damping (0-1):", 0.0, 1.0,
            CONFIG.rotation_force.xy_limit.force_mode_damping, 2, 0.05, tooltip=TT["force_mode_damping"]
        )
        self.force_xy_gain_input = self._add_spin(
            self.force_xy_limit_group, "Gain Scaling (0-2):", 0.0, 2.0,
            CONFIG.rotation_force.xy_limit.force_mode_gain_scaling, 2, 0.1, tooltip=TT["force_mode_gain_scaling"]
        )
        self.force_params_group.addWidget(self.force_xy_limit_group)

        layout.addWidget(self.force_params_group)

        # =====================================================
        # TRAVERSE PARAMETERS (Hybrid/Force only)
        # =====================================================
        self.replay_group = CollapsibleGroupBox("Traverse Parameters (servoPath)", expanded=False)

        traverse_method_layout = QHBoxLayout()
        traverse_method_layout.addWidget(QLabel("Method:"))
        self.traverse_method_combo = QComboBox()
        self.traverse_method_combo.addItem("moveLPath (smooth)", "moveLPath")
        self.traverse_method_combo.addItem("servoPath (precise)", "servoPath")
        self.traverse_method_combo.addItem("movePath (straight)", "movePath")
        self.traverse_method_combo.setCurrentIndex(1)
        self.traverse_method_combo.setToolTip(
            "moveLPath: Smooth path following with blend radius\n"
            "servoPath: Precise timing control with servo commands\n"
            "movePath: Single moveL to end (straight line)"
        )
        self.traverse_method_combo.currentIndexChanged.connect(self._on_traverse_method_changed)
        traverse_method_layout.addWidget(self.traverse_method_combo)
        self.replay_group.addLayout(traverse_method_layout)

        self.replay_speed_input = self._add_spin(
            self.replay_group, "Speed (m/s):", 0.001, 1.0,
            CONFIG.traverse_servopath.speed, 3, 0.01, tooltip=TT["speed"]
        )
        self.replay_accel_input = self._add_spin(
            self.replay_group, "Accel (m/s²):", 0.01, 2.0,
            CONFIG.traverse_servopath.acceleration, 2, 0.1, tooltip=TT["acceleration"]
        )

        # moveLPath-specific: Blend radius
        self.blend_widget = QWidget()
        blend_layout = QHBoxLayout(self.blend_widget)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        blend_layout.addWidget(QLabel("Blend (m):"))
        self.replay_blend_input = QDoubleSpinBox()
        self.replay_blend_input.setRange(0.0, 0.1)
        self.replay_blend_input.setDecimals(3)
        self.replay_blend_input.setValue(CONFIG.traverse_movelpath.blend)
        self.replay_blend_input.setSingleStep(0.005)
        self.replay_blend_input.setMaximumWidth(80)
        self.replay_blend_input.setToolTip(TT["blend"])
        blend_layout.addWidget(self.replay_blend_input)
        self.replay_group.addWidget(self.blend_widget)
        self.blend_widget.setVisible(False)

        # servoPath-specific parameters
        self.servo_params_group = CollapsibleGroupBox("Servo Parameters", expanded=False)

        self.servo_dt_input = self._add_spin(
            self.servo_params_group, "Control dt (s):", 0.002, 0.05,
            CONFIG.traverse_servopath.dt, 3, 0.002, tooltip=TT["servo_dt"]
        )
        self.servo_lookahead_input = self._add_spin(
            self.servo_params_group, "Lookahead (s):", 0.05, 0.5,
            CONFIG.traverse_servopath.lookahead_time, 2, 0.05, tooltip=TT["lookahead"]
        )
        self.servo_gain_input = self._add_spin(
            self.servo_params_group, "Servo Gain:", 50, 500,
            CONFIG.traverse_servopath.gain, 0, 50, tooltip=TT["servo_gain"]
        )
        self.servo_ramp_input = self._add_spin(
            self.servo_params_group, "Ramp-up (s):", 0.1, 2.0,
            CONFIG.traverse_servopath.ramp_up_time, 2, 0.1, tooltip=TT["ramp_up"]
        )

        self.replay_group.addWidget(self.servo_params_group)

        self.end_force_control_checkbox = QCheckBox("Enable Torque Limit (Mz)")
        self.end_force_control_checkbox.setChecked(True)
        self.replay_group.addWidget(self.end_force_control_checkbox)

        layout.addWidget(self.replay_group)

        # =====================================================
        # LOG FILENAME
        # =====================================================
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log:"))
        self.log_filename_input = QLineEdit()
        self.log_filename_input.setPlaceholderText("Enter filename (no extension)")
        self.log_filename_input.textChanged.connect(self._on_log_filename_changed)
        log_layout.addWidget(self.log_filename_input)
        layout.addLayout(log_layout)

        # =====================================================
        # BUTTONS
        # =====================================================
        button_layout = QHBoxLayout()

        self.set_path_btn = QPushButton("Set Path")
        self.set_path_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_has_path_style = "background-color: #1464A0; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setEnabled(False)
        self.set_path_btn.clicked.connect(self.collectWaypoints)
        button_layout.addWidget(self.set_path_btn)

        self.run_btn = QPushButton("Run")
        self.run_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.run_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self.run_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        # Apply initial visibility for selected method
        self._apply_method_visibility(self.method_combo.currentData())

    # =====================================================
    # HELPERS
    # =====================================================

    def _add_spin(self, group, label, min_val, max_val, default, decimals, step, tooltip=None):
        """Create a labelled QDoubleSpinBox inside a CollapsibleGroupBox."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setMaximumWidth(80)
        if tooltip:
            spin.setToolTip(tooltip)
        row.addWidget(spin)
        group.addLayout(row)
        return spin

    def _get_current_method_module(self):
        """Get the currently selected method's module."""
        method = self.method_combo.currentData()
        return METHOD_MODULES.get(method, rotation_direct)

    def _is_path_method(self):
        """Return True if the current method uses Set Path / Run workflow."""
        return self.method_combo.currentData() in ("hybrid", "force")

    def _apply_method_visibility(self, method):
        """Show/hide UI sections based on the selected method."""
        is_path = method in ("hybrid", "force")

        # Method-specific collection parameter groups
        self.direct_params_group.setVisible(method == "direct")
        self.hybrid_params_group.setVisible(method == "hybrid")
        self.force_params_group.setVisible(method == "force")

        # Traverse parameters and Set Path button: only for path methods
        self.replay_group.setVisible(is_path)
        self.set_path_btn.setVisible(is_path)

    # =====================================================
    # SLOTS
    # =====================================================

    def _on_method_changed(self, index):
        """Show/hide method-specific UI and refresh button states."""
        method = self.method_combo.currentData()
        self._apply_method_visibility(method)
        self._on_direction_changed(None, None)

    def _on_traverse_method_changed(self, index):
        """Update traverse group title and show/hide method-specific parameters."""
        traverse_method = self.traverse_method_combo.currentData()
        method_names = {
            "moveLPath": "Traverse Parameters (moveLPath)",
            "servoPath": "Traverse Parameters (servoPath)",
            "movePath": "Traverse Parameters (movePath)"
        }
        self.replay_group.setTitle(method_names.get(traverse_method, "Traverse Parameters"))

        self.blend_widget.setVisible(traverse_method == "moveLPath")
        self.servo_params_group.setVisible(traverse_method == "servoPath")

        if traverse_method == "servoPath":
            self.replay_speed_input.setValue(CONFIG.traverse_servopath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_servopath.acceleration)
        elif traverse_method == "movePath":
            self.replay_speed_input.setValue(CONFIG.traverse_movepath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_movepath.acceleration)
        else:
            self.replay_speed_input.setValue(CONFIG.traverse_movelpath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_movelpath.acceleration)

    def _on_direction_changed(self, button, checked):
        """Update button states based on direction, method, path existence, and log filename."""
        checked_button = self.direction_group.checkedButton()
        has_direction = checked_button is not None
        has_log_filename = bool(self.log_filename_input.text().strip())
        is_path = self._is_path_method()

        if is_path:
            # --- Hybrid / Force: Set Path + Run workflow ---
            self.set_path_btn.setEnabled(has_direction)

            if has_direction:
                direction = checked_button.direction
                method_module = self._get_current_method_module()
                path_file_exists = method_module.path_exists(direction)

                if path_file_exists:
                    self.set_path_btn.setStyleSheet(self.set_path_btn_has_path_style)
                    can_run = has_log_filename
                    self.run_btn.setEnabled(can_run)
                    self.run_btn.setStyleSheet(
                        self.run_btn_default_style if can_run else self.run_btn_disabled_style
                    )
                    # Update waypoint visualization only when direction changed (not on log filename keystroke)
                    if button is not None and hasattr(self.app, '_motion_path_visible') and self.app._motion_path_visible:
                        path_file = method_module.get_path_filename(direction)
                        if hasattr(self.app, 'visualize_waypoints'):
                            self.app.visualize_waypoints(path_file)
                else:
                    self.set_path_btn.setStyleSheet(self.set_path_btn_default_style)
                    self.run_btn.setEnabled(False)
                    self.run_btn.setStyleSheet(self.run_btn_disabled_style)
                    self._hide_path_visualization()
            else:
                self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
                self.run_btn.setEnabled(False)
                self.run_btn.setStyleSheet(self.run_btn_disabled_style)
                self._hide_path_visualization()
        else:
            # --- Direct: single Run button ---
            can_run = has_direction and has_log_filename
            self.run_btn.setEnabled(can_run)
            self.run_btn.setStyleSheet(
                self.run_btn_default_style if can_run else self.run_btn_disabled_style
            )
            if not has_direction:
                self._hide_path_visualization()

        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()

    def _on_log_filename_changed(self, text):
        """Update button states when log filename changes."""
        self._on_direction_changed(None, None)

    def clear_direction(self):
        """Clear the direction selection."""
        self.direction_group.setExclusive(False)
        self.left_arrow.setChecked(False)
        self.right_arrow.setChecked(False)
        self.direction_group.setExclusive(True)

        self.set_path_btn.setEnabled(False)
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self._hide_path_visualization()

        self.log_filename_input.clear()

        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()

    def _hide_path_visualization(self):
        """Hide path visualization."""
        if hasattr(self.app, 'hide_waypoints_visualization'):
            self.app.hide_waypoints_visualization()
        if hasattr(self.app, '_motion_path_visible'):
            self.app._motion_path_visible = False
        if hasattr(self.app, 'motion_show_path_btn'):
            self.app.motion_show_path_btn.setText("○")
        self.path_visible = False

    def update_path_buttons(self):
        """Update path button states - called by parent."""
        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()

    def on_path_saved(self, direction: str):
        """Called when a path has been saved — refresh button states."""
        checked_button = self.direction_group.checkedButton()
        if checked_button and checked_button.direction == direction:
            self._on_direction_changed(None, None)

    # =====================================================
    # RUN DISPATCHER
    # =====================================================

    def _on_run_clicked(self):
        """Route to the correct execution method based on selected method."""
        if self._is_path_method():
            self.traverseWaypoints()
        else:
            self.execute_rotation_direct()

    # =====================================================
    # DIRECT EXECUTION (no path)
    # =====================================================

    def execute_rotation_direct(self):
        """Execute the Direct axial rotation (single run, no path recording)."""
        angle_deg = self.angle_input.value()
        speed = self.speed_input.value()
        accel = self.accel_input.value()
        max_moment = self.max_moment_input.value()
        direction = DIRECTION_MAP[self.direction_group.checkedButton().direction]

        log_filename = self.log_filename_input.text().strip()
        if not log_filename:
            print("Error: Log filename is required")
            return

        self.run_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")

        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            self.reset_button_style()
            return

        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress. Please wait for it to complete.")
            self.reset_button_style()
            return

        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("rotation")

        try:
            start_position = list(self.app.robot.rtdeReceive.getActualTCPPose())
            print(f"\n=== Axial Rotation (direct) ===")
            print(f"Starting TCP pose:")
            print(f"  Position: [{start_position[0]:.4f}, {start_position[1]:.4f}, {start_position[2]:.4f}]")
            print(f"  Orientation (axis-angle): [{start_position[3]:.4f}, {start_position[4]:.4f}, {start_position[5]:.4f}]")

            new_pose = rotate_pose_around_tcp_z(start_position, angle_deg * direction)
            print(f"\nRotating TCP frame {angle_deg * direction}° around TCP z-axis (local)...")

            widget = self

            def on_movement_started():
                print("Nulling force sensor...")
                self.app.stop_btn.setEnabled(True)

            def on_completed(success, message):
                print(message)
                self.app.stop_btn.setEnabled(False)
                widget.reset_button_style()
                widget.update_path_buttons()
                self.app.async_motion_runner = None

            method_kwargs = {
                'force_mode_z_limit': self.direct_z_limit_input.value(),
                'force_mode_z_damping': self.direct_z_damping_input.value(),
                'force_mode_z_gain_scaling': self.direct_z_gain_input.value(),
                'force_mode_xy_limit': self.direct_xy_limit_input.value(),
                'force_mode_xy_damping': self.direct_xy_damping_input.value(),
                'force_mode_xy_gain_scaling': self.direct_xy_gain_input.value(),
            }

            self.app.async_motion_runner = AsyncMotionRunner(
                mode=AsyncMotionRunner.MODE_COLLECT,
                robot=self.app.robot,
                func=rotation_direct,
                start_position=start_position,
                new_pose=new_pose,
                speed=speed,
                accel=accel,
                max_moment=max_moment,
                direction=self.direction_group.checkedButton().direction,
                motionLogFile=getLogfilePath(log_filename),
                **method_kwargs
            )
            self.app.async_motion_runner.movement_started.connect(on_movement_started)
            self.app.async_motion_runner.movement_progress.connect(print)
            self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
            self.app.async_motion_runner.movement_completed.connect(on_completed)
            self.app.async_motion_runner.start()

        except Exception as e:
            print(f"Error during axial rotation: {e}")
            import traceback
            traceback.print_exc()
            self.reset_button_style()

    # =====================================================
    # COLLECT WAYPOINTS (Hybrid / Force — Set Path)
    # =====================================================

    def collectWaypoints(self):
        """Execute rotation movement and save path (Hybrid/Force methods)."""
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return

        direction = checked_button.direction
        angle_deg = self.angle_input.value()
        speed = self.speed_input.value()
        accel = self.accel_input.value()
        max_moment = self.max_moment_input.value()
        direction_multiplier = DIRECTION_MAP[direction]
        method = self.method_combo.currentData()
        method_module = self._get_current_method_module()
        path_file = method_module.get_path_filename(direction)

        # Build method-specific kwargs
        if method == "hybrid":
            method_kwargs = {
                'force_mode_z_limit': self.hybrid_z_limit_input.value(),
                'force_mode_z_damping': self.hybrid_z_damping_input.value(),
                'force_mode_z_gain_scaling': self.hybrid_z_gain_input.value(),
                'force_mode_xy_limit': self.hybrid_xy_limit_input.value(),
                'force_mode_xy_damping': self.hybrid_xy_damping_input.value(),
                'force_mode_xy_gain_scaling': self.hybrid_xy_gain_input.value(),
            }
        else:  # force
            method_kwargs = {
                'force_mode_z_limit': self.force_z_limit_input.value(),
                'force_mode_z_damping': self.force_z_damping_input.value(),
                'force_mode_z_gain_scaling': self.force_z_gain_input.value(),
                'force_mode_xy_limit': self.force_xy_limit_input.value(),
                'force_mode_xy_damping': self.force_xy_damping_input.value(),
                'force_mode_xy_gain_scaling': self.force_xy_gain_input.value(),
                'control_loop_dt': self.force_control_dt_input.value(),
                'rotation_speed_factor': self.force_rot_speed_factor_input.value(),
            }

        # Visual feedback
        self.set_path_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
        self.set_path_btn.setEnabled(False)
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self.run_btn.setEnabled(False)

        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            self.reset_button_style()
            return

        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress. Please wait for it to complete.")
            self.reset_button_style()
            return

        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("rotation set path")

        try:
            start_position = list(self.app.robot.rtdeReceive.getActualTCPPose())
            new_pose = rotate_pose_around_tcp_z(start_position, angle_deg * direction_multiplier)

            print(f"\n=== Axial Rotation Set Path ({method}) ===")
            print(f"Starting TCP pose: [{start_position[0]:.4f}, {start_position[1]:.4f}, {start_position[2]:.4f}]")
            print(f"Rotating {angle_deg * direction_multiplier}° around TCP z-axis...")
            print(f"Saving path to: {path_file}")

            widget = self

            def on_movement_started():
                print("Nulling force sensor...")
                self.app.stop_btn.setEnabled(True)

            def on_completed(success, message):
                print(message)
                self.app.stop_btn.setEnabled(False)
                if success:
                    widget.on_path_saved(direction)
                widget.reset_button_style()
                self.app.async_motion_runner = None

            self.app.async_motion_runner = AsyncMotionRunner(
                mode=AsyncMotionRunner.MODE_COLLECT,
                robot=self.app.robot,
                func=method_module,
                start_position=start_position,
                new_pose=new_pose,
                speed=speed,
                accel=accel,
                max_moment=max_moment,
                path_file=path_file,
                collection_method=method,
                traverseMethod=self.traverse_method_combo.currentData(),
                blend=self.replay_blend_input.value(),
                retrace_speed=self.replay_speed_input.value(),
                retrace_acceleration=self.replay_accel_input.value(),
                **method_kwargs
            )
            self.app.async_motion_runner.movement_started.connect(on_movement_started)
            self.app.async_motion_runner.movement_progress.connect(print)
            self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
            self.app.async_motion_runner.movement_completed.connect(on_completed)
            self.app.async_motion_runner.start()

        except Exception as e:
            print(f"Error during rotation set path: {e}")
            import traceback
            traceback.print_exc()
            self.reset_button_style()

    # =====================================================
    # TRAVERSE WAYPOINTS (Hybrid / Force — Run)
    # =====================================================

    def traverseWaypoints(self):
        """Execute path replay (Hybrid/Force methods)."""
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return

        direction = checked_button.direction
        method_module = self._get_current_method_module()
        path_file = method_module.get_path_filename(direction)

        if not method_module.path_exists(direction):
            print(f"No path found for {direction} direction")
            return

        log_filename = self.log_filename_input.text().strip()
        if not log_filename:
            print("Error: Log filename is required")
            return

        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            return

        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress. Please wait for it to complete.")
            return

        # Visual feedback
        self.run_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
        self.run_btn.setEnabled(False)
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setEnabled(False)

        replay_speed = self.replay_speed_input.value()
        enable_end_force = self.end_force_control_checkbox.isChecked()
        moment_limit = self.max_moment_input.value()

        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("rotation path replay")

        print(f"\n=== Path Replay (Axial Rotation) ===")
        print(f"Replay speed: {replay_speed} m/s")
        print(f"Torque limit control: {'enabled' if enable_end_force else 'disabled'}")
        if enable_end_force:
            print(f"Max moment (Mz): {moment_limit} Nm")

        widget = self

        def on_completed(success, message):
            print(message)
            self.app.stop_btn.setEnabled(False)
            widget.reset_button_style()
            self.app.async_motion_runner = None

        self.app.async_motion_runner = AsyncMotionRunner(
            mode=AsyncMotionRunner.MODE_TRAVERSE,
            robot=self.app.robot,
            path_file=path_file,
            speed=replay_speed,
            acceleration=self.replay_accel_input.value(),
            blend=self.replay_blend_input.value(),
            traverseMethod=self.traverse_method_combo.currentData(),
            enableForceControl=enable_end_force,
            forceLimit=moment_limit,
            forceAxis='mz',
            direction=direction,
            autoReturn=True,
            motionLogFile=getLogfilePath(log_filename)
        )
        self.app.async_motion_runner.movement_progress.connect(print)
        self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
        self.app.async_motion_runner.movement_completed.connect(on_completed)
        self.app.async_motion_runner.start()
        self.app.stop_btn.setEnabled(True)

    # =====================================================
    # RESET
    # =====================================================

    def reset_button_style(self):
        """Reset button styles after movement."""
        self.clear_direction()


def rotate_pose_around_tcp_z(pose: List[float], angle_deg: float) -> List[float]:
    """
    Rotate the TCP frame around its own (local) z-axis while keeping position fixed.

    Args:
        pose: Current TCP pose [x, y, z, rx, ry, rz]
        angle_deg: Rotation angle in degrees (positive = counterclockwise about TCP z)

    Returns:
        New TCP pose [x, y, z, rx_new, ry_new, rz_new] with rotated orientation
    """
    x, y, z = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]

    R_current = axis_angle_to_rotation_matrix(rx, ry, rz)

    angle_rad = math.radians(angle_deg)
    R_tcp_z = Rotation.from_euler('z', angle_rad, degrees=False).as_matrix()

    R_new = R_current @ R_tcp_z

    rx_new, ry_new, rz_new = rotation_matrix_to_axis_angle(R_new)

    return [x, y, z, rx_new, ry_new, rz_new]


MotionWidget = AxialRotationWidget
