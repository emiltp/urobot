"""Arc (Fz-constrained) movement widget - orbit around ref frame with constant TCP Fz."""

import math
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QComboBox, QButtonGroup, QLineEdit,
)
from PyQt6.QtCore import Qt

from src.ui import ArrowButton, CircleWidget, CollapsibleGroupBox
from src.utils import orbit_tcp_around_ref, axis_angle_to_rotation_matrix
from src.parameter_tooltips import PARAMETER_TOOLTIPS as TT
from src.objects.actors.endpoint_axes_actor import EndpointAxesActor
from src.motion_logger import getLogfilePath

from src.movements.async_motion_runner import AsyncMotionRunner
from . import servol as arc_servol
from . import speedl as arc_speedl
from . import movel as arc_movel
from . import get_path_filename as arc_get_path
from . import path_exists as arc_path_exists
from config import defaults as CONFIG

DIRECTION_MAP = {"left": -1, "right": 1}

METHOD_MODULES = {
    "servol": arc_servol,
    "speedl": arc_speedl,
    "movel": arc_movel,
}


class ArcForceWidget(QWidget):
    """Widget for arc movement with Fz constraint (orbit around ref frame)."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._endpoint_actor = None
        self.path_visible = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        info_label = QLabel(
            "Moves TCP along an arc (orbit around ref frame) while keeping Fz constant. "
            "Radius/center may vary; robot may not reach endpoint."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        layout.addWidget(info_label)

        # Direction selection (like new_x)
        direction_layout = QHBoxLayout()
        direction_layout.setSpacing(2)

        self.direction_group = QButtonGroup(self)
        self.direction_group.setExclusive(True)

        self.left_arrow = ArrowButton("left", label="CCW")
        self.left_arrow.setFixedSize(80, 55)
        self.circle = CircleWidget(top_label="Ref", bottom_label="Axis")
        self.circle.setFixedSize(50, 70)
        self.right_arrow = ArrowButton("right", label="CW")
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

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("servoL (pose following)", "servol")
        self.method_combo.addItem("speedL (velocity control)", "speedl")
        self.method_combo.addItem("moveL (path)", "movel")
        self.method_combo.setCurrentIndex(2)  # moveL as default
        self.method_combo.setToolTip(
            "servoL: Precise path tracking, best for variable radius/curved surfaces.\n"
            "speedL: Velocity-based, predictable speed profile.\n"
            "moveL: Simplest, pre-computed path, no mid-flight adaptation."
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # Axis selection
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Ref axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("X", "x")
        self.axis_combo.addItem("Y", "y")
        self.axis_combo.addItem("Z", "z")
        self.axis_combo.setToolTip("Reference frame axis to rotate around")
        self.axis_combo.currentIndexChanged.connect(self._update_endpoint_visualization)
        axis_layout.addWidget(self.axis_combo)
        layout.addLayout(axis_layout)

        # Movement Parameters
        self.movement_group = CollapsibleGroupBox("Movement Parameters", expanded=False)
        self.angle_input = self._add_spin(
            self.movement_group, "Angle (deg):", 0.0, 180.0,
            CONFIG.arc_force.angle, 1, 5.0, TT["angle"]
        )
        self.angle_input.valueChanged.connect(self._update_endpoint_visualization)
        self.speed_input = self._add_spin(
            self.movement_group, "Speed (m/s):", 0.001, 1.0,
            CONFIG.arc_force.speed, 4, 0.01, TT["speed"]
        )
        self.accel_input = self._add_spin(
            self.movement_group, "Acceleration (m/s²):", 0.001, 2.0,
            CONFIG.arc_force.acceleration, 3, 0.01, TT["acceleration"]
        )
        self.max_moment_input = self._add_spin(
            self.movement_group, "Max Moment Mx (Nm):", 0.1, 50.0,
            CONFIG.arc_force.max_moment, 2, 0.1, TT["arc_force_max_moment"]
        )
        self.fy_tolerance_input = self._add_spin(
            self.movement_group, "Max TCP Fy (N):", 0.5, 50.0,
            CONFIG.arc_force.fy_tolerance, 1, 1.0, TT["fy_tolerance"]
        )
        layout.addWidget(self.movement_group)

        # Force Mode Parameters
        self.force_group = CollapsibleGroupBox("Force Mode Parameters", expanded=False)
        force_baseline_layout = QHBoxLayout()
        force_baseline_layout.addWidget(QLabel("Force baseline:"))
        self.force_baseline_combo = QComboBox()
        self.force_baseline_combo.addItem("Null sensor", "null")
        self.force_baseline_combo.addItem("Subtract initial", "subtract_initial")
        default_baseline = getattr(CONFIG.arc_force, 'force_baseline', 'null')
        idx = self.force_baseline_combo.findData(default_baseline)
        if idx >= 0:
            self.force_baseline_combo.setCurrentIndex(idx)
        self.force_baseline_combo.setToolTip(TT["force_baseline"])
        force_baseline_layout.addWidget(self.force_baseline_combo)
        self.force_group.addLayout(force_baseline_layout)
        self.z_limit_input = self._add_spin(
            self.force_group, "Z limit (m/s):", 0.01, 0.2,
            CONFIG.arc_force.force_mode_z_limit, 3, 0.01, TT["force_mode_z_limit"]
        )
        self.damping_input = self._add_spin(
            self.force_group, "Damping:", 0.0, 1.0,
            CONFIG.arc_force.force_mode_damping, 2, 0.05, TT["force_mode_damping"]
        )
        self.gain_input = self._add_spin(
            self.force_group, "Gain scaling:", 0.0, 2.0,
            CONFIG.arc_force.force_mode_gain_scaling, 2, 0.1, TT["force_mode_gain_scaling"]
        )
        layout.addWidget(self.force_group)

        # Termination Parameters
        self.termination_group = CollapsibleGroupBox("Termination Parameters", expanded=False)
        self.timeout_input = self._add_spin(
            self.termination_group, "Timeout (s):", 5.0, 300.0,
            CONFIG.arc_force.timeout, 1, 10.0, TT["timeout"]
        )
        self.pos_tolerance_input = self._add_spin(
            self.termination_group, "Pos tolerance (m):", 0.0005, 0.01,
            CONFIG.arc_force.pos_tolerance, 4, 0.0005, TT["pos_tolerance"]
        )
        self.rot_tolerance_input = self._add_spin(
            self.termination_group, "Rot tolerance (rad):", 0.005, 0.1,
            CONFIG.arc_force.rot_tolerance, 3, 0.01, TT["rot_tolerance"]
        )
        layout.addWidget(self.termination_group)

        # Waypoint Parameters (servoL/moveL)
        self.waypoint_group = CollapsibleGroupBox("Waypoint Parameters", expanded=False)
        self.waypoint_count_input = self._add_spin(
            self.waypoint_group, "Waypoint count:", 10, 200,
            CONFIG.arc_force.waypoint_count, 0, 10, TT["waypoint_count"]
        )
        self.target_distance_input = self._add_spin(
            self.waypoint_group, "Target distance (m):", 0.0005, 0.01,
            CONFIG.arc_force.target_distance, 4, 0.001, TT["target_distance"]
        )
        layout.addWidget(self.waypoint_group)
        self.waypoint_group.setVisible(True)  # Visible for servol/movel

        # Traverse Parameters
        self.replay_group = CollapsibleGroupBox("Traverse Parameters (servoPath)", expanded=False)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
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
        method_row.addWidget(self.traverse_method_combo)
        self.replay_group.addLayout(method_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed (m/s):"))
        self.replay_speed_input = QDoubleSpinBox()
        self.replay_speed_input.setRange(0.001, 1.0)
        self.replay_speed_input.setDecimals(3)
        self.replay_speed_input.setValue(CONFIG.traverse_servopath.speed)
        self.replay_speed_input.setSingleStep(0.01)
        self.replay_speed_input.setMaximumWidth(80)
        speed_row.addWidget(self.replay_speed_input)
        self.replay_group.addLayout(speed_row)
        accel_row = QHBoxLayout()
        accel_row.addWidget(QLabel("Accel (m/s²):"))
        self.replay_accel_input = QDoubleSpinBox()
        self.replay_accel_input.setRange(0.01, 2.0)
        self.replay_accel_input.setDecimals(2)
        self.replay_accel_input.setValue(CONFIG.traverse_servopath.acceleration)
        self.replay_accel_input.setSingleStep(0.1)
        self.replay_accel_input.setMaximumWidth(80)
        accel_row.addWidget(self.replay_accel_input)
        self.replay_group.addLayout(accel_row)
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
        blend_layout.addWidget(self.replay_blend_input)
        self.replay_group.addWidget(self.blend_widget)
        self.blend_widget.setVisible(False)
        self.servo_params_group = CollapsibleGroupBox("Servo Parameters", expanded=False)
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Control dt (s):"))
        self.servo_dt_input = QDoubleSpinBox()
        self.servo_dt_input.setRange(0.002, 0.05)
        self.servo_dt_input.setDecimals(3)
        self.servo_dt_input.setValue(CONFIG.traverse_servopath.dt)
        self.servo_dt_input.setSingleStep(0.002)
        self.servo_dt_input.setMaximumWidth(80)
        dt_layout.addWidget(self.servo_dt_input)
        self.servo_params_group.addLayout(dt_layout)
        lookahead_layout = QHBoxLayout()
        lookahead_layout.addWidget(QLabel("Lookahead (s):"))
        self.servo_lookahead_input = QDoubleSpinBox()
        self.servo_lookahead_input.setRange(0.05, 0.5)
        self.servo_lookahead_input.setDecimals(2)
        self.servo_lookahead_input.setValue(CONFIG.traverse_servopath.lookahead_time)
        self.servo_lookahead_input.setSingleStep(0.05)
        self.servo_lookahead_input.setMaximumWidth(80)
        lookahead_layout.addWidget(self.servo_lookahead_input)
        self.servo_params_group.addLayout(lookahead_layout)
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Servo Gain:"))
        self.servo_gain_input = QDoubleSpinBox()
        self.servo_gain_input.setRange(50, 500)
        self.servo_gain_input.setDecimals(0)
        self.servo_gain_input.setValue(CONFIG.traverse_servopath.gain)
        self.servo_gain_input.setSingleStep(50)
        self.servo_gain_input.setMaximumWidth(80)
        gain_layout.addWidget(self.servo_gain_input)
        self.servo_params_group.addLayout(gain_layout)
        ramp_layout = QHBoxLayout()
        ramp_layout.addWidget(QLabel("Ramp-up (s):"))
        self.servo_ramp_input = QDoubleSpinBox()
        self.servo_ramp_input.setRange(0.1, 2.0)
        self.servo_ramp_input.setDecimals(2)
        self.servo_ramp_input.setValue(CONFIG.traverse_servopath.ramp_up_time)
        self.servo_ramp_input.setSingleStep(0.1)
        self.servo_ramp_input.setMaximumWidth(80)
        ramp_layout.addWidget(self.servo_ramp_input)
        self.servo_params_group.addLayout(ramp_layout)
        self.replay_group.addWidget(self.servo_params_group)
        layout.addWidget(self.replay_group)

        # Log filename (for Run / motion logging)
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log:"))
        self.log_filename_input = QLineEdit()
        self.log_filename_input.setPlaceholderText("Enter filename (no extension)")
        self.log_filename_input.textChanged.connect(self._on_log_filename_changed)
        log_layout.addWidget(self.log_filename_input)
        layout.addLayout(log_layout)

        # Buttons: Set Path + Run
        button_layout = QHBoxLayout()
        self.set_path_btn = QPushButton("Set Path")
        self.set_path_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_has_path_style = "background-color: #1464A0; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setEnabled(False)
        self.set_path_btn.clicked.connect(self._on_set_path_clicked)
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

        self._on_method_changed()

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
        return METHOD_MODULES.get(method, arc_servol)

    def _on_method_changed(self):
        """Update visibility of method-specific parameter groups."""
        method = self.method_combo.currentData()
        self.waypoint_group.setVisible(method in ("servol", "movel"))

    def _on_direction_changed(self, button, checked):
        """Handle direction selection change."""
        self._update_button_states()

    def _update_endpoint_visualization(self):
        """Compute and display the arc endpoint (orbit around ref frame)."""
        if not self.app.connected or self.app.robot is None:
            return
        ref_offset = self.app.robot.refFrameOffset
        tcp_pose = self.app.robot.tcpPose
        if tcp_pose is None or len(tcp_pose) < 6:
            return
        if ref_offset is None:
            self._remove_endpoint_visualization()
            return

        ref_pose = self.app.robot._calculateRefFramePose(tcp_pose, ref_offset)
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return
        direction_multiplier = DIRECTION_MAP[checked_button.direction]
        angle_deg = self.angle_input.value() * direction_multiplier
        angle_rad = math.radians(angle_deg)
        axis = self.axis_combo.currentData()

        endpoint_pose = orbit_tcp_around_ref(tcp_pose, ref_pose, angle_rad, axis)

        origin = endpoint_pose[:3]
        rot = axis_angle_to_rotation_matrix(endpoint_pose[3], endpoint_pose[4], endpoint_pose[5])

        renderer = getattr(self.app, 'renderer', None)
        if renderer is None:
            return

        if self._endpoint_actor is None:
            self._endpoint_actor = EndpointAxesActor(origin, rot)
            self._endpoint_actor.addToRenderer(renderer)
        else:
            self._endpoint_actor.updatePose(origin, rot)

        self.app.vtk_widget.GetRenderWindow().Render()

    def _remove_endpoint_visualization(self):
        """Remove the endpoint axes from the renderer."""
        if self._endpoint_actor is not None:
            self._endpoint_actor.removeFromRenderer()
            self._endpoint_actor = None
            if hasattr(self.app, 'vtk_widget'):
                self.app.vtk_widget.GetRenderWindow().Render()

    def _hide_path_visualization(self):
        """Hide path/endpoint visualization (called when switching away from arc_force)."""
        self.path_visible = False
        self._remove_endpoint_visualization()

    def _on_traverse_method_changed(self, index):
        """Update traverse group title and show/hide method-specific parameters."""
        traverse_method = self.traverse_method_combo.currentData()
        method_names = {
            "moveLPath": "Traverse Parameters (moveLPath)",
            "servoPath": "Traverse Parameters (servoPath)",
            "movePath": "Traverse Parameters (movePath)",
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

    def _on_log_filename_changed(self, text):
        """Update Run button state when log filename changes."""
        self._update_button_states()

    def _update_button_states(self):
        """Enable Set Path / Run buttons based on connection, direction, path existence, and log."""
        has_direction = self.direction_group.checkedButton() is not None
        connected = self.app.connected and self.app.robot is not None
        has_log_filename = bool(self.log_filename_input.text().strip())

        self.set_path_btn.setEnabled(connected and has_direction)
        if connected and has_direction:
            direction = self.direction_group.checkedButton().direction
            path_file_exists = arc_path_exists(direction)
            if path_file_exists:
                self.set_path_btn.setStyleSheet(self.set_path_btn_has_path_style)
                self.run_btn.setEnabled(has_log_filename)
                self.run_btn.setStyleSheet(
                    self.run_btn_default_style if has_log_filename else self.run_btn_disabled_style
                )
            else:
                self.set_path_btn.setStyleSheet(self.set_path_btn_default_style)
                self.run_btn.setEnabled(False)
                self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        else:
            self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
            self.run_btn.setEnabled(False)
            self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self._update_endpoint_visualization()
        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()

    def reset_button_style(self):
        """Reset Set Path and Run buttons after movement."""
        self._update_button_states()

    def _on_set_path_clicked(self):
        """Execute arc movement and record path (Set Path)."""
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            print("Error: Select direction (CCW or CW)")
            return
        if not self.app.connected or self.app.robot is None:
            print("Error: Not connected to robot")
            return
        if self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: RTDE interfaces not available")
            return
        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress.")
            return

        ref_offset = self.app.robot.refFrameOffset
        if ref_offset is None:
            print("Error: Ref frame offset not set. Set ref frame in TCP panel first.")
            return

        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("arc force")

        direction = checked_button.direction
        path_file = arc_get_path(direction)
        try:
            start_position = list(self.app.robot.rtdeReceive.getActualTCPPose())
            ref_pose = self.app.robot._calculateRefFramePose(start_position, ref_offset)

            direction_multiplier = DIRECTION_MAP[checked_button.direction]
            angle_deg = self.angle_input.value() * direction_multiplier
            angle_rad = math.radians(angle_deg)
            axis = self.axis_combo.currentData()
            end_pose = orbit_tcp_around_ref(start_position, ref_pose, angle_rad, axis)

            print(f"\n=== Arc (Fz-constrained) ===")
            print(f"Method: {self.method_combo.currentData()}, Axis: {axis}, Direction: {checked_button.direction}")
            print(f"Angle: {angle_deg} deg, Start: [{start_position[0]:.4f}, {start_position[1]:.4f}, {start_position[2]:.4f}]")
            try:
                rtde_r = self.app.robot.rtdeReceive
                if rtde_r is not None and hasattr(rtde_r, 'getPayload'):
                    payload = rtde_r.getPayload()
                    cog = rtde_r.getPayloadCog()
                    # RTDE can return invalid values on some robot firmware (e.g. negative mass, CoG in km)
                    valid_payload = 0 <= payload <= 500  # kg
                    valid_cog = (len(cog) >= 3 and
                                all(-2 <= v <= 2 for v in cog[:3]))
                    if valid_payload and valid_cog:
                        cog_str = f"[{cog[0]:.4f}, {cog[1]:.4f}, {cog[2]:.4f}]"
                        print(f"Payload: {payload:.4f} kg, CoG: {cog_str} m")
                    else:
                        print("Payload: (RTDE returned invalid values — verify in Polyscope Installation)")
                else:
                    print("Payload: (unavailable)")
            except Exception as e:
                print(f"Payload: (unavailable: {e})")

            method_module = self._get_current_method_module()
            method = self.method_combo.currentData()

            method_kwargs = {
                'start_position': start_position,
                'ref_pose': ref_pose,
                'end_pose': end_pose,
                'angle_rad': angle_rad,
                'axis': axis,
                'direction_multiplier': direction_multiplier,
                'force_baseline': self.force_baseline_combo.currentData(),
                'speed': self.speed_input.value(),
                'accel': self.accel_input.value(),
                'max_moment': self.max_moment_input.value(),
                'force_mode_z_limit': self.z_limit_input.value(),
                'force_mode_damping': self.damping_input.value(),
                'force_mode_gain_scaling': self.gain_input.value(),
                'timeout': self.timeout_input.value(),
                'fy_tolerance': self.fy_tolerance_input.value(),
                'pos_tolerance': self.pos_tolerance_input.value(),
                'rot_tolerance': self.rot_tolerance_input.value(),
                'waypoint_count': int(self.waypoint_count_input.value()),
                'target_distance': self.target_distance_input.value(),
                'collection_method': 'original',
                'path_file': path_file,
                'retrace_speed': self.replay_speed_input.value(),
                'retrace_acceleration': self.replay_accel_input.value(),
                'traverseMethod': self.traverse_method_combo.currentData(),
                'blend': self.replay_blend_input.value(),
            }

            self.set_path_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
            self.set_path_btn.setEnabled(False)
            self.run_btn.setStyleSheet(self.run_btn_disabled_style)
            self.run_btn.setEnabled(False)
            self.app.stop_btn.setEnabled(True)

            def on_completed(success, message):
                print(message)
                self.app.stop_btn.setEnabled(False)
                self.reset_button_style()
                self.app.async_motion_runner = None

            self.app.async_motion_runner = AsyncMotionRunner(
                mode=AsyncMotionRunner.MODE_COLLECT,
                robot=self.app.robot,
                func=method_module,
                **method_kwargs
            )
            self.app.async_motion_runner.movement_started.connect(
                lambda: print("Starting arc force (Set Path)...")
            )
            self.app.async_motion_runner.movement_progress.connect(print)
            self.app.async_motion_runner.pose_updated.connect(
                lambda _: self.app.vtk_widget.GetRenderWindow().Render()
            )
            self.app.async_motion_runner.movement_completed.connect(on_completed)
            self.app.async_motion_runner.path_saved.connect(lambda _: self._update_button_states())
            self.app.async_motion_runner.start()

        except Exception as e:
            print(f"Error during arc force Set Path: {e}")
            import traceback
            traceback.print_exc()
            self.reset_button_style()

    def _on_run_clicked(self):
        """Execute path replay (Run / traverse)."""
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            print("Error: Select direction (CCW or CW)")
            return
        direction = checked_button.direction
        path_file = arc_get_path(direction)

        if not arc_path_exists(direction):
            print("No path found. Click 'Set Path' to record first.")
            return
        log_filename = self.log_filename_input.text().strip()
        if not log_filename:
            print("Error: Log filename is required for Run")
            return
        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            return
        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress.")
            return

        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("arc force traverse")

        self.run_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
        self.run_btn.setEnabled(False)
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setEnabled(False)

        print(f"\n=== Arc Force Traverse (Run) ===")
        print(f"Path: {path_file}")
        print(f"Speed: {self.replay_speed_input.value()} m/s")

        def on_completed(success, message):
            print(message)
            self.app.stop_btn.setEnabled(False)
            self.reset_button_style()
            self.app.async_motion_runner = None

        self.app.async_motion_runner = AsyncMotionRunner(
            mode=AsyncMotionRunner.MODE_TRAVERSE,
            robot=self.app.robot,
            path_file=path_file,
            speed=self.replay_speed_input.value(),
            acceleration=self.replay_accel_input.value(),
            blend=self.replay_blend_input.value(),
            traverseMethod=self.traverse_method_combo.currentData(),
            enableForceControl=False,
            motionLogFile=getLogfilePath(log_filename),
        )
        self.app.async_motion_runner.movement_progress.connect(print)
        self.app.async_motion_runner.pose_updated.connect(
            lambda _: self.app.vtk_widget.GetRenderWindow().Render()
        )
        self.app.async_motion_runner.movement_completed.connect(on_completed)
        self.app.async_motion_runner.start()
        self.app.stop_btn.setEnabled(True)


MotionWidget = ArcForceWidget
