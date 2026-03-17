"""Arc (y) movement widget — moveL around ref Y axis, Fz=0, Fx/My limits, null sensing."""

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
from . import movel as arc_y_movel
from . import get_path_filename as arc_y_get_path
from . import path_exists as arc_y_path_exists
from config import defaults as CONFIG

DIRECTION_MAP = {"left": -1, "right": 1}


class ArcYWidget(QWidget):
    """Widget for Arc (y): moveL around ref Y axis, Fz=0, Fx and My limits, null sensing."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._endpoint_actor = None
        self.path_visible = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        info_label = QLabel(
            "Arc (y): moveL orbit around ref Y axis. Fz=0, Fx and My limits, null sensing."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        layout.addWidget(info_label)

        # Direction selection (same labels as New (y))
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

        # Movement Parameters
        arc_cfg = getattr(CONFIG, 'arc_y', CONFIG.arc_force)
        self.movement_group = CollapsibleGroupBox("Movement Parameters", expanded=False)
        self.angle_input = self._add_spin(
            self.movement_group, "Angle (deg):", 0.0, 180.0,
            arc_cfg.angle, 1, 5.0, TT["angle"]
        )
        self.angle_input.valueChanged.connect(self._update_endpoint_visualization)
        self.speed_input = self._add_spin(
            self.movement_group, "Speed (m/s):", 0.001, 1.0,
            arc_cfg.speed, 4, 0.01, TT["speed"]
        )
        self.accel_input = self._add_spin(
            self.movement_group, "Acceleration (m/s²):", 0.001, 2.0,
            arc_cfg.acceleration, 3, 0.01, TT["acceleration"]
        )
        self.max_moment_input = self._add_spin(
            self.movement_group, "Max Moment My (Nm):", 0.1, 50.0,
            arc_cfg.max_moment, 2, 0.1, TT["arc_force_max_moment"]
        )
        self.fx_tolerance_input = self._add_spin(
            self.movement_group, "Max TCP Fx (N):", 0.5, 50.0,
            getattr(arc_cfg, 'fx_tolerance', 5.0), 1, 1.0, TT["fx_tolerance"]
        )
        layout.addWidget(self.movement_group)

        # Collection Method Parameters (force mode + waypoints as submenu)
        self.collection_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)
        self.z_limit_input = self._add_spin(
            self.collection_group, "Z limit (m/s):", 0.01, 0.2,
            arc_cfg.force_mode_z_limit, 3, 0.01, TT["force_mode_z_limit"]
        )
        self.xz_limit_input = self._add_spin(
            self.collection_group, "XZ limit (m/s):", 0.01, 0.2,
            getattr(arc_cfg, 'force_mode_xz_limit', 0.05), 3, 0.01, TT.get("force_mode_xy_limit", TT["force_mode_z_limit"])
        )
        self.damping_input = self._add_spin(
            self.collection_group, "Damping:", 0.0, 1.0,
            arc_cfg.force_mode_damping, 2, 0.05, TT["force_mode_damping"]
        )
        self.gain_input = self._add_spin(
            self.collection_group, "Gain scaling:", 0.0, 2.0,
            arc_cfg.force_mode_gain_scaling, 2, 0.1, TT["force_mode_gain_scaling"]
        )
        self.waypoint_group = CollapsibleGroupBox("Waypoint Parameters", expanded=False)
        self.waypoint_count_input = self._add_spin(
            self.waypoint_group, "Waypoint count:", 10, 200,
            arc_cfg.waypoint_count, 0, 10, TT["waypoint_count"]
        )
        self.target_distance_input = self._add_spin(
            self.waypoint_group, "Target distance (m):", 0.0005, 0.01,
            arc_cfg.target_distance, 4, 0.001, TT["target_distance"]
        )
        self.collection_group.addWidget(self.waypoint_group)
        layout.addWidget(self.collection_group)

        # Traverse Parameters
        self.replay_group = CollapsibleGroupBox("Traverse Parameters (servoPath)", expanded=False)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.traverse_method_combo = QComboBox()
        self.traverse_method_combo.addItem("moveLPath (smooth)", "moveLPath")
        self.traverse_method_combo.addItem("servoPath (precise)", "servoPath")
        self.traverse_method_combo.addItem("movePath (straight)", "movePath")
        self.traverse_method_combo.setCurrentIndex(1)
        self.traverse_method_combo.currentIndexChanged.connect(self._on_traverse_method_changed)
        method_row.addWidget(self.traverse_method_combo)
        self.replay_group.addLayout(method_row)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed (m/s):"))
        self.replay_speed_input = QDoubleSpinBox()
        self.replay_speed_input.setRange(0.001, 1.0)
        self.replay_speed_input.setDecimals(3)
        self.replay_speed_input.setValue(CONFIG.traverse_servopath.speed)
        speed_row.addWidget(self.replay_speed_input)
        self.replay_group.addLayout(speed_row)
        accel_row = QHBoxLayout()
        accel_row.addWidget(QLabel("Accel (m/s²):"))
        self.replay_accel_input = QDoubleSpinBox()
        self.replay_accel_input.setRange(0.01, 2.0)
        self.replay_accel_input.setDecimals(2)
        self.replay_accel_input.setValue(CONFIG.traverse_servopath.acceleration)
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
        blend_layout.addWidget(self.replay_blend_input)
        self.replay_group.addWidget(self.blend_widget)
        self.blend_widget.setVisible(False)
        self.servo_params_group = CollapsibleGroupBox("Servo Parameters", expanded=False)
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Control dt (s):"))
        self.servo_dt_input = QDoubleSpinBox()
        self.servo_dt_input.setRange(0.002, 0.05)
        self.servo_dt_input.setValue(CONFIG.traverse_servopath.dt)
        dt_layout.addWidget(self.servo_dt_input)
        self.servo_params_group.addLayout(dt_layout)
        lookahead_layout = QHBoxLayout()
        lookahead_layout.addWidget(QLabel("Lookahead (s):"))
        self.servo_lookahead_input = QDoubleSpinBox()
        self.servo_lookahead_input.setRange(0.05, 0.5)
        self.servo_lookahead_input.setValue(CONFIG.traverse_servopath.lookahead_time)
        lookahead_layout.addWidget(self.servo_lookahead_input)
        self.servo_params_group.addLayout(lookahead_layout)
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Servo Gain:"))
        self.servo_gain_input = QDoubleSpinBox()
        self.servo_gain_input.setRange(50, 500)
        self.servo_gain_input.setValue(CONFIG.traverse_servopath.gain)
        gain_layout.addWidget(self.servo_gain_input)
        self.servo_params_group.addLayout(gain_layout)
        ramp_layout = QHBoxLayout()
        ramp_layout.addWidget(QLabel("Ramp-up (s):"))
        self.servo_ramp_input = QDoubleSpinBox()
        self.servo_ramp_input.setRange(0.1, 2.0)
        self.servo_ramp_input.setValue(CONFIG.traverse_servopath.ramp_up_time)
        ramp_layout.addWidget(self.servo_ramp_input)
        self.servo_params_group.addLayout(ramp_layout)
        self.replay_group.addWidget(self.servo_params_group)
        layout.addWidget(self.replay_group)

        # Log filename
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log:"))
        self.log_filename_input = QLineEdit()
        self.log_filename_input.setPlaceholderText("Enter filename (no extension)")
        self.log_filename_input.textChanged.connect(self._on_log_filename_changed)
        log_layout.addWidget(self.log_filename_input)
        layout.addLayout(log_layout)

        # Buttons
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

    def _add_spin(self, group, label, min_val, max_val, default, decimals, step, tooltip=None):
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
        from . import movel
        return movel

    def _on_direction_changed(self, button, checked):
        self._update_button_states()

    def _update_endpoint_visualization(self):
        if not self.app.connected or self.app.robot is None:
            return
        ref_offset = self.app.robot.refFrameOffset
        tcp_pose = self.app.robot.tcpPose
        if tcp_pose is None or len(tcp_pose) < 6 or ref_offset is None:
            self._remove_endpoint_visualization()
            return
        ref_pose = self.app.robot._calculateRefFramePose(tcp_pose, ref_offset)
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return
        direction_multiplier = DIRECTION_MAP[checked_button.direction]
        angle_deg = self.angle_input.value() * direction_multiplier
        angle_rad = math.radians(angle_deg)
        axis = 'y'
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
        if self._endpoint_actor is not None:
            self._endpoint_actor.removeFromRenderer()
            self._endpoint_actor = None
            if hasattr(self.app, 'vtk_widget'):
                self.app.vtk_widget.GetRenderWindow().Render()

    def _hide_path_visualization(self):
        self.path_visible = False
        self._remove_endpoint_visualization()

    def _on_traverse_method_changed(self, index):
        traverse_method = self.traverse_method_combo.currentData()
        titles = {"moveLPath": "Traverse Parameters (moveLPath)",
                  "servoPath": "Traverse Parameters (servoPath)",
                  "movePath": "Traverse Parameters (movePath)"}
        self.replay_group.setTitle(titles.get(traverse_method, "Traverse Parameters"))
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
        self._update_button_states()

    def _update_button_states(self):
        has_direction = self.direction_group.checkedButton() is not None
        connected = self.app.connected and self.app.robot is not None
        has_log_filename = bool(self.log_filename_input.text().strip())
        self.set_path_btn.setEnabled(connected and has_direction)
        if connected and has_direction:
            direction = self.direction_group.checkedButton().direction
            path_file_exists = arc_y_path_exists(direction)
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
        self._update_button_states()

    def _on_set_path_clicked(self):
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            print("Error: Select direction (LEFT or RIGHT)")
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
            self.app._disable_freedrive_for_movement("arc y")

        direction = checked_button.direction
        path_file = arc_y_get_path(direction)
        arc_cfg = getattr(CONFIG, 'arc_y', CONFIG.arc_force)

        try:
            start_position = list(self.app.robot.rtdeReceive.getActualTCPPose())
            ref_pose = self.app.robot._calculateRefFramePose(start_position, ref_offset)
            direction_multiplier = DIRECTION_MAP[checked_button.direction]
            angle_deg = self.angle_input.value() * direction_multiplier
            angle_rad = math.radians(angle_deg)
            axis = 'y'
            end_pose = orbit_tcp_around_ref(start_position, ref_pose, angle_rad, axis)

            print(f"\n=== Arc (y) — Set Path ===")
            print(f"moveL around ref Y, null sensing. Direction: {checked_button.direction}")

            method_kwargs = {
                'start_position': start_position,
                'ref_pose': ref_pose,
                'end_pose': end_pose,
                'angle_rad': angle_rad,
                'axis': axis,
                'direction_multiplier': direction_multiplier,
                'force_baseline': 'null',
                'speed': self.speed_input.value(),
                'accel': self.accel_input.value(),
                'max_moment': self.max_moment_input.value(),
                'force_mode_z_limit': self.z_limit_input.value(),
                'force_mode_xz_limit': self.xz_limit_input.value(),
                'force_mode_damping': self.damping_input.value(),
                'force_mode_gain_scaling': self.gain_input.value(),
                'fx_tolerance': self.fx_tolerance_input.value(),
                'pos_tolerance': getattr(arc_cfg, 'pos_tolerance', 0.002),
                'rot_tolerance': getattr(arc_cfg, 'rot_tolerance', 0.02),
                'waypoint_count': int(self.waypoint_count_input.value()),
                'target_distance': self.target_distance_input.value(),
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
                func=arc_y_movel,
                **method_kwargs
            )
            self.app.async_motion_runner.movement_started.connect(lambda: print("Starting Arc (y) Set Path..."))
            self.app.async_motion_runner.movement_progress.connect(print)
            self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
            self.app.async_motion_runner.movement_completed.connect(on_completed)
            self.app.async_motion_runner.path_saved.connect(lambda _: self._update_button_states())
            self.app.async_motion_runner.start()

        except Exception as e:
            print(f"Error during Arc (y) Set Path: {e}")
            import traceback
            traceback.print_exc()
            self.reset_button_style()

    def _on_run_clicked(self):
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            print("Error: Select direction (LEFT or RIGHT)")
            return
        direction = checked_button.direction
        path_file = arc_y_get_path(direction)
        if not arc_y_path_exists(direction):
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
            self.app._disable_freedrive_for_movement("arc y traverse")

        self.run_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
        self.run_btn.setEnabled(False)
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setEnabled(False)

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
        self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
        self.app.async_motion_runner.movement_completed.connect(on_completed)
        self.app.async_motion_runner.start()
        self.app.stop_btn.setEnabled(True)


MotionWidget = ArcYWidget
