"""FlexionY movement widget for embedding in the main window."""

import os
import math
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QCheckBox, QButtonGroup, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt
from scipy.spatial.transform import Rotation

from src.ui import ArrowButton, CircleWidget, CollapsibleGroupBox
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from src.motion_logger import getLogfilePath

from src.movements.async_motion_runner import AsyncMotionRunner
from . import original as flexion_y_original
from . import hybrid as flexion_y_hybrid
from . import force as flexion_y_force
from src.movements.waypoint_collector import (
    calculateWaypointsDistance, smoothWaypoints, interpolateWaypoints,
    smoothWaypointsByFlange, interpolateWaypointsByFlange, calculateFlangeDistance
)
from config import defaults as CONFIG
from src.parameter_tooltips import PARAMETER_TOOLTIPS as TT

DIRECTION_MAP = {"left": -1, "right": 1}

# Method modules mapping
METHOD_MODULES = {
    "original": flexion_y_original,
    "hybrid": flexion_y_hybrid,
    "force": flexion_y_force,
}


class FlexionYWidget(QWidget):
    """Widget containing flexion-y movement controls (rotation about TCP y-axis, controlling Fx)."""
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.path_visible = False
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Info label - full width, centered
        info_label = QLabel("Rotates TCP around its own y-axis while maintaining zero force in TCP z-direction.")
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        layout.addWidget(info_label)
        
        # Direction selection - centered
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
        self.method_combo.addItem("Original", "original")
        self.method_combo.addItem("Hybrid (force mode + moveL)", "hybrid")
        self.method_combo.addItem("Force (force mode + speedL)", "force")
        self.method_combo.setCurrentIndex(2)  # Default to Force
        self.method_combo.setToolTip(
            "Original: Software-based Fz compensation with stop-adjust-restart\n"
            "Hybrid: Force mode for Fz=0 compliance with moveL trajectory\n"
            "Force: Force mode with real-time speedL control loop"
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)
        
        # Flexion_Y Movement Parameters (expanded by default)
        self.movement_group = CollapsibleGroupBox("Flexion (y) Movement Parameters", expanded=False)
        
        # Angle input
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Rotation Angle (deg):"))
        self.angle_input = QDoubleSpinBox()
        self.angle_input.setRange(0.0, 90.0)
        self.angle_input.setDecimals(1)
        self.angle_input.setValue(CONFIG.flexion_y.angle)
        self.angle_input.setSingleStep(5.0)
        self.angle_input.setMaximumWidth(80)
        self.angle_input.setToolTip(TT["angle"])
        angle_layout.addWidget(self.angle_input)
        self.movement_group.addLayout(angle_layout)
        
        # Speed input
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed (m/s):"))
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.001, 1.0)
        self.speed_input.setDecimals(4)
        self.speed_input.setValue(CONFIG.flexion_y.speed)
        self.speed_input.setSingleStep(0.01)
        self.speed_input.setMaximumWidth(80)
        self.speed_input.setToolTip(TT["speed"])
        speed_layout.addWidget(self.speed_input)
        self.movement_group.addLayout(speed_layout)
        
        # Acceleration input
        accel_layout = QHBoxLayout()
        accel_layout.addWidget(QLabel("Acceleration (m/s²):"))
        self.accel_input = QDoubleSpinBox()
        self.accel_input.setRange(0.001, 2.0)
        self.accel_input.setDecimals(4)
        self.accel_input.setValue(CONFIG.flexion_y.acceleration)
        self.accel_input.setSingleStep(0.01)
        self.accel_input.setMaximumWidth(80)
        self.accel_input.setToolTip(TT["acceleration"])
        accel_layout.addWidget(self.accel_input)
        self.movement_group.addLayout(accel_layout)
        
        # Max TCP Fx (common to all methods)
        fx_layout = QHBoxLayout()
        fx_layout.addWidget(QLabel("Max TCP Fx (N):"))
        self.fx_limit_input = QDoubleSpinBox()
        self.fx_limit_input.setRange(0.1, 100.0)
        self.fx_limit_input.setDecimals(2)
        self.fx_limit_input.setValue(CONFIG.flexion_y.force_limit_x)
        self.fx_limit_input.setSingleStep(0.1)
        self.fx_limit_input.setMaximumWidth(80)
        self.fx_limit_input.setToolTip(TT["force_limit"])
        fx_layout.addWidget(self.fx_limit_input)
        self.movement_group.addLayout(fx_layout)
        
        # Max Moment My (common to all methods) - directional stopping criteria
        flexion_cfg = getattr(CONFIG, 'flexion_y', None)
        arc_cfg = getattr(CONFIG, 'arc_y', CONFIG.arc_force)
        max_moment_default = getattr(flexion_cfg, 'max_moment', None) or getattr(arc_cfg, 'max_moment', 3.0)
        my_layout = QHBoxLayout()
        my_layout.addWidget(QLabel("Max Moment My (Nm):"))
        self.my_limit_input = QDoubleSpinBox()
        self.my_limit_input.setRange(0.1, 50.0)
        self.my_limit_input.setDecimals(2)
        self.my_limit_input.setValue(max_moment_default)
        self.my_limit_input.setSingleStep(0.1)
        self.my_limit_input.setMaximumWidth(80)
        self.my_limit_input.setToolTip(TT["arc_force_max_moment"])
        my_layout.addWidget(self.my_limit_input)
        self.movement_group.addLayout(my_layout)
        
        layout.addWidget(self.movement_group)
        
        # Collection Method Parameters - Original (software Fz compensation)
        self.original_params_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)
        
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Force Control Gain (m/N):"))
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(0.0001, 0.05)
        self.gain_input.setDecimals(4)
        self.gain_input.setValue(CONFIG.flexion_y_original.force_control_gain)
        self.gain_input.setSingleStep(0.0005)
        self.gain_input.setMaximumWidth(80)
        self.gain_input.setToolTip(TT["force_control_gain"])
        gain_layout.addWidget(self.gain_input)
        self.original_params_group.addLayout(gain_layout)
        
        deadband_layout = QHBoxLayout()
        deadband_layout.addWidget(QLabel("Force Deadband (N):"))
        self.deadband_input = QDoubleSpinBox()
        self.deadband_input.setRange(0.01, 1.0)
        self.deadband_input.setDecimals(2)
        self.deadband_input.setValue(CONFIG.flexion_y_original.force_deadband)
        self.deadband_input.setSingleStep(0.01)
        self.deadband_input.setMaximumWidth(80)
        self.deadband_input.setToolTip(TT["force_deadband"])
        deadband_layout.addWidget(self.deadband_input)
        self.original_params_group.addLayout(deadband_layout)
        
        max_adj_layout = QHBoxLayout()
        max_adj_layout.addWidget(QLabel("Max Adj per Step (m):"))
        self.max_adj_input = QDoubleSpinBox()
        self.max_adj_input.setRange(0.001, 0.02)
        self.max_adj_input.setDecimals(3)
        self.max_adj_input.setValue(CONFIG.flexion_y_original.max_adjustment_per_step)
        self.max_adj_input.setSingleStep(0.001)
        self.max_adj_input.setMaximumWidth(80)
        self.max_adj_input.setToolTip(TT["max_adj_per_step"])
        max_adj_layout.addWidget(self.max_adj_input)
        self.original_params_group.addLayout(max_adj_layout)
        
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Min Adj Interval (s):"))
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setRange(0.02, 1.0)
        self.interval_input.setDecimals(2)
        self.interval_input.setValue(CONFIG.flexion_y_original.min_adjustment_interval)
        self.interval_input.setSingleStep(0.02)
        self.interval_input.setMaximumWidth(80)
        self.interval_input.setToolTip(TT["min_adj_interval"])
        interval_layout.addWidget(self.interval_input)
        self.original_params_group.addLayout(interval_layout)
        
        layout.addWidget(self.original_params_group)
        self.original_params_group.setVisible(False)  # Hidden by default (Force selected)
        
        # Collection Method Parameters - Hybrid (forceMode + moveL)
        self.hybrid_params_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)
        
        hybrid_z_limit_layout = QHBoxLayout()
        hybrid_z_limit_layout.addWidget(QLabel("Force Mode Z Limit (m/s):"))
        self.hybrid_z_limit_input = QDoubleSpinBox()
        self.hybrid_z_limit_input.setRange(0.01, 0.2)
        self.hybrid_z_limit_input.setDecimals(3)
        self.hybrid_z_limit_input.setValue(CONFIG.flexion_y_hybrid.force_mode_z_limit)
        self.hybrid_z_limit_input.setSingleStep(0.01)
        self.hybrid_z_limit_input.setMaximumWidth(80)
        self.hybrid_z_limit_input.setToolTip(TT["force_mode_z_limit"])
        hybrid_z_limit_layout.addWidget(self.hybrid_z_limit_input)
        self.hybrid_params_group.addLayout(hybrid_z_limit_layout)
        
        hybrid_damping_layout = QHBoxLayout()
        hybrid_damping_layout.addWidget(QLabel("Force Mode Damping:"))
        self.hybrid_damping_input = QDoubleSpinBox()
        self.hybrid_damping_input.setRange(0.0, 1.0)
        self.hybrid_damping_input.setDecimals(2)
        self.hybrid_damping_input.setValue(CONFIG.flexion_y_hybrid.force_mode_damping)
        self.hybrid_damping_input.setSingleStep(0.05)
        self.hybrid_damping_input.setMaximumWidth(80)
        self.hybrid_damping_input.setToolTip(TT["force_mode_damping"])
        hybrid_damping_layout.addWidget(self.hybrid_damping_input)
        self.hybrid_params_group.addLayout(hybrid_damping_layout)
        
        layout.addWidget(self.hybrid_params_group)
        self.hybrid_params_group.setVisible(False)
        
        # Collection Method Parameters - Force (forceMode + speedL)
        self.force_params_group = CollapsibleGroupBox("Collection Method Parameters", expanded=False)
        
        force_z_limit_layout = QHBoxLayout()
        force_z_limit_layout.addWidget(QLabel("Force Mode Z Limit (m/s):"))
        self.force_z_limit_input = QDoubleSpinBox()
        self.force_z_limit_input.setRange(0.01, 0.2)
        self.force_z_limit_input.setDecimals(3)
        self.force_z_limit_input.setValue(CONFIG.flexion_y_force.force_mode_z_limit)
        self.force_z_limit_input.setSingleStep(0.01)
        self.force_z_limit_input.setMaximumWidth(80)
        self.force_z_limit_input.setToolTip(TT["force_mode_z_limit"])
        force_z_limit_layout.addWidget(self.force_z_limit_input)
        self.force_params_group.addLayout(force_z_limit_layout)
        
        force_damping_layout = QHBoxLayout()
        force_damping_layout.addWidget(QLabel("Force Mode Damping:"))
        self.force_damping_input = QDoubleSpinBox()
        self.force_damping_input.setRange(0.0, 1.0)
        self.force_damping_input.setDecimals(2)
        self.force_damping_input.setValue(CONFIG.flexion_y_force.force_mode_damping)
        self.force_damping_input.setSingleStep(0.05)
        self.force_damping_input.setMaximumWidth(80)
        self.force_damping_input.setToolTip(TT["force_mode_damping"])
        force_damping_layout.addWidget(self.force_damping_input)
        self.force_params_group.addLayout(force_damping_layout)
        
        force_gain_layout = QHBoxLayout()
        force_gain_layout.addWidget(QLabel("Force Mode Gain Scaling:"))
        self.force_gain_scaling_input = QDoubleSpinBox()
        self.force_gain_scaling_input.setRange(0.0, 2.0)
        self.force_gain_scaling_input.setDecimals(2)
        self.force_gain_scaling_input.setValue(CONFIG.flexion_y_force.force_mode_gain_scaling)
        self.force_gain_scaling_input.setSingleStep(0.1)
        self.force_gain_scaling_input.setMaximumWidth(80)
        self.force_gain_scaling_input.setToolTip(TT["force_mode_gain_scaling"])
        force_gain_layout.addWidget(self.force_gain_scaling_input)
        self.force_params_group.addLayout(force_gain_layout)
        
        control_dt_layout = QHBoxLayout()
        control_dt_layout.addWidget(QLabel("Control Loop dt (s):"))
        self.control_dt_input = QDoubleSpinBox()
        self.control_dt_input.setRange(0.002, 0.05)
        self.control_dt_input.setDecimals(3)
        self.control_dt_input.setValue(CONFIG.flexion_y_force.control_loop_dt)
        self.control_dt_input.setSingleStep(0.002)
        self.control_dt_input.setMaximumWidth(80)
        self.control_dt_input.setToolTip(TT["control_loop_dt"])
        control_dt_layout.addWidget(self.control_dt_input)
        self.force_params_group.addLayout(control_dt_layout)
        
        rot_speed_layout = QHBoxLayout()
        rot_speed_layout.addWidget(QLabel("Rotation Speed Factor:"))
        self.rot_speed_factor_input = QDoubleSpinBox()
        self.rot_speed_factor_input.setRange(1.0, 20.0)
        self.rot_speed_factor_input.setDecimals(1)
        self.rot_speed_factor_input.setValue(CONFIG.flexion_y_force.rotation_speed_factor)
        self.rot_speed_factor_input.setSingleStep(1.0)
        self.rot_speed_factor_input.setMaximumWidth(80)
        self.rot_speed_factor_input.setToolTip(TT["rotation_speed_factor"])
        rot_speed_layout.addWidget(self.rot_speed_factor_input)
        self.force_params_group.addLayout(rot_speed_layout)
        
        layout.addWidget(self.force_params_group)
        # Force params visible by default (Force method selected)
        
        # Path Replay Parameters (expanded by default)
        self.replay_group = CollapsibleGroupBox("Traverse Parameters (servoPath)", expanded=False)
        
        # Traverse method selector - FIRST
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.traverse_method_combo = QComboBox()
        self.traverse_method_combo.addItem("moveLPath (smooth)", "moveLPath")
        self.traverse_method_combo.addItem("servoPath (precise)", "servoPath")
        self.traverse_method_combo.addItem("movePath (straight)", "movePath")
        self.traverse_method_combo.setCurrentIndex(1)  # Default to servoPath
        self.traverse_method_combo.setToolTip(
            "moveLPath: Smooth path following with blend radius\n"
            "servoPath: Precise timing control with servo commands\n"
            "movePath: Single moveL to end (straight line)"
        )
        self.traverse_method_combo.currentIndexChanged.connect(self._on_traverse_method_changed)
        method_layout.addWidget(self.traverse_method_combo)
        self.replay_group.addLayout(method_layout)
        
        # Common parameters: Speed and Acceleration
        replay_speed_layout = QHBoxLayout()
        replay_speed_layout.addWidget(QLabel("Speed (m/s):"))
        self.replay_speed_input = QDoubleSpinBox()
        self.replay_speed_input.setRange(0.001, 1.0)
        self.replay_speed_input.setDecimals(3)
        self.replay_speed_input.setValue(CONFIG.traverse_servopath.speed)  # Default to servoPath
        self.replay_speed_input.setSingleStep(0.01)
        self.replay_speed_input.setMaximumWidth(80)
        self.replay_speed_input.setToolTip(TT["speed"])
        replay_speed_layout.addWidget(self.replay_speed_input)
        self.replay_group.addLayout(replay_speed_layout)
        
        accel_layout = QHBoxLayout()
        accel_layout.addWidget(QLabel("Accel (m/s²):"))
        self.replay_accel_input = QDoubleSpinBox()
        self.replay_accel_input.setRange(0.01, 2.0)
        self.replay_accel_input.setDecimals(2)
        self.replay_accel_input.setValue(CONFIG.traverse_servopath.acceleration)  # Default to servoPath
        self.replay_accel_input.setSingleStep(0.1)
        self.replay_accel_input.setMaximumWidth(80)
        self.replay_accel_input.setToolTip(TT["acceleration"])
        accel_layout.addWidget(self.replay_accel_input)
        self.replay_group.addLayout(accel_layout)
        
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
        self.blend_widget.setVisible(False)  # Hidden by default (servoPath selected)
        
        # servoPath-specific parameters in a collapsible group
        self.servo_params_group = CollapsibleGroupBox("Servo Parameters", expanded=False)
        
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Control dt (s):"))
        self.servo_dt_input = QDoubleSpinBox()
        self.servo_dt_input.setRange(0.002, 0.05)
        self.servo_dt_input.setDecimals(3)
        self.servo_dt_input.setValue(CONFIG.traverse_servopath.dt)
        self.servo_dt_input.setSingleStep(0.002)
        self.servo_dt_input.setMaximumWidth(80)
        self.servo_dt_input.setToolTip(TT["servo_dt"])
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
        self.servo_lookahead_input.setToolTip(TT["lookahead"])
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
        self.servo_gain_input.setToolTip(TT["servo_gain"])
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
        self.servo_ramp_input.setToolTip(TT["ramp_up"])
        ramp_layout.addWidget(self.servo_ramp_input)
        self.servo_params_group.addLayout(ramp_layout)
        
        self.replay_group.addWidget(self.servo_params_group)
        # Servo params visible by default (servoPath selected)
        
        # Path processing buttons: TCP and Flange rows
        # TCP row
        tcp_process_layout = QHBoxLayout()
        tcp_process_layout.addWidget(QLabel("TCP:"))
        
        self.smooth_tcp_btn = QPushButton("Smooth")
        self.smooth_tcp_btn.setToolTip("Smooth waypoints based on TCP positions")
        self.smooth_tcp_btn.setEnabled(False)
        self.smooth_tcp_btn.clicked.connect(self._on_smooth_tcp_clicked)
        tcp_process_layout.addWidget(self.smooth_tcp_btn)
        
        self.interpolate_tcp_btn = QPushButton("Interpolate")
        self.interpolate_tcp_btn.setToolTip("Resample to uniform TCP distance intervals")
        self.interpolate_tcp_btn.setEnabled(False)
        self.interpolate_tcp_btn.clicked.connect(self._on_interpolate_tcp_clicked)
        tcp_process_layout.addWidget(self.interpolate_tcp_btn)
        
        self.replay_group.addLayout(tcp_process_layout)
        
        # Flange row
        flange_process_layout = QHBoxLayout()
        flange_process_layout.addWidget(QLabel("Flange:"))
        
        self.smooth_flange_btn = QPushButton("Smooth")
        self.smooth_flange_btn.setToolTip("Smooth waypoints based on flange positions")
        self.smooth_flange_btn.setEnabled(False)
        self.smooth_flange_btn.clicked.connect(self._on_smooth_flange_clicked)
        flange_process_layout.addWidget(self.smooth_flange_btn)
        
        self.interpolate_flange_btn = QPushButton("Interpolate")
        self.interpolate_flange_btn.setToolTip("Resample to uniform flange distance intervals")
        self.interpolate_flange_btn.setEnabled(False)
        self.interpolate_flange_btn.clicked.connect(self._on_interpolate_flange_clicked)
        flange_process_layout.addWidget(self.interpolate_flange_btn)
        
        self.replay_group.addLayout(flange_process_layout)
        
        self.end_force_control_checkbox = QCheckBox("Enable End Force Control")
        self.end_force_control_checkbox.setChecked(True)
        self.replay_group.addWidget(self.end_force_control_checkbox)
        
        layout.addWidget(self.replay_group)
        
        # Log filename input
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
        self.set_path_btn.clicked.connect(self.collectWaypoints)
        button_layout.addWidget(self.set_path_btn)
        
        self.run_btn = QPushButton("Run")
        self.run_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.run_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.traverseWaypoints)
        button_layout.addWidget(self.run_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def _get_current_method_module(self):
        """Get the currently selected method's module."""
        method = self.method_combo.currentData()
        return METHOD_MODULES.get(method, flexion_y_original)
    
    def _on_method_changed(self, index):
        """Update button states and show/hide method-specific parameters."""
        method = self.method_combo.currentData()
        
        # Show/hide method-specific parameter groups
        self.original_params_group.setVisible(method == "original")
        self.hybrid_params_group.setVisible(method == "hybrid")
        self.force_params_group.setVisible(method == "force")
        
        self._on_direction_changed(None, None)
    
    def _on_traverse_method_changed(self, index):
        """Update traverse group title, show/hide method-specific parameters, and update defaults."""
        traverse_method = self.traverse_method_combo.currentData()
        method_names = {
            "moveLPath": "Traverse Parameters (moveLPath)",
            "servoPath": "Traverse Parameters (servoPath)",
            "movePath": "Traverse Parameters (movePath)"
        }
        self.replay_group.setTitle(method_names.get(traverse_method, "Traverse Parameters"))
        
        # Show/hide method-specific parameters
        self.blend_widget.setVisible(traverse_method == "moveLPath")
        self.servo_params_group.setVisible(traverse_method == "servoPath")
        
        # Update speed/accel defaults based on selected method
        if traverse_method == "servoPath":
            self.replay_speed_input.setValue(CONFIG.traverse_servopath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_servopath.acceleration)
        elif traverse_method == "movePath":
            self.replay_speed_input.setValue(CONFIG.traverse_movepath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_movepath.acceleration)
        else:  # moveLPath
            self.replay_speed_input.setValue(CONFIG.traverse_movelpath.speed)
            self.replay_accel_input.setValue(CONFIG.traverse_movelpath.acceleration)
    
    def _on_direction_changed(self, button, checked):
        """Update button states based on direction selection."""
        checked_button = self.direction_group.checkedButton()
        has_direction = checked_button is not None
        has_log_filename = bool(self.log_filename_input.text().strip())
        
        self.set_path_btn.setEnabled(has_direction)
        
        if has_direction:
            direction = checked_button.direction
            method_module = self._get_current_method_module()
            path_file_exists = method_module.path_exists(direction)
            
            if path_file_exists:
                self.set_path_btn.setStyleSheet(self.set_path_btn_has_path_style)
                # Run button is enabled only if log filename is provided
                can_run = has_log_filename
                self.run_btn.setEnabled(can_run)
                self.run_btn.setStyleSheet(self.run_btn_default_style if can_run else self.run_btn_disabled_style)
                # Smooth and Interpolate buttons enabled when path exists
                self.smooth_tcp_btn.setEnabled(True)
                self.interpolate_tcp_btn.setEnabled(True)
                self.smooth_flange_btn.setEnabled(True)
                self.interpolate_flange_btn.setEnabled(True)
                
                # Update waypoint visualization only when direction changed (not on log filename keystroke)
                if button is not None and hasattr(self.app, '_motion_path_visible') and self.app._motion_path_visible:
                    path_file = method_module.get_path_filename(direction)
                    if hasattr(self.app, 'visualize_waypoints'):
                        self.app.visualize_waypoints(path_file)
            else:
                self.set_path_btn.setStyleSheet(self.set_path_btn_default_style)
                self.run_btn.setEnabled(False)
                self.run_btn.setStyleSheet(self.run_btn_disabled_style)
                self.smooth_tcp_btn.setEnabled(False)
                self.interpolate_tcp_btn.setEnabled(False)
                self.smooth_flange_btn.setEnabled(False)
                self.interpolate_flange_btn.setEnabled(False)
                self._hide_path_visualization()
        else:
            self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
            self.run_btn.setEnabled(False)
            self.run_btn.setStyleSheet(self.run_btn_disabled_style)
            self.smooth_tcp_btn.setEnabled(False)
            self.interpolate_tcp_btn.setEnabled(False)
            self.smooth_flange_btn.setEnabled(False)
            self.interpolate_flange_btn.setEnabled(False)
            self._hide_path_visualization()
        
        # Notify parent to update path buttons
        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()
    
    def _on_log_filename_changed(self, text):
        """Update Run button state when log filename changes."""
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
        self.smooth_tcp_btn.setEnabled(False)
        self.interpolate_tcp_btn.setEnabled(False)
        self.smooth_flange_btn.setEnabled(False)
        self.interpolate_flange_btn.setEnabled(False)
        self._hide_path_visualization()
        
        # Clear log filename
        self.log_filename_input.clear()
        
        # Notify parent to update path buttons
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
    
    def collectWaypoints(self):
        """Execute flexion-y movement and save path."""
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return
        
        direction = checked_button.direction
        angle_deg = self.angle_input.value()
        speed = self.speed_input.value()
        accel = self.accel_input.value()
        force_limit_x = self.fx_limit_input.value()
        max_moment = self.my_limit_input.value()
        direction_multiplier = DIRECTION_MAP[direction]
        method_module = self._get_current_method_module()
        method = self.method_combo.currentData()
        path_file = method_module.get_path_filename(direction)
        
        # Build method-specific kwargs
        method_kwargs = {}
        if method == "original":
            method_kwargs = {
                'force_control_gain': self.gain_input.value(),
                'force_deadband': self.deadband_input.value(),
                'max_adjustment_per_step': self.max_adj_input.value(),
                'min_adjustment_interval': self.interval_input.value(),
            }
        elif method == "hybrid":
            method_kwargs = {
                'force_mode_z_limit': self.hybrid_z_limit_input.value(),
                'force_mode_damping': self.hybrid_damping_input.value(),
            }
        elif method == "force":
            method_kwargs = {
                'force_mode_z_limit': self.force_z_limit_input.value(),
                'force_mode_damping': self.force_damping_input.value(),
                'force_mode_gain_scaling': self.force_gain_scaling_input.value(),
                'control_loop_dt': self.control_dt_input.value(),
                'rotation_speed_factor': self.rot_speed_factor_input.value(),
            }
        
        # Visual feedback
        self.set_path_btn.setStyleSheet("background-color: orange; font-weight: bold; color: black; padding: 6px 16px;")
        self.set_path_btn.setEnabled(False)
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self.run_btn.setEnabled(False)
        
        # Check connection
        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            self.reset_button_style()
            return
        
        # Check if a movement is already in progress
        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            print("Warning: A movement is already in progress. Please wait for it to complete.")
            self.reset_button_style()
            return
        
        # Disable freedrive mode if active
        if hasattr(self.app, '_disable_freedrive_for_movement'):
            self.app._disable_freedrive_for_movement("flexion-y movement")
        
        try:
            # Get current TCP pose
            start_position = list(self.app.robot.rtdeReceive.getActualTCPPose())
            print(f"\n=== Flexion-Y Movement (Set Path) ===")
            print(f"Starting TCP pose: [{start_position[0]:.4f}, {start_position[1]:.4f}, {start_position[2]:.4f}]")
            
            # Calculate new pose (rotation about TCP y-axis)
            new_pose = rotate_pose_around_tcp_y(start_position, angle_deg * direction_multiplier)
            new_pose[1] = start_position[1]  # Keep y position constant (xz-plane constraint)
            
            print(f"Rotating TCP frame {angle_deg * direction_multiplier}° around TCP y-axis...")
            print(f"Saving path to: {path_file}")
            
            # Create and start movement worker thread
            def on_movement_started():
                print("Nulling force sensor...")
                self.app.stop_btn.setEnabled(True)
            
            def on_movement_progress(message):
                print(message)
            
            def on_pose_updated(pose):
                self.app.vtk_widget.GetRenderWindow().Render()
            
            def on_movement_completed(success, message):
                print(message)
                self.app.stop_btn.setEnabled(False)
                # Update widget button states
                if success:
                    self.on_path_saved(direction)
                self.reset_button_style()
                # Clean up worker
                self.app.async_motion_runner = None
                        
            # Get the main module for the selected method
            method_main = method_module
            
            self.app.async_motion_runner = AsyncMotionRunner(
                mode=AsyncMotionRunner.MODE_COLLECT,
                robot=self.app.robot,
                func=method_main,
                start_position=start_position,
                new_pose=new_pose,
                speed=speed,
                accel=accel,
                force_limit_x=force_limit_x,
                max_moment=max_moment,
                direction_multiplier=direction_multiplier,
                path_file=path_file,
                collection_method=method,  # "original", "hybrid", or "force"
                traverseMethod=self.traverse_method_combo.currentData(),
                blend=self.replay_blend_input.value(),
                retrace_speed=self.replay_speed_input.value(),
                retrace_acceleration=self.replay_accel_input.value(),
                **method_kwargs
            )
            self.app.async_motion_runner.movement_started.connect(on_movement_started)
            self.app.async_motion_runner.movement_progress.connect(on_movement_progress)
            self.app.async_motion_runner.pose_updated.connect(on_pose_updated)
            self.app.async_motion_runner.movement_completed.connect(on_movement_completed)
            self.app.async_motion_runner.start()
            
        except Exception as e:
            print(f"Error during flexion-y set path: {e}")
            import traceback
            traceback.print_exc()
            self.reset_button_style()
    
    def traverseWaypoints(self):
        """Execute path replay using AsyncMotionRunner."""
        
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return
        
        direction = checked_button.direction
        method_module = self._get_current_method_module()
        path_file = method_module.get_path_filename(direction)
        
        # Check path exists
        if not method_module.path_exists(direction):
            print(f"No path found for {direction} direction")
            return
        
        # Get log filename
        log_filename = self.log_filename_input.text().strip()
        if not log_filename:
            print("Error: Log filename is required")
            return
        
        # Check app connection
        if not self.app.connected or self.app.robot.rtdeControl is None or self.app.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            return
        
        # Check if a movement is already in progress
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
        force_limit = self.fx_limit_input.value()
        
        # Disable freedrive mode if active
        self.app._disable_freedrive_for_movement("path replay")
        
        print(f"\n=== Path Replay (Flexion Y) ===")
        print(f"Replay speed: {replay_speed} m/s")
        print(f"Force limit control: {'enabled' if enable_end_force else 'disabled'}")
        if enable_end_force:
            print(f"Force limit Fx: {force_limit} N")
        
        widget = self
        
        def on_completed(success, message):
            print(message)
            self.app.stop_btn.setEnabled(False)
            widget.reset_button_style()
            self.app.async_motion_runner = None
        
        # Create and start AsyncMotionRunner
        self.app.async_motion_runner = AsyncMotionRunner(
            mode=AsyncMotionRunner.MODE_TRAVERSE,
            robot=self.app.robot,
            path_file=path_file,
            speed=replay_speed,
            acceleration=self.replay_accel_input.value(),
            blend=self.replay_blend_input.value(),
            traverseMethod=self.traverse_method_combo.currentData(),
            enableForceControl=enable_end_force,
            forceLimit=force_limit,
            forceAxis='x',  # Monitor Fx for flexion Y
            direction=direction,
            autoReturn=True,
            motionLogFile=getLogfilePath(log_filename)
        )
        self.app.async_motion_runner.movement_progress.connect(print)
        self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
        self.app.async_motion_runner.movement_completed.connect(on_completed)
        self.app.async_motion_runner.start()
        self.app.stop_btn.setEnabled(True)
    
    def reset_button_style(self):
        """Reset button styles after movement."""
        self.clear_direction()
    
    def on_path_saved(self, direction: str):
        """Called when path is saved."""
        checked_button = self.direction_group.checkedButton()
        if checked_button and checked_button.direction == direction:
            self._on_direction_changed(None, None)
    
    def _process_path(self, process_func, process_name: str, distance_func=calculateWaypointsDistance):
        """Common path processing logic."""
        import numpy as np
        
        checked_button = self.direction_group.checkedButton()
        if not checked_button:
            return
        
        direction = checked_button.direction
        method_module = self._get_current_method_module()
        if not method_module.path_exists(direction):
            return
        
        try:
            filepath = method_module.get_path_filename(direction)
            with np.load(filepath) as data:
                waypoints = np.array(data['poses'])
                timestamps = np.array(data['timestamps'])

            original_count = len(waypoints)
            processed_waypoints, processed_timestamps = process_func(waypoints, timestamps)
            
            np.savez_compressed(filepath, poses=processed_waypoints, timestamps=processed_timestamps)
            
            distance = distance_func(processed_waypoints)
            print(f"Path {process_name.lower()}: {original_count} -> {len(processed_waypoints)} waypoints ({distance*1000:.1f}mm)")
            
            if hasattr(self.app, '_motion_path_visible') and self.app._motion_path_visible:
                if hasattr(self.app, 'visualize_waypoints'):
                    self.app.visualize_waypoints(filepath)
                    
        except Exception as e:
            print(f"Error {process_name.lower()} path: {e}")
    
    def _on_smooth_tcp_clicked(self):
        """Apply TCP-based smoothing."""
        self._process_path(smoothWaypoints, "Smoothed (TCP)", calculateWaypointsDistance)
    
    def _on_interpolate_tcp_clicked(self):
        """Apply TCP-based interpolation."""
        self._process_path(interpolateWaypoints, "Interpolated (TCP)", calculateWaypointsDistance)
    
    def _on_smooth_flange_clicked(self):
        """Apply flange-based smoothing."""
        self._process_path(smoothWaypointsByFlange, "Smoothed (Flange)", calculateFlangeDistance)
    
    def _on_interpolate_flange_clicked(self):
        """Apply flange-based interpolation."""
        self._process_path(interpolateWaypointsByFlange, "Interpolated (Flange)", calculateFlangeDistance)


def rotate_pose_around_tcp_y(pose: List[float], angle_deg: float) -> List[float]:
    """
    Rotate the TCP frame around its own (local) y-axis while keeping position fixed.

    Args:
        pose: Current TCP pose [x, y, z, rx, ry, rz]
        angle_deg: Rotation angle in degrees (positive = counterclockwise about TCP y)

    Returns:
        New TCP pose [x, y, z, rx_new, ry_new, rz_new] with rotated orientation
    """
    x, y, z = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]

    # Current orientation
    R_current = axis_angle_to_rotation_matrix(rx, ry, rz)

    # Local TCP y rotation
    angle_rad = math.radians(angle_deg)
    R_tcp_y = Rotation.from_euler('y', angle_rad, degrees=False).as_matrix()

    # Apply local rotation: R_new = R_current @ R_tcp_y
    R_new = R_current @ R_tcp_y

    # Back to axis-angle
    rx_new, ry_new, rz_new = rotation_matrix_to_axis_angle(R_new)

    return [x, y, z, rx_new, ry_new, rz_new]


MotionWidget = FlexionYWidget
