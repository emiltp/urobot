"""Freemove widget for collecting waypoints during freedrive mode."""

import time
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, QTimer

from src.ui import CollapsibleGroupBox
from src.motion_logger import getLogfilePath
from src.movements.waypoint_collector import (
    calculateWaypointsDistance, smoothWaypoints, interpolateWaypoints,
    smoothWaypointsByFlange, interpolateWaypointsByFlange, calculateFlangeDistance
)
from src.movements.async_motion_runner import AsyncMotionRunner
from src.movements import freemove
from config import defaults as CONFIG
from src.parameter_tooltips import PARAMETER_TOOLTIPS as TT


class FreemoveWidget(QWidget):
    """Widget for freemove waypoint collection and traversal."""
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        
        # Collection state
        self._collecting = False
        self._collection_timer: Optional[QTimer] = None
        self._waypoints_list = []
        self._timestamps_list = []
        self._start_time = None
        self._last_collected_position = None
        self._collection_threshold = CONFIG.tracking.tcp_threshold  # 1mm
        
        self._setup_ui()
        self._update_button_states()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # === Traverse Parameters ===
        self.traverse_group = CollapsibleGroupBox("Traverse Parameters (moveLPath)", expanded=False)
        
        # Traverse method selector - FIRST
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("moveLPath (smooth)", "moveLPath")
        self.method_combo.addItem("servoPath (precise)", "servoPath")
        self.method_combo.addItem("movePath (straight)", "movePath")
        self.method_combo.addItem("forceHybrid (Fz=0 + moveLPath)", "forceHybrid")
        self.method_combo.addItem("forceSpeedL (Fz=0 + speedL)", "forceSpeedL")
        self.method_combo.setToolTip(
            "moveLPath: Smooth path following with blend radius\n"
            "servoPath: Precise timing control with servo commands\n"
            "movePath: Single moveL to end (straight line)\n"
            "forceHybrid: Force mode Fz=0 compliance + moveLPath\n"
            "forceSpeedL: Force mode Fz=0 compliance + speedL control"
        )
        self.method_combo.currentIndexChanged.connect(self._on_traverse_method_changed)
        method_row.addWidget(self.method_combo)
        self.traverse_group.addLayout(method_row)
        
        # Common parameters: Speed and Acceleration
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed (m/s):"))
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.001, 1.0)
        self.speed_input.setDecimals(3)
        self.speed_input.setSingleStep(0.01)
        self.speed_input.setValue(CONFIG.traverse_movelpath.speed)
        self.speed_input.setMaximumWidth(80)
        self.speed_input.setToolTip(TT["speed"])
        speed_row.addWidget(self.speed_input)
        self.traverse_group.addLayout(speed_row)
        
        accel_row = QHBoxLayout()
        accel_row.addWidget(QLabel("Accel (m/s²):"))
        self.accel_input = QDoubleSpinBox()
        self.accel_input.setRange(0.01, 2.0)
        self.accel_input.setDecimals(2)
        self.accel_input.setSingleStep(0.1)
        self.accel_input.setValue(CONFIG.traverse_movelpath.acceleration)
        self.accel_input.setMaximumWidth(80)
        self.accel_input.setToolTip(TT["acceleration"])
        accel_row.addWidget(self.accel_input)
        self.traverse_group.addLayout(accel_row)
        
        # moveLPath-specific: Blend radius
        self.blend_widget = QWidget()
        blend_layout = QHBoxLayout(self.blend_widget)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        blend_layout.addWidget(QLabel("Blend (m):"))
        self.blend_input = QDoubleSpinBox()
        self.blend_input.setRange(0.0, 0.1)
        self.blend_input.setDecimals(3)
        self.blend_input.setSingleStep(0.005)
        self.blend_input.setValue(CONFIG.traverse_movelpath.blend)
        self.blend_input.setMaximumWidth(80)
        self.blend_input.setToolTip(TT["blend"])
        blend_layout.addWidget(self.blend_input)
        self.traverse_group.addWidget(self.blend_widget)
        
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
        
        self.traverse_group.addWidget(self.servo_params_group)
        self.servo_params_group.setVisible(False)  # Hidden by default (moveLPath selected)
        
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
        
        self.traverse_group.addLayout(tcp_process_layout)
        
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
        
        self.traverse_group.addLayout(flange_process_layout)
        
        # Auto return checkbox
        self.auto_return_cb = QCheckBox("Auto Return (backward traverse)")
        self.auto_return_cb.setChecked(True)
        self.auto_return_cb.setToolTip("Automatically return through waypoints after forward traverse")
        self.traverse_group.addWidget(self.auto_return_cb)
        
        layout.addWidget(self.traverse_group)
        
        # Log filename input
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log:"))
        self.log_filename_input = QLineEdit()
        self.log_filename_input.setPlaceholderText("Enter filename (no extension)")
        self.log_filename_input.textChanged.connect(self._on_log_filename_changed)
        log_layout.addWidget(self.log_filename_input)
        layout.addLayout(log_layout)
        
        # === Buttons ===
        button_layout = QHBoxLayout()
        
        # Set Path button
        self.set_path_btn = QPushButton("Set Path")
        self.set_path_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_has_path_style = "background-color: #1464A0; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn_collecting_style = "background-color: #FF9800; color: white; font-weight: bold; padding: 6px 16px;"
        self.set_path_btn.setStyleSheet(self.set_path_btn_disabled_style)
        self.set_path_btn.setCheckable(True)
        self.set_path_btn.setEnabled(False)
        self.set_path_btn.clicked.connect(self._on_set_path_toggled)
        button_layout.addWidget(self.set_path_btn)
        
        # Run button
        self.run_btn = QPushButton("Run")
        self.run_btn_default_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px;"
        self.run_btn_disabled_style = "background-color: #555555; color: #888888; font-weight: bold; padding: 6px 16px;"
        self.run_btn.setStyleSheet(self.run_btn_disabled_style)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self.run_btn)
        
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Add stretch
        layout.addStretch()
    
    def _update_button_states(self):
        """Update button states based on current state."""
        is_connected = self.app.connected if hasattr(self.app, 'connected') else False
        has_path = freemove.path_exists()
        has_log_filename = bool(self.log_filename_input.text().strip())
        
        # Set Path button
        if self._collecting:
            self.set_path_btn.setEnabled(True)
            self.set_path_btn.setStyleSheet(self.set_path_btn_collecting_style)
            self.set_path_btn.setText("Stop Collection")
        else:
            self.set_path_btn.setEnabled(is_connected)
            if has_path:
                self.set_path_btn.setStyleSheet(self.set_path_btn_has_path_style if is_connected else self.set_path_btn_disabled_style)
            else:
                self.set_path_btn.setStyleSheet(self.set_path_btn_default_style if is_connected else self.set_path_btn_disabled_style)
            self.set_path_btn.setText("Set Path")
            self.set_path_btn.setChecked(False)
        
        # Run button - enabled if connected, has path, and has log filename
        can_run = is_connected and has_path and has_log_filename and not self._collecting
        self.run_btn.setEnabled(can_run)
        self.run_btn.setStyleSheet(self.run_btn_default_style if can_run else self.run_btn_disabled_style)
        
        # Smooth and Interpolate buttons - enabled if path exists and not collecting
        path_buttons_enabled = has_path and not self._collecting
        self.smooth_tcp_btn.setEnabled(path_buttons_enabled)
        self.interpolate_tcp_btn.setEnabled(path_buttons_enabled)
        self.smooth_flange_btn.setEnabled(path_buttons_enabled)
        self.interpolate_flange_btn.setEnabled(path_buttons_enabled)
        
        # Update status with path info
        if has_path and not self._collecting:
            try:
                import numpy as np
                with np.load(freemove.get_path_filename()) as data:
                    waypoints = np.array(data['poses'])
                distance = calculateWaypointsDistance(waypoints)
                self.status_label.setText(f"Path: {len(waypoints)} waypoints ({distance*1000:.1f}mm)")
            except Exception:
                self.status_label.setText("Path exists")
        elif not self._collecting:
            self.status_label.setText("No path - click 'Set Path' to record")
        
        # Update parent path buttons
        if hasattr(self.app, '_update_motion_path_buttons'):
            self.app._update_motion_path_buttons()
    
    def _on_log_filename_changed(self, text):
        """Update Run button state when log filename changes."""
        self._update_button_states()
    
    def _on_traverse_method_changed(self, index):
        """Update traverse group title, show/hide method-specific parameters, and update defaults."""
        traverse_method = self.method_combo.currentData()
        method_names = {
            "moveLPath": "Traverse Parameters (moveLPath)",
            "servoPath": "Traverse Parameters (servoPath)",
            "movePath": "Traverse Parameters (movePath)",
            "forceHybrid": "Traverse Parameters (forceHybrid)",
            "forceSpeedL": "Traverse Parameters (forceSpeedL)"
        }
        self.traverse_group.setTitle(method_names.get(traverse_method, "Traverse Parameters"))
        
        # Show/hide method-specific parameters
        # moveLPath and forceHybrid show blend, servoPath shows servo params
        self.blend_widget.setVisible(traverse_method in ("moveLPath", "forceHybrid"))
        self.servo_params_group.setVisible(traverse_method == "servoPath")
        
        # Update speed/accel defaults based on selected method
        if traverse_method == "servoPath":
            self.speed_input.setValue(CONFIG.traverse_servopath.speed)
            self.accel_input.setValue(CONFIG.traverse_servopath.acceleration)
        elif traverse_method == "movePath":
            self.speed_input.setValue(CONFIG.traverse_movepath.speed)
            self.accel_input.setValue(CONFIG.traverse_movepath.acceleration)
        elif traverse_method == "forceSpeedL":
            self.speed_input.setValue(CONFIG.traverse_servopath.speed)
            self.accel_input.setValue(CONFIG.traverse_servopath.acceleration)
        else:  # moveLPath or forceHybrid
            self.speed_input.setValue(CONFIG.traverse_movelpath.speed)
            self.accel_input.setValue(CONFIG.traverse_movelpath.acceleration)
    
    def _process_path(self, process_func, process_name: str, distance_func=calculateWaypointsDistance):
        """Common path processing logic."""
        import numpy as np
        
        if not freemove.path_exists():
            self.status_label.setText(f"No path to {process_name.lower()}")
            return
        
        try:
            filepath = freemove.get_path_filename()
            with np.load(filepath) as data:
                waypoints = np.array(data['poses'])
                timestamps = np.array(data['timestamps'])

            original_count = len(waypoints)
            processed_waypoints, processed_timestamps = process_func(waypoints, timestamps)
            
            np.savez_compressed(filepath, poses=processed_waypoints, timestamps=processed_timestamps)
            
            distance = distance_func(processed_waypoints)
            self.status_label.setText(f"{process_name}: {len(processed_waypoints)} waypoints ({distance*1000:.1f}mm)")
            print(f"Path {process_name.lower()}: {original_count} -> {len(processed_waypoints)} waypoints")
            
            if hasattr(self.app, '_motion_path_visible') and self.app._motion_path_visible:
                if hasattr(self.app, 'visualize_waypoints'):
                    self.app.visualize_waypoints(filepath)
            
            if hasattr(self.app, '_update_motion_path_buttons'):
                self.app._update_motion_path_buttons()
                
        except Exception as e:
            self.status_label.setText(f"{process_name} failed: {str(e)}")
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
    
    def _on_set_path_toggled(self, checked: bool):
        """Handle Set Path button toggle."""
        if checked:
            self._start_collection()
        else:
            self._stop_collection()
    
    def _start_collection(self):
        """Start waypoint collection during freedrive."""
        if not self.app.connected or self.app.robot is None:
            self.set_path_btn.setChecked(False)
            self.status_label.setText("Not connected to robot")
            return
        
        # Start freedrive mode
        if not self.app.robot.startFreedrive():
            self.set_path_btn.setChecked(False)
            self.status_label.setText("Failed to start freedrive")
            return
        
        # Update Free Move button in main app to show ON state
        self.app.free_move_btn.setText("Free Move: ON")
        self.app.free_move_btn.setStyleSheet("background-color: green;")
        
        # Initialize collection state
        self._collecting = True
        self._waypoints_list = []
        self._timestamps_list = []
        self._start_time = time.time()
        self._last_collected_position = None
        
        # Update UI
        self._update_button_states()
        self.status_label.setText("Freedrive active - move robot to collect waypoints")
        
        # Enable stop button
        self.app.stop_btn.setEnabled(True)
        
        # Start collection timer (sample every 50ms)
        self._collection_timer = QTimer()
        self._collection_timer.timeout.connect(self._collect_waypoint)
        self._collection_timer.start(50)
        
        print("Freemove collection started - move robot manually")
    
    def _collect_waypoint(self):
        """Collect current pose as waypoint (called by timer)."""
        if not self._collecting or self.app.robot is None:
            return
        
        try:
            pose = self.app.robot.getTcpPose()
            if pose is None:
                return
            
            position = tuple(pose[:3])
            
            # Check if we've moved enough from last collected position
            if self._last_collected_position is not None:
                distance = sum((a - b) ** 2 for a, b in zip(position, self._last_collected_position)) ** 0.5
                if distance < self._collection_threshold:
                    return  # Haven't moved enough
            
            # Collect waypoint
            timestamp = time.time() - self._start_time
            self._waypoints_list.append(list(pose))
            self._timestamps_list.append(timestamp)
            self._last_collected_position = position
            
            # Update status
            count = len(self._waypoints_list)
            self.status_label.setText(f"Collecting: {count} waypoints")
            
        except Exception as e:
            print(f"Error collecting waypoint: {e}")
    
    def _stop_collection(self):
        """Stop waypoint collection and save."""
        # Stop timer
        if self._collection_timer is not None:
            self._collection_timer.stop()
            self._collection_timer = None
        
        # End freedrive mode
        if self.app.robot is not None:
            self.app.robot.stopFreedrive()
        
        # Update Free Move button in main app to show OFF state
        self.app.free_move_btn.setText("Free Move: OFF")
        self.app.free_move_btn.setStyleSheet("")
        
        # Save waypoints if we collected any
        if self._collecting and len(self._waypoints_list) > 0:
            import numpy as np
            import os
            
            waypoints = np.array(self._waypoints_list, dtype=np.float64)
            timestamps = np.array(self._timestamps_list, dtype=np.float64)
            
            filepath = freemove.get_path_filename()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, poses=waypoints, timestamps=timestamps)
            
            distance = calculateWaypointsDistance(waypoints)
            self.status_label.setText(
                f"Saved {len(self._waypoints_list)} waypoints ({distance*1000:.1f}mm)"
            )
            print(f"Freemove path saved: {len(self._waypoints_list)} waypoints, {distance*1000:.1f}mm")
        elif self._collecting:
            self.status_label.setText("No waypoints collected")
        
        self._collecting = False
        self._waypoints_list = []
        self._timestamps_list = []
        
        # Disable stop button
        self.app.stop_btn.setEnabled(False)
        
        # Update button states
        self._update_button_states()
    
    def stop_if_collecting(self) -> bool:
        """Stop collection if active. Returns True if collection was stopped."""
        if self._collecting:
            self._stop_collection()
            return True
        return False
    
    def _on_run_clicked(self):
        """Handle Run button click - start path traversal."""
        if not self.app.connected or self.app.robot is None:
            self.status_label.setText("Not connected")
            return
        
        if not freemove.path_exists():
            self.status_label.setText("No saved path")
            return
        
        # Check if already running
        if self.app.async_motion_runner is not None and self.app.async_motion_runner.isRunning():
            self.status_label.setText("Movement in progress")
            return
        
        # Get motion log file
        log_filename = self.log_filename_input.text().strip()
        motion_log_file = getLogfilePath(log_filename) if log_filename else None
        
        # Create async motion runner for traversal
        self.app.async_motion_runner = AsyncMotionRunner(
            mode=AsyncMotionRunner.MODE_TRAVERSE,
            robot=self.app.robot,
            path_file=freemove.get_path_filename(),
            speed=self.speed_input.value(),
            acceleration=self.accel_input.value(),
            blend=self.blend_input.value(),
            traverseMethod=self.method_combo.currentData(),
            autoReturn=self.auto_return_cb.isChecked(),
            motionLogFile=motion_log_file
        )
        
        # Connect signals
        self.app.async_motion_runner.movement_started.connect(self._on_movement_started)
        self.app.async_motion_runner.movement_progress.connect(self._on_movement_progress)
        self.app.async_motion_runner.movement_completed.connect(self._on_movement_completed)
        self.app.async_motion_runner.pose_updated.connect(lambda _: self.app.vtk_widget.GetRenderWindow().Render())
        
        # Start
        self.app.async_motion_runner.start()
        
        # Update UI
        self.run_btn.setEnabled(False)
        self.set_path_btn.setEnabled(False)
        self.app.stop_btn.setEnabled(True)
        self.status_label.setText("Starting traversal...")
    
    def _on_movement_started(self):
        """Handle movement started signal."""
        self.status_label.setText("Traversing path...")
    
    def _on_movement_progress(self, message: str):
        """Handle movement progress signal."""
        self.status_label.setText(message)
        print(f"Freemove: {message}")
    
    def _on_movement_completed(self, success: bool, message: str):
        """Handle movement completed signal."""
        self.status_label.setText(message)
        print(f"Freemove completed: {message}")
        
        # Re-enable UI
        self._update_button_states()
        self.app.stop_btn.setEnabled(False)
        
        # Hide waypoint visualization
        self._hide_path_visualization()
    
    def _hide_path_visualization(self):
        """Hide waypoint path visualization."""
        if hasattr(self.app, 'hide_waypoints_visualization'):
            self.app.hide_waypoints_visualization()
        if hasattr(self.app, '_motion_path_visible'):
            self.app._motion_path_visible = False
        if hasattr(self.app, 'motion_show_path_btn'):
            self.app.motion_show_path_btn.setText("○")
    
    def _update_status_from_saved_path(self):
        """Update UI based on whether a saved path exists."""
        self._update_button_states()
    
    def on_connected(self):
        """Called when robot connects."""
        self._update_button_states()
    
    def on_disconnected(self):
        """Called when robot disconnects."""
        self.stop_if_collecting()
        self._update_button_states()


MotionWidget = FreemoveWidget
