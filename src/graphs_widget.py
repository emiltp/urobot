"""Real-time graphs widget for visualizing robot TCP velocity and cumulative distances."""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QFrame, QPushButton, QComboBox, QGroupBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: pyqtgraph not available. Install with: pip install pyqtgraph")


# Color scheme for the plots
COLORS = {
    # Velocity - Translation (warm/cool distinct)
    'tx': '#FF3333',  # Bright Red
    'ty': '#33FF33',  # Bright Green
    'tz': '#3399FF',  # Bright Blue
    # Velocity - Rotation (orange/purple/cyan)
    'rx': '#FF9900',  # Orange
    'ry': '#CC33FF',  # Magenta/Purple
    'rz': '#00FFFF',  # Cyan
    # Force (same as translation - warm/cool)
    'Fx': '#FF2222',  # Red
    'Fy': '#22DD22',  # Green
    'Fz': '#1144FF',  # Blue
    # Torque (same as rotation)
    'Tx': '#FF9900',  # Orange
    'Ty': '#CC33FF',  # Magenta/Purple
    'Tz': '#00FFFF',  # Cyan
    # Magnitudes
    'force_mag': '#FFFFFF',   # White
    'torque_mag': '#FFCC00',  # Gold
    'vel_mag': '#00FF00',     # Bright green
}

WINDOW_SECONDS_MAX = 30.0  # Maximum rolling window size
WINDOW_SECONDS_MIN = 5.0   # Minimum window size to start with
UPDATE_RATE_HZ = 15        # Graph update rate (reduced for performance)
MAX_DISPLAY_POINTS = 500   # Maximum points to display per curve (downsampling)


class RealTimeGraphsWidget(QWidget):
    """Widget displaying real-time velocity and cumulative distance graphs."""
    
    # Signal emitted when force zeroing button is clicked
    zero_force_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if not PYQTGRAPH_AVAILABLE:
            self._setup_fallback_ui()
            return
        
        # Configure pyqtgraph for performance
        # Disable antialiasing for better performance (major improvement)
        # OpenGL disabled - causes errors on some systems (context returns None)
        pg.setConfigOptions(
            antialias=False,  # Disabled for performance
            background='#1E2630',
            foreground='#CCCCCC',
            useOpenGL=False,  # Disabled - causes errors on macOS
        )
        
        # Data buffers (time, values)
        self._max_samples = int(WINDOW_SECONDS_MAX * 100)  # Assume max 100 Hz sampling
        self._times = deque(maxlen=self._max_samples)
        self._velocities = {
            'tx': deque(maxlen=self._max_samples),
            'ty': deque(maxlen=self._max_samples),
            'tz': deque(maxlen=self._max_samples),
            'rx': deque(maxlen=self._max_samples),
            'ry': deque(maxlen=self._max_samples),
            'rz': deque(maxlen=self._max_samples),
        }
        # Force/torque data [Fx, Fy, Fz, Tx, Ty, Tz]
        self._forces = {
            'Fx': deque(maxlen=self._max_samples),
            'Fy': deque(maxlen=self._max_samples),
            'Fz': deque(maxlen=self._max_samples),
            'Tx': deque(maxlen=self._max_samples),
            'Ty': deque(maxlen=self._max_samples),
            'Tz': deque(maxlen=self._max_samples),
        }
        # Force magnitudes
        self._force_magnitudes = {
            'force': deque(maxlen=self._max_samples),   # sqrt(Fx² + Fy² + Fz²)
            'torque': deque(maxlen=self._max_samples),  # sqrt(Tx² + Ty² + Tz²)
        }
        # Velocity magnitudes (for stats display)
        self._velocity_magnitudes = {
            'trans': deque(maxlen=self._max_samples),  # sqrt(tx² + ty² + tz²)
            'rot': deque(maxlen=self._max_samples),    # sqrt(rx² + ry² + rz²) * weight
            'total': deque(maxlen=self._max_samples),  # sqrt(trans² + rot²)
        }
        
        # State tracking
        self._last_pose: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._rotation_weight = 0.1  # Will be updated from TCP offset
        self._selected_window_size = WINDOW_SECONDS_MIN  # Current window size selection
        
        # Performance optimization: track update cycles for less frequent operations
        self._update_cycle = 0
        self._autoscale_interval = 1  # Auto-scale every update for responsiveness
        
        # Build UI
        self._setup_ui()
        
        # Update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_plots)
        self._update_timer.start(int(1000 / UPDATE_RATE_HZ))
    
    def _setup_fallback_ui(self):
        """Setup fallback UI when pyqtgraph is not available."""
        layout = QVBoxLayout(self)
        label = QLabel("Install pyqtgraph for real-time graphs:\npip install pyqtgraph")
        label.setStyleSheet("color: #888; padding: 20px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
    
    def _setup_ui(self):
        """Setup the graphs UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create group box container
        group_box = QGroupBox("Real-Time Graphs")
        layout = QVBoxLayout(group_box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header with controls
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addStretch()
        
        # Window size selector
        window_label = QLabel("Window:")
        window_label.setStyleSheet("color: #888; font-size: 10px;")
        header.addWidget(window_label)
        
        self._window_combo = QComboBox()
        self._window_combo.addItems([" 5s", "10s", "15s", "20s", "30s"])
        self._window_combo.setCurrentIndex(0)  # Select first item (5s)
        self._window_combo.setFixedWidth(65)
        self._window_combo.setStyleSheet("QComboBox QAbstractItemView { min-width: 60px; }")
        self._window_combo.currentIndexChanged.connect(self._on_window_size_changed)
        header.addWidget(self._window_combo)
        
        # Line width selector
        line_label = QLabel("Line:")
        line_label.setStyleSheet("color: #888; font-size: 10px;")
        header.addWidget(line_label)
        
        self._line_width = 2.0  # Default line width
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 10.0)
        self._line_width_spin.setDecimals(1)
        self._line_width_spin.setValue(self._line_width)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setFixedWidth(55)
        self._line_width_spin.setToolTip("Graph line width")
        self._line_width_spin.valueChanged.connect(self._on_line_width_changed)
        header.addWidget(self._line_width_spin)
        
        # Pause/Play button
        self._paused = False
        self._pause_btn = QPushButton("\u23F8")  # ⏸ pause symbol (playing state)
        self._pause_btn.setFixedSize(23, 23)
        self._pause_btn.setToolTip("Pause graph updates")
        self._pause_btn.clicked.connect(self._toggle_pause)
        header.addWidget(self._pause_btn)
        
        # Force zeroing button
        self._zero_force_btn = QPushButton("\u2296")  # ⊖ circled minus symbol
        self._zero_force_btn.setFixedSize(23, 23)
        self._zero_force_btn.setToolTip("Zero force/torque sensor")
        self._zero_force_btn.clicked.connect(self._on_zero_force_clicked)
        header.addWidget(self._zero_force_btn)
        
        # Reset button (matches connection button style)
        self._reset_btn = QPushButton("↻")
        self._reset_btn.setFixedSize(23, 23)
        self._reset_btn.setToolTip("Reset graphs")
        self._reset_btn.clicked.connect(self.reset)
        header.addWidget(self._reset_btn)
        
        layout.addLayout(header)
        
        mono_font = QFont("Courier", 11)
        
        # === Velocity Plot with dual Y-axes ===
        self._vel_graphics = pg.GraphicsLayoutWidget()
        self._vel_plot = self._vel_graphics.addPlot(row=0, col=0)
        self._vel_plot.setLabel('left', 'Linear velocity', units='m/s', color=COLORS['tx'])
        self._vel_plot.showGrid(x=True, y=True, alpha=0.3)
        self._vel_plot.setXRange(-WINDOW_SECONDS_MIN, 0, padding=0)
        self._vel_plot.setYRange(-0.1, 0.1, padding=0)
        # Disable mouse interaction for better performance
        self._vel_plot.setMouseEnabled(x=False, y=False)
        self._vel_plot.hideButtons()
        self._vel_plot.setClipToView(True)  # Only render visible data
        self._vel_plot.disableAutoRange()   # We control the range manually
        
        # Left legend for translation (top-left)
        self._vel_legend_left = self._vel_plot.addLegend(offset=(10, 5), labelTextSize='9pt')
        
        # Create secondary Y-axis for rotation (right side)
        self._vel_viewbox_rot = pg.ViewBox()
        self._vel_plot.scene().addItem(self._vel_viewbox_rot)
        self._vel_plot.getAxis('right').linkToView(self._vel_viewbox_rot)
        self._vel_viewbox_rot.setXLink(self._vel_plot)
        self._vel_viewbox_rot.disableAutoRange()  # Manual Y-range control
        self._vel_viewbox_rot.setYRange(-0.1, 0.1)  # Initial range
        self._vel_plot.getAxis('right').setLabel('Angular velocity', units='rad/s', color=COLORS['rx'])
        self._vel_plot.showAxis('right')
        
        # Right legend for rotation (inside plot, top-right area)
        self._vel_legend_right = pg.LegendItem(labelTextSize='9pt')
        self._vel_legend_right.setParentItem(self._vel_plot.vb)
        self._vel_legend_right.anchor((1, 0), (1, 0), offset=(-5, 5))
        
        # Velocity curves - translation on left axis
        self._vel_curves = {}
        for key, color in [('tx', COLORS['tx']), ('ty', COLORS['ty']), ('tz', COLORS['tz'])]:
            self._vel_curves[key] = self._vel_plot.plot(
                [], [], pen=pg.mkPen(color, width=self._line_width), name=key
            )
        
        # Velocity curves - rotation on right axis (add to right legend)
        for key, color in [('rx', COLORS['rx']), ('ry', COLORS['ry']), ('rz', COLORS['rz'])]:
            curve = pg.PlotCurveItem([], [], pen=pg.mkPen(color, width=self._line_width), name=key)
            self._vel_viewbox_rot.addItem(curve)
            self._vel_curves[key] = curve
            self._vel_legend_right.addItem(curve, key)
        
        # Connect viewbox resize
        self._vel_plot.vb.sigResized.connect(self._update_vel_viewbox)
        
        layout.addWidget(self._vel_graphics, stretch=1)  # Expand to fill
        
        # Velocity stats (under velocity graph)
        vel_stats_frame = QFrame()
        vel_stats_frame.setStyleSheet("background: #252F3A; border-radius: 3px;")
        vel_stats_frame.setMaximumHeight(32)
        vel_stats_layout = QHBoxLayout(vel_stats_frame)
        vel_stats_layout.setContentsMargins(6, 2, 6, 2)
        vel_stats_layout.setSpacing(12)
        
        vel_stats_layout.addStretch()
        
        self._trans_vel_label = QLabel("|T|: 0.0 mm/s")
        self._trans_vel_label.setFont(mono_font)
        self._trans_vel_label.setStyleSheet(f"color: {COLORS['tx']};")
        vel_stats_layout.addWidget(self._trans_vel_label)
        
        self._rot_vel_label = QLabel("|R|: 0.0 rad/s")
        self._rot_vel_label.setFont(mono_font)
        self._rot_vel_label.setStyleSheet(f"color: {COLORS['rx']};")
        vel_stats_layout.addWidget(self._rot_vel_label)
        
        self._total_vel_label = QLabel("|∑|: 0.0 mm/s")
        self._total_vel_label.setFont(mono_font)
        self._total_vel_label.setStyleSheet(f"color: {COLORS['vel_mag']};")
        vel_stats_layout.addWidget(self._total_vel_label)
        
        vel_stats_layout.addStretch()
        layout.addWidget(vel_stats_frame)
        
        # Vertical spacing between graphs
        layout.addSpacing(8)
        
        # === TCP Force Plot with dual Y-axes ===
        self._force_graphics = pg.GraphicsLayoutWidget()
        self._force_plot = self._force_graphics.addPlot(row=0, col=0)
        self._force_plot.setLabel('left', 'Force', units='N', color=COLORS['Fx'])
        self._force_plot.showGrid(x=True, y=True, alpha=0.3)
        self._force_plot.setXRange(-WINDOW_SECONDS_MIN, 0, padding=0)
        self._force_plot.setYRange(-50, 50, padding=0)
        # Disable mouse interaction for better performance
        self._force_plot.setMouseEnabled(x=False, y=False)
        self._force_plot.hideButtons()
        self._force_plot.setClipToView(True)  # Only render visible data
        self._force_plot.disableAutoRange()   # We control the range manually
        
        # Left legend for force (top-left)
        self._force_legend_left = self._force_plot.addLegend(offset=(10, 5), labelTextSize='9pt')
        
        # Create secondary Y-axis for torque (right side)
        self._force_viewbox_torque = pg.ViewBox()
        self._force_plot.scene().addItem(self._force_viewbox_torque)
        self._force_plot.getAxis('right').linkToView(self._force_viewbox_torque)
        self._force_viewbox_torque.setXLink(self._force_plot)
        self._force_viewbox_torque.disableAutoRange()  # Manual Y-range control
        self._force_viewbox_torque.setYRange(-1, 1)  # Initial range
        self._force_plot.getAxis('right').setLabel('Torque', units='Nm', color=COLORS['Tx'])
        self._force_plot.showAxis('right')
        
        # Right legend for torque (inside plot, top-right area)
        self._force_legend_right = pg.LegendItem(labelTextSize='9pt')
        self._force_legend_right.setParentItem(self._force_plot.vb)
        self._force_legend_right.anchor((1, 0), (1, 0), offset=(-5, 5))
        
        # Force curves - forces on left axis
        self._force_curves = {}
        for key, color in [('Fx', COLORS['Fx']), ('Fy', COLORS['Fy']), ('Fz', COLORS['Fz'])]:
            self._force_curves[key] = self._force_plot.plot(
                [], [], pen=pg.mkPen(color, width=self._line_width), name=key
            )
        
        # Force curves - torques on right axis (add to right legend)
        for key, color in [('Tx', COLORS['Tx']), ('Ty', COLORS['Ty']), ('Tz', COLORS['Tz'])]:
            curve = pg.PlotCurveItem([], [], pen=pg.mkPen(color, width=self._line_width), name=key)
            self._force_viewbox_torque.addItem(curve)
            self._force_curves[key] = curve
            self._force_legend_right.addItem(curve, key)
        
        # Connect viewbox resize
        self._force_plot.vb.sigResized.connect(self._update_force_viewbox)
        
        layout.addWidget(self._force_graphics, stretch=1)  # Expand to fill
        
        # Force stats (under force graph)
        force_stats_frame = QFrame()
        force_stats_frame.setStyleSheet("background: #252F3A; border-radius: 3px;")
        force_stats_frame.setMaximumHeight(32)
        force_stats_layout = QHBoxLayout(force_stats_frame)
        force_stats_layout.setContentsMargins(6, 2, 6, 2)
        force_stats_layout.setSpacing(12)
        
        force_stats_layout.addStretch()
        
        self._force_label = QLabel("|F|: 0.0 N")
        self._force_label.setFont(mono_font)
        self._force_label.setStyleSheet(f"color: {COLORS['Fx']};")
        force_stats_layout.addWidget(self._force_label)
        
        self._torque_label = QLabel("|τ|: 0.0 Nm")
        self._torque_label.setFont(mono_font)
        self._torque_label.setStyleSheet(f"color: {COLORS['Tx']};")
        force_stats_layout.addWidget(self._torque_label)
        
        force_stats_layout.addStretch()
        layout.addWidget(force_stats_frame)
        
        # Add group box to main layout
        main_layout.addWidget(group_box)
    
    def _update_vel_viewbox(self):
        """Update the rotation viewbox geometry to match the velocity plot."""
        self._vel_viewbox_rot.setGeometry(self._vel_plot.vb.sceneBoundingRect())
    
    def _update_force_viewbox(self):
        """Update the torque viewbox geometry to match the force plot."""
        self._force_viewbox_torque.setGeometry(self._force_plot.vb.sceneBoundingRect())
    
    def _on_window_size_changed(self, index: int):
        """Handle window size selection change."""
        # Map index to window size: 0=5s, 1=10s, 2=15s, 3=20s, 4=30s
        sizes = [5.0, 10.0, 15.0, 20.0, 30.0]
        if 0 <= index < len(sizes):
            self._selected_window_size = sizes[index]
        else:
            self._selected_window_size = WINDOW_SECONDS_MIN
    
    def _on_line_width_changed(self, value: float):
        """Handle line width change."""
        self._line_width = value
        # Update all curve pen widths
        if PYQTGRAPH_AVAILABLE:
            for key, curve in self._vel_curves.items():
                color = COLORS[key]
                curve.setPen(pg.mkPen(color, width=value))
            for key, curve in self._force_curves.items():
                color = COLORS[key]
                curve.setPen(pg.mkPen(color, width=value))
    
    def _toggle_pause(self):
        """Toggle pause/play state for graph updates."""
        self._paused = not self._paused
        if self._paused:
            # Now paused - show play symbol
            self._pause_btn.setText("\u25B6")  # ▶ play symbol
            self._pause_btn.setToolTip("Resume graph updates")
        else:
            # Now playing - show pause symbol and reset data
            self._pause_btn.setText("\u23F8")  # ⏸ pause symbol
            self._pause_btn.setToolTip("Pause graph updates")
            self.reset()  # Reset data when resuming
    
    def _on_zero_force_clicked(self):
        """Handle force zeroing button click."""
        self.zero_force_requested.emit()
    
    def set_rotation_weight(self, weight: float):
        """Set the rotation weight (typically TCP offset magnitude)."""
        self._rotation_weight = max(weight, 0.01)
    
    def update_pose(self, tcp_pose: Tuple[float, ...], flange_pose: Optional[Tuple[float, ...]] = None, tcp_force: Optional[Tuple[float, ...]] = None):
        """Update with new pose and force data.
        
        Args:
            tcp_pose: TCP pose [x, y, z, rx, ry, rz]
            flange_pose: Flange pose (optional)
            tcp_force: TCP force/torque [Fx, Fy, Fz, Tx, Ty, Tz] (optional)
        """
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Don't update data when paused
        if self._paused:
            return
        
        current_time = time.time()
        pose = np.array(tcp_pose)
        
        if self._start_time is None:
            self._start_time = current_time
        
        relative_time = current_time - self._start_time
        
        if self._last_pose is not None and self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 0.001:  # Avoid division by very small dt
                # Calculate velocities (per component)
                diff = pose - self._last_pose
                velocities = diff / dt
                
                # Store velocities
                self._times.append(relative_time)
                for i, key in enumerate(['tx', 'ty', 'tz', 'rx', 'ry', 'rz']):
                    self._velocities[key].append(velocities[i])
                
                # Calculate velocity magnitudes
                trans_speed = np.sqrt(np.sum(velocities[:3]**2))
                rot_speed = np.sqrt(np.sum(velocities[3:]**2))
                # Weighted total for combined metric
                weighted_rot = rot_speed * self._rotation_weight
                total_vel = np.sqrt(trans_speed**2 + weighted_rot**2)
                self._velocity_magnitudes['trans'].append(trans_speed)
                self._velocity_magnitudes['rot'].append(rot_speed)
                self._velocity_magnitudes['total'].append(total_vel)
                
                # Store force/torque data
                if tcp_force is not None and len(tcp_force) >= 6:
                    force = np.array(tcp_force)
                    for i, key in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
                        self._forces[key].append(force[i])
                    
                    # Calculate force magnitudes
                    force_mag = np.sqrt(np.sum(force[:3]**2))
                    torque_mag = np.sqrt(np.sum(force[3:]**2))
                    self._force_magnitudes['force'].append(force_mag)
                    self._force_magnitudes['torque'].append(torque_mag)
                else:
                    # No force data - append zeros
                    for key in self._forces:
                        self._forces[key].append(0.0)
                    self._force_magnitudes['force'].append(0.0)
                    self._force_magnitudes['torque'].append(0.0)
        
        self._last_pose = pose
        self._last_time = current_time
    
    def _update_plots(self):
        """Update the plot displays (called by timer)."""
        if not PYQTGRAPH_AVAILABLE or len(self._times) < 2:
            return
        
        # Skip updates if widget is not visible (major performance gain)
        if not self.isVisible():
            return
        
        self._update_cycle += 1
        
        times = np.array(self._times)
        if len(times) == 0:
            return
        
        # Shift times so current time is 0
        current_time = times[-1]
        relative_times = times - current_time
        
        # Downsample data if too many points for better performance
        n_points = len(times)
        if n_points > MAX_DISPLAY_POINTS:
            # Use strided indexing for efficient downsampling
            step = n_points // MAX_DISPLAY_POINTS
            indices = slice(None, None, step)
            display_times = relative_times[indices]
        else:
            indices = slice(None)
            display_times = relative_times
        
        # Update velocity curves with downsampled data
        for key, curve in self._vel_curves.items():
            if len(self._velocities[key]) > 0:
                data = np.array(self._velocities[key])[indices]
                curve.setData(display_times, data, skipFiniteCheck=True)
        
        # Update force curves with downsampled data
        for key, curve in self._force_curves.items():
            if len(self._forces[key]) > 0:
                data = np.array(self._forces[key])[indices]
                curve.setData(display_times, data, skipFiniteCheck=True)
        
        # Use the user-selected window size (fixed, not growing)
        window_size = self._selected_window_size
        self._vel_plot.setXRange(-window_size, 0, padding=0)
        self._force_plot.setXRange(-window_size, 0, padding=0)
        
        # Auto-scale Y axes less frequently for better performance
        if self._update_cycle % self._autoscale_interval == 0:
            self._autoscale_y_axes()
        
        # Update stats labels (lightweight operation)
        if len(self._velocity_magnitudes['total']) > 0:
            trans_vel = self._velocity_magnitudes['trans'][-1] * 1000  # m/s to mm/s
            rot_vel = self._velocity_magnitudes['rot'][-1]  # rad/s
            total_vel = self._velocity_magnitudes['total'][-1] * 1000  # m/s to mm/s
            self._trans_vel_label.setText(f"|T|: {trans_vel:.1f} mm/s")
            self._rot_vel_label.setText(f"|R|: {rot_vel:.2f} rad/s")
            self._total_vel_label.setText(f"|∑|: {total_vel:.1f} mm/s")
        
        if len(self._force_magnitudes['force']) > 0:
            force_mag = self._force_magnitudes['force'][-1]
            torque_mag = self._force_magnitudes['torque'][-1]
            self._force_label.setText(f"|F|: {force_mag:.1f} N")
            self._torque_label.setText(f"|τ|: {torque_mag:.2f} Nm")
    
    def _autoscale_y_axes(self):
        """Auto-scale Y axes for all plots based on visible window + 5s buffer."""
        if len(self._times) == 0:
            return
        
        # Get time range for visible window + 5 second buffer
        times = np.array(self._times)
        current_time = times[-1] if len(times) > 0 else 0
        # Time is displayed relative to current (0 = now, negative = past)
        # Visible window is from -window_size to 0, plus 5s buffer
        window_with_buffer = self._selected_window_size + 5.0
        min_time = current_time - window_with_buffer
        
        # Create mask for data within the time window
        mask = times >= min_time
        
        # Auto-scale Y axes for velocity (left: translation, right: rotation)
        if len(self._velocities['tx']) > 0:
            # Translation (left axis) - only data in visible window + buffer
            trans_data = np.concatenate([
                np.array(self._velocities[key])[mask] for key in ['tx', 'ty', 'tz']
            ])
            if len(trans_data) > 0:
                max_trans = max(np.abs(trans_data).max(), 0.01)
                self._vel_plot.setYRange(-max_trans * 1.2, max_trans * 1.2, padding=0)
            
            # Rotation (right axis)
            rot_data = np.concatenate([
                np.array(self._velocities[key])[mask] for key in ['rx', 'ry', 'rz']
            ])
            if len(rot_data) > 0:
                max_rot = max(np.abs(rot_data).max(), 0.01)
                self._vel_viewbox_rot.setYRange(-max_rot * 1.2, max_rot * 1.2)
        
        # Auto-scale Y axes for force (left: force, right: torque)
        if len(self._forces['Fx']) > 0:
            # Force (left axis)
            force_data = np.concatenate([
                np.array(self._forces[key])[mask] for key in ['Fx', 'Fy', 'Fz']
            ])
            if len(force_data) > 0:
                max_force = max(np.abs(force_data).max(), 1.0)
                self._force_plot.setYRange(-max_force * 1.2, max_force * 1.2, padding=0)
            
            # Torque (right axis)
            torque_data = np.concatenate([
                np.array(self._forces[key])[mask] for key in ['Tx', 'Ty', 'Tz']
            ])
            if len(torque_data) > 0:
                max_torque = max(np.abs(torque_data).max(), 0.1)
                self._force_viewbox_torque.setYRange(-max_torque * 1.2, max_torque * 1.2)
    
    def reset(self):
        """Reset all data."""
        self._times.clear()
        for key in self._velocities:
            self._velocities[key].clear()
        for key in self._forces:
            self._forces[key].clear()
        for key in self._force_magnitudes:
            self._force_magnitudes[key].clear()
        for key in self._velocity_magnitudes:
            self._velocity_magnitudes[key].clear()
        
        self._last_pose = None
        self._last_time = None
        self._start_time = None
        
        # Clear curves
        if PYQTGRAPH_AVAILABLE:
            for curve in self._vel_curves.values():
                curve.setData([], [])
            for curve in self._force_curves.values():
                curve.setData([], [])
            
            # Reset velocity stats
            self._trans_vel_label.setText("|T|: 0.0 mm/s")
            self._rot_vel_label.setText("|R|: 0.0 rad/s")
            self._total_vel_label.setText("|∑|: 0.0 mm/s")
            # Reset force stats
            self._force_label.setText("|F|: 0.0 N")
            self._torque_label.setText("|τ|: 0.0 Nm")
    
    def stop_updates(self):
        """Stop the update timer."""
        if hasattr(self, '_update_timer'):
            self._update_timer.stop()
    
    def start_updates(self):
        """Start the update timer."""
        if hasattr(self, '_update_timer'):
            self._update_timer.start(int(1000 / UPDATE_RATE_HZ))

