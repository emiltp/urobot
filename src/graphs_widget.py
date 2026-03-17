"""Real-time graphs widget for visualizing robot TCP force and Ref frame force."""

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
    # Force (warm/cool distinct)
    'Fx': '#FF2222',  # Red
    'Fy': '#22DD22',  # Green
    'Fz': '#1144FF',  # Blue
    # Torque
    'Tx': '#FF9900',  # Orange
    'Ty': '#CC33FF',  # Magenta/Purple
    'Tz': '#00FFFF',  # Cyan
    # Magnitudes
    'force_mag': '#FFFFFF',   # White
    'torque_mag': '#FFCC00',  # Gold
}

WINDOW_SECONDS_MAX = 30.0  # Maximum rolling window size
WINDOW_SECONDS_MIN = 5.0   # Minimum window size to start with
UPDATE_RATE_HZ = 15        # Graph update rate (reduced for performance)
MAX_DISPLAY_POINTS = 500   # Maximum points to display per curve (downsampling)


class RealTimeGraphsWidget(QWidget):
    """Widget displaying real-time TCP force (TCP frame) and Ref frame force (Ref frame) graphs."""
    
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
        # TCP force in TCP frame [Fx, Fy, Fz, Tx, Ty, Tz]
        self._tcp_forces = {
            'Fx': deque(maxlen=self._max_samples),
            'Fy': deque(maxlen=self._max_samples),
            'Fz': deque(maxlen=self._max_samples),
            'Tx': deque(maxlen=self._max_samples),
            'Ty': deque(maxlen=self._max_samples),
            'Tz': deque(maxlen=self._max_samples),
        }
        # Ref frame force in Ref frame [Fx, Fy, Fz, Tx, Ty, Tz]
        self._forces = {
            'Fx': deque(maxlen=self._max_samples),
            'Fy': deque(maxlen=self._max_samples),
            'Fz': deque(maxlen=self._max_samples),
            'Tx': deque(maxlen=self._max_samples),
            'Ty': deque(maxlen=self._max_samples),
            'Tz': deque(maxlen=self._max_samples),
        }
        # TCP force magnitudes
        self._tcp_force_magnitudes = {
            'force': deque(maxlen=self._max_samples),
            'torque': deque(maxlen=self._max_samples),
        }
        # Ref frame force magnitudes
        self._force_magnitudes = {
            'force': deque(maxlen=self._max_samples),
            'torque': deque(maxlen=self._max_samples),
        }
        
        # State tracking
        self._last_pose: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._start_time: Optional[float] = None
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
        
        # === TCP Force Plot (TCP frame) with dual Y-axes ===
        self._tcp_force_graphics = pg.GraphicsLayoutWidget()
        self._tcp_force_plot = self._tcp_force_graphics.addPlot(row=0, col=0)
        self._tcp_force_plot.setLabel('left', 'Force (TCP frame)', units='N', color=COLORS['Fx'])
        self._tcp_force_plot.showGrid(x=True, y=True, alpha=0.3)
        self._tcp_force_plot.setXRange(-WINDOW_SECONDS_MIN, 0, padding=0)
        self._tcp_force_plot.setYRange(-50, 50, padding=0)
        self._tcp_force_plot.setMouseEnabled(x=False, y=False)
        self._tcp_force_plot.hideButtons()
        self._tcp_force_plot.setClipToView(True)
        self._tcp_force_plot.disableAutoRange()
        
        self._tcp_force_legend_left = self._tcp_force_plot.addLegend(offset=(10, 5), labelTextSize='9pt')
        
        self._tcp_force_viewbox_torque = pg.ViewBox()
        self._tcp_force_plot.scene().addItem(self._tcp_force_viewbox_torque)
        self._tcp_force_plot.getAxis('right').linkToView(self._tcp_force_viewbox_torque)
        self._tcp_force_viewbox_torque.setXLink(self._tcp_force_plot)
        self._tcp_force_viewbox_torque.disableAutoRange()
        self._tcp_force_viewbox_torque.setYRange(-1, 1)
        self._tcp_force_plot.getAxis('right').setLabel('Torque (TCP frame)', units='Nm', color=COLORS['Tx'])
        self._tcp_force_plot.showAxis('right')
        
        self._tcp_force_legend_right = pg.LegendItem(labelTextSize='9pt')
        self._tcp_force_legend_right.setParentItem(self._tcp_force_plot.vb)
        self._tcp_force_legend_right.anchor((1, 0), (1, 0), offset=(-5, 5))
        
        self._tcp_force_curves = {}
        for key, color in [('Fx', COLORS['Fx']), ('Fy', COLORS['Fy']), ('Fz', COLORS['Fz'])]:
            self._tcp_force_curves[key] = self._tcp_force_plot.plot(
                [], [], pen=pg.mkPen(color, width=self._line_width), name=key
            )
        for key, color in [('Tx', COLORS['Tx']), ('Ty', COLORS['Ty']), ('Tz', COLORS['Tz'])]:
            curve = pg.PlotCurveItem([], [], pen=pg.mkPen(color, width=self._line_width), name=key)
            self._tcp_force_viewbox_torque.addItem(curve)
            self._tcp_force_curves[key] = curve
            self._tcp_force_legend_right.addItem(curve, key)
        
        self._tcp_force_plot.vb.sigResized.connect(self._update_tcp_force_viewbox)
        self._tcp_force_graphics.setToolTip("Wrench at TCP in TCP frame coordinates")
        layout.addWidget(self._tcp_force_graphics, stretch=1)
        
        # TCP force stats
        tcp_force_stats_frame = QFrame()
        tcp_force_stats_frame.setStyleSheet("background: #252F3A; border-radius: 3px;")
        tcp_force_stats_frame.setMaximumHeight(32)
        tcp_force_stats_layout = QHBoxLayout(tcp_force_stats_frame)
        tcp_force_stats_layout.setContentsMargins(6, 2, 6, 2)
        tcp_force_stats_layout.setSpacing(12)
        tcp_force_stats_layout.addStretch()
        self._tcp_force_label = QLabel("|F|: 0.0 N")
        self._tcp_force_label.setFont(mono_font)
        self._tcp_force_label.setStyleSheet(f"color: {COLORS['Fx']};")
        tcp_force_stats_layout.addWidget(self._tcp_force_label)
        self._tcp_torque_label = QLabel("|τ|: 0.0 Nm")
        self._tcp_torque_label.setFont(mono_font)
        self._tcp_torque_label.setStyleSheet(f"color: {COLORS['Tx']};")
        tcp_force_stats_layout.addWidget(self._tcp_torque_label)
        tcp_force_stats_layout.addStretch()
        layout.addWidget(tcp_force_stats_frame)
        
        # Vertical spacing between graphs
        layout.addSpacing(8)
        
        # === TCP Force Plot with dual Y-axes ===
        self._force_graphics = pg.GraphicsLayoutWidget()
        self._force_plot = self._force_graphics.addPlot(row=0, col=0)
        self._force_plot.setLabel('left', 'Force (Ref frame)', units='N', color=COLORS['Fx'])
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
        self._force_plot.getAxis('right').setLabel('Torque (Ref frame)', units='Nm', color=COLORS['Tx'])
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
        
        self._force_graphics.setToolTip("Wrench at Ref frame in Ref frame coordinates")
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
    
    def _update_tcp_force_viewbox(self):
        """Update the torque viewbox geometry to match the TCP force plot."""
        self._tcp_force_viewbox_torque.setGeometry(self._tcp_force_plot.vb.sceneBoundingRect())
    
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
            for key, curve in self._tcp_force_curves.items():
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
    
    def update_pose(self, tcp_pose: Tuple[float, ...], flange_pose: Optional[Tuple[float, ...]] = None,
                    tcp_force_in_tcp_frame: Optional[Tuple[float, ...]] = None,
                    ref_frame_force: Optional[Tuple[float, ...]] = None):
        """Update with new pose and force data.
        
        Args:
            tcp_pose: TCP pose [x, y, z, rx, ry, rz]
            flange_pose: Flange pose (optional)
            tcp_force_in_tcp_frame: Wrench at TCP in TCP frame [Fx, Fy, Fz, Tx, Ty, Tz] (optional)
            ref_frame_force: Wrench at Ref frame in Ref frame [Fx, Fy, Fz, Tx, Ty, Tz] (optional)
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
            if dt > 0.001:
                self._times.append(relative_time)
                
                # Store TCP force in TCP frame
                if tcp_force_in_tcp_frame is not None and len(tcp_force_in_tcp_frame) >= 6:
                    tcp_f = np.array(tcp_force_in_tcp_frame)
                    for i, key in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
                        self._tcp_forces[key].append(tcp_f[i])
                    self._tcp_force_magnitudes['force'].append(np.sqrt(np.sum(tcp_f[:3]**2)))
                    self._tcp_force_magnitudes['torque'].append(np.sqrt(np.sum(tcp_f[3:]**2)))
                else:
                    for key in self._tcp_forces:
                        self._tcp_forces[key].append(0.0)
                    self._tcp_force_magnitudes['force'].append(0.0)
                    self._tcp_force_magnitudes['torque'].append(0.0)
                
                # Store Ref frame force
                if ref_frame_force is not None and len(ref_frame_force) >= 6:
                    ref_f = np.array(ref_frame_force)
                    for i, key in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
                        self._forces[key].append(ref_f[i])
                    self._force_magnitudes['force'].append(np.sqrt(np.sum(ref_f[:3]**2)))
                    self._force_magnitudes['torque'].append(np.sqrt(np.sum(ref_f[3:]**2)))
                else:
                    for key in self._forces:
                        self._forces[key].append(0.0)
                    self._force_magnitudes['force'].append(0.0)
                    self._force_magnitudes['torque'].append(0.0)
        
        self._last_pose = pose
        self._last_time = current_time
    
    def _update_plots(self):
        """Update the plot displays (called by timer)."""
        if not PYQTGRAPH_AVAILABLE or len(self._times) < 1:
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
            step = n_points // MAX_DISPLAY_POINTS
            indices = slice(None, None, step)
            display_times = relative_times[indices]
        else:
            indices = slice(None)
            display_times = relative_times
        
        # Update TCP force curves
        for key, curve in self._tcp_force_curves.items():
            if len(self._tcp_forces[key]) > 0:
                data = np.array(self._tcp_forces[key])[indices]
                curve.setData(display_times, data, skipFiniteCheck=True)
        
        # Update Ref frame force curves
        for key, curve in self._force_curves.items():
            if len(self._forces[key]) > 0:
                data = np.array(self._forces[key])[indices]
                curve.setData(display_times, data, skipFiniteCheck=True)
        
        # Use the user-selected window size (fixed, not growing)
        window_size = self._selected_window_size
        self._tcp_force_plot.setXRange(-window_size, 0, padding=0)
        self._force_plot.setXRange(-window_size, 0, padding=0)
        
        # Auto-scale Y axes less frequently for better performance
        if self._update_cycle % self._autoscale_interval == 0:
            self._autoscale_y_axes()
        
        # Update TCP force stats
        if len(self._tcp_force_magnitudes['force']) > 0:
            self._tcp_force_label.setText(f"|F|: {self._tcp_force_magnitudes['force'][-1]:.1f} N")
            self._tcp_torque_label.setText(f"|τ|: {self._tcp_force_magnitudes['torque'][-1]:.2f} Nm")
        
        # Update Ref frame force stats
        if len(self._force_magnitudes['force']) > 0:
            self._force_label.setText(f"|F|: {self._force_magnitudes['force'][-1]:.1f} N")
            self._torque_label.setText(f"|τ|: {self._force_magnitudes['torque'][-1]:.2f} Nm")
    
    def _autoscale_y_axes(self):
        """Auto-scale Y axes for all plots based on visible window + 5s buffer."""
        if len(self._times) == 0:
            return
        
        times = np.array(self._times)
        current_time = times[-1] if len(times) > 0 else 0
        window_with_buffer = self._selected_window_size + 5.0
        min_time = current_time - window_with_buffer
        mask = times >= min_time
        
        # Auto-scale TCP force plot (left: force, right: torque)
        if len(self._tcp_forces['Fx']) > 0:
            force_data = np.concatenate([
                np.array(self._tcp_forces[key])[mask] for key in ['Fx', 'Fy', 'Fz']
            ])
            if len(force_data) > 0:
                max_force = max(np.abs(force_data).max(), 1.0)
                self._tcp_force_plot.setYRange(-max_force * 1.2, max_force * 1.2, padding=0)
            torque_data = np.concatenate([
                np.array(self._tcp_forces[key])[mask] for key in ['Tx', 'Ty', 'Tz']
            ])
            if len(torque_data) > 0:
                max_torque = max(np.abs(torque_data).max(), 0.1)
                self._tcp_force_viewbox_torque.setYRange(-max_torque * 1.2, max_torque * 1.2)
        
        # Auto-scale Ref frame force plot (left: force, right: torque)
        if len(self._forces['Fx']) > 0:
            force_data = np.concatenate([
                np.array(self._forces[key])[mask] for key in ['Fx', 'Fy', 'Fz']
            ])
            if len(force_data) > 0:
                max_force = max(np.abs(force_data).max(), 1.0)
                self._force_plot.setYRange(-max_force * 1.2, max_force * 1.2, padding=0)
            torque_data = np.concatenate([
                np.array(self._forces[key])[mask] for key in ['Tx', 'Ty', 'Tz']
            ])
            if len(torque_data) > 0:
                max_torque = max(np.abs(torque_data).max(), 0.1)
                self._force_viewbox_torque.setYRange(-max_torque * 1.2, max_torque * 1.2)
    
    def reset(self):
        """Reset all data."""
        self._times.clear()
        for key in self._tcp_forces:
            self._tcp_forces[key].clear()
        for key in self._forces:
            self._forces[key].clear()
        for key in self._tcp_force_magnitudes:
            self._tcp_force_magnitudes[key].clear()
        for key in self._force_magnitudes:
            self._force_magnitudes[key].clear()
        
        self._last_pose = None
        self._last_time = None
        self._start_time = None
        
        # Clear curves
        if PYQTGRAPH_AVAILABLE:
            for curve in self._tcp_force_curves.values():
                curve.setData([], [])
            for curve in self._force_curves.values():
                curve.setData([], [])
            self._tcp_force_label.setText("|F|: 0.0 N")
            self._tcp_torque_label.setText("|τ|: 0.0 Nm")
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

