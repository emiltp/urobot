"""
GUI Application to visualize UR robot base frame and live TCP orientation.

This application uses VTK and PyQt6 for 3D visualization:
- Shows the robot's base frame (X, Y, Z axes)
- Displays live TCP position and orientation
- Updates in real-time as the robot moves

Requirements:
    pip install vtk pyqt6 numpy scipy
    pip install ur-rtde  # For robot freedrive mode control

Usage:
    python3 tcp_visualizer.py
"""

import sys
import os
import math
import socket
import threading
import time
import io
import qdarkstyle
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, QDialog,
    QComboBox, QStackedWidget, QMenu, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QTextCursor, QAction
import vtk

# Handle different VTK versions for PyQt6 integration
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    try:
        # Alternative import path for some VTK versions
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError:
        raise ImportError(
            "Could not import QVTKRenderWindowInteractor. "
            "Please ensure VTK is properly installed with PyQt6 support."
        )

from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from src.dialogboxes.tcpoffset import TCPOffsetDialog
from src.dialogboxes.refframe import RefFrameOffsetDialog
from src.terminalstream import TerminalStream
from src import sphere
from src.objects import WireframeSphereActor, TrackedPointsActor, UniversalRobot, RobotUpdateThread
from src.movements import get_registered_motions, home
from src.graphs_widget import RealTimeGraphsWidget
from config import defaults, runtime_home_position, runtime_tcp_offset
from version import __version__

# Config shortcuts for cleaner code
CONFIG = defaults



# Set monospace font for consistent column alignment
if sys.platform == "win32":
    MONOSPACE_QFONT = QFont("Courier New", 8)
elif sys.platform == "darwin":  # macOS
    MONOSPACE_QFONT = QFont("Menlo", 11)  # macOS default monospace font
else:  # Linux and other Unix-like systems
    MONOSPACE_QFONT = QFont("Courier New", 11)  # Fallback to Courier New



class TCPVisualizer(QMainWindow):
    """Main window for TCP visualization."""
    
    # Signal for protective stop events (emitted from background thread)
    protective_stop_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"UR Robot Controller (v{__version__})")

        # Center the window on the screen
        screen = QApplication.primaryScreen().geometry()
        width, height = CONFIG.visualization.window_width, CONFIG.visualization.window_height
        x = int((screen.width() - width) / 2)
        y = int((screen.height() - height) / 2)
        self.setGeometry(x, y, width, height)
        
        # Robot connection settings
        self.robot_ip = CONFIG.robot.ip
        self.connected = False
        self.update_thread = None
        self.protective_stopped = False  # Track protective stop state
        
        # Connect protective stop signal to handler
        self.protective_stop_signal.connect(self._handle_protective_stop)
        
        # Movement worker thread (property setter auto-connects error_dialog_requested)
        self._async_motion_runner = None
        
        # Universal Robot instance (handles connection, movement control, state reading, and visualization)
        self.robot: Optional[UniversalRobot] = None
        
        # Pose lock for thread-safe access
        self.pose_lock = threading.Lock()
        self.tcp_points_actor: Optional[TrackedPointsActor] = None  # Actor for tracked points using glyphs
        self.fitted_sphere: Optional[WireframeSphereActor] = None  # Actor for fitted sphere
        
        
        # Point visualization toggle
        self.show_points = False  # Show/hide TCP tracked points
        self.show_flange_points = False  # Show/hide flange tracked points
        
        # Point actors (initialized when first point is added)
        self.tcp_points_actor = None
        self.flange_points_actor = None
        
        # Sphere visualization toggle
        self.show_sphere = False  # Sphere not visible by default
        
        # TCP tracking settings
        self.tcp_tracking_enabled = False
        self.tcp_points = []  # List of [x, y, z] positions
        self.last_tracked_position = None  # Last tracked position for threshold checking
        self.tcp_tracking_threshold = CONFIG.tracking.tcp_threshold
        self.tcp_tracking_lock = threading.Lock()  # Lock for thread-safe TCP tracking access
        
        # Flange tracking settings
        self.flange_tracking_enabled = False
        self.flange_points = []  # List of [x, y, z] positions for flange
        self.last_flange_tracked_position = None  # Last tracked flange position for threshold checking
        self.flange_tracking_threshold = CONFIG.tracking.flange_threshold
        self.flange_tracking_lock = threading.Lock()  # Lock for thread-safe flange tracking access
        self._last_flange_points_added = 0  # Track how many flange points have been added to visualization
        
        # Home position [x, y, z, rx, ry, rz] - loaded from config.yaml defaults
        self.home_position: Optional[List[float]] = runtime_home_position.copy()
        
        # Terminal output stream handlers
        self.terminal_stream_stdout = None
        self.terminal_stream_stderr = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Setup UI
        self.setup_ui()
        
        # Setup VTK
        self.setup_vtk()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(CONFIG.visualization.update_interval_ms)

    @property
    def async_motion_runner(self):
        return self._async_motion_runner

    @async_motion_runner.setter
    def async_motion_runner(self, runner):
        self._async_motion_runner = runner
        if runner is not None and hasattr(runner, 'error_dialog_requested'):
            runner.error_dialog_requested.connect(self._show_config_error_dialog)

    def _show_config_error_dialog(self, message: str):
        """Show a critical dialog for config/file errors."""
        QMessageBox.critical(self, "Configuration Error", message)
        
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout is vertical: top content area + bottom terminal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left side panel container (vertical layout for Connection and Control panels)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Connection Panel
        connection_panel = QGroupBox("Connection Panel")
        connection_layout = QVBoxLayout()
        connection_layout.setSpacing(4)  # Reduce spacing between elements
        connection_layout.setContentsMargins(8, 8, 8, 8)  # Reduce margins
        
        # IP address input
        ip_row = QHBoxLayout()
        ip_label = QLabel("Robot IP:")
        self.ip_input = QLineEdit(self.robot_ip)
        ip_row.addWidget(ip_label)
        ip_row.addWidget(self.ip_input)
        
        # Connection button with unicode symbol (U+1F504)
        self.connection_btn = QPushButton("\u21BB")
        self.connection_btn.setFixedSize(23, 23)  # Make it squared
        self.connection_btn.clicked.connect(self.toggle_connection)
        self.connection_btn.setToolTip("Connect to robot")
        
        ip_row.addWidget(self.connection_btn)
        connection_layout.addLayout(ip_row)

        # Status
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        connection_layout.addWidget(self.status_label)
        
        connection_panel.setLayout(connection_layout)
        left_layout.addWidget(connection_panel)
        
        # TCP Panel - Tool Center Point Panel
        tcp_panel = QGroupBox("Tool Center Point Panel")
        tcp_layout = QVBoxLayout()
        tcp_layout.setSpacing(4)
        tcp_layout.setContentsMargins(8, 8, 8, 8)
        
        # TCP Pose Header with buttons
        pose_header_row = QHBoxLayout()
        pose_header_row.setSpacing(2)
        
        self.pose_collapse_btn = QPushButton("▾")
        self.pose_collapse_btn.setFixedSize(20, 20)
        self.pose_collapse_btn.setToolTip("Collapse/expand TCP pose display")
        self.pose_collapse_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; font-size: 16px; color: #AAA; padding: 0px; }
            QPushButton:hover { color: #FFF; }
        """)
        self.pose_collapse_btn.clicked.connect(self._toggle_pose_collapse)
        pose_header_row.addWidget(self.pose_collapse_btn)
        
        pose_label = QLabel("TCP:")
        pose_label.setStyleSheet("font-weight: bold;")
        pose_header_row.addWidget(pose_label)
        pose_header_row.addStretch()
        
        # Graphs toggle button (U+223F - sine wave)
        self.toggle_graphs_btn = QPushButton("\u223F")  # ∿
        self.toggle_graphs_btn.setFixedSize(20, 20)
        self.toggle_graphs_btn.setCheckable(True)
        self.toggle_graphs_btn.setChecked(False)  # Graphs hidden by default
        self.toggle_graphs_btn.clicked.connect(self.toggle_graphs_visibility)
        self.toggle_graphs_btn.setToolTip("Toggle graphs visibility")
        self.toggle_graphs_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
                padding: 0px;
                padding-bottom: 2px;
                qproperty-iconSize: 0px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:checked {
                color: #4ADE80;
            }
        """)
        pose_header_row.addWidget(self.toggle_graphs_btn)
        
        # Vertical separator line
        separator = QLabel("|")
        separator.setStyleSheet("color: #5A6578; font-size: 16px; padding-bottom: 3px;")
        separator.setFixedWidth(12)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pose_header_row.addWidget(separator)
        
        # Home button (U+2302)
        self.home_go_btn = QPushButton("\u2302")  # ⌂
        self.home_go_btn.setFixedSize(20, 20)  # Make it squared
        self.home_go_btn.clicked.connect(lambda: home.goDirect(self))
        self.home_go_btn.setToolTip("Go to home position")
        self.home_go_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        pose_header_row.addWidget(self.home_go_btn)
        
        # Set button (U+2699) - right-click for context menu to save as default
        self.home_set_btn = QPushButton("\u2699")  # ⚙
        self.home_set_btn.setFixedSize(20, 20)  # Make it squared
        self.home_set_btn.clicked.connect(self.set_home)
        self.home_set_btn.setToolTip("Set home position (right-click to save as default)")
        self.home_set_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.home_set_btn.customContextMenuRequested.connect(self._show_home_context_menu)
        self.home_set_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        pose_header_row.addWidget(self.home_set_btn)
        
        # Clear button (U+2A2F)
        self.home_reset_btn = QPushButton("\u2A2F")  # ⨯
        self.home_reset_btn.setFixedSize(20, 20)  # Make it squared
        self.home_reset_btn.clicked.connect(self.reset_home_to_default)
        self.home_reset_btn.setToolTip("Reset home to default")
        self.home_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        pose_header_row.addWidget(self.home_reset_btn)
        pose_header_row.setContentsMargins(0, 0, 0, 0)
        
        tcp_layout.addLayout(pose_header_row)
        
        # TCP pose display using QTextEdit (position and orientation side by side)
        self.pose_text = QTextEdit()
        self.pose_text.setReadOnly(True)
        self.pose_text.setMaximumHeight(70)
        
        self.pose_text.setFont(MONOSPACE_QFONT)
        self.pose_text.setText("Not available")
        tcp_layout.addWidget(self.pose_text)
        
        # TCP Offset display section
        offset_header_row = QHBoxLayout()
        offset_header_row.setSpacing(2)
        
        self.offset_collapse_btn = QPushButton("▸")
        self.offset_collapse_btn.setFixedSize(20, 20)
        self.offset_collapse_btn.setToolTip("Collapse/expand TCP offset display")
        self.offset_collapse_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; font-size: 16px; color: #AAA; padding: 0px; }
            QPushButton:hover { color: #FFF; }
        """)
        self.offset_collapse_btn.clicked.connect(self._toggle_offset_collapse)
        offset_header_row.addWidget(self.offset_collapse_btn)
        
        offset_label = QLabel("Offset:")
        offset_label.setStyleSheet("font-weight: bold;")
        offset_header_row.addWidget(offset_label)
        offset_header_row.addStretch()

        # Align TCP orientation to base at current position (squared ⟂ U+27C2)
        self.align_tcp_btn = QPushButton("⟂")  # U+27C2
        self.align_tcp_btn.setFixedSize(20, 20)  # Make it squared
        self.align_tcp_btn.clicked.connect(self.align_tcp_axes_to_base)
        self.align_tcp_btn.setToolTip("Align TCP orientation to base axes at current position")
        self.align_tcp_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:disabled {
                background-color: #2A3548;
                border: 1px solid #3A4558;
                color: #5A6578;
            }
        """)
        self.align_tcp_btn.setEnabled(False)  # Disabled until connected
        offset_header_row.addWidget(self.align_tcp_btn)
        
        # Set tool point offset button (U+2699) - right-click for context menu to save as default
        self.set_tcp_offset_btn = QPushButton("\u2699")  # ⚙
        self.set_tcp_offset_btn.setFixedSize(20, 20)  # Make it squared
        self.set_tcp_offset_btn.clicked.connect(self.show_tcp_offset_dialog)
        self.set_tcp_offset_btn.setToolTip("Set tool point offset (right-click to save as default)")
        self.set_tcp_offset_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.set_tcp_offset_btn.customContextMenuRequested.connect(self._show_tcp_offset_context_menu)
        self.set_tcp_offset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        offset_header_row.addWidget(self.set_tcp_offset_btn)

        
        
        # Reset offset button (U+2A2F)
        self.reset_tcp_offset_btn_panel = QPushButton("\u2A2F")  # ⨯
        self.reset_tcp_offset_btn_panel.setFixedSize(20, 20)  # Make it squared
        self.reset_tcp_offset_btn_panel.clicked.connect(self.reset_tcp_offset_to_default)
        self.reset_tcp_offset_btn_panel.setToolTip("Reset TCP offset to default")
        self.reset_tcp_offset_btn_panel.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        offset_header_row.addWidget(self.reset_tcp_offset_btn_panel)
        offset_header_row.setContentsMargins(0, 0, 0, 0)
        
        tcp_layout.addLayout(offset_header_row)
        
        # TCP offset display using QTextEdit (position and orientation side by side)
        self.tcp_offset_text = QTextEdit()
        self.tcp_offset_text.setReadOnly(True)
        self.tcp_offset_text.setMaximumHeight(70)
        # Set monospace font for consistent column alignment
        self.tcp_offset_text.setFont(MONOSPACE_QFONT)
        self.tcp_offset_text.setText("Not available")
        self.tcp_offset_text.setVisible(False)
        tcp_layout.addWidget(self.tcp_offset_text)
        
        # Reference Frame display section
        ref_header_row = QHBoxLayout()
        ref_header_row.setSpacing(2)
        
        self.ref_collapse_btn = QPushButton("▸")
        self.ref_collapse_btn.setFixedSize(20, 20)
        self.ref_collapse_btn.setToolTip("Collapse/expand ref frame display")
        self.ref_collapse_btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; font-size: 16px; color: #AAA; padding: 0px; }
            QPushButton:hover { color: #FFF; }
        """)
        self.ref_collapse_btn.clicked.connect(self._toggle_ref_collapse)
        ref_header_row.addWidget(self.ref_collapse_btn)
        
        ref_label = QLabel("Ref frame:")
        ref_label.setStyleSheet("font-weight: bold;")
        ref_header_row.addWidget(ref_label)
        ref_header_row.addStretch()

        # Align ref frame orientation to base at current TCP position
        self.align_ref_btn = QPushButton("⟂")
        self.align_ref_btn.setFixedSize(20, 20)
        self.align_ref_btn.clicked.connect(self.align_ref_frame_to_base)
        self.align_ref_btn.setToolTip("Align ref frame orientation to base axes at current TCP position")
        self.align_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:disabled {
                background-color: #2A3548;
                border: 1px solid #3A4558;
                color: #5A6578;
            }
        """)
        self.align_ref_btn.setEnabled(False)
        ref_header_row.addWidget(self.align_ref_btn)
        
        # Set ref frame offset button
        self.set_ref_btn = QPushButton("\u2699")
        self.set_ref_btn.setFixedSize(20, 20)
        self.set_ref_btn.clicked.connect(self.show_ref_frame_dialog)
        self.set_ref_btn.setToolTip("Set ref frame offset relative to TCP")
        self.set_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        self.set_ref_btn.setEnabled(False)
        ref_header_row.addWidget(self.set_ref_btn)

        # Clear ref frame offset button
        self.clear_ref_btn = QPushButton("\u2A2F")
        self.clear_ref_btn.setFixedSize(20, 20)
        self.clear_ref_btn.clicked.connect(self.clear_ref_frame)
        self.clear_ref_btn.setToolTip("Clear ref frame offset")
        self.clear_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
        """)
        self.clear_ref_btn.setEnabled(False)
        ref_header_row.addWidget(self.clear_ref_btn)
        ref_header_row.setContentsMargins(0, 0, 0, 0)
        
        tcp_layout.addLayout(ref_header_row)
        
        # Ref frame offset display
        self.ref_frame_text = QTextEdit()
        self.ref_frame_text.setReadOnly(True)
        self.ref_frame_text.setMaximumHeight(70)
        self.ref_frame_text.setFont(MONOSPACE_QFONT)
        self.ref_frame_text.setText("Not set")
        self.ref_frame_text.setVisible(False)
        tcp_layout.addWidget(self.ref_frame_text)
        
        tcp_panel.setLayout(tcp_layout)
        tcp_panel.setMaximumWidth(300)
        tcp_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        left_layout.addWidget(tcp_panel)
        
        # Control panel
        control_panel = QGroupBox("Control Panel")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(4)
        control_layout.setContentsMargins(8, 8, 8, 8)
        
        # Free move toggle button
        self.free_move_btn = QPushButton("Free Move: OFF")
        self.free_move_btn.clicked.connect(self.toggle_free_move)
        self.free_move_btn.setStyleSheet("")
        self.free_move_btn.setEnabled(False)  # Disabled until connected
        control_layout.addWidget(self.free_move_btn)
        
        # Tracking button row
        # TCP Tracking row
        tcp_tracking_row = QHBoxLayout()
        tcp_tracking_row.setSpacing(2)  # Reduce spacing between squared buttons
        
        # TCP Tracking toggle button
        self.tcp_tracking_btn = QPushButton("TCP Tracking")
        self.tcp_tracking_btn.clicked.connect(self.toggle_tcp_tracking)
        self.tcp_tracking_btn.setEnabled(False)  # Disabled until connected
        tcp_tracking_row.addWidget(self.tcp_tracking_btn)
        
        # Show points toggle button (squared with symbol)
        self.show_points_btn = QPushButton("○")  # Hollow circle for OFF
        self.show_points_btn.setFixedSize(20, 20)  # Make it squared
        self.show_points_btn.clicked.connect(self.toggle_show_points)
        self.show_points_btn.setToolTip("Show tracked points")
        
        self.show_points_btn.setEnabled(False)  # Disabled until connected
        tcp_tracking_row.addWidget(self.show_points_btn)
        
        # Clear tracked points button (squared with symbol)
        self.clear_points_btn = QPushButton("\u2A2F")  # ⨯ symbol (U+2A2F)
        self.clear_points_btn.setFixedSize(20, 20)  # Make it squared
        self.clear_points_btn.clicked.connect(self.clear_tcp_points)
        self.clear_points_btn.setToolTip("Clear tracked points")
        self.clear_points_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:disabled {
                background-color: #2A3548;
                border: 1px solid #3A4558;
                color: #5A6578;
            }
        """)
        self.clear_points_btn.setEnabled(False)  # Disabled until connected
        tcp_tracking_row.addWidget(self.clear_points_btn)
        
        control_layout.addLayout(tcp_tracking_row)
        
        # Flange Tracking row
        flange_tracking_row = QHBoxLayout()
        flange_tracking_row.setSpacing(2)  # Reduce spacing between squared buttons
        
        # Flange Tracking toggle button
        self.flange_tracking_btn = QPushButton("Flange Tracking")
        self.flange_tracking_btn.clicked.connect(self.toggle_flange_tracking)
        self.flange_tracking_btn.setEnabled(False)  # Disabled until connected
        flange_tracking_row.addWidget(self.flange_tracking_btn)
        
        # Show flange points toggle button (squared with symbol)
        self.show_flange_points_btn = QPushButton("○")  # Hollow circle for OFF
        self.show_flange_points_btn.setFixedSize(20, 20)  # Make it squared
        self.show_flange_points_btn.clicked.connect(self.toggle_show_flange_points)
        self.show_flange_points_btn.setToolTip("Show tracked flange points")
        self.show_flange_points_btn.setEnabled(False)  # Disabled until connected
        flange_tracking_row.addWidget(self.show_flange_points_btn)
        
        # Clear flange tracked points button (squared with symbol)
        self.clear_flange_points_btn = QPushButton("\u2A2F")  # ⨯ symbol (U+2A2F)
        self.clear_flange_points_btn.setFixedSize(20, 20)  # Make it squared
        self.clear_flange_points_btn.clicked.connect(self.clear_flange_points)
        self.clear_flange_points_btn.setToolTip("Clear tracked flange points")
        self.clear_flange_points_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:disabled {
                background-color: #2A3548;
                border: 1px solid #3A4558;
                color: #5A6578;
            }
        """)
        self.clear_flange_points_btn.setEnabled(False)  # Disabled until connected
        flange_tracking_row.addWidget(self.clear_flange_points_btn)
        
        control_layout.addLayout(flange_tracking_row)
        
        # Fit sphere button row
        sphere_row = QHBoxLayout()
        sphere_row.setSpacing(2)  # Reduce spacing between squared buttons
        
        # Fit sphere button
        self.fit_sphere_btn = QPushButton("Fit Sphere")
        self.fit_sphere_btn.clicked.connect(self.fit_and_visualize_sphere)
        #self.fit_sphere_btn.setStyleSheet("background-color: lightcyan;")
        self.fit_sphere_btn.setEnabled(False)  # Disabled until connected
        sphere_row.addWidget(self.fit_sphere_btn)
        
        # Show sphere toggle button (squared with symbol)
        self.show_sphere_btn = QPushButton("○")  # Hollow circle for OFF (default)
        self.show_sphere_btn.setFixedSize(20, 20)  # Make it squared
        self.show_sphere_btn.clicked.connect(self.toggle_show_sphere)
        self.show_sphere_btn.setToolTip("Show fitted sphere")
        self.show_sphere_btn.setEnabled(False)  # Disabled until sphere is fitted
        sphere_row.addWidget(self.show_sphere_btn)
        
        # Delete sphere button (squared with symbol)
        self.delete_sphere_btn = QPushButton("\u2A2F")  # ⨯ symbol (U+2A2F)
        self.delete_sphere_btn.setFixedSize(20, 20)  # Make it squared
        self.delete_sphere_btn.clicked.connect(self.delete_sphere)
        self.delete_sphere_btn.setToolTip("Delete fitted sphere")
        self.delete_sphere_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5A6578;
                border: 1px solid #6A7588;
            }
            QPushButton:pressed {
                background-color: #3A4558;
                border: 1px solid #4A5568;
            }
            QPushButton:disabled {
                background-color: #2A3548;
                border: 1px solid #3A4558;
                color: #5A6578;
            }
        """)
        self.delete_sphere_btn.setEnabled(False)  # Disabled until sphere is fitted
        sphere_row.addWidget(self.delete_sphere_btn)
        
        control_layout.addLayout(sphere_row)
        
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)
        control_panel.setMaximumHeight(180)  # Fixed height to contain only buttons
        left_layout.addWidget(control_panel)
        
        # Motion Control panel with dropdown
        motion_panel = QGroupBox("Motion Control")
        motion_layout = QVBoxLayout()
        motion_layout.setSpacing(4)
        motion_layout.setContentsMargins(8, 8, 8, 8)
        
        # Motion type dropdown with path buttons
        motion_type_layout = QHBoxLayout()
        motion_type_layout.setSpacing(4)
        self.motion_combo = QComboBox()
        self.motion_combo.addItem("Select Motion...", None)
        self.motion_widgets = {}
        self.motion_ids = []
        
        # Create path buttons before widgets (freemove widget calls _update_motion_path_buttons in __init__)
        self.motion_show_path_btn = QPushButton("○")
        self.motion_show_path_btn.setFixedSize(24, 24)
        self.motion_show_path_btn.clicked.connect(self._toggle_motion_path_visualization)
        self.motion_show_path_btn.setToolTip("Show/hide path")
        self.motion_show_path_btn.setEnabled(False)
        self.motion_clear_path_btn = QPushButton("\u2A2F")
        self.motion_clear_path_btn.setFixedSize(24, 24)
        self.motion_clear_path_btn.clicked.connect(self._delete_motion_path)
        self.motion_clear_path_btn.setToolTip("Delete path")
        self.motion_clear_path_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A5568;
                border: 1px solid #5A6578;
                border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #5A6578; }
            QPushButton:disabled { background-color: #2A3548; color: #5A6578; }
        """)
        self.motion_clear_path_btn.setEnabled(False)
        
        # Container for motion-specific widgets
        self.motion_stack = QStackedWidget()
        empty_widget = QWidget()
        self.motion_stack.addWidget(empty_widget)
        
        for motion_id, display_name, widget_cls in get_registered_motions():
            self.motion_combo.addItem(display_name, motion_id)
            widget = widget_cls(self)
            self.motion_widgets[motion_id] = widget
            self.motion_ids.append(motion_id)
            self.motion_stack.addWidget(widget)
        
        self.motion_combo.currentIndexChanged.connect(self._on_motion_type_changed)
        self.motion_combo.setEnabled(False)
        motion_type_layout.addWidget(self.motion_combo, stretch=1)
        motion_type_layout.addWidget(self.motion_show_path_btn)
        motion_type_layout.addWidget(self.motion_clear_path_btn)
        
        motion_layout.addLayout(motion_type_layout)
        motion_layout.addWidget(self.motion_stack)
        
        motion_panel.setLayout(motion_layout)
        motion_panel.setMaximumWidth(300)
        left_layout.addWidget(motion_panel)
        
        # Stop button (outside the Run Motion group box)
        # Add small spacing before the button
        #left_layout.addSpacing(5)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_movement)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: white;
                border: 2px solid #CC0000;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 6px 20px;
                min-width: 200px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #FF3333;
                border: 2px solid #FF0000;
            }
            QPushButton:pressed {
                background-color: #CC0000;
                border: 2px solid #990000;
            }
            QPushButton:disabled {
                background-color: #808080;
                border: 2px solid #606060;
                color: #CCCCCC;
            }
        """)
        self.stop_btn.setEnabled(False)  # Disabled until a movement starts
        self.stop_btn.setMaximumWidth(260)
        
        # Center the button by adding it to a horizontal layout with stretch
        stop_button_container = QHBoxLayout()
        stop_button_container.addWidget(self.stop_btn)
        left_layout.addLayout(stop_button_container)
        
        # Add small spacing after the button
        left_layout.addSpacing(10)
        
        left_panel.setMaximumWidth(300)
        
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        
        # Terminal output panel
        terminal_panel = QGroupBox("Terminal Output")
        terminal_layout = QVBoxLayout()
        terminal_layout.setContentsMargins(8, 8, 8, 8)
        terminal_layout.setSpacing(4)
        
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(MONOSPACE_QFONT)
        self.terminal_output.setStyleSheet("""
            QTextEdit {
                background-color: #1E2630;
                color: #d4d4d4;
                border: 1px solid #3A4350;
                border-radius: 4px;
            }
        """)
        terminal_layout.addWidget(self.terminal_output)
        terminal_panel.setLayout(terminal_layout)
        terminal_panel.setMinimumHeight(180)
        terminal_panel.setMaximumHeight(220)
        
        # Create center layout: VTK on top, terminal at bottom
        center_side = QWidget()
        center_layout = QVBoxLayout(center_side)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.vtk_widget, stretch=1)
        center_layout.addWidget(terminal_panel)
        
        # Real-time graphs widget (between controls and VTK)
        self.graphs_widget = RealTimeGraphsWidget()
        self.graphs_widget.setMinimumWidth(280)
        self.graphs_widget.setMaximumWidth(400)
        self.graphs_widget.setVisible(False)  # Hidden by default
        self.graphs_widget.zero_force_requested.connect(self._zero_force_sensor)
        
        # Create horizontal container for left panel, graphs, and center (VTK + terminal)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(left_panel)
        content_layout.addSpacing(4)
        content_layout.addWidget(self.graphs_widget)
        content_layout.addSpacing(4)
        content_layout.addWidget(center_side, stretch=1)
        
        # Add content area to main layout
        main_layout.addWidget(content_widget, stretch=1)
        
        # Set up terminal stream handlers with log file
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'terminal.log')
        # Clear log at startup
        with open(log_file_path, 'w', encoding='utf-8'):
            pass
        self.terminal_stream_stdout = TerminalStream(self.terminal_output, log_file_path)
        self.terminal_stream_stderr = TerminalStream(self.terminal_output, log_file_path)
        sys.stdout = self.terminal_stream_stdout
        sys.stderr = self.terminal_stream_stderr
        
    def setup_vtk(self):
        """Setup VTK renderer and scene."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.118, 0.149, 0.188)  # Dark slate blue (#1E2630)
        
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Initialize robot reference frame actors
        self.initialize_robot()
        
        # Initialize camera with custom view:
        # - World Y axis points left on screen
        # - World Z axis points up on screen
        # This is achieved by placing the camera on the negative X axis,
        # looking toward the origin, with view-up along +Z.
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(-1.0, 0.0, 0.0)
        camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def initialize_robot(self):
        """Initialize the Universal Robot instance."""
        # Remove existing robot actor if present
        if self.robot is not None:
            self.robot.removeFromRenderer()
        
        # Create Universal Robot with visualization
        self.robot = UniversalRobot(
            ip=self.robot_ip,
            autoConnect=False,  # We'll connect manually via toggle_connection
            tcpOffset=runtime_tcp_offset.copy(),
            baseScale=0.5,
            flangeScale=0.1,
            tcpScale=0.1
        )
        self.robot.addToRenderer(self.renderer)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera_to_show_all(self):
        """Reset camera to show both base frame and TCP axes."""
        # Reset camera to fit all visible actors
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        
        # Get the bounds of all visible actors
        bounds = self.renderer.ComputeVisiblePropBounds()
        
        # Check if bounds are valid (not infinite or zero)
        if (bounds[0] <= bounds[1] and bounds[2] <= bounds[3] and bounds[4] <= bounds[5] and
            abs(bounds[1] - bounds[0]) > 1e-6 and 
            abs(bounds[3] - bounds[2]) > 1e-6 and 
            abs(bounds[5] - bounds[4]) > 1e-6):
            
            # Calculate center and distance
            center_x = (bounds[0] + bounds[1]) / 2.0
            center_y = (bounds[2] + bounds[3]) / 2.0
            center_z = (bounds[4] + bounds[5]) / 2.0
            
            # Calculate the maximum extent
            max_extent = max(
                bounds[1] - bounds[0],  # x extent
                bounds[3] - bounds[2],  # y extent
                bounds[5] - bounds[4]   # z extent
            )
            
            # Add some padding (20% margin)
            max_extent *= 1.2
            
            # Set focal point to center of all visible objects
            camera.SetFocalPoint(center_x, center_y, center_z)
            
            # Position camera to maintain the same view direction (from negative X, looking at origin)
            # Adjust distance based on extent
            distance = max(max_extent * 1.5, 0.5)  # Minimum distance of 0.5
            camera.SetPosition(center_x - distance, center_y, center_z)
            camera.SetViewUp(0.0, 0.0, 1.0)
        else:
            # Fallback: use default camera position if bounds are invalid
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetPosition(-1.0, 0.0, 0.0)
            camera.SetViewUp(0.0, 0.0, 1.0)
        
        # Reset clipping range
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
    
        
    def update_visualization(self):
        """Update the visualization with current TCP pose."""
        # Guard against uninitialized robot
        if self.robot is None:
            return
        
        # Get pose from UniversalRobot
        if not (pose := self.robot.tcpPose): return
        
        # Extract TCP position and orientation
        x, y, z = pose[0], pose[1], pose[2]
        rx, ry, rz = pose[3], pose[4], pose[5]
        
        # Update tracked points (only add new ones)
        self.update_points_visualization()
        
        # Update TCP Panel pose text - orientation to the right of position
        # Use fixed-width formatting with monospace font for proper column alignment
        pose_text = f"Position (m)       Orientation (deg)\n"
        pose_text += f" X: {x:>9.4f}       RX: {np.rad2deg(rx):>9.4f}\n"
        pose_text += f" Y: {y:>9.4f}       RY: {np.rad2deg(ry):>9.4f}\n"
        pose_text += f" Z: {z:>9.4f}       RZ: {np.rad2deg(rz):>9.4f}"
        self.pose_text.setText(pose_text)
        
        # Update home set button state based on pose comparison
        self.update_home_button_states()
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_connection(self):
        """Toggle robot connection."""
        if not self.connected:
            # Connect
            self.robot_ip = self.ip_input.text().strip()
            if not self.robot_ip:
                self.status_label.setText("Status: Error - Enter IP address")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return
            
            self.status_label.setText("Status: Connecting...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            
            # Connect via UniversalRobot
            if self.robot is not None:
                # Get current TCP offset to restore, or use default
                tcp_offset_to_set = self.robot.tcpOffset if self.robot.tcpOffset is not None else runtime_tcp_offset.copy()
                
                if self.robot.connect(self.robot_ip, tcp_offset_to_set):
                    self.connected = True
                    
                    # Reset and start graphs
                    if hasattr(self, 'graphs_widget'):
                        self.graphs_widget.reset()
                        self.graphs_widget.start_updates()
                    
                    # Start update thread
                    self.update_thread = RobotUpdateThread(self.robot)
                    self.update_thread.pose_updated.connect(self._handle_pose_update)
                    self.update_thread.error_occurred.connect(self._handle_protective_stop)
                    self.update_thread.start()
                    
                    self.status_label.setText("Status: Connected")
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    
                    # Print initial TCP offset
                    tcp_offset = self.robot.getTcpOffset()
                    if tcp_offset:
                        print(f"\n=== Initial TCP Offset ===")
                        print(f"TCP Offset: [{tcp_offset[0]:.6f}, {tcp_offset[1]:.6f}, {tcp_offset[2]:.6f}, {tcp_offset[3]:.6f}, {tcp_offset[4]:.6f}, {tcp_offset[5]:.6f}]")
                        print(f"  Position (x, y, z): [{tcp_offset[0]:.6f}, {tcp_offset[1]:.6f}, {tcp_offset[2]:.6f}] m")
                        print(f"  Orientation (rx, ry, rz): [{tcp_offset[3]:.6f}, {tcp_offset[4]:.6f}, {tcp_offset[5]:.6f}] rad")
                        print(f"=== End of Initial TCP Offset ===")
                    
                    # Enable control buttons
                    self.free_move_btn.setEnabled(True)
                    self.tcp_tracking_btn.setEnabled(True)
                    self.flange_tracking_btn.setEnabled(True)
                    self.show_points_btn.setEnabled(True)
                    self.show_flange_points_btn.setEnabled(True)
                    self.align_tcp_btn.setEnabled(True)
                    self.clear_points_btn.setEnabled(True)
                    self.clear_flange_points_btn.setEnabled(True)
                    self.update_clear_button_state()
                    self.update_clear_flange_button_state()
                    self.set_tcp_offset_btn.setEnabled(True)
                    self.reset_tcp_offset_btn_panel.setEnabled(True)
                    self.align_ref_btn.setEnabled(True)
                    self.set_ref_btn.setEnabled(True)
                    self.clear_ref_btn.setEnabled(True)
                    self.motion_combo.setEnabled(True)
                    self.home_go_btn.setEnabled(self.home_position is not None)
                    self.home_set_btn.setEnabled(True)
                    
                    # Update freemove widget button states
                    freemove_widget = self.motion_widgets.get('freemove')
                    if freemove_widget and hasattr(freemove_widget, '_update_button_states'):
                        freemove_widget._update_button_states()
                    
                    # Update TCP offset display after connection
                    self.update_tcp_offset_display()
                    
                    # Reset camera to show both base and TCP axes after a short delay
                    QTimer.singleShot(500, self.reset_camera_to_show_all)
                else:
                    self.status_label.setText("Status: Connection Failed")
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.status_label.setText("Status: Robot not initialized")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            # Disconnect
            # Disable freedrive mode if active
            if self.robot is not None and self.robot.isFreedriveActive():
                self.robot.stopFreedrive()
                self.free_move_btn.setText("Free Move: OFF")
                self.free_move_btn.setStyleSheet("")
            
            # Stop the update thread
            if self.update_thread is not None and self.update_thread.isRunning():
                self.update_thread.stop()
                self.update_thread = None
            
            # Disconnect via UniversalRobot
            if self.robot is not None:
                self.robot.disconnect()
            
            self.connected = False
            self.status_label.setText("Status: Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Disable all control buttons when disconnected
            self.free_move_btn.setEnabled(False)
            self.tcp_tracking_btn.setEnabled(False)
            self.show_points_btn.setEnabled(False)
            self.show_flange_points_btn.setEnabled(False)
            self.clear_points_btn.setEnabled(False)
            self.clear_flange_points_btn.setEnabled(False)
            self.fit_sphere_btn.setEnabled(False)
            self.show_sphere_btn.setEnabled(False)
            self.delete_sphere_btn.setEnabled(False)
            self.set_tcp_offset_btn.setEnabled(False)
            self.reset_tcp_offset_btn_panel.setEnabled(False)
            self.align_ref_btn.setEnabled(False)
            self.set_ref_btn.setEnabled(False)
            self.clear_ref_btn.setEnabled(False)
            self.motion_combo.setEnabled(False)
            self.home_go_btn.setEnabled(False)
            self.home_set_btn.setEnabled(False)
            
            # Clear TCP tracking data
            self.tcp_tracking_enabled = False
            with self.tcp_tracking_lock:
                self.tcp_points.clear()
                self.last_tracked_position = None
            
            # Clear flange tracking data
            self.flange_tracking_enabled = False
            with self.flange_tracking_lock:
                self.flange_points.clear()
                self.last_flange_tracked_position = None
            if self.tcp_points_actor is not None:
                self.tcp_points_actor.clearPoints()
            
            # Clear fitted sphere
            if self.fitted_sphere is not None:
                self.fitted_sphere.removeFromRenderer()
                self.fitted_sphere = None
            self.show_sphere_btn.setEnabled(False)
            self.delete_sphere_btn.setEnabled(False)
            
            # Reset show_points state
            self.show_points = False
            # Reset show_sphere state
            self.show_sphere = False
            self.show_sphere_btn.setText("○")  # Reset to hollow circle
            
            # Reset robot visualization (sets to origin and hides)
            if self.robot is not None:
                self.robot.resetVisualization()

            self.vtk_widget.GetRenderWindow().Render()

            # Reset TCP Panel display
            self.pose_text.setText("Not available")
            self.tcp_offset_text.setText("Not available")
            self.ref_frame_text.setText("Not set")
    
    def _handle_pose_update(self, tcp_pose: tuple, flange_pose: tuple):
        """Handle pose updates from the robot update thread.
        
        This is called from RobotUpdateThread via signal when new pose data is available.
        Runs on the main thread so it's safe to update UI/tracking.
        """
        # Reset protective stop flag since we're receiving valid data
        self.protective_stopped = False
        
        # Check if we should track TCP position
        if self.tcp_tracking_enabled:
            self.check_and_track_position(tcp_pose[:3])
        
        # Check if we should track flange position
        if self.flange_tracking_enabled:
            self.check_and_track_flange_position(flange_pose[:3])
        
        # Update real-time graphs with pose and force data
        if hasattr(self, 'graphs_widget'):
            tcp_force = self.robot.getTcpForceInTcpFrame() if self.robot else None
            self.graphs_widget.update_pose(tcp_pose, flange_pose, tcp_force)
    
    def _handle_protective_stop(self, stop_type: str):
        """Handle protective stop, emergency stop, or connection loss events.
        
        This is called from the update loop via signal when the robot enters
        a protective stop, emergency stop, or loses connection.
        It updates the UI to show the error state and allows the user to reconnect.
        """
        print(f"Robot {stop_type} detected!")
        
        # Disable freedrive mode if active
        if self.robot is not None and self.robot.isFreedriveActive():
            self.robot.stopFreedrive()
            self.free_move_btn.setText("Free Move: OFF")
            self.free_move_btn.setStyleSheet("")
        
        # Stop the update thread (called via signal so this runs on main thread)
        if self.update_thread is not None and self.update_thread.isRunning():
            self.update_thread.stop()
            self.update_thread = None
        
        # Disconnect via UniversalRobot
        if self.robot is not None:
            self.robot.disconnect()
        
        # Update connection state
        self.connected = False
        self.protective_stopped = False  # Reset flag for next connection
        
        # Update status label with specific stop type
        self.status_label.setText(f"Status: {stop_type}")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # Update connection button to show reconnect option
        self.connection_btn.setToolTip("Reconnect to robot")
        
        # Disable control buttons when not connected
        self.free_move_btn.setEnabled(False)
        self.tcp_tracking_btn.setEnabled(False)
        self.show_points_btn.setEnabled(False)
        self.show_flange_points_btn.setEnabled(False)
        self.clear_points_btn.setEnabled(False)
        self.clear_flange_points_btn.setEnabled(False)
        self.set_tcp_offset_btn.setEnabled(False)
        self.reset_tcp_offset_btn_panel.setEnabled(False)
        self.align_ref_btn.setEnabled(False)
        self.set_ref_btn.setEnabled(False)
        self.clear_ref_btn.setEnabled(False)
        self.motion_combo.setEnabled(False)
        self.home_go_btn.setEnabled(False)
        self.home_set_btn.setEnabled(False)
        self.align_tcp_btn.setEnabled(False)
        
        # Render the updated state
        self.vtk_widget.GetRenderWindow().Render()
    
    def _on_motion_type_changed(self, index):
        """Handle motion type dropdown change."""
        motion_type = self.motion_combo.itemData(index)
        
        self.hide_waypoints_visualization()
        self._motion_path_visible = False
        self.motion_show_path_btn.setText("○")
        
        # Clear visualization on previously visible widget
        current_idx = self.motion_stack.currentIndex()
        if current_idx > 0 and current_idx - 1 < len(self.motion_ids):
            prev_id = self.motion_ids[current_idx - 1]
            widget = self.motion_widgets.get(prev_id)
            if widget:
                if hasattr(widget, '_remove_endpoint_visualization'):
                    widget._remove_endpoint_visualization()
                widget._hide_path_visualization()
        
        if motion_type and motion_type in self.motion_ids:
            stack_idx = self.motion_ids.index(motion_type) + 1
            self.motion_stack.setCurrentIndex(stack_idx)
            widget = self.motion_widgets[motion_type]
            if hasattr(widget, '_update_button_states'):
                widget._update_button_states()
        else:
            self.motion_stack.setCurrentIndex(0)
        
        self._update_motion_path_buttons()
    
    def _update_motion_path_buttons(self):
        """Update path button states based on current motion widget state."""
        motion_type = self.motion_combo.currentData()
        if not motion_type or motion_type not in self.motion_widgets:
            self.motion_show_path_btn.setEnabled(False)
            self.motion_clear_path_btn.setEnabled(False)
            return
        widget = self.motion_widgets[motion_type]
        if motion_type == "axial_rotation":
            self.motion_show_path_btn.setEnabled(False)
            self.motion_clear_path_btn.setEnabled(False)
            return
        if motion_type == "freemove":
            from src.movements import freemove
            has_path = freemove.path_exists()
            self.motion_show_path_btn.setEnabled(has_path)
            self.motion_clear_path_btn.setEnabled(has_path)
            return
        if hasattr(widget, 'direction_group') and hasattr(widget, '_get_current_method_module'):
            checked_btn = widget.direction_group.checkedButton()
            if checked_btn:
                method_module = widget._get_current_method_module()
                has_path = method_module.path_exists(checked_btn.direction)
                self.motion_show_path_btn.setEnabled(has_path)
                self.motion_clear_path_btn.setEnabled(has_path)
            else:
                self.motion_show_path_btn.setEnabled(False)
                self.motion_clear_path_btn.setEnabled(False)
        else:
            self.motion_show_path_btn.setEnabled(False)
            self.motion_clear_path_btn.setEnabled(False)
    
    def _get_path_file_for_current_motion(self):
        """Get path file for current motion, or None if not applicable."""
        motion_type = self.motion_combo.currentData()
        if not motion_type or motion_type not in self.motion_widgets or motion_type == "axial_rotation":
            return None
        widget = self.motion_widgets[motion_type]
        if motion_type == "freemove":
            from src.movements import freemove
            return freemove.get_path_filename()
        if hasattr(widget, 'direction_group') and hasattr(widget, '_get_current_method_module'):
            checked_btn = widget.direction_group.checkedButton()
            if checked_btn:
                method_module = widget._get_current_method_module()
                return method_module.get_path_filename(checked_btn.direction)
        return None

    def _toggle_motion_path_visualization(self):
        """Toggle path visualization for current motion."""
        if hasattr(self, '_motion_path_visible') and self._motion_path_visible:
            self.hide_waypoints_visualization()
            self._motion_path_visible = False
            self.motion_show_path_btn.setText("○")
            return
        path_file = self._get_path_file_for_current_motion()
        if path_file and os.path.exists(path_file):
            if self.visualize_waypoints(path_file):
                self._motion_path_visible = True
                self.motion_show_path_btn.setText("◉")
    
    def _delete_motion_path(self):
        """Delete the current motion's path file."""
        from src.movements.waypoint_collector import deleteWaypointsFile
        motion_type = self.motion_combo.currentData()
        if not motion_type or motion_type == "axial_rotation":
            return
        path_file = self._get_path_file_for_current_motion()
        if not path_file:
            return
        if deleteWaypointsFile(path_file):
            print(f"Deleted path file: {path_file}")
            self.clear_waypoints_visualization()
            self._motion_path_visible = False
            self.motion_show_path_btn.setText("○")
            widget = self.motion_widgets.get(motion_type)
            if widget:
                if hasattr(widget, '_update_status_from_saved_path'):
                    widget._update_status_from_saved_path()
                elif hasattr(widget, '_on_direction_changed'):
                    widget._on_direction_changed(None, None)
        self._update_motion_path_buttons()
    
    def check_and_track_position(self, position: List[float]):
        """Check if TCP position should be tracked (1mm threshold) and track it."""
        if not self.tcp_tracking_enabled:
            return
        
        point_added = False
        with self.tcp_tracking_lock:
            if self.last_tracked_position is None:
                # Track first position (position is already a new tuple from slice)
                self.tcp_points.append(position)
                self.last_tracked_position = position
                point_added = True
                print(f"Tracked TCP point {len(self.tcp_points)}: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            else:
                # Calculate distance from last tracked position
                distance = math.sqrt(
                    (position[0] - self.last_tracked_position[0])**2 +
                    (position[1] - self.last_tracked_position[1])**2 +
                    (position[2] - self.last_tracked_position[2])**2
                )
                
                # Track if movement exceeds threshold
                if distance >= self.tcp_tracking_threshold:
                    self.tcp_points.append(position)
                    self.last_tracked_position = position
                    point_added = True
                    print(f"Tracked TCP point {len(self.tcp_points)}: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        
        # Update clear button state if a point was added
        if point_added:
            self.update_clear_button_state()
            self.update_clear_flange_button_state()

    def align_tcp_axes_to_base(self):
        """
        Align TCP orientation to base axes at current position by updating TCP offset only.
        Robot does not move; only the TCP offset is changed so TCP frame aligns with base.
        """
        if not self.connected:
            print("Error: Not connected; cannot align TCP orientation.")
            return

        # Avoid aligning while a movement worker is running
        if self.async_motion_runner is not None and self.async_motion_runner.isRunning():
            print("Warning: Cannot align TCP while a movement is running. Stop the movement first.")
            return

        # Get current TCP pose
        tcp_pose = self.robot.tcpPose
        flange_pose = self.robot.flangePose

        if tcp_pose is None or len(tcp_pose) < 6:
            print("Error: No valid TCP pose available to align.")
            return

        if flange_pose is None or len(flange_pose) < 6:
            print("Error: Could not determine flange pose; cannot align TCP offset.")
            return

        # Build transforms
        p_f = np.array(flange_pose[:3])
        rx_f, ry_f, rz_f = flange_pose[3:6]
        R_f = axis_angle_to_rotation_matrix(rx_f, ry_f, rz_f)

        p_tcp_des = np.array(tcp_pose[:3])  # Keep position
        R_tcp_des = np.eye(3)               # Align orientation to base

        # Compute offset transform: T_off = T_f^{-1} * T_desired
        R_off = R_f.T @ R_tcp_des
        p_off = R_f.T @ (p_tcp_des - p_f)

        # Convert R_off to axis-angle
        rx_off, ry_off, rz_off = rotation_matrix_to_axis_angle(R_off)
        new_tcp_offset = [float(p_off[0]), float(p_off[1]), float(p_off[2]), float(rx_off), float(ry_off), float(rz_off)]

        # Apply offset via UniversalRobot (handles both RTDE and internal state)
        if self.robot.isConnected():
            if not self.robot.setTcpOffset(new_tcp_offset):
                print("Error: Failed to set TCP offset for alignment")
                return

        # Update visualization/UI
        self.update_tcp_offset_display()
        self.update_visualization()

        # Track position if TCP tracking is enabled (position unchanged)
        if self.tcp_tracking_enabled:
            self.check_and_track_position(p_tcp_des.tolist())

        print(f"Aligned TCP orientation to base axes at position [{p_tcp_des[0]:.4f}, {p_tcp_des[1]:.4f}, {p_tcp_des[2]:.4f}] by updating TCP offset.")
    
    def check_and_track_flange_position(self, position: List[float]):
        """Check if flange position should be tracked (1mm threshold) and track it."""
        if not self.flange_tracking_enabled:
            return
        
        point_added = False
        with self.flange_tracking_lock:
            if self.last_flange_tracked_position is None:
                # Track first position (position is already a new tuple from slice)
                self.flange_points.append(position)
                self.last_flange_tracked_position = position
                point_added = True
                print(f"Tracked flange point {len(self.flange_points)}: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            else:
                # Calculate distance from last tracked position
                distance = math.sqrt(
                    (position[0] - self.last_flange_tracked_position[0])**2 +
                    (position[1] - self.last_flange_tracked_position[1])**2 +
                    (position[2] - self.last_flange_tracked_position[2])**2
                )
                
                # Track if movement exceeds threshold
                if distance >= self.flange_tracking_threshold:
                    self.flange_points.append(position)
                    self.last_flange_tracked_position = position
                    point_added = True
                    print(f"Tracked flange point {len(self.flange_points)}: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        
        # Update clear button state if a point was added
        if point_added:
            self.update_clear_button_state()
    
    def update_points_visualization(self):
        """Update tracked points visualization using separate actors for TCP and flange points."""
        # Safety check: ensure renderer is available
        if not hasattr(self, 'renderer') or self.renderer is None:
            return
        
        # Track how many points have been added to each actor
        if not hasattr(self, '_last_tcp_points_added'):
            self._last_tcp_points_added = 0
        if not hasattr(self, '_last_flange_points_added'):
            self._last_flange_points_added = 0
        
        # Get new TCP points
        with self.tcp_tracking_lock:
            tcp_current_count = len(self.tcp_points)
            if tcp_current_count > self._last_tcp_points_added:
                new_tcp_points = [list(self.tcp_points[i]) for i in range(self._last_tcp_points_added, tcp_current_count)]
                self._last_tcp_points_added = tcp_current_count
            else:
                new_tcp_points = []
        
        # Get new flange points
        with self.flange_tracking_lock:
            flange_current_count = len(self.flange_points)
            if flange_current_count > self._last_flange_points_added:
                new_flange_points = [list(self.flange_points[i]) for i in range(self._last_flange_points_added, flange_current_count)]
                self._last_flange_points_added = flange_current_count
            else:
                new_flange_points = []
        
        # Initialize TCP points actor if needed and we have TCP points
        if self.tcp_points_actor is None and (new_tcp_points or self.show_points):
            try:
                self.tcp_points_actor = TrackedPointsActor(color=[1, 1, 0], radius=0.005, resolution=20)  # Yellow
                self.tcp_points_actor.addToRenderer(self.renderer)
            except Exception as e:
                print(f"Error creating TCP points actor: {e}")
                self.tcp_points_actor = None
        
        # Initialize flange points actor if needed and we have flange points
        if self.flange_points_actor is None and (new_flange_points or self.show_flange_points):
            try:
                self.flange_points_actor = TrackedPointsActor(color=[0.1, 0.1, 0.1], radius=0.005, resolution=20)  # Dark gray
                self.flange_points_actor.addToRenderer(self.renderer)
            except Exception as e:
                print(f"Error creating flange points actor: {e}")
                self.flange_points_actor = None
        
        # Update TCP points actor visibility and add new points
        if self.tcp_points_actor is not None:
            self.tcp_points_actor.setVisibility(self.show_points)
            if new_tcp_points and self.show_points:
                try:
                    self.tcp_points_actor.addPoints(new_tcp_points)
                except Exception as e:
                    print(f"Error adding TCP points: {e}")
        
        # Update flange points actor visibility and add new points
        if self.flange_points_actor is not None:
            self.flange_points_actor.setVisibility(self.show_flange_points)
            if new_flange_points and self.show_flange_points:
                try:
                    self.flange_points_actor.addPoints(new_flange_points)
                except Exception as e:
                    print(f"Error adding flange points: {e}")
    
    def toggle_tcp_tracking(self):
        """Toggle TCP position tracking on/off."""
        if not self.connected:
            return
        
        self.tcp_tracking_enabled = not self.tcp_tracking_enabled
        
        if self.tcp_tracking_enabled:
            self.tcp_tracking_btn.setText("TCP Tracking")
            self.tcp_tracking_btn.setStyleSheet("background-color: blue;")
            # Reset TCP tracking state
            with self.tcp_tracking_lock:
                self.last_tracked_position = None
            # Automatically show points when tracking is enabled
            if not self.show_points:
                self.show_points = True
                self.show_points_btn.setText("◉")  # Filled circle for ON
                
            print("TCP Tracking enabled - TCP positions will be tracked for each 1mm movement")
        else:
            self.tcp_tracking_btn.setText("TCP Tracking")
            self.tcp_tracking_btn.setStyleSheet("")
            with self.tcp_tracking_lock:
                num_points = len(self.tcp_points)
            print(f"TCP Tracking disabled - {num_points} points were tracked")
        
        # Update clear button states
        self.update_clear_button_state()
        self.update_clear_flange_button_state()
    
    def toggle_flange_tracking(self):
        """Toggle flange position tracking on/off."""
        if not self.connected:
            return
        
        self.flange_tracking_enabled = not self.flange_tracking_enabled
        
        if self.flange_tracking_enabled:
            self.flange_tracking_btn.setText("Flange Tracking")
            self.flange_tracking_btn.setStyleSheet("background-color: blue;")
            # Reset flange tracking state
            with self.flange_tracking_lock:
                self.last_flange_tracked_position = None
            # Automatically show flange points when tracking is enabled
            if not self.show_flange_points:
                self.show_flange_points = True
                self.show_flange_points_btn.setText("◉")  # Filled circle for ON
                
            print("Flange Tracking enabled - Flange positions will be tracked for each 1mm movement")
        else:
            self.flange_tracking_btn.setText("Flange Tracking")
            self.flange_tracking_btn.setStyleSheet("")
            with self.flange_tracking_lock:
                num_points = len(self.flange_points)
            print(f"Flange Tracking disabled - {num_points} points were tracked")
        
        # Update clear button states
        self.update_clear_button_state()
        self.update_clear_flange_button_state()
    
    def clear_tcp_points(self):
        """Clear TCP tracked points and their visualization."""
        if not self.connected:
            return
        
        # Clear TCP tracked points data
        with self.tcp_tracking_lock:
            num_points = len(self.tcp_points)
            self.tcp_points.clear()
            self.last_tracked_position = None
        
        # Clear TCP points visualization
        if self.tcp_points_actor is not None:
            self.tcp_points_actor.clearPoints()
        self.vtk_widget.GetRenderWindow().Render()
        
        # Reset TCP points tracking counter
        self._last_tcp_points_added = 0
        
        # Update clear button states
        self.update_clear_button_state()
        self.update_clear_flange_button_state()
        
        print(f"Cleared {num_points} TCP tracked points")
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_show_sphere(self):
        """Toggle visualization of fitted sphere."""
        if self.fitted_sphere is None:
            return
        
        self.show_sphere = not self.show_sphere
        
        if self.show_sphere:
            self.show_sphere_btn.setText("◉")  # Filled circle for ON
            # Show sphere actor if it exists
            if self.fitted_sphere is not None:
                self.fitted_sphere.setVisibility(True)
        else:
            self.show_sphere_btn.setText("○")  # Hollow circle for OFF
            # Hide sphere actor
            if self.fitted_sphere is not None:
                self.fitted_sphere.setVisibility(False)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def delete_sphere(self):
        """Delete the fitted sphere."""
        if self.fitted_sphere is None:
            return
        
        # Remove sphere actor from renderer
        self.fitted_sphere.removeFromRenderer()
        self.fitted_sphere = None
        self.show_sphere = False  # Reset to default state (not visible)
        
        # Disable buttons
        self.show_sphere_btn.setEnabled(False)
        self.delete_sphere_btn.setEnabled(False)
        self.show_sphere_btn.setText("○")  # Reset to hollow circle
        
        self.vtk_widget.GetRenderWindow().Render()
        print("Fitted sphere deleted")
    
    def _toggle_pose_collapse(self):
        """Toggle visibility of the TCP pose data field."""
        visible = not self.pose_text.isVisible()
        self.pose_text.setVisible(visible)
        self.pose_collapse_btn.setText("▾" if visible else "▸")
    
    def _toggle_offset_collapse(self):
        """Toggle visibility of the TCP offset data field."""
        visible = not self.tcp_offset_text.isVisible()
        self.tcp_offset_text.setVisible(visible)
        self.offset_collapse_btn.setText("▾" if visible else "▸")
    
    def _toggle_ref_collapse(self):
        """Toggle visibility of the ref frame data field."""
        visible = not self.ref_frame_text.isVisible()
        self.ref_frame_text.setVisible(visible)
        self.ref_collapse_btn.setText("▾" if visible else "▸")
    
    def toggle_graphs_visibility(self):
        """Toggle the graphs widget visibility."""
        is_visible = self.graphs_widget.isVisible()
        self.graphs_widget.setVisible(not is_visible)
    
    def toggle_show_points(self):
        """Toggle visualization of TCP tracked points."""
        if not self.connected:
            return
        
        self.show_points = not self.show_points
        
        if self.show_points:
            self.show_points_btn.setText("◉")  # Filled circle for ON
        else:
            self.show_points_btn.setText("○")  # Hollow circle for OFF
        
        # Update TCP points visibility
        if self.tcp_points_actor is not None:
            self.tcp_points_actor.setVisibility(self.show_points)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def toggle_show_flange_points(self):
        """Toggle visualization of flange tracked points."""
        if not self.connected:
            return
        
        self.show_flange_points = not self.show_flange_points
        
        if self.show_flange_points:
            self.show_flange_points_btn.setText("◉")  # Filled circle for ON
        else:
            self.show_flange_points_btn.setText("○")  # Hollow circle for OFF
        
        # Update flange points visibility
        if self.flange_points_actor is not None:
            self.flange_points_actor.setVisibility(self.show_flange_points)
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def clear_flange_points(self):
        """Clear only flange tracked points and their visualization."""
        if not self.connected:
            return
        
        # Clear flange tracked points data
        with self.flange_tracking_lock:
            flange_num_points = len(self.flange_points)
            self.flange_points.clear()
            self.last_flange_tracked_position = None
        
        # Clear flange points visualization
        if self.flange_points_actor is not None:
            self.flange_points_actor.clearPoints()
        self.vtk_widget.GetRenderWindow().Render()
        
        # Reset flange points tracking counter
        self._last_flange_points_added = 0
        
        # Update clear button states
        self.update_clear_button_state()
        self.update_clear_flange_button_state()
        
        print(f"Cleared {flange_num_points} flange tracked points")
    
    def update_clear_button_state(self):
        """Update the TCP clear button and fit sphere button states."""
        with self.tcp_tracking_lock:
            has_tcp_points = len(self.tcp_points) > 0
            has_enough_tcp_points = len(self.tcp_points) > 10
        # Enable TCP clear button if there are TCP points and connected
        self.clear_points_btn.setEnabled(has_tcp_points and self.connected)
        # Update fit sphere button state - enable if there are more than 10 TCP points and connected
        self.fit_sphere_btn.setEnabled(has_enough_tcp_points and self.connected)

    def update_clear_flange_button_state(self):
        """Update the flange clear button enabled state."""
        with self.flange_tracking_lock:
            has_flange_points = len(self.flange_points) > 0
        # Enable flange clear button if there are flange points and connected
        self.clear_flange_points_btn.setEnabled(has_flange_points and self.connected)
    
    def visualize_waypoints(self, path_file: str) -> bool:
        """Load and visualize a path file as magenta points.
        
        Args:
            path_file: Full path to the .npz path file
            
        Returns:
            True if path was loaded and visualized, False otherwise
        """
        from src.movements.waypoint_collector import loadWaypoints, getWaypointsDisplayName
        
        if not os.path.exists(path_file):
            print(f"No waypoints file found: {path_file}")
            return False
        
        # Load waypoints directly (no need for full WaypointCollector)
        poses, timestamps = loadWaypoints(path_file)
        
        if poses is None or len(poses) == 0:
            print(f"Path file is empty or invalid: {path_file}")
            return False
        
        # Extract positions from poses (first 3 elements)
        tcp_positions = poses[:, :3].tolist()
        # Convert TCP poses to flange positions
        flange_positions = []
        for pose in poses:
            flange_pose = self.robot._calculateFlangePoseFromTcp(pose, self.robot.tcpOffset)
            flange_positions.append(flange_pose[:3])
        positions = np.vstack((tcp_positions, flange_positions))
        
        
        # Create or update path points actor (white, semi-transparent)
        if not hasattr(self, 'waypoint_actor') or self.waypoint_actor is None:
            self.waypoint_actor = TrackedPointsActor(color=[1.0, 1.0, 1.0], radius=0.003, resolution=16, opacity=0.5)
            self.waypoint_actor.addToRenderer(self.renderer)
        
        # Set the points
        self.waypoint_actor.setPoints(positions)
        self.waypoint_actor.setVisibility(True)
        
        self.vtk_widget.GetRenderWindow().Render()
        display_name = getWaypointsDisplayName(path_file)
        print(f"Visualized path '{display_name}' with {len(positions)} points")
        return True
    
    def hide_waypoints_visualization(self):
        """Hide the path visualization."""
        if hasattr(self, 'waypoint_actor') and self.waypoint_actor is not None:
            self.waypoint_actor.setVisibility(False)
            self.vtk_widget.GetRenderWindow().Render()
    
    def clear_waypoints_visualization(self):
        """Clear the path visualization (but don't delete file)."""
        if hasattr(self, 'waypoint_actor') and self.waypoint_actor is not None:
            self.waypoint_actor.clearPoints()
            self.waypoint_actor.setVisibility(False)
            self.vtk_widget.GetRenderWindow().Render()
    
    def is_waypoints_visible(self) -> bool:
        """Check if path visualization is currently visible."""
        if hasattr(self, 'waypoint_actor') and self.waypoint_actor is not None:
            return self.waypoint_actor.getActor().GetVisibility() == 1
        return False
    
    def fit_and_visualize_sphere(self):
        """Fit a sphere to TCP tracked points and visualize it."""
        if not self.connected:
            return
        
        # Get TCP points (sphere fitting uses TCP points)
        with self.tcp_tracking_lock:  # Hold lock to ensure consistent data
            if len(self.tcp_points) <= 10:
                print("Error: Need more than 10 TCP points to fit a sphere")
                return
            points = [list(point) for point in self.tcp_points]
        
        # Convert points to numpy array (sphere.py expects points in meters)
        points_array = np.array(points)
        
        try:
            # Fit sphere using the lstfit function
            radius, center = sphere.lstfit(points_array, plot=False)
            
            print(f"Fitted sphere - Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m, Radius: {radius:.4f} m")
            
            # Remove existing fitted sphere if present
            if self.fitted_sphere is not None:
                self.fitted_sphere.removeFromRenderer()
            
            # Create wireframe sphere actor (center is already in meters)
            self.fitted_sphere = WireframeSphereActor(
                list(center), radius, [1.0, 0.0, 1.0], resolution=20, linewidth=2  # Magenta color for fitted sphere
            )
            # Make sphere visible when fitted
            self.show_sphere = True
            self.fitted_sphere.addToRenderer(self.renderer)
            self.vtk_widget.GetRenderWindow().Render()
            
            # Enable the show and delete sphere buttons and update show button text
            self.show_sphere_btn.setEnabled(True)
            self.show_sphere_btn.setText("◉")  # Filled circle for ON
            self.delete_sphere_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error fitting sphere: {e}")
    
    def reset_tcp_offset_to_default(self):
        """Reset TCP offset to the saved default value (from settings)."""
        # Need RTDE interfaces
        if not self.connected or self.robot.rtdeControl is None or self.robot.rtdeReceive is None:
            print("Error: Not connected or RTDE interfaces not available")
            return
        
        try:
            # Set TCP offset to default (from config.yaml)
            default_offset = runtime_tcp_offset.copy()
            self.robot.setTcpOffset(default_offset)
            
            # Wait a moment for the robot controller to process the TCP offset change
            time.sleep(0.1)
            
            # Get the updated TCP pose immediately after setting offset
            try:                
                print(f"TCP offset reset to default: {default_offset}")
                print(f"  TCP pose after reset: [{', '.join(f'{x:.4f}' for x in self.robot.tcpPose)}]")
                
                # Force visualization update
                self.vtk_widget.GetRenderWindow().Render()
                
            except Exception as e:
                print(f"Warning: Could not verify TCP pose after reset: {e}")
            
            # Update the offset display
            self.update_tcp_offset_display()
                
        except Exception as e:
            print(f"Error resetting TCP offset: {e}")
            import traceback; traceback.print_exc()
    
    def update_tcp_offset_display(self):
        """Update the TCP offset display in the panel."""
        # Get TCP offset from robot
        tcp_offset = self.robot.tcpOffset if self.robot else None
        
        if tcp_offset is None:
            self.tcp_offset_text.setText("Not available")
            return
        
        try:
            # Update TCP offset display - orientation to the right of position
            x, y, z = tcp_offset[0], tcp_offset[1], tcp_offset[2]
            rx, ry, rz = tcp_offset[3], tcp_offset[4], tcp_offset[5]
            
            offset_text = f"Position (m)     Orientation (deg)\n"
            offset_text += f" X: {x:>9.4f}     RX: {np.rad2deg(rx):>9.4f}\n"
            offset_text += f" Y: {y:>9.4f}     RY: {np.rad2deg(ry):>9.4f}\n"
            offset_text += f" Z: {z:>9.4f}     RZ: {np.rad2deg(rz):>9.4f}"
            self.tcp_offset_text.setText(offset_text)
            
            # Update graphs widget rotation weight based on TCP offset magnitude
            if hasattr(self, 'graphs_widget'):
                tcp_offset_magnitude = np.sqrt(x**2 + y**2 + z**2)
                self.graphs_widget.set_rotation_weight(tcp_offset_magnitude)
            
        except Exception as e:
            print(f"Warning: Could not read TCP offset for display: {e}")
            self.tcp_offset_text.setText("Not available")
    
    def show_tcp_offset_dialog(self):
        """Show the TCP offset dialog."""
        dialog = TCPOffsetDialog(self)
        dialog.exec()
    
    # =========================================================================
    # REFERENCE FRAME METHODS
    # =========================================================================
    
    def align_ref_frame_to_base(self):
        """
        Set the ref frame offset so that the ref frame axes align with base axes
        at the current TCP position. The offset captures the rotation difference
        between TCP orientation and base (identity).
        """
        if not self.connected:
            print("Error: Not connected; cannot align ref frame.")
            return
        
        tcp_pose = self.robot.tcpPose
        if tcp_pose is None or len(tcp_pose) < 6:
            print("Error: No valid TCP pose available to align ref frame.")
            return
        
        R_tcp = axis_angle_to_rotation_matrix(tcp_pose[3], tcp_pose[4], tcp_pose[5])
        
        # We want the ref frame at the TCP position with base orientation.
        # RefFrame = TCP @ Offset  =>  R_ref = R_tcp @ R_offset
        # For R_ref = I (base-aligned): R_offset = R_tcp^T
        R_offset = R_tcp.T
        rx_off, ry_off, rz_off = rotation_matrix_to_axis_angle(R_offset)
        
        ref_offset = [0.0, 0.0, 0.0, float(rx_off), float(ry_off), float(rz_off)]
        self.robot.setRefFrameOffset(ref_offset)
        self.update_ref_frame_display()
        self.update_visualization()
        print("Ref frame aligned to base axes at TCP position.")
    
    def show_ref_frame_dialog(self):
        """Show the ref frame offset dialog."""
        dialog = RefFrameOffsetDialog(self)
        dialog.exec()
    
    def clear_ref_frame(self):
        """Clear the ref frame offset and hide the ref frame visualization."""
        if self.robot:
            self.robot.clearRefFrameOffset()
        self.update_ref_frame_display()
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_ref_frame_display(self):
        """Update the ref frame offset display in the panel."""
        ref_offset = self.robot.refFrameOffset if self.robot else None
        
        if ref_offset is None:
            self.ref_frame_text.setText("Not set")
            return
        
        x, y, z = ref_offset[0], ref_offset[1], ref_offset[2]
        rx, ry, rz = ref_offset[3], ref_offset[4], ref_offset[5]
        
        text = f"Position (m)     Orientation (deg)\n"
        text += f" X: {x:>9.4f}     RX: {np.rad2deg(rx):>9.4f}\n"
        text += f" Y: {y:>9.4f}     RY: {np.rad2deg(ry):>9.4f}\n"
        text += f" Z: {z:>9.4f}     RZ: {np.rad2deg(rz):>9.4f}"
        self.ref_frame_text.setText(text)
    
    def set_tcp_offset_from_sphere(self):
        """Set TCP offset using the fitted sphere center."""
        try:
            if self.fitted_sphere is None:
                print("Error: No fitted sphere available. Please fit a sphere first.")
                return
            
            # Get sphere center from the actor
            fitted_sphere_center = self.fitted_sphere.getCenter()
            if fitted_sphere_center is None or len(fitted_sphere_center) < 3 or not np.all(np.isfinite(fitted_sphere_center[:3])):
                print(f"Error: Invalid sphere center: {fitted_sphere_center}")
                return

            if not self.robot.isConnected():
                print("Error: Not connected to robot")
                return

            # Get current TCP pose from robot
            if not (current_tcp_pose := self.robot.tcpPose):
                print("Error: No TCP pose available")
                return
            
            if len(current_tcp_pose) < 6 or not np.all(np.isfinite(current_tcp_pose[:6])):
                print(f"Error: Invalid TCP pose: {current_tcp_pose}")
                return
            
            tcp_position = np.array(current_tcp_pose[:3])
            tcp_orientation = current_tcp_pose[3:6]
            
            # Calculate offset vector in base frame: from TCP to sphere center
            sphere_center_base = np.array(fitted_sphere_center[:3])
            offset_base = sphere_center_base - tcp_position
            
            # Transform offset from base frame to flange frame
            R_tcp_to_base = axis_angle_to_rotation_matrix(
                tcp_orientation[0], tcp_orientation[1], tcp_orientation[2]
            )
            if R_tcp_to_base is None or R_tcp_to_base.shape != (3, 3) or not np.all(np.isfinite(R_tcp_to_base)):
                print(f"Error: Failed to compute rotation matrix from orientation: {tcp_orientation}")
                return
            
            offset_flange = R_tcp_to_base.T @ offset_base
            if not np.all(np.isfinite(offset_flange)):
                print(f"Error: Calculated offset has invalid values: {offset_flange}")
                return
            
            # TCP offset format: [x, y, z, rx, ry, rz] relative to flange
            tcp_offset = [float(offset_flange[0]), float(offset_flange[1]), float(offset_flange[2]), 0.0, 0.0, 0.0]
            
            print(f"  Current TCP pose before offset change: [{current_tcp_pose[0]:.4f}, {current_tcp_pose[1]:.4f}, {current_tcp_pose[2]:.4f}]")
            print(f"  Sphere center (base frame): [{sphere_center_base[0]:.4f}, {sphere_center_base[1]:.4f}, {sphere_center_base[2]:.4f}] m")
            print(f"  Offset in base frame (TCP to sphere): [{offset_base[0]:.4f}, {offset_base[1]:.4f}, {offset_base[2]:.4f}] m")
            print(f"  TCP offset (flange frame): [{tcp_offset[0]:.4f}, {tcp_offset[1]:.4f}, {tcp_offset[2]:.4f}, {tcp_offset[3]:.4f}, {tcp_offset[4]:.4f}, {tcp_offset[5]:.4f}]")
            
            # Set the TCP offset on the robot
            if not self.robot.setTcpOffset(tcp_offset):
                print("Error: Failed to set TCP offset on robot")
                return
            time.sleep(0.1)
            
            # Get the updated TCP pose immediately after setting offset
            try:
                tcp_pose_after = self.robot.tcpPose
                
                print("TCP offset set from sphere center:")
                print(f"  Current TCP position (base frame): [{tcp_position[0]:.4f}, {tcp_position[1]:.4f}, {tcp_position[2]:.4f}] m")
                print(f"  Sphere center (base frame): [{sphere_center_base[0]:.4f}, {sphere_center_base[1]:.4f}, {sphere_center_base[2]:.4f}] m")
                print(f"  Offset in base frame (TCP to sphere): [{offset_base[0]:.4f}, {offset_base[1]:.4f}, {offset_base[2]:.4f}] m")
                print(f"  TCP offset (flange frame): [{tcp_offset[0]:.4f}, {tcp_offset[1]:.4f}, {tcp_offset[2]:.4f}, {tcp_offset[3]:.4f}, {tcp_offset[4]:.4f}, {tcp_offset[5]:.4f}]")
                print(f"  TCP pose after offset change: [{tcp_pose_after[0]:.4f}, {tcp_pose_after[1]:.4f}, {tcp_pose_after[2]:.4f}]")
                print(f"  Distance from sphere center: {math.sqrt((tcp_pose_after[0] - sphere_center_base[0])**2 + (tcp_pose_after[1] - sphere_center_base[1])**2 + (tcp_pose_after[2] - sphere_center_base[2])**2):.4f} m")
                
                # Force visualization update
                self.vtk_widget.GetRenderWindow().Render()
                
            except Exception as e:
                print(f"Warning: Could not verify TCP pose after offset change: {e}")
                import traceback; traceback.print_exc()
            
            # Update the offset display
            try:
                self.update_tcp_offset_display()
            except Exception as e:
                print(f"Warning: Could not update TCP offset display: {e}")
                import traceback; traceback.print_exc()
            
        except Exception as e:
            print(f"Error setting TCP offset from sphere: {e}")
            import traceback; traceback.print_exc()
    
    def update_home_button_states(self):
        """Update the home button states based on current pose comparison.
        
        - Go Home button: disabled when already at home position
        - Set Home button: disabled when already at home position
        """
        if not self.connected or self.robot is None:
            # When not connected, keep buttons disabled
            return
        
        if self.home_position is None:
            # No home position set yet
            self.home_go_btn.setEnabled(False)  # Can't go home if no home is set
            self.home_set_btn.setEnabled(True)  # Can set home
            return
        
        # Check if current pose matches home pose. Use stricter tolerance (0.5mm, ~0.3°)
        # so we only disable the home button when truly at home, avoiding "sometimes
        # doesn't work" when the user is close but not exactly there.
        if self.robot.isEqualToTcpPose(
            self.home_position,
            position_tolerance=0.0005,
            orientation_tolerance=0.005,
        ):
            # Already at home - disable both buttons
            self.home_go_btn.setEnabled(False)
            self.home_set_btn.setEnabled(False)
        else:
            # Not at home - enable both buttons
            self.home_go_btn.setEnabled(True)
            self.home_set_btn.setEnabled(True)
    
    def reset_home_to_default(self):
        """Reset the home position to the default value from config.yaml."""
        self.home_position = runtime_home_position.copy()
        print(f"Home position reset to default: [{', '.join(f'{x:.4f}' for x in self.home_position)}]")
        
        # Update button state if connected and we have a current pose
        if self.connected and self.robot is not None:
            self.update_home_button_states()
    
    def set_home(self):
        """Set the current TCP position as the home position."""
        if not self.connected or self.robot is None:
            print("Error: Not connected to robot")
            return
        
        # Get current TCP pose from robot
        pose = self.robot.tcpPose
        if pose is None:
            print("Error: No TCP pose available")
            return
        
        self.home_position = list(pose)
        
        # Update button states immediately (will disable both since poses now match)
        self.update_home_button_states()
        print(f"Home position set: [{', '.join(f'{x:.4f}' for x in self.home_position)}]")
    
    def _show_home_context_menu(self, pos):
        """Show context menu for home position button."""
        menu = QMenu(self)
        
        # Save current home position as default in config.yaml
        save_action = QAction("Hardcode current position as default", self)
        save_action.triggered.connect(self._save_home_as_default)
        menu.addAction(save_action)
        
        menu.exec(self.home_set_btn.mapToGlobal(pos))
    
    def _save_home_as_default(self):
        """Save the current robot position as the default home position by updating config.yaml."""
        # First, get the current robot position and set it as home
        if not self.connected or self.robot is None:
            print("Error: Not connected to robot")
            return
        
        pose = self.robot.tcpPose
        if pose is None:
            print("Error: No TCP pose available")
            return
                
        # Set as current session's home position
        self.home_position = list(pose)
        self.update_home_button_states()
        
        # Now save to config.yaml
        try:
            from config import update_home_position_in_config
            update_home_position_in_config(self.home_position)
            print(f"Home position saved to config.yaml: [{', '.join(f'{x:.4f}' for x in self.home_position)}]")
        except Exception as e:
            print(f"Error saving home position to config.yaml: {e}")
    
    def _show_tcp_offset_context_menu(self, pos):
        """Show context menu for TCP offset button."""
        menu = QMenu(self)
        
        # Save current TCP offset as default in config.yaml
        save_action = QAction("Hardcode current TCP as default", self)
        save_action.triggered.connect(self._save_tcp_offset_as_default)
        menu.addAction(save_action)
        
        menu.exec(self.set_tcp_offset_btn.mapToGlobal(pos))
    
    def _save_tcp_offset_as_default(self):
        """Save the current TCP offset as the default by updating config.yaml."""
        if self.robot.tcpOffset is None:
            print("Error: No TCP offset set. Set a TCP offset first.")
            return
        
        try:
            from config import update_tcp_offset_in_config
            update_tcp_offset_in_config(self.robot.tcpOffset)
            print(f"TCP offset saved to config.yaml: [{', '.join(f'{x:.4f}' for x in self.robot.tcpOffset)}]")
        except Exception as e:
            print(f"Error saving TCP offset to config.yaml: {e}")
    
    def stop_movement(self):
        """Stop any currently running movement or freemove collection."""
        stopped_something = False
        
        # Check if freemove widget is collecting
        freemove_widget = self.motion_widgets.get('freemove') if hasattr(self, 'motion_widgets') else None
        if freemove_widget and hasattr(freemove_widget, 'stop_if_collecting') and freemove_widget.stop_if_collecting():
            print("Freemove collection stopped")
            stopped_something = True
        
        # Check if async motion runner is running
        if self.async_motion_runner is not None and self.async_motion_runner.isRunning():
            print("Stopping movement...")
            self.async_motion_runner.stop()
            # Try to stop the robot movement if RTDE control is available.
            # Only call stopL when the program is actually running to avoid
            # "RTDE control script is not running!" when the script is already idle.
            if self.robot.rtdeControl is not None:
                try:
                    if self.robot.rtdeControl.isProgramRunning():
                        self.robot.rtdeControl.stopL()
                        print("Robot movement stopped")
                except Exception as e:
                    print(f"Warning: Error stopping robot movement: {e}")
            # Wait for worker to finish (with timeout)
            if not self.async_motion_runner.wait(1000):  # Wait up to 1 second
                print("Warning: Movement worker did not stop within timeout")
            self.async_motion_runner = None
            self.stop_btn.setEnabled(False)
            print("Movement stopped")
            stopped_something = True
        
        if not stopped_something:
            print("No movement in progress")
    
    def _zero_force_sensor(self):
        """Zero the force/torque sensor."""
        if not self.connected or self.robot is None:
            print("Not connected to robot")
            return
        
        if self.robot.zeroFtSensor():
            print("Force/torque sensor zeroed")
        else:
            print("Error zeroing force/torque sensor")

    def toggle_free_move(self):
        """Toggle robot freedrive mode on/off."""
        if not self.connected or self.robot is None:
            return
        
        if not self.robot.isFreedriveActive():
            # Enable freedrive mode
            if self.robot.startFreedrive():
                self.free_move_btn.setText("Free Move: ON")
                self.free_move_btn.setStyleSheet("background-color: green;")
                print("Freedrive mode enabled - you can now manually move the robot")
            else:
                print("Error enabling freedrive mode")
        else:
            # Disable freedrive mode
            if self.robot.stopFreedrive():
                self.free_move_btn.setText("Free Move: OFF")
                self.free_move_btn.setStyleSheet("")
                print("Freedrive mode disabled")
            else:
                print("Error disabling freedrive mode")
    
    def _disable_freedrive_for_movement(self, movement_name: str = "movement"):
        """Disable freedrive mode if active (required before robot movements).
        
        Args:
            movement_name: Name of movement for logging purposes
        """
        if self.robot is not None and self.robot.isFreedriveActive():
            if self.robot.stopFreedrive():
                self.free_move_btn.setText("Free Move: OFF")
                self.free_move_btn.setStyleSheet("")
                print(f"Freedrive mode disabled for {movement_name}")
            else:
                print(f"Warning: Error disabling freedrive mode for {movement_name}")
    
    
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop graphs updates
        if hasattr(self, 'graphs_widget'):
            self.graphs_widget.stop_updates()
        
        # Disable freedrive mode if active and disconnect via UniversalRobot
        if self.robot is not None:
            if self.robot.isFreedriveActive():
                self.robot.stopFreedrive()
            self.robot.disconnect()
        
        # Clear tracked points visualization
        if self.tcp_points_actor is not None:
            self.tcp_points_actor.removeFromRenderer()
            self.tcp_points_actor = None
        if self.flange_points_actor is not None:
            self.flange_points_actor.removeFromRenderer()
            self.flange_points_actor = None
        
        # Stop any running movement worker
        if self.async_motion_runner is not None and self.async_motion_runner.isRunning():
            self.async_motion_runner.stop()
            self.async_motion_runner.wait(1000)  # Wait up to 1 second for thread to finish
        
        self.connected = False
        if self.update_thread is not None and self.update_thread.isRunning():
            self.update_thread.stop()
        
        # Close log files and restore original stdout/stderr
        if self.terminal_stream_stdout:
            self.terminal_stream_stdout.close()
            sys.stdout = self.original_stdout
        if self.terminal_stream_stderr:
            self.terminal_stream_stderr.close()
            sys.stderr = self.original_stderr
        
        event.accept()


def main():
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

    window = TCPVisualizer()
    window.show()
    
    app.exec()


if __name__ == "__main__":
    main()


