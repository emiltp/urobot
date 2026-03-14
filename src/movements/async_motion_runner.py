"""
Async Motion Runner - Handles robot movements with waypoint collection and traversal.

Supports two modes:
- COLLECT: Execute movement while collecting waypoints
- TRAVERSE: Smoothly traverse a previously recorded path (with optional auto-return)
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from typing import Optional, Callable, List

from src.movements.waypoint_collector import WaypointCollector, DEFAULT_WAYPOINTS_FILE
from config import defaults as CONFIG, ConfigKeyNotFoundError


class AsyncMotionRunner(QThread):
    """Worker thread for executing robot movements with waypoint collection and traversal.
    
    Modes:
        - 'collect': Execute movement while collecting waypoints
        - 'traverse': Smoothly traverse a recorded path (with optional auto-return and motion logging)
    """
    
    # Mode constants
    MODE_COLLECT = 'collect'
    MODE_TRAVERSE = 'traverse'
    MODE_RETURN_TO_HOME = 'return_to_home'
    MODE_FREEMOVE = 'freemove'
    MODE_DIRECT_RETURN_TO_HOME = 'direct_return_to_home'
    
    # Signals for communication with main thread
    movement_started = pyqtSignal()
    movement_progress = pyqtSignal(str)  # Progress message
    movement_completed = pyqtSignal(bool, str)  # success, message
    error_dialog_requested = pyqtSignal(str)   # message for config/file errors (show QMessageBox)
    pose_updated = pyqtSignal(list)  # Updated pose
    path_saved = pyqtSignal(str)  # Emitted when path is saved (filepath)
    
    def __init__(self, mode: str, robot, func: Optional[Callable] = None, **kwargs):
        """
        Initialize the async motion runner.
        
        Args:
            mode: 'collect' or 'traverse'
            robot: UniversalRobot instance
            func: Execute function for collect mode (receives self as argument)
            **kwargs: Mode-specific parameters:
            
            For COLLECT mode:
                func: The execute function to run (e.g., flexion_x.execute)
                path_file: Optional path to save collected waypoints
                ... movement-specific parameters
                
            For TRAVERSE mode:
                path_file: Path to the waypoints file
                speed: Traversal speed in m/s
                acceleration: Acceleration in m/s²
                blend: Blend radius in meters
                traverseMethod: 'moveLPath', 'servoPath', or 'movePath'
                enableForceControl: Enable force limit checking
                forceLimit: Force limit in N
                forceAxis: 'x' or 'y' for force monitoring
                direction: 'left' or 'right' for direction-dependent limits
                autoReturn: Whether to auto-return after traversal (default: False)
                motionLogFile: Optional path for motion logging
        """
        super().__init__()
        self.mode = mode
        self.robot = robot
        self.rtde_c = robot.rtdeControl
        self.rtde_r = robot.rtdeReceive
        self.kwargs = kwargs
        self._stop_requested = False
        
        # The execute function for collect mode
        self.func = func
        
        # Waypoint collection (for collect mode)
        self.waypointCollector: Optional[WaypointCollector] = None
        
        # Motion logger (for traverse mode)
        self._motionLogger = None
    
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True
    
    def run(self):
        """Execute based on mode."""

        motionLogFile = self.kwargs.get('motionLogFile')
        
        # Start motion logging if specified
        if motionLogFile:
            try:
                from src.motion_logger import MotionLogger
                self._motionLogger = MotionLogger(self.robot, motionLogFile)
                self._motionLogger.start()
            except Exception as e:
                self.movement_progress.emit(f"Warning: Could not start motion logging: {e}")
                return

        try:
            match self.mode:
                case self.MODE_COLLECT:
                    self._runCollectMode()
                case self.MODE_TRAVERSE:
                    self._runTraverseMode(forward=True, backward=True)
                case self.MODE_DIRECT_RETURN_TO_HOME:
                    self._runDirectReturnHomeMode()
                case _:
                    self.movement_completed.emit(False, f"Unknown mode: {self.mode}")
        except ConfigKeyNotFoundError as e:
            self.error_dialog_requested.emit(str(e))
            self.movement_completed.emit(False, str(e))
        except Exception as e:
            self.movement_completed.emit(False, f"Error: {str(e)}")

        finally:
            # Stop motion logging
            if self._motionLogger is not None:
                self._motionLogger.stop()
    
    def _runCollectMode(self):
        """Execute movement while collecting waypoints."""
        if self.func is None:
            self.movement_completed.emit(False, "No execute function provided for collect mode")
            return
        
        # Initialize waypoint collector
        self.waypointCollector = WaypointCollector(self)
        self.waypointCollector.start()
        
        executionError = None
        try:
            # Execute the provided function
            self.func.execute(self)
        except Exception as e:
            executionError = e
            self.movement_progress.emit(f"Execution error: {e}")
        finally:
            # Stop collecting
            self.waypointCollector.stop()
        
        # Check if any waypoints were collected
        if self.waypointCollector.getWaypointCount() == 0: 
            self.movement_completed.emit(False, "No waypoints collected")
            return

        # Save waypoints if path_file specified
        pathFile = self.kwargs.get('path_file')
        if pathFile:
            self.waypointCollector.save(pathFile)
            self.path_saved.emit(pathFile)

        # Run backward traverse to return home
        self._runBackwardTraverseMode(self.waypointCollector, **self.kwargs)
            
    
    def _runTraverseMode(self, forward=None, backward=None, collector=None):
        """Smoothly traverse a recorded path with optional motion logging and auto-return."""
        
        # Load waypoints from file or use provided arrays
        collector = collector or WaypointCollector.load(self, self.kwargs.get('path_file'))
        if collector is None:
            self.movement_completed.emit(False, f"Could not load waypoints from {self.kwargs.get('path_file')}")
            return
        
        autoReturn = self.kwargs.get('autoReturn', False)
        willBackward = backward or autoReturn

        if forward:
            # Don't emit completion if backward traverse will follow
            success = self._runForwardTraverseMode(collector, emitCompletion=not willBackward, **self.kwargs)
            if not success: return
            
            if self._stop_requested:
                self.movement_completed.emit(False, "Traversal stopped by user")
                return  
        
        if autoReturn:
            # Auto-return via backward traverse (unless stopped by user)
            self.movement_progress.emit("Waiting 1 second before returning home...")
            time.sleep(1.0)

        if willBackward:
            success = self._runBackwardTraverseMode(collector, **self.kwargs)
        
        return success

    
    def _runForwardTraverseMode(self, collector, emitCompletion=True, **kwargs):
        """Forward traverse the recorded path.
        
        Args:
            collector: WaypointCollector with waypoints to traverse
            emitCompletion: If True, emit movement_completed signal. Set to False when
                           backward traverse will follow.
        """
        try:
            self.movement_started.emit()
            self.movement_progress.emit(f"Starting forward traverse ({collector.getWaypointCount()} waypoints)...")

            # Execute traversal
            success = collector.forwardTraverse(
                speed=kwargs.get('speed'),
                acceleration=kwargs.get('acceleration'),
                blend=kwargs.get('blend'),
                traverseMethod=kwargs.get('traverseMethod'),
                enableForceControl=kwargs.get('enableForceControl'),
                forceLimit=kwargs.get('forceLimit'),
                forceAxis=kwargs.get('forceAxis'),
                direction=kwargs.get('direction'))
            
            # Store traveled path for potential return
            if success:
                self.previousCollector = collector
                self.path_saved.emit(DEFAULT_WAYPOINTS_FILE)
            
            # Only emit completion if this is a standalone forward traverse
            if emitCompletion:
                if success:
                    self.movement_completed.emit(True, "Traversal completed successfully")
                elif self._stop_requested:
                    self.movement_completed.emit(False, "Traversal stopped by user")
                else:
                    self.movement_completed.emit(False, f"Traversal stopped (force limit at index {collector.traverseStopIndex})")
            else:
                # Just emit progress for combined traverse
                if not success and not self._stop_requested:
                    self.movement_progress.emit(f"Forward traverse stopped (force limit at index {collector.traverseStopIndex})")
        
        except Exception as e:
            self.movement_completed.emit(False, f"Error during forward traversal: {e}")
            return False
        
        return True

    def _runBackwardTraverseMode(self, collector, **kwargs):
        """Backward traverse the recorded path.
        
        For hybrid/force collection methods, uses force-compliant return (Fz=0).
        For original method, uses standard moveLPath return.
        """
        try:
            self.movement_started.emit()
            
            # Determine actual waypoint count for backward traverse
            waypointCount = collector.getWaypointCount()
            if collector.traverseStopIndex is not None and 0 <= collector.traverseStopIndex < waypointCount - 1:
                waypointCount = collector.traverseStopIndex + 1
            
            # Check collection method to determine return strategy
            collection_method = kwargs.get('collection_method', 'original')
            retrace_speed = kwargs.get('retrace_speed') or kwargs.get('speed')
            retrace_acceleration = kwargs.get('retrace_acceleration') or kwargs.get('acceleration')

            if collection_method in ('hybrid', 'force'):
                # Use force-compliant return for hybrid/force methods
                self.movement_progress.emit(f"Starting force-compliant backward traverse ({waypointCount} waypoints)...")
                success = collector.forceCompliantBackwardTraverse(
                    speed=retrace_speed,
                    acceleration=retrace_acceleration
                )
            else:
                # Use selected traverse method for original method (moveLPath, servoPath, movePath)
                self.movement_progress.emit(f"Starting backward traverse ({waypointCount} waypoints)...")
                success = collector.backwardTraverse(
                    speed=retrace_speed,
                    acceleration=retrace_acceleration,
                    blend=kwargs.get('blend'),
                    traverseMethod=kwargs.get('traverseMethod')
                )
            
            if success:
                self.movement_completed.emit(True, "Traversal completed successfully")
            elif self._stop_requested:
                self.movement_completed.emit(False, "Traversal stopped by user")
            else:
                self.movement_completed.emit(False, f"Traversal stopped (force limit at index {collector.traverseStopIndex})")
        
        except Exception as e:
            self.movement_completed.emit(False, f"Error during backward traversal: {e}")
            return False
        
        return True

    
    def collectWaypoint(self, pose):
        """Collect a waypoint during collect mode.
        
        Call this from movement functions to collect waypoints.
        
        Args:
            pose: TCP pose [x, y, z, rx, ry, rz]
        """
        if self.waypointCollector and self.waypointCollector.collecting:
            self.waypointCollector.collect(pose)
    
    def getCollectedWaypoints(self):
        """Get the recorded path after collect mode completes.
        
        Returns:
            Tuple of (poses, timestamps) or (None, None)
        """
        if self.recordedPoses is not None and self.recordedTimestamps is not None:
            return self.recordedPoses, self.recordedTimestamps
        return None, None
    def _runDirectReturnHomeMode(self):
        """Execute direct return to home."""
        from src.movements import home
        home.executeDirectReturnHome(self)

