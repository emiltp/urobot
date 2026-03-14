import numpy as np
import time
import math
from PyQt6.QtCore import QTimer
from config import defaults as CONFIG

from src.movements.waypoint_collector import WaypointCollector, getReversedWaypoints


def goDirect(app):
    """Move the TCP to the home position.
    
    Args:
        app: The main application instance
    """
    
    if app.home_position is None:
        print("Error: No home position set. Please set home position first.")
        return
    
    # Need RTDE interfaces
    if not app.connected or app.robot.rtdeControl is None or app.robot.rtdeReceive is None:
        print("Error: Not connected or RTDE interfaces not available")
        return
    
    # Check if a movement is already in progress
    if app.async_motion_runner is not None and app.async_motion_runner.isRunning():
        print("Warning: A movement is already in progress. Please wait for it to complete.")
        return
    
    # Visual feedback: change button appearance when pressed
    
    original_style = app.home_go_btn.styleSheet()
    app.home_go_btn.setStyleSheet("""
        QPushButton {
            background-color: #FF8C00;
            border: 1px solid #FF9C10;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #FF9C10;
            border: 1px solid #FFAC20;
        }
        QPushButton:pressed {
            background-color: #FF7C00;
            border: 1px solid #FF8C00;
        }
    """)
    app.home_go_btn.setToolTip("Moving to home position...")
    
    
    # Disable freedrive mode if active (required for movement)
    app._disable_freedrive_for_movement("home movement")
    
    # Force visualization update
    app.vtk_widget.GetRenderWindow().Render()
    
    # Create and start movement worker thread
    speed = CONFIG.movement.speed
    accel = CONFIG.movement.acceleration
    
    def on_movement_started():
        print(f"Moving to home position...")
        app.stop_btn.setEnabled(True)  # Enable stop button when movement starts
    
    def on_movement_progress(message):
        print(message)
    
    def on_pose_updated(pose):
        # Check for TCP tracking if enabled
        if app.tcp_tracking_enabled:
            position = [pose[0], pose[1], pose[2]]
            app.check_and_track_position(position)
        
        # Check for flange tracking if enabled
        if app.flange_tracking_enabled:
            # Try to get flange pose from RTDE
            flange_position = None
            if app.robot.rtdeReceive is not None:
                try:
                    flange_pose = list(app.robot.rtdeReceive.getActualToolFlangePose())
                    flange_position = [flange_pose[0], flange_pose[1], flange_pose[2]]
                except (AttributeError, Exception):
                    # If flange pose not available, calculate from TCP pose and offset
                    if app.robot.tcpOffset is not None and app.robot is not None:
                        try:
                            flange_pose = app.robot._calculateFlangePoseFromTcp(pose, app.robot.tcpOffset)
                            flange_position = [flange_pose[0], flange_pose[1], flange_pose[2]]
                        except Exception:
                            pass
            
            if flange_position is not None:
                app.check_and_track_flange_position(flange_position)
        app.vtk_widget.GetRenderWindow().Render()
    
    def on_movement_completed(success, message):
        print(message)
        app.stop_btn.setEnabled(False)  # Disable stop button when movement completes
        if success:
            # Restore button appearance after a short delay
            QTimer.singleShot(1000, lambda: app.home_go_btn.setStyleSheet(original_style))
            QTimer.singleShot(1000, lambda: app.home_go_btn.setToolTip("Go to home position"))
            # Update button states now that we're at home (disable go home since we're there)
            app.update_home_button_states()
        else:
            # Restore button appearance immediately on failure
            app.home_go_btn.setStyleSheet(original_style)
            app.home_go_btn.setToolTip("Go to home position")
        # Clean up worker
        app.async_motion_runner = None
    
    # Lazy import to avoid circular imports
    from src.movements.async_motion_runner import AsyncMotionRunner
    from src.movements import home

    app.async_motion_runner = AsyncMotionRunner(
        mode=AsyncMotionRunner.MODE_DIRECT_RETURN_TO_HOME,
        robot=app.robot,
        func=home,
        home_position=app.home_position,
        speed=speed,
        accel=accel
    )
    app.async_motion_runner.movement_started.connect(on_movement_started)
    app.async_motion_runner.movement_progress.connect(on_movement_progress)
    app.async_motion_runner.pose_updated.connect(on_pose_updated)
    app.async_motion_runner.movement_completed.connect(on_movement_completed)
    app.async_motion_runner.start()


def executeDirectReturnHome(self):
    print("Executing home movement directly...")
    
    """Execute home movement - uses recorded path if available and close to end position."""
    homePosition = self.kwargs.get('home_position')
    speed = self.kwargs.get('speed', CONFIG.movement.speed)
    accel = self.kwargs.get('accel', CONFIG.movement.acceleration)
    
    self.movement_started.emit()
    
    try:
        # Get current position
        currentPose = list(self.rtde_r.getActualTCPPose())
        
        # Try to load collected waypoints
        collector = WaypointCollector.load(self)
        
        useCollectedWaypoints = False
        if collector is not None and collector.getWaypointCount() > 0:
            waypoints = collector.getWaypoints()
            # Get end position of collected waypoints (where robot stopped)
            endPose = waypoints[-1]
            
            # Calculate distance from current position to end of collected waypoints
            positionDiff = np.array(currentPose[:3]) - np.array(endPose[:3])
            distance = np.linalg.norm(positionDiff)
            
            # Calculate orientation difference
            currentOrientation = currentPose[3:6]
            endOrientation = endPose[3:6]
            orientationDiff = computeOrientationDifference(currentOrientation, endOrientation)
            orientationDiffDeg = np.degrees(orientationDiff)
            
            # Position threshold: 1cm, Orientation threshold: 5 degrees
            positionOk = distance < 0.01
            orientationOk = orientationDiffDeg < 5.0
            
            # If within thresholds for both position and orientation, use collected waypoints
            if positionOk and orientationOk:
                useCollectedWaypoints = True
                self.movement_progress.emit(f"Close to waypoints end (pos: {distance*1000:.1f}mm, rot: {orientationDiffDeg:.1f}°) - retracing waypoints for return")
            elif positionOk and not orientationOk:
                self.movement_progress.emit(f"Position close ({distance*1000:.1f}mm) but orientation too different ({orientationDiffDeg:.1f}°) - using direct move")
            elif not positionOk and orientationOk:
                self.movement_progress.emit(f"Orientation close ({orientationDiffDeg:.1f}°) but position too far ({distance*1000:.1f}mm) - using direct move")
        
        if useCollectedWaypoints:
            # Use collected waypoints for smooth return
            self.movement_progress.emit("Retracing collected waypoints...")
            reversedPoses, reversedTimestamps = getReversedWaypoints(
                targetAngularVelocity=CONFIG.movement.return_angular_velocity
            )
            
            if reversedPoses is not None:
                # Create a temporary collector for retrace
                retraceCollector = WaypointCollector(self)
                retraceCollector.waypoints = reversedPoses
                retraceCollector.timestamps = reversedTimestamps
                
                success = retraceCollector.backwardTraverse(speed=speed)
                
                if success:
                    self.movement_progress.emit("Successfully retraced waypoints")
                    actualPose = list(self.rtde_r.getActualTCPPose())
                    self.pose_updated.emit(actualPose)
                    self.movement_completed.emit(True, "Movement completed successfully")
                else:
                    self.movement_progress.emit("Retrace interrupted - completing with direct move")
                    # Fall back to direct move to home
                    self.rtde_c.moveL(homePosition, speed, accel)
                    actualPose = list(self.rtde_r.getActualTCPPose())
                    self.pose_updated.emit(actualPose)
                    self.movement_completed.emit(True, "Movement completed")
            else:
                # Fallback to direct move
                self.movement_progress.emit("No valid collected waypoints - using direct move")
                success = self.rtde_c.moveL(homePosition, speed, accel)
                if success:
                    actualPose = list(self.rtde_r.getActualTCPPose())
                    self.pose_updated.emit(actualPose)
                self.movement_completed.emit(success, "Movement completed" if success else "Movement failed")
        else:
            # Direct move to home position
            self.movement_progress.emit("Moving to home position (direct)...")
            success = self.rtde_c.moveL(homePosition, speed, accel)
            if success:
                self.movement_progress.emit("Successfully moved to home position")
                try:
                    actualPose = list(self.rtde_r.getActualTCPPose())
                    self.pose_updated.emit(actualPose)
                except Exception as e:
                    self.movement_progress.emit(f"Warning: Could not get actual pose: {e}")
                self.movement_completed.emit(True, "Movement completed successfully")
            else:
                self.movement_completed.emit(False, "Movement command returned False")
    except Exception as e:
        self.movement_completed.emit(False, f"Error moving to home: {str(e)}")


def computeOrientationDifference(orientation1, orientation2):
    """Compute the angular difference between two axis-angle orientations.
    
    Args:
        orientation1: [rx, ry, rz] axis-angle representation
        orientation2: [rx, ry, rz] axis-angle representation
        
    Returns:
        Angular difference in radians
    """
    def axisAngleToRotationMatrix(rx, ry, rz):
        angle = np.sqrt(rx**2 + ry**2 + rz**2)
        if angle < 1e-10:
            return np.eye(3)
        axis = np.array([rx, ry, rz]) / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    R1 = axisAngleToRotationMatrix(*orientation1)
    R2 = axisAngleToRotationMatrix(*orientation2)
    
    # Compute relative rotation: R_diff = R1^T @ R2
    R_diff = R1.T @ R2
    
    # Extract angle from rotation matrix using trace
    traceVal = np.trace(R_diff)
    cosAngle = np.clip((traceVal - 1) / 2, -1.0, 1.0)
    angleDiff = np.arccos(cosAngle)
    
    return angleDiff
