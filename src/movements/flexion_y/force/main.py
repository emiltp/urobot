"""
Flexion Y Force - Full force mode with speedL control loop.

This approach uses force mode for Fz=0 compliance combined with continuous
speedL commands for rotation control around TCP Y-axis. Provides maximum 
reactivity and smooth motion.
"""

import time
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from config import defaults as CONFIG


def execute(self):
    """Execute flexion-y movement using force mode with speedL control loop."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.flexion_y.speed)
    accel = self.kwargs.get('accel', CONFIG.flexion_y.acceleration)
    forceLimitX = self.kwargs.get('force_limit_x', CONFIG.flexion_y.force_limit_x)
    momentLimitY = self.kwargs.get('max_moment', getattr(CONFIG.flexion_y, 'max_moment', 3.0))
    pathFile = self.kwargs.get('path_file', None)
    
    # Force mode parameters
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.flexion_y_force.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.flexion_y_force.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.flexion_y_force.force_mode_gain_scaling)
    controlLoopDt = self.kwargs.get('control_loop_dt', CONFIG.flexion_y_force.control_loop_dt)
    rotationSpeedFactor = self.kwargs.get('rotation_speed_factor', CONFIG.flexion_y_force.rotation_speed_factor)
    
    # Determine direction from pathFile
    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'
    
    self.movement_started.emit()
    self.movement_progress.emit("Starting flexion-y movement (force mode + speedL)...")
    
    try:
        # Zero the force sensor
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)
        
        # Collect initial position
        self.waypointCollector.collect(startPosition)
        
        # =====================================================
        # FORCE MODE SETUP
        # =====================================================
        selection_vector = [0, 0, 1, 0, 0, 0]  # Only Z is force-controlled
        target_wrench = [0, 0, 0, 0, 0, 0]  # Target Fz = 0
        force_type = 2  # Frame not transformed
        limits = [0.1, 0.1, forceModeZLimit, 0.5, 0.5, 0.5]
        
        # Configure force mode parameters
        self.rtde_c.forceModeSetDamping(forceModeDamping)
        self.rtde_c.forceModeSetGainScaling(forceModeGainScaling)
        
        self.movement_progress.emit(f"Force mode enabled: Fz compliance active")
        
        # =====================================================
        # CALCULATE TARGET ROTATION
        # =====================================================
        startOrientation = np.array(startPosition[3:6])
        targetOrientation = np.array(newPose[3:6])
        initialRotationError = _calculate_rotation_error(startOrientation, targetOrientation)
        
        # Angular speed based on linear speed parameter
        angularSpeed = speed * rotationSpeedFactor  # rad/s
        
        self.movement_progress.emit(f"Rotating toward target at {angularSpeed:.3f} rad/s...")
        
        # =====================================================
        # CONTROL LOOP
        # =====================================================
        currentPose = list(self.rtde_r.getActualTCPPose())
        lastWaypointTime = time.time()
        waypointInterval = 0.05  # Collect waypoints every 50ms
        
        # Enter force mode
        self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)
        
        while not self._stop_requested:
            t_start = self.rtde_c.initPeriod()
            
            # Get current state
            currentPose = list(self.rtde_r.getActualTCPPose())
            currentOrientation = np.array(currentPose[3:6])
            
            # Calculate rotation error
            rotationError = _calculate_rotation_error(currentOrientation, targetOrientation)
            
            # Check if we've reached the target (within ~0.5 degrees)
            if rotationError < 0.01:
                self.movement_progress.emit("Target rotation reached!")
                break
            
            # =====================================================
            # FORCE MONITORING: force limit uses Ref frame, moment limit uses TCP
            # =====================================================
            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            refWrench = self.robot.getRefFrameForceInRefFrame(self.robot.getRefFrameRelativeTo())
            if refWrench is None:
                refWrench = [0.0] * 6
            forceXRef = refWrench[0]  # Force limit uses Ref frame Fx
            forceZTcp = tcpForce[2]
            momentYTcp = tcpForce[4]  # Moment limit uses TCP My
            
            # Check limits: force in Ref frame, moment in TCP frame
            forceExceeded = False
            momentExceeded = False
            if direction == 'left':
                forceLimitDisplay = f"limit: < {forceLimitX:.2f} N"                ###  forceLimitDisplay = f"limit: > -{forceLimitX:.2f} N"
                momentLimitDisplay = f"limit: < {momentLimitY:.2f} Nm"
                if forceXRef > forceLimitX:                                        ###  forceXRef < -forceLimitX:
                    forceExceeded = True
                if momentLimitY > 0 and momentYTcp > momentLimitY:
                    momentExceeded = True

            elif direction == 'right':
                forceLimitDisplay = f"limit: > -{forceLimitX:.2f} N"             ###   forceLimitDisplay = f"limit: < {forceLimitX:.2f} N"
                momentLimitDisplay = f"limit: > -{momentLimitY:.2f} Nm"
                if forceXRef < - forceLimitX:                                        ###  forceXRef > forceLimitX:
                    forceExceeded = True
                if momentLimitY > 0 and momentYTcp < -momentLimitY:
                    momentExceeded = True
            else:
                forceLimitDisplay = f"limit: ±{forceLimitX:.2f} N"
                momentLimitDisplay = f"limit: ±{momentLimitY:.2f} Nm"
                if abs(forceXRef) > forceLimitX:
                    forceExceeded = True
                if momentLimitY > 0 and abs(momentYTcp) > momentLimitY:
                    momentExceeded = True
            
            if forceExceeded or momentExceeded:
                msg = f"Limit exceeded! Fx: {forceXRef:.3f} N ({forceLimitDisplay}), My: {momentYTcp:.3f} Nm ({momentLimitDisplay}). Stopping."
                self.movement_progress.emit(msg)
                break
            
            # =====================================================
            # UPDATE TASK FRAME
            # =====================================================
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)
            
            # =====================================================
            # COMPUTE SPEED COMMAND
            # =====================================================
            # Direction toward target
            rotationDirection = _normalize_rotation_vector(targetOrientation - currentOrientation)
            
            # Slow down as we approach target
            speedScale = min(1.0, rotationError / 0.3)  # Slow down within 0.3 rad
            angularVelocity = rotationDirection * angularSpeed * speedScale
            
            # Speed vector: [vx, vy, vz, ωrx, ωry, ωrz] in base frame
            speedVector = [0, 0, 0, angularVelocity[0], angularVelocity[1], angularVelocity[2]]
            
            # Apply speed command
            self.rtde_c.speedL(speedVector, accel, controlLoopDt)
            
            # =====================================================
            # WAYPOINT COLLECTION
            # =====================================================
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime
                
                # Progress update
                progressPct = (1.0 - rotationError / initialRotationError) * 100 if initialRotationError > 0 else 100
                self.movement_progress.emit(
                    f"Progress: {progressPct:.1f}% | Fx: {forceXRef:.2f} N ({forceLimitDisplay}) | "
                    f"Fz: {forceZTcp:.3f} N (force-controlled) | My: {momentYTcp:.2f} Nm ({momentLimitDisplay})"
                )
            
            self.rtde_c.waitPeriod(t_start)
        
        # =====================================================
        # CLEANUP
        # =====================================================
        self.rtde_c.speedStop()
        self.rtde_c.forceModeStop()
        
        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return
        
        self.movement_progress.emit("Movement completed.")
        
        # Final waypoint
        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")
        
    except Exception as e:
        # Ensure we exit force mode on error
        try:
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during flexion-y force movement: {str(e)}")
        raise


def _calculate_rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    """Calculate the rotation error between two axis-angle representations."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    
    # Relative rotation
    R_error = R_target @ R_current.T
    
    # Extract angle from relative rotation
    r = Rotation.from_matrix(R_error)
    angle = np.linalg.norm(r.as_rotvec())
    
    return angle


def _normalize_rotation_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a rotation vector, handling zero case."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros(3)
    return vec / norm

