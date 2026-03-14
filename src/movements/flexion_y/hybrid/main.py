"""
Flexion Y Hybrid - Force mode wrapper around moveL.

This approach uses force mode to maintain Fz=0 while executing a standard moveL
for the rotation around TCP Y-axis. The robot follows the planned trajectory 
with Z-axis compliance.
"""

import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix, transform_wrench
from config import defaults as CONFIG


def execute(self):
    """Execute flexion-y movement using hybrid approach (force mode + moveL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.flexion_y.speed)
    accel = self.kwargs.get('accel', CONFIG.flexion_y.acceleration)
    forceLimitX = self.kwargs.get('force_limit_x', CONFIG.flexion_y.force_limit_x)
    pathFile = self.kwargs.get('path_file', None)
    
    # Force mode parameters
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.flexion_y_hybrid.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.flexion_y_hybrid.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.flexion_y_hybrid.force_mode_gain_scaling)
    
    # Determine direction from pathFile
    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'
    
    self.movement_started.emit()
    self.movement_progress.emit("Starting flexion-y movement (hybrid: force mode + moveL)...")
    
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
        
        self.movement_progress.emit(f"Force mode enabled: Fz compliance active (Z limit: {forceModeZLimit} m/s)")
        
        # Get initial task frame (current TCP pose)
        initialPose = list(self.rtde_r.getActualTCPPose())
        
        # Enter force mode with initial task frame
        self.rtde_c.forceMode(initialPose, selection_vector, target_wrench, force_type, limits)
        
        # =====================================================
        # EXECUTE MOVEL WITH FORCE MODE ACTIVE
        # =====================================================
        self.movement_progress.emit(f"Moving to target pose (speed: {speed} m/s, accel: {accel} m/s²)...")
        success = self.rtde_c.moveL(newPose, speed, accel, asynchronous=True)
        
        if not success:
            self.rtde_c.forceModeStop()
            self.movement_progress.emit("Movement command returned False")
            return
        
        self.movement_progress.emit("Asynchronous operation started with force compliance...")
        
        lastWaypointTime = time.time()
        waypointInterval = 0.05  # Collect waypoints every 50ms
        
        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            # Get current state
            currentPose = list(self.rtde_r.getActualTCPPose())
            
            # Update task frame to current TCP pose (keeps Z-compliance in TCP frame)
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)
            
            # Read forces in TCP frame
            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpForce = transform_wrench(currentPose, tcpWrenchInBase)
            forceXTcp = tcpForce[0]
            forceZTcp = tcpForce[2]
            forceMagnitude = math.sqrt(tcpForce[0]**2 + tcpForce[1]**2 + tcpForce[2]**2)
            
            # Build limit display string (flexion-y monitors Fx)
            if direction == 'left':
                limitDisplay = f"limit: < {forceLimitX:.2f} N"
            elif direction == 'right':
                limitDisplay = f"limit: > -{forceLimitX:.2f} N"
            else:
                limitDisplay = f"limit: ±{forceLimitX:.2f} N"
            
            # Check force limit (Fx in TCP frame for flexion-y)
            forceExceeded = False
            if direction == 'left':
                if forceXTcp > forceLimitX:
                    forceExceeded = True
            elif direction == 'right':
                if forceXTcp < -forceLimitX:
                    forceExceeded = True
            else:
                if abs(forceXTcp) > forceLimitX:
                    forceExceeded = True
            
            if forceExceeded:
                self.movement_progress.emit(
                    f"Force threshold exceeded! Fx: {forceXTcp:.3f} N ({limitDisplay}). Stopping movement."
                )
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                self.rtde_c.forceModeStop()
                time.sleep(0.5)
                return  # Runner handles retrace
            
            # Collect waypoint at intervals
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime
                
                # Progress update
                progress = self.rtde_c.getAsyncOperationProgress()
                self.movement_progress.emit(
                    f"Progress: {progress} | Fx: {forceXTcp:.2f} N ({limitDisplay}) | "
                    f"Fz: {forceZTcp:.3f} N (force-controlled)"
                )
            
            time.sleep(0.02)  # 50Hz monitoring loop
        
        # =====================================================
        # CLEANUP
        # =====================================================
        self.rtde_c.forceModeStop()
        
        if self._stop_requested:
            try:
                self.rtde_c.stopL()
            except Exception:
                pass
            self.movement_progress.emit("Movement stopped by user")
            return
        
        self.movement_progress.emit("Movement completed with force compliance.")
        
        # Final waypoint
        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")
        
    except Exception as e:
        # Ensure we exit force mode on error
        try:
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during flexion-y hybrid movement: {str(e)}")
        raise

