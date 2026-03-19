import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix
from config import defaults as CONFIG

def execute(self):
    """Execute flexion-y movement (rotation about TCP y-axis, controlling Fx while maintaining Fz=0)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    if startPosition is None:
        self.movement_progress.emit("Error: start_position is required")
        return
    if newPose is None:
        self.movement_progress.emit("Error: new_pose is required")
        return
    speed = self.kwargs.get('speed', CONFIG.flexion_y.speed)
    accel = self.kwargs.get('accel', CONFIG.flexion_y.acceleration)
    forceControlGain = self.kwargs.get('force_control_gain', CONFIG.flexion_y_original.force_control_gain)
    forceDeadband = self.kwargs.get('force_deadband', CONFIG.flexion_y_original.force_deadband)
    maxAdjustmentPerStep = self.kwargs.get('max_adjustment_per_step', CONFIG.flexion_y_original.max_adjustment_per_step)
    minAdjustmentInterval = self.kwargs.get('min_adjustment_interval', CONFIG.flexion_y_original.min_adjustment_interval)
    forceLimitX = self.kwargs.get('force_limit_x', CONFIG.flexion_y.force_limit_x)
    momentLimitY = self.kwargs.get('max_moment', getattr(CONFIG.flexion_y, 'max_moment', 3.0))
    pathFile = self.kwargs.get('path_file', None)
    
    # Determine direction from pathFile for direction-dependent force limit
    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'
    
    self.movement_started.emit()
    self.movement_progress.emit("Starting flexion-y movement (rotation about TCP y-axis)...")
    
    try:
        # Zero the force sensor
        self.rtde_c.zeroFtSensor()
        time.sleep(0.1)
        
        # Collect initial position
        self.waypointCollector.collect(startPosition)
        
        # Execute movement (asynchronous)
        currentTargetPose = newPose.copy()
        adjustmentCount = 0
        lastAdjustmentTime = time.time()
        
        self.movement_progress.emit(f"Moving to new pose (speed: {speed} m/s, accel: {accel} m/s²)...")
        success = self.rtde_c.moveL(currentTargetPose, speed, accel, asynchronous=True)
        
        if not success:
            self.movement_progress.emit("Movement command returned False")
            return
        
        self.movement_progress.emit("Asynchronous operation started...")
        self.movement_progress.emit("Active force control: Maintaining force_z_tcp at 0.0 N")
        
        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            progress = self.rtde_c.getAsyncOperationProgress()
            if progress >= 0:
                self.movement_progress.emit(f"Asynchronous operation progress: {progress}")
            
            tcpPose = list(self.rtde_r.getActualTCPPose())
            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            refWrench = self.robot.getRefFrameForceInRefFrame(self.robot.getRefFrameRelativeTo())
            if refWrench is None:
                refWrench = [0.0] * 6
            forceXRef = refWrench[0]  # Force limit uses Ref frame Fx
            forceZTcp = tcpForce[2]
            momentYTcp = tcpForce[4]  # Moment limit uses TCP My
            forceMagnitude = math.sqrt(tcpForce[0]**2 + tcpForce[1]**2 + tcpForce[2]**2)
            
            # Check limits: force in Ref frame, moment in TCP frame
            forceExceeded = False
            momentExceeded = False
            if direction == 'left':
                forceLimitDisplay = f"limit: < {forceLimitX:.2f} N"
                momentLimitDisplay = f"limit: < {momentLimitY:.2f} Nm"
                if forceXRef > forceLimitX:
                    forceExceeded = True
                if momentLimitY > 0 and momentYTcp > momentLimitY:
                    momentExceeded = True
            elif direction == 'right':
                forceLimitDisplay = f"limit: > -{forceLimitX:.2f} N"
                momentLimitDisplay = f"limit: > -{momentLimitY:.2f} Nm"
                if forceXRef < -forceLimitX:
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
            
            self.movement_progress.emit(
                f"Force (Ref frame Fx for limit):\n"
                f"  Fx: {forceXRef:.3f} N ({forceLimitDisplay})\n"
                f"  Fz: {forceZTcp:.3f} N (target: 0.0 N)\n"
                f"  My: {momentYTcp:.3f} Nm ({momentLimitDisplay})\n"
                f"  Total: {forceMagnitude:.2f} N"
            )
            
            if forceExceeded or momentExceeded:
                msg = f"Limit exceeded! Fx: {forceXRef:.3f} N ({forceLimitDisplay}), My: {momentYTcp:.3f} Nm ({momentLimitDisplay}). Stopping movement."
                self.movement_progress.emit(msg)
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                time.sleep(1.0)
                return  # Runner handles retrace
            
            # Collect waypoint
            currentPose = list(self.rtde_r.getActualTCPPose())
            self.pose_updated.emit(currentPose)
            self.waypointCollector.collect(currentPose)
            
            # Active force control - adjust for Fz deviation
            currentTime = time.time()
            if abs(forceZTcp) > forceDeadband and (currentTime - lastAdjustmentTime) >= minAdjustmentInterval:
                if self._stop_requested:
                    break
                
                self.movement_progress.emit(f"Force deviation detected ({forceZTcp:.3f} N). Adjusting path...")
                
                rx, ry, rz = currentPose[3], currentPose[4], currentPose[5]
                R_tcp_to_base = axis_angle_to_rotation_matrix(rx, ry, rz)
                tcpZDirectionBase = R_tcp_to_base[:, 2]
                
                # XZ plane adjustment for flexion-y
                tcpZXz = np.array([tcpZDirectionBase[0], 0, tcpZDirectionBase[2]])
                tcpZXzNorm = np.linalg.norm(tcpZXz)
                
                if tcpZXzNorm > 0.01:
                    tcpZXzUnit = tcpZXz / tcpZXzNorm
                else:
                    tcpZXzUnit = np.array([0, 0, -1])
                
                adjustmentMagnitude = min(abs(forceZTcp) * forceControlGain, maxAdjustmentPerStep)
                adjustmentDirection = np.sign(forceZTcp) * tcpZXzUnit
                positionAdjustment = adjustmentDirection * adjustmentMagnitude
                
                adjustedTarget = currentTargetPose.copy()
                adjustedTarget[0] += positionAdjustment[0]  # Adjust x
                adjustedTarget[2] += positionAdjustment[2]  # Adjust z
                adjustedTarget[1] = startPosition[1]  # Keep y unchanged
                
                self.movement_progress.emit(f"Adjusting target: x += {positionAdjustment[0]:.6f} m, z += {positionAdjustment[2]:.6f} m")
                
                self.rtde_c.stopL()
                time.sleep(0.1)
                
                if self._stop_requested:
                    break
                
                currentTargetPose = adjustedTarget.copy()
                adjustmentCount += 1
                lastAdjustmentTime = currentTime
                
                self.movement_progress.emit(f"Resuming movement with adjusted target (adjustment #{adjustmentCount})...")
                success = self.rtde_c.moveL(currentTargetPose, speed, accel, asynchronous=True)
                
                if not success:
                    self.movement_progress.emit("Warning: Failed to start adjusted movement. Stopping.")
                    return
            
            time.sleep(0.1)
        
        # Stop robot if stop was requested
        if self._stop_requested:
            try:
                self.rtde_c.stopL()
            except Exception:
                pass
            self.movement_progress.emit("Movement stopped by user")
            return
        
        self.movement_progress.emit("Asynchronous operation completed.")
        
        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")
        # Runner handles waypoint save and retrace
        
    except Exception as e:
        self.movement_progress.emit(f"Error during flexion-y movement: {str(e)}")
        raise
