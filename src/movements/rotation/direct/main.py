"""
Axial Rotation - Force mode wrapper around moveL.

This approach uses force mode to maintain Fz=0 (TCP z-axis compliance) while
executing a standard moveL for the rotation around TCP z-axis. The robot follows
the planned rotational trajectory while the TCP z-axis remains compliant.
"""

import time
import math
from config import defaults as CONFIG


def execute(self):
    """Execute rotation around TCP z-axis with force mode for Fz compliance."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.rotation.speed)
    accel = self.kwargs.get('accel', CONFIG.rotation.acceleration)
    maxMoment = self.kwargs.get('max_moment', CONFIG.rotation.max_moment)
    
    # Force mode parameters
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.rotation.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.rotation.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.rotation.force_mode_gain_scaling)
    
    self.movement_started.emit()
    self.movement_progress.emit("Starting axial rotation (with Fz compliance)...")
    
    try:
        # Zero force sensor
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
        self.movement_progress.emit(f"Moving to new pose (speed: {speed} m/s, accel: {accel} m/s²)...")
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
            
            # Read forces in TCP frame (wrench at TCP in TCP frame)
            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            torqueMagnitudeZ = abs(tcpForce[5])
            forceZTcp = tcpForce[2]
            
            # Check torque limit (Mz in TCP frame)
            if torqueMagnitudeZ > maxMoment:
                self.movement_progress.emit(
                    f"Contact detected! Torque {torqueMagnitudeZ:.2f} Nm exceeds limit {maxMoment:.2f} Nm. Stopping."
                )
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                self.rtde_c.forceModeStop()
                time.sleep(0.5)
                # Collect final pose at contact point
                contactPose = list(self.rtde_r.getActualTCPPose())
                self.waypointCollector.collect(contactPose)
                self.pose_updated.emit(contactPose)
                self.movement_progress.emit(f"Contact pose: [{contactPose[0]:.4f}, {contactPose[1]:.4f}, {contactPose[2]:.4f}]")
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
                    f"Progress: {progress} | Mz: {torqueMagnitudeZ:.2f} Nm (limit: {maxMoment:.2f}) | "
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
        self.movement_progress.emit(f"Error during axial rotation: {str(e)}")
        raise
