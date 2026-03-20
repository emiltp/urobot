"""
Axial Rotation Force - Full force mode with speedL control loop.

Uses force mode for full translational compliance (Fx=Fy=Fz=0) combined with
continuous speedL commands for rotation control around TCP z-axis. All three
positional axes float freely, providing maximum reactivity and smooth motion.
"""

import time
import os
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils import axis_angle_to_rotation_matrix
from config import defaults as CONFIG


def execute(self):
    """Execute axial rotation with full translational compliance (force mode + speedL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    if startPosition is None:
        self.movement_progress.emit("Error: start_position is required")
        return
    if newPose is None:
        self.movement_progress.emit("Error: new_pose is required")
        return
    speed = self.kwargs.get('speed', CONFIG.rotation.speed)
    accel = self.kwargs.get('accel', CONFIG.rotation.acceleration)
    momentLimitZ = self.kwargs.get('max_moment', CONFIG.rotation.max_moment)

    # Determine direction from path_file (for directional moment limits)
    direction = None
    pathFile = self.kwargs.get('path_file')
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    # Force mode parameters (defaults from rotation: z_limit / xy_limit in config.yaml)
    z_cfg = CONFIG.rotation.z_limit
    xy_cfg = CONFIG.rotation.xy_limit
    forceModeXYLimit = self.kwargs.get('force_mode_xy_limit', xy_cfg.force_mode_xy_limit)
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', z_cfg.force_mode_z_limit)
    zDamping = self.kwargs.get('force_mode_z_damping', z_cfg.force_mode_damping)
    xyDamping = self.kwargs.get('force_mode_xy_damping', xy_cfg.force_mode_damping)
    zGain = self.kwargs.get('force_mode_z_gain_scaling', z_cfg.force_mode_gain_scaling)
    xyGain = self.kwargs.get('force_mode_xy_gain_scaling', xy_cfg.force_mode_gain_scaling)
    forceModeDamping = (zDamping + xyDamping) / 2
    forceModeGainScaling = (zGain + xyGain) / 2
    controlLoopDt = self.kwargs.get('control_loop_dt', CONFIG.rotation_force.control_loop_dt)
    rotationSpeedFactor = self.kwargs.get('rotation_speed_factor', CONFIG.rotation_force.rotation_speed_factor)

    self.movement_started.emit()
    self.movement_progress.emit("Starting axial rotation (force: full compliance + speedL)...")

    try:
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)

        self.waypointCollector.collect(startPosition)

        # =====================================================
        # FORCE MODE SETUP — full translational compliance
        # =====================================================
        selection_vector = [1, 1, 1, 0, 0, 0]  # Fx, Fy, Fz force-controlled
        target_wrench = [0, 0, 0, 0, 0, 0]     # Target all forces = 0
        force_type = 2  # Frame not transformed
        limits = [forceModeXYLimit, forceModeXYLimit, forceModeZLimit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(forceModeDamping)
        self.rtde_c.forceModeSetGainScaling(forceModeGainScaling)

        self.movement_progress.emit(
            f"Force mode enabled: full compliance (XY limit: {forceModeXYLimit} m/s, Z limit: {forceModeZLimit} m/s)"
        )

        # =====================================================
        # CALCULATE TARGET ROTATION
        # =====================================================
        startOrientation = np.array(startPosition[3:6])
        targetOrientation = np.array(newPose[3:6])
        initialRotationError = _calculate_rotation_error(startOrientation, targetOrientation)

        angularSpeed = speed * rotationSpeedFactor  # rad/s

        self.movement_progress.emit(f"Rotating toward target at {angularSpeed:.3f} rad/s...")

        # =====================================================
        # CONTROL LOOP
        # =====================================================
        currentPose = list(self.rtde_r.getActualTCPPose())
        lastWaypointTime = time.time()
        waypointInterval = 0.05

        self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

        while not self._stop_requested:
            t_start = self.rtde_c.initPeriod()

            currentPose = list(self.rtde_r.getActualTCPPose())
            currentOrientation = np.array(currentPose[3:6])

            rotationError = _calculate_rotation_error(currentOrientation, targetOrientation)

            if rotationError < 0.01:
                self.movement_progress.emit("Target rotation reached!")
                break

            # =====================================================
            # FORCE / TORQUE MONITORING (wrench at TCP in TCP frame)
            # Directional moment limits like flexion_x/flexion_y
            # =====================================================
            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            momentZTcp = tcpForce[5]  # Mz in TCP frame
            forceXTcp = tcpForce[0]
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]

            momentExceeded = False
            if direction == 'left':
                momentLimitDisplay = f"limit: > -{momentLimitZ:.2f} Nm"
                if momentLimitZ > 0 and momentZTcp < -momentLimitZ:
                    momentExceeded = True
            elif direction == 'right':
                momentLimitDisplay = f"limit: < {momentLimitZ:.2f} Nm"
                if momentLimitZ > 0 and momentZTcp > momentLimitZ:
                    momentExceeded = True
            else:
                momentLimitDisplay = f"limit: ±{momentLimitZ:.2f} Nm"
                if momentLimitZ > 0 and abs(momentZTcp) > momentLimitZ:
                    momentExceeded = True

            if momentExceeded:
                self.movement_progress.emit(
                    f"Limit exceeded! Mz: {momentZTcp:.3f} Nm ({momentLimitDisplay}). Stopping."
                )
                break

            # =====================================================
            # UPDATE TASK FRAME
            # =====================================================
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            # =====================================================
            # COMPUTE SPEED COMMAND
            # =====================================================
            rotationDirection = _get_rotation_direction(currentOrientation, targetOrientation)

            speedScale = min(1.0, rotationError / 0.3)
            angularVelocity = rotationDirection * angularSpeed * speedScale

            speedVector = [0, 0, 0, angularVelocity[0], angularVelocity[1], angularVelocity[2]]
            self.rtde_c.speedL(speedVector, accel, controlLoopDt)

            # =====================================================
            # WAYPOINT COLLECTION
            # =====================================================
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progressPct = (1.0 - rotationError / initialRotationError) * 100 if initialRotationError > 0 else 100
                self.movement_progress.emit(
                    f"Progress: {progressPct:.1f}% | Mz: {momentZTcp:.2f} Nm ({momentLimitDisplay}) | "
                    f"Fx: {forceXTcp:.3f} Fy: {forceYTcp:.3f} Fz: {forceZTcp:.3f} N (all force-controlled)"
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

        self.movement_progress.emit("Movement completed with full translational compliance.")

        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(
            f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]"
        )

    except Exception as e:
        try:
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during rotation force movement: {str(e)}")
        raise


def _calculate_rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    """Calculate the rotation error between two axis-angle representations."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])

    R_error = R_target @ R_current.T

    r = Rotation.from_matrix(R_error)
    angle = np.linalg.norm(r.as_rotvec())

    return angle


def _normalize_rotation_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a rotation vector, handling zero case."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros(3)
    return vec / norm


def _get_rotation_direction(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return geodesic rotation direction from current to target. Avoids axis-angle subtraction near π."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    R_error = R_target @ R_current.T
    rotvec = Rotation.from_matrix(R_error).as_rotvec()
    norm = np.linalg.norm(rotvec)
    if norm < 1e-6:
        return np.zeros(3)
    return rotvec / norm
