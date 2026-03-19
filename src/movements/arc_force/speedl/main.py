"""
Arc (Fz-constrained) - Force mode + speedL control loop.

Uses force mode with Z-only compliance and speedL for velocity control
toward the target pose. Similar to rotation/force pattern.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils import orbit_tcp_around_ref, transform_wrench, pose_near
from src.utils import axis_angle_to_rotation_matrix
from config import defaults as CONFIG


def _rotation_error_to_omega(current_aa: np.ndarray, target_aa: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Compute angular velocity from orientation error (axis-angle)."""
    R_cur = axis_angle_to_rotation_matrix(current_aa[0], current_aa[1], current_aa[2])
    R_tgt = axis_angle_to_rotation_matrix(target_aa[0], target_aa[1], target_aa[2])
    R_err = R_tgt @ R_cur.T
    r = Rotation.from_matrix(R_err)
    rotvec = r.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-6:
        return np.zeros(3)
    axis = rotvec / angle
    omega = axis * min(angle * gain, 0.5)
    return omega


def execute(self):
    """Execute arc movement with force mode + speedL."""
    start_position = self.kwargs.get('start_position')
    ref_pose = self.kwargs.get('ref_pose')
    end_pose = self.kwargs.get('end_pose')
    angle_rad = self.kwargs.get('angle_rad')
    if start_position is None:
        self.movement_progress.emit("Error: start_position is required")
        return
    if ref_pose is None:
        self.movement_progress.emit("Error: ref_pose is required")
        return
    if end_pose is None:
        self.movement_progress.emit("Error: end_pose is required")
        return
    if angle_rad is None:
        self.movement_progress.emit("Error: angle_rad is required")
        return
    axis = self.kwargs.get('axis', 'z')
    direction_multiplier = self.kwargs.get('direction_multiplier', 1)
    speed = self.kwargs.get('speed', CONFIG.arc_force.speed)
    accel = self.kwargs.get('accel', CONFIG.arc_force.acceleration)
    max_moment = self.kwargs.get('max_moment', CONFIG.arc_force.max_moment)
    force_mode_z_limit = self.kwargs.get('force_mode_z_limit', CONFIG.arc_force.force_mode_z_limit)
    force_mode_damping = self.kwargs.get('force_mode_damping', CONFIG.arc_force.force_mode_damping)
    force_mode_gain_scaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.arc_force.force_mode_gain_scaling)
    timeout = self.kwargs.get('timeout', CONFIG.arc_force.timeout)
    fy_tolerance = self.kwargs.get('fy_tolerance', CONFIG.arc_force.fy_tolerance)
    pos_tolerance = self.kwargs.get('pos_tolerance', CONFIG.arc_force.pos_tolerance)
    rot_tolerance = self.kwargs.get('rot_tolerance', CONFIG.arc_force.rot_tolerance)
    control_loop_dt = self.kwargs.get('control_loop_dt', 0.008)
    force_baseline = self.kwargs.get('force_baseline', 'null')

    self.movement_started.emit()
    self.movement_progress.emit("Starting arc (speedL, Fz-constrained)...")

    try:
        # Always zero - UR force mode requires it; without zero the TCP drops
        self.movement_progress.emit("Nulling force sensor...")
        self.rtde_c.zeroFtSensor()
        time.sleep(0.3)

        initial_wrench_base = None
        if force_baseline == 'subtract_initial':
            # Capture post-zero residual (gravity, drift) as baseline for target and stop criteria
            self.movement_progress.emit("Capturing baseline for subtract-initial...")
            time.sleep(0.2)
            initial_wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(initial_wrench_base) < 6:
                initial_wrench_base = list(initial_wrench_base) + [0.0] * (6 - len(initial_wrench_base))

        self.waypointCollector.collect(start_position)

        selection_vector = [0, 0, 1, 0, 0, 0]
        force_type = 2
        limits = [0.1, 0.1, force_mode_z_limit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(force_mode_damping)
        self.rtde_c.forceModeSetGainScaling(force_mode_gain_scaling)

        self.movement_progress.emit("Force mode enabled (Z compliant)")

        current_pose = list(self.rtde_r.getActualTCPPose())
        initial_pose = list(current_pose)
        start_time = time.time()
        last_waypoint_time = start_time
        waypoint_interval = 0.05
        tcp_offset = self.robot.getTcpOffset() if self.robot else None

        # Target wrench: zero for null, or initial (base frame) transformed to current TCP frame
        flange_pose = self.robot._calculateFlangePoseFromTcp(current_pose, tcp_offset) if tcp_offset else None
        target_wrench = (
            transform_wrench(current_pose, initial_wrench_base, flange_pose)
            if initial_wrench_base is not None
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.rtde_c.forceMode(current_pose, selection_vector, target_wrench, force_type, limits)

        while not self._stop_requested:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.movement_progress.emit(f"Timeout ({timeout}s) reached. Stopping.")
                break

            # Target wrench: initial in base transformed to current TCP frame
            flange_pose = self.robot._calculateFlangePoseFromTcp(current_pose, tcp_offset) if tcp_offset else None
            target_wrench = (
                transform_wrench(current_pose, initial_wrench_base, flange_pose)
                if initial_wrench_base is not None
                else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.rtde_c.forceMode(current_pose, selection_vector, target_wrench, force_type, limits)

            # Check Fy (tangential) and Mx - directional based on motion direction
            wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(wrench_base) < 6:
                wrench_base = wrench_base + [0.0] * (6 - len(wrench_base))
            if initial_wrench_base is not None:
                wrench_base = [c - i for c, i in zip(wrench_base, initial_wrench_base)]
            wrench_tcp = transform_wrench(current_pose, wrench_base, flange_pose)
            fy = wrench_tcp[1]
            mx = wrench_tcp[3]

            fy_exceeded = (direction_multiplier > 0 and fy > fy_tolerance) or (
                direction_multiplier < 0 and fy < -fy_tolerance
            )
            mx_exceeded = (direction_multiplier > 0 and mx < -max_moment) or (
                direction_multiplier < 0 and mx > max_moment
            )
            if fy_exceeded:
                self.movement_progress.emit(f"Fy (tangential) {fy:.2f} N exceeds tolerance. Stopping.")
                break
            if mx_exceeded:
                self.movement_progress.emit(f"Mx {mx:.2f} Nm exceeds limit. Stopping.")
                break

            if pose_near(current_pose, end_pose, pos_tolerance, rot_tolerance):
                self.movement_progress.emit("Endpoint reached.")
                break

            # Velocity toward end_pose
            pos_err = np.array(end_pose[:3]) - np.array(current_pose[:3])
            pos_dist = np.linalg.norm(pos_err)
            if pos_dist > 1e-6:
                vel_trans = (pos_err / pos_dist) * min(speed, pos_dist / control_loop_dt)
            else:
                vel_trans = np.zeros(3)
            vel_trans[2] = 0  # Z is force-controlled

            omega = _rotation_error_to_omega(
                np.array(current_pose[3:6]), np.array(end_pose[3:6]), gain=3.0
            )
            speed_scale = min(1.0, (pos_dist / 0.05) + (np.linalg.norm(omega) / 0.3))
            vel_trans *= speed_scale
            omega *= speed_scale

            speed_vector = [float(vel_trans[0]), float(vel_trans[1]), float(vel_trans[2]),
                           float(omega[0]), float(omega[1]), float(omega[2])]
            t_start = self.rtde_c.initPeriod()
            self.rtde_c.speedL(speed_vector, accel, control_loop_dt)

            now = time.time()
            if now - last_waypoint_time >= waypoint_interval:
                self.pose_updated.emit(current_pose)
                self.waypointCollector.collect(current_pose)
                last_waypoint_time = now
                self.movement_progress.emit(f"Fy: {fy:.3f} N | Mx: {mx:.2f} Nm")

            current_pose = list(self.rtde_r.getActualTCPPose())
            self.rtde_c.waitPeriod(t_start)

        self.rtde_c.speedStop()
        self.rtde_c.forceModeStop()

        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Arc (speedL) completed.")
        final_pose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(final_pose)
        self.waypointCollector.collect(final_pose)

    except Exception as e:
        try:
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during arc speedL: {str(e)}")
        raise
