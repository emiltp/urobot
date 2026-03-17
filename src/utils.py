import math
from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation

def axis_angle_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix."""
    angle = math.sqrt(rx**2 + ry**2 + rz**2)
    if angle < 1e-6:
        return np.eye(3)
    axis = np.array([rx, ry, rz]) / angle
    r = Rotation.from_rotvec(axis * angle)
    return r.as_matrix()


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        (rx, ry, rz) axis-angle components
    """
    r = Rotation.from_matrix(R)
    rotvec = r.as_rotvec()
    return float(rotvec[0]), float(rotvec[1]), float(rotvec[2])


def rotate_pose_around_base_z(pose: List[float], angle_deg: float) -> List[float]:
    """
    Rotate the TCP frame around the base z-axis while keeping position fixed.
    
    Args:
        pose: Current TCP pose [x, y, z, rx, ry, rz]
        angle_deg: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        New TCP pose [x, y, z, rx_new, ry_new, rz_new] with rotated orientation
    """
    x, y, z = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]
    
    # Convert current orientation from axis-angle to rotation matrix
    R_current = axis_angle_to_rotation_matrix(rx, ry, rz)
    
    # Create rotation matrix for rotation around base z-axis
    angle_rad = math.radians(angle_deg)
    R_z_rotation = Rotation.from_euler('z', angle_rad, degrees=False).as_matrix()
    
    # Apply rotation: R_new = R_z_rotation @ R_current
    # This rotates the current orientation around base z-axis
    R_new = R_z_rotation @ R_current
    
    # Convert back to axis-angle
    rx_new, ry_new, rz_new = rotation_matrix_to_axis_angle(R_new)
    
    # Return new pose with original position and rotated orientation
    return [x, y, z, rx_new, ry_new, rz_new]


def rotate_pose_around_base_x(pose: List[float], angle_deg: float) -> List[float]:
    """
    Rotate the TCP frame around the base x-axis while keeping position fixed.
    
    Args:
        pose: Current TCP pose [x, y, z, rx, ry, rz]
        angle_deg: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        New TCP pose [x, y, z, rx_new, ry_new, rz_new] with rotated orientation
    """
    x, y, z = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]
    
    # Convert current orientation from axis-angle to rotation matrix
    R_current = axis_angle_to_rotation_matrix(rx, ry, rz)
    
    # Create rotation matrix for rotation around base x-axis
    angle_rad = math.radians(angle_deg)
    R_x_rotation = Rotation.from_euler('x', angle_rad, degrees=False).as_matrix()
    
    # Apply rotation: R_new = R_x_rotation @ R_current
    # This rotates the current orientation around base x-axis
    R_new = R_x_rotation @ R_current
    
    # Convert back to axis-angle
    rx_new, ry_new, rz_new = rotation_matrix_to_axis_angle(R_new)
    
    # Return new pose with original position and rotated orientation
    return [x, y, z, rx_new, ry_new, rz_new]



def transform_wrench(pose: List[float], wrench: List[float], flange_pose: Optional[List[float]] = None) -> List[float]:
    """
    Transform a wrench (force + moment) from base frame into the TCP frame.

    When flange_pose is provided and the wrench is measured at the flange,
    the moment is translated from flange to TCP before rotation, so the
    result is the wrench at TCP in TCP frame (same as getTcpForceInTcpFrame).

    Args:
        pose: TCP pose [x, y, z, rx, ry, rz] in base frame (axis-angle orientation)
        wrench: [Fx, Fy, Fz, Mx, My, Mz] in base frame (at flange if flange_pose given)
        flange_pose: Optional [x, y, z, rx, ry, rz] of flange in base frame.
            When provided, moment is translated flange -> TCP before rotation.

    Returns:
        Wrench in TCP frame as [Fx, Fy, Fz, Mx, My, Mz]
    """
    if len(wrench) < 3:
        return wrench

    rx, ry, rz = pose[3], pose[4], pose[5]
    R_base_tcp = axis_angle_to_rotation_matrix(rx, ry, rz)
    R_tcp_base = R_base_tcp.T  # inverse for force/moment transform

    force_base = np.array(wrench[:3], dtype=float)
    force_tcp = R_tcp_base @ force_base

    if len(wrench) >= 6:
        moment_base = np.array(wrench[3:6], dtype=float)
        # Translate moment from flange to TCP when flange pose available
        if flange_pose is not None and len(flange_pose) >= 3:
            r_base = np.array(pose[:3]) - np.array(flange_pose[:3])  # tcp - flange
            moment_base = moment_base - np.cross(r_base, force_base)
        moment_tcp = R_tcp_base @ moment_base
        return [
            float(force_tcp[0]), float(force_tcp[1]), float(force_tcp[2]),
            float(moment_tcp[0]), float(moment_tcp[1]), float(moment_tcp[2])
        ]

    return [float(force_tcp[0]), float(force_tcp[1]), float(force_tcp[2])]


def orbit_tcp_around_ref(tcp_pose: List[float], ref_pose: List[float],
                         angle_rad: float, axis: str = 'z') -> List[float]:
    """Orbit the TCP around the Ref frame's local axis.

    The TCP endpoint is placed at the same distance from the Ref frame origin
    as the current TCP, but rotated by angle_rad around the Ref frame's
    local axis. The TCP orientation is rotated by the same amount.

    Args:
        tcp_pose: Current TCP pose [x, y, z, rx, ry, rz] in base frame.
        ref_pose: Ref frame pose [x, y, z, rx, ry, rz] in base frame.
        angle_rad: Rotation angle in radians around the Ref frame axis.
        axis: 'x', 'y', or 'z' — which Ref frame axis to rotate around.

    Returns:
        Endpoint pose [x, y, z, rx, ry, rz] in base frame.
    """
    ref_pos = np.array(ref_pose[:3])
    R_ref = axis_angle_to_rotation_matrix(ref_pose[3], ref_pose[4], ref_pose[5])

    tcp_pos = np.array(tcp_pose[:3])
    R_tcp = axis_angle_to_rotation_matrix(tcp_pose[3], tcp_pose[4], tcp_pose[5])

    offset_base = tcp_pos - ref_pos
    offset_local = R_ref.T @ offset_base

    if axis == 'x':
        R_axis = Rotation.from_rotvec([angle_rad, 0, 0]).as_matrix()
    elif axis == 'y':
        R_axis = Rotation.from_rotvec([0, angle_rad, 0]).as_matrix()
    else:  # 'z' default
        R_axis = Rotation.from_rotvec([0, 0, angle_rad]).as_matrix()

    offset_local_rotated = R_axis @ offset_local
    new_pos = ref_pos + R_ref @ offset_local_rotated

    R_base_rotation = R_ref @ R_axis @ R_ref.T
    R_new = R_base_rotation @ R_tcp

    rx_new, ry_new, rz_new = rotation_matrix_to_axis_angle(R_new)
    return [float(new_pos[0]), float(new_pos[1]), float(new_pos[2]),
            float(rx_new), float(ry_new), float(rz_new)]


def pose_near(pose_a: List[float], pose_b: List[float],
              pos_tol: float = 0.002, rot_tol: float = 0.02) -> bool:
    """Check if two poses are near (position and orientation within tolerance).

    Args:
        pose_a: Pose [x, y, z, rx, ry, rz]
        pose_b: Pose [x, y, z, rx, ry, rz]
        pos_tol: Max position distance (m) to consider equal
        rot_tol: Max rotation error (rad) to consider equal

    Returns:
        True if poses are within tolerance.
    """
    pos_diff = np.sqrt(sum((pose_a[i] - pose_b[i]) ** 2 for i in range(3)))
    if pos_diff > pos_tol:
        return False

    R_a = axis_angle_to_rotation_matrix(pose_a[3], pose_a[4], pose_a[5])
    R_b = axis_angle_to_rotation_matrix(pose_b[3], pose_b[4], pose_b[5])
    R_err = R_b @ R_a.T
    r = Rotation.from_matrix(R_err)
    rot_angle = np.linalg.norm(r.as_rotvec())
    return rot_angle <= rot_tol