import math
from typing import List
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



def transform_wrench(pose: List[float], wrench: List[float]) -> List[float]:
    """
    Transform a wrench (force + moment) expressed in the base frame
    into the TCP frame given the TCP pose (axis-angle orientation).
    
    Args:
        pose: TCP pose [x, y, z, rx, ry, rz] in base frame (axis-angle orientation)
        wrench_base: [Fx, Fy, Fz, Mx, My, Mz] expressed in base frame
    
    Returns:
        Wrench expressed in TCP frame as [Fx, Fy, Fz, Mx, My, Mz]
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
        moment_tcp = R_tcp_base @ moment_base
        return [
            float(force_tcp[0]), float(force_tcp[1]), float(force_tcp[2]),
            float(moment_tcp[0]), float(moment_tcp[1]), float(moment_tcp[2])
        ]
    
    return [float(force_tcp[0]), float(force_tcp[1]), float(force_tcp[2])]