"""
Universal Robot class for controlling and visualizing a UR robot.

This class encapsulates:
- RTDE connection (control and receive interfaces)
- Robot control (movement, freedrive, stop)
- Pose calculations (TCP offset, flange pose)
- Visualization via UniversalRobotActor
"""

import threading
from typing import List, Optional, Tuple, Callable
import numpy as np
import vtk
import math

from .actors.universal_robot_actor import UniversalRobotActor
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from config import defaults as CONFIG

# Try to import RTDE interfaces
try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False


class UniversalRobot:
    """
    Universal Robot class containing RTDE connection, control, calculations, and visualization.
    
    This class provides a unified interface to:
    - Connect/disconnect from the robot via RTDE
    - Read robot state (TCP pose, flange pose, joint positions, forces)
    - Control robot movements (moveL, moveJ, freedrive, stop)
    - Manage TCP offset
    - Visualize robot state via UniversalRobotActor
    """
    
    def __init__(self, 
                 ip: Optional[str] = None,
                 autoConnect: bool = False,
                 tcpOffset: Optional[List[float]] = None,
                 baseScale: float = 0.5,
                 flangeScale: float = 0.2,
                 tcpScale: float = 0.25):
        """
        Initialize the Universal Robot.
        
        Args:
            ip: Robot IP address (optional, can be set later via connect())
            autoConnect: If True and ip is provided, automatically connect
            tcpOffset: Initial TCP offset [x, y, z, rx, ry, rz] (optional)
            baseScale: Scale factor for base axes visualization
            flangeScale: Scale factor for flange axes visualization
            tcpScale: Scale factor for TCP axes visualization
        """
        # Connection state
        self._ip: Optional[str] = ip
        self._connected: bool = False
        self._rtdeC = None  # RTDE control interface
        self._rtdeR = None  # RTDE receive interface
        self._connectionLock = threading.Lock()
        
        # Robot state
        self._tcpPose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz]
        self._flangePose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz]
        self._tcpOffset: Optional[List[float]] = tcpOffset.copy() if tcpOffset else None
        self._refFrameOffset: Optional[List[float]] = None  # [x, y, z, rx, ry, rz] relative to TCP
        self._jointPositions: Optional[List[float]] = None  # [q0, q1, q2, q3, q4, q5]
        self._tcpForce: Optional[List[float]] = None  # [Fx, Fy, Fz, Mx, My, Mz]
        self._stateLock = threading.Lock()
        
        # Freedrive state
        self._freedriveActive: bool = False
        
        # Protective stop state
        self._protectiveStopped: bool = False
        self._emergencyStopped: bool = False
        
        # Visualization actor
        self._actor = UniversalRobotActor(
            baseScale=baseScale,
            flangeScale=flangeScale,
            tcpScale=tcpScale
        )
        
        # Auto-connect if requested
        if autoConnect and ip:
            self.connect(ip)
    
    # =========================================================================
    # CONNECTION METHODS
    # =========================================================================
    
    def connect(self, ip: Optional[str] = None, tcpOffset: Optional[List[float]] = None) -> bool:
        """
        Connect to the robot via RTDE.
        
        Args:
            ip: Robot IP address (uses stored IP if not provided)
            tcpOffset: TCP offset to set on connection (optional)
        
        Returns:
            True if connection successful, False otherwise
        """
        if not RTDE_AVAILABLE:
            print("Error: ur-rtde library not available")
            return False
        
        if ip:
            self._ip = ip
        
        if not self._ip:
            print("Error: No IP address provided")
            return False
        
        with self._connectionLock:
            try:
                # Create RTDE interfaces
                self._rtdeR = rtde_receive.RTDEReceiveInterface(self._ip)
                self._rtdeC = rtde_control.RTDEControlInterface(self._ip)
                
                if not self._rtdeR.isConnected():
                    self._rtdeR = None
                    self._rtdeC = None
                    print(f"Error: Could not connect to robot at {self._ip}")
                    return False
                
                self._connected = True
                
                # Set TCP offset if provided
                if tcpOffset is not None:
                    self.setTcpOffset(tcpOffset)
                elif self._tcpOffset is not None:
                    # Restore existing TCP offset
                    self.setTcpOffset(self._tcpOffset)
                
                # Read initial state
                self._updateState()
                
                print(f"Connected to robot at {self._ip}")
                return True
                
            except Exception as e:
                print(f"Error connecting to robot: {e}")
                self._rtdeR = None
                self._rtdeC = None
                self._connected = False
                return False
    
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        with self._connectionLock:
            # End freedrive mode if active
            if self._freedriveActive and self._rtdeC is not None:
                try:
                    self._rtdeC.endFreedriveMode()
                except Exception:
                    pass
                self._freedriveActive = False
            
            # Disconnect control interface
            if self._rtdeC is not None:
                try:
                    self._rtdeC.disconnect()
                except Exception:
                    pass
                self._rtdeC = None
            
            # Disconnect receive interface
            if self._rtdeR is not None:
                try:
                    self._rtdeR.disconnect()
                except Exception:
                    pass
                self._rtdeR = None
            
            self._connected = False
            print("Disconnected from robot")
    
    def reconnect(self) -> bool:
        """Reconnect to the robot using the stored IP address."""
        self.disconnect()
        return self.connect()
    
    def isConnected(self) -> bool:
        """Check if connected to the robot."""
        if not self._connected or self._rtdeR is None:
            return False
        try:
            return self._rtdeR.isConnected()
        except Exception:
            return False
    
    @property
    def ip(self) -> Optional[str]:
        """Get the robot IP address."""
        return self._ip
    
    # =========================================================================
    # STATE READING METHODS
    # =========================================================================
    
    def _updateState(self) -> None:
        """Update internal state from robot (called internally)."""
        if not self.isConnected():
            return
        
        with self._stateLock:
            try:
                # Update protective stop state
                self._protectiveStopped = self._rtdeR.isProtectiveStopped()
                self._emergencyStopped = self._rtdeR.isEmergencyStopped()
                
                if self._protectiveStopped or self._emergencyStopped:
                    return
                
                # Read TCP pose
                self._tcpPose = list(self._rtdeR.getActualTCPPose())
                
                # Try to read flange pose
                try:
                    self._flangePose = list(self._rtdeR.getActualToolFlangePose())
                except (AttributeError, Exception):
                    # Calculate flange pose from TCP pose and offset
                    if self._tcpPose and self._tcpOffset:
                        self._flangePose = self._calculateFlangePoseFromTcp(
                            self._tcpPose, self._tcpOffset
                        )
                
                # Read joint positions
                try:
                    self._jointPositions = list(self._rtdeR.getActualQ())
                except Exception:
                    pass
                
                # Read TCP force
                try:
                    self._tcpForce = list(self._rtdeR.getActualTCPForce())
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"Error updating robot state: {e}")
    
    def update(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Update robot state and visualization.
        
        Returns:
            Tuple of (TCP pose, flange pose) if available, (None, None) otherwise
        """
        self._updateState()
        
        if self._tcpPose is not None:
            # Calculate flange pose if we have TCP offset but no flange pose
            flangePose = self._flangePose
            if flangePose is None and self._tcpOffset is not None:
                flangePose = self._calculateFlangePoseFromTcp(self._tcpPose, self._tcpOffset)
            
            # Update actor visualization with pre-calculated poses
            self._actor.updatePoses(self._tcpPose, flangePose)
            
            # Update ref frame visualization if offset is set
            if self._refFrameOffset is not None:
                refPose = self._calculateRefFramePose(self._tcpPose, self._refFrameOffset)
                self._actor.updateRefFramePose(refPose)
        
        return self._tcpPose, self._flangePose
    
    def getTcpPose(self) -> Optional[List[float]]:
        """
        Get the current TCP pose.
        
        Returns:
            TCP pose [x, y, z, rx, ry, rz] or None if not available
        """
        if not self.isConnected():
            return self._tcpPose
        
        try:
            self._tcpPose = list(self._rtdeR.getActualTCPPose())
            return self._tcpPose.copy()
        except Exception as e:
            print(f"Error reading TCP pose: {e}")
            return None
    
    def getFlangePose(self) -> Optional[List[float]]:
        """
        Get the current flange (tool flange) pose.
        
        Returns:
            Flange pose [x, y, z, rx, ry, rz] or None if not available
        """
        if not self.isConnected():
            return self._flangePose
        
        try:
            self._flangePose = list(self._rtdeR.getActualToolFlangePose())
            return self._flangePose.copy()
        except (AttributeError, Exception):
            # Calculate from TCP pose and offset if method not available
            if self._tcpPose and self._tcpOffset:
                self._flangePose = self._calculateFlangePoseFromTcp(self._tcpPose, self._tcpOffset)
                return self._flangePose.copy() if self._flangePose else None
            return None
    
    def getJointPositions(self) -> Optional[List[float]]:
        """
        Get the current joint positions.
        
        Returns:
            Joint positions [q0, q1, q2, q3, q4, q5] in radians or None
        """
        if not self.isConnected():
            return self._jointPositions
        
        try:
            self._jointPositions = list(self._rtdeR.getActualQ())
            return self._jointPositions.copy()
        except Exception:
            return None
    
    def getTcpForce(self) -> Optional[List[float]]:
        """
        Get the current TCP force/torque.
        
        Returns:
            TCP force [Fx, Fy, Fz, Mx, My, Mz] or None if not available
        """
        if not self.isConnected():
            return self._tcpForce
        
        try:
            self._tcpForce = list(self._rtdeR.getActualTCPForce())
            return self._tcpForce.copy()
        except Exception:
            return None
    
    def isProtectiveStopped(self) -> bool:
        """Check if the robot is in protective stop."""
        if not self.isConnected():
            return self._protectiveStopped
        try:
            self._protectiveStopped = self._rtdeR.isProtectiveStopped()
            return self._protectiveStopped
        except Exception:
            return self._protectiveStopped
    
    def isEmergencyStopped(self) -> bool:
        """Check if the robot is in emergency stop."""
        if not self.isConnected():
            return self._emergencyStopped
        try:
            self._emergencyStopped = self._rtdeR.isEmergencyStopped()
            return self._emergencyStopped
        except Exception:
            return self._emergencyStopped

    def isEqualToTcpPose(self, pose: List[float], position_tolerance: float = None, orientation_tolerance: float = None) -> bool:
        """Check if the current TCP pose is equal to the given TCP pose.
        
        Optional tighter tolerances can be passed for stricter equality checks
        (e.g. only consider "at home" when very close).
        """
        return _isEqualPoses(self.tcpPose, pose, position_tolerance, orientation_tolerance)
    
    # =========================================================================
    # TCP OFFSET METHODS
    # =========================================================================
    
    def getTcpOffset(self) -> Optional[List[float]]:
        """
        Get the current TCP offset.
        
        Returns:
            TCP offset [x, y, z, rx, ry, rz] or None
        """
        if self.isConnected() and self._rtdeC is not None:
            try:
                self._tcpOffset = list(self._rtdeC.getTCPOffset())
            except (AttributeError, Exception):
                pass  # Method might not exist in all RTDE versions
        
        return self._tcpOffset.copy() if self._tcpOffset else None
    
    def setTcpOffset(self, tcpOffset: List[float]) -> bool:
        """
        Set the TCP offset on the robot.
        
        Args:
            tcpOffset: TCP offset [x, y, z, rx, ry, rz]
        
        Returns:
            True if successful, False otherwise
        """
        if len(tcpOffset) != 6:
            print("Error: TCP offset must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        
        self._tcpOffset = tcpOffset.copy()
        
        if self.isConnected() and self._rtdeC is not None:
            try:
                self._rtdeC.setTcp(tcpOffset)
                print(f"TCP offset set: {tcpOffset}")
                return True
            except Exception as e:
                print(f"Error setting TCP offset: {e}")
                return False
        
        return True  # Stored for later
    
    # =========================================================================
    # REFERENCE FRAME OFFSET METHODS
    # =========================================================================
    
    def getRefFrameOffset(self) -> Optional[List[float]]:
        """
        Get the current reference frame offset relative to TCP.
        
        Returns:
            Ref frame offset [x, y, z, rx, ry, rz] or None
        """
        return self._refFrameOffset.copy() if self._refFrameOffset else None
    
    def setRefFrameOffset(self, offset: List[float]) -> bool:
        """
        Set the reference frame offset relative to TCP.
        
        Args:
            offset: Ref frame offset [x, y, z, rx, ry, rz] relative to TCP
        
        Returns:
            True if successful
        """
        if len(offset) != 6:
            print("Error: Ref frame offset must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        
        self._refFrameOffset = offset.copy()
        
        # Update visualization immediately if TCP pose is available
        if self._tcpPose is not None:
            refPose = self._calculateRefFramePose(self._tcpPose, self._refFrameOffset)
            self._actor.updateRefFramePose(refPose)
        
        print(f"Ref frame offset set: {offset}")
        return True
    
    def clearRefFrameOffset(self) -> None:
        """Clear the reference frame offset and hide the ref frame."""
        self._refFrameOffset = None
        self._actor.hideRefFrame()
        print("Ref frame offset cleared")
    
    @property
    def refFrameOffset(self) -> Optional[List[float]]:
        """Current ref frame offset [x, y, z, rx, ry, rz] relative to TCP."""
        return self._refFrameOffset.copy() if self._refFrameOffset else None
    
    # =========================================================================
    # CALCULATION METHODS
    # =========================================================================
    
    def _calculateFlangePoseFromTcp(self, tcpPose: List[float], tcpOffset: List[float]) -> List[float]:
        """
        Calculate flange pose from TCP pose and TCP offset.
        
        Args:
            tcpPose: TCP pose [x, y, z, rx, ry, rz] in base frame
            tcpOffset: TCP offset [x, y, z, rx, ry, rz] relative to flange
        
        Returns:
            Flange pose [x, y, z, rx, ry, rz] in base frame
        """
        # Extract TCP position and orientation
        tcpPos = np.array(tcpPose[:3])
        tcpRot = axis_angle_to_rotation_matrix(
            tcpPose[3], tcpPose[4], tcpPose[5]
        )
        
        # Extract TCP offset (in flange frame)
        offsetPos = np.array(tcpOffset[:3])
        offsetRot = axis_angle_to_rotation_matrix(
            tcpOffset[3], tcpOffset[4], tcpOffset[5]
        )
        
        # Transform offset from flange frame to base frame
        # Relationship: TCP = Flange @ Offset (in homogeneous transformation)
        # So: R_tcp_to_base = R_flange_to_base @ R_offset
        # Therefore: R_flange_to_base = R_tcp_to_base @ R_offset^T
        ROffsetInv = offsetRot.T
        RFlangeToBase = tcpRot @ ROffsetInv
        
        # Position: TCP_position_base = Flange_position_base + R_flange_to_base @ offset_position_flange
        # Therefore: Flange_position_base = TCP_position_base - R_flange_to_base @ offset_position_flange
        offsetPosBase = RFlangeToBase @ offsetPos
        flangePos = tcpPos - offsetPosBase
        
        # Convert flange rotation to axis-angle
        flangeRx, flangeRy, flangeRz = rotation_matrix_to_axis_angle(RFlangeToBase)
        
        return [
            float(flangePos[0]),
            float(flangePos[1]),
            float(flangePos[2]),
            float(flangeRx),
            float(flangeRy),
            float(flangeRz)
        ]
    
    def _calculateRefFramePose(self, tcpPose: List[float], refFrameOffset: List[float]) -> List[float]:
        """
        Calculate reference frame pose in base frame from TCP pose and ref frame offset.
        
        The ref frame offset is defined relative to the TCP frame:
            RefFrame = TCP @ Offset
        
        Args:
            tcpPose: TCP pose [x, y, z, rx, ry, rz] in base frame
            refFrameOffset: Offset [x, y, z, rx, ry, rz] relative to TCP
        
        Returns:
            Ref frame pose [x, y, z, rx, ry, rz] in base frame
        """
        tcpPos = np.array(tcpPose[:3])
        tcpRot = axis_angle_to_rotation_matrix(tcpPose[3], tcpPose[4], tcpPose[5])
        
        offsetPos = np.array(refFrameOffset[:3])
        offsetRot = axis_angle_to_rotation_matrix(refFrameOffset[3], refFrameOffset[4], refFrameOffset[5])
        
        # RefFrame_position_base = TCP_position_base + R_tcp_to_base @ offset_position_tcp
        refPos = tcpPos + tcpRot @ offsetPos
        
        # RefFrame_rotation_base = R_tcp_to_base @ R_offset
        refRot = tcpRot @ offsetRot
        
        rx, ry, rz = rotation_matrix_to_axis_angle(refRot)
        return [float(refPos[0]), float(refPos[1]), float(refPos[2]), float(rx), float(ry), float(rz)]
    
    def _calculateTcpOffsetFromPoses(self, tcpPose: List[float], flangePose: List[float]) -> List[float]:
        """
        Calculate TCP offset from TCP and flange poses.
        
        Args:
            tcpPose: TCP pose [x, y, z, rx, ry, rz] in base frame
            flangePose: Flange pose [x, y, z, rx, ry, rz] in base frame
        
        Returns:
            TCP offset [x, y, z, rx, ry, rz] relative to flange
        """
        # Extract positions and orientations
        tcpPos = np.array(tcpPose[:3])
        flangePos = np.array(flangePose[:3])
        
        RTcpToBase = axis_angle_to_rotation_matrix(
            tcpPose[3], tcpPose[4], tcpPose[5]
        )
        RFlangeToBase = axis_angle_to_rotation_matrix(
            flangePose[3], flangePose[4], flangePose[5]
        )
        
        # Offset in base frame: from flange to TCP
        offsetBase = tcpPos - flangePos
        
        # Transform offset from base frame to flange frame
        offsetFlange = RFlangeToBase.T @ offsetBase
        
        # Relative rotation: R_offset = R_flange_to_base^T @ R_tcp_to_base
        ROffset = RFlangeToBase.T @ RTcpToBase
        rxOffset, ryOffset, rzOffset = rotation_matrix_to_axis_angle(ROffset)
        
        return [
            float(offsetFlange[0]),
            float(offsetFlange[1]),
            float(offsetFlange[2]),
            float(rxOffset),
            float(ryOffset),
            float(rzOffset)
        ]
    
    def calculateTcpOffsetForAlignment(self, targetCenter: List[float], targetAxis: List[float]) -> Optional[List[float]]:
        """
        Calculate TCP offset to align with a target point and axis.
        
        Args:
            targetCenter: Target center point [x, y, z]
            targetAxis: Target axis direction [ax, ay, az]
        
        Returns:
            New TCP offset [x, y, z, rx, ry, rz] or None on error
        """
        tcpPose = self.getTcpPose()
        flangePose = self.getFlangePose()
        
        if tcpPose is None or flangePose is None:
            print("Error: Could not read poses for alignment calculation")
            return None
        
        # Current TCP offset
        currentOffset = self._tcpOffset if self._tcpOffset else [0, 0, 0, 0, 0, 0]
        
        # Get current flange rotation
        RFlange = axis_angle_to_rotation_matrix(
            flangePose[3], flangePose[4], flangePose[5]
        )
        
        # Calculate new offset position (in flange frame)
        targetCenterArr = np.array(targetCenter)
        flangePos = np.array(flangePose[:3])
        offsetBase = targetCenterArr - flangePos
        offsetFlange = RFlange.T @ offsetBase
        
        # Calculate new offset orientation (align TCP Z with target axis)
        targetAxisArr = np.array(targetAxis)
        targetAxisArr = targetAxisArr / np.linalg.norm(targetAxisArr)
        
        # Current TCP Z axis in base frame
        RTcp = axis_angle_to_rotation_matrix(tcpPose[3], tcpPose[4], tcpPose[5])
        tcpZBase = RTcp[:, 2]
        
        # Calculate rotation to align with target axis
        # This is a simplification - you might want more sophisticated alignment
        currentOffsetRot = axis_angle_to_rotation_matrix(
            currentOffset[3], currentOffset[4], currentOffset[5]
        )
        
        newTcpOffset = [
            float(offsetFlange[0]),
            float(offsetFlange[1]),
            float(offsetFlange[2]),
            currentOffset[3],  # Keep current orientation offset for now
            currentOffset[4],
            currentOffset[5]
        ]
        
        return newTcpOffset
    
    # =========================================================================
    # MOVEMENT CONTROL METHODS
    # =========================================================================
    
    def moveL(self, targetPose: List[float], speed: float = 0.1, acceleration: float = 0.2, asynchronous: bool = False) -> bool:
        """
        Linear move to target pose.
        
        Args:
            targetPose: Target pose [x, y, z, rx, ry, rz]
            speed: Movement speed in m/s
            acceleration: Movement acceleration in m/s²
            asynchronous: If True, return immediately without waiting
        
        Returns:
            True if move started/completed successfully, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            print("Error: Not connected to robot")
            return False
        
        try:
            if asynchronous:
                return self._rtdeC.moveL(targetPose, speed, acceleration, True)
            else:
                return self._rtdeC.moveL(targetPose, speed, acceleration)
        except Exception as e:
            print(f"Error executing moveL: {e}")
            return False
    
    def moveJ(self, targetJoints: List[float], speed: float = 1.0, acceleration: float = 1.0, asynchronous: bool = False) -> bool:
        """
        Joint move to target joint positions.
        
        Args:
            targetJoints: Target joint positions [q0, q1, q2, q3, q4, q5] in radians
            speed: Joint speed in rad/s
            acceleration: Joint acceleration in rad/s²
            asynchronous: If True, return immediately without waiting
        
        Returns:
            True if move started/completed successfully, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            print("Error: Not connected to robot")
            return False
        
        try:
            if asynchronous:
                return self._rtdeC.moveJ(targetJoints, speed, acceleration, True)
            else:
                return self._rtdeC.moveJ(targetJoints, speed, acceleration)
        except Exception as e:
            print(f"Error executing moveJ: {e}")
            return False
    
    def moveLPath(self, path: List[List[float]], speed: float = 0.1, acceleration: float = 0.2, blend: float = 0.0) -> bool:
        """
        Execute a path of linear moves.
        
        Args:
            path: List of poses [[x, y, z, rx, ry, rz], ...]
            speed: Movement speed in m/s
            acceleration: Movement acceleration in m/s²
            blend: Blend radius in meters
        
        Returns:
            True if path execution successful, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            print("Error: Not connected to robot")
            return False
        
        try:
            # Build path with parameters
            pathWithParams = []
            for pose in path:
                pathWithParams.append(pose + [speed, acceleration, blend])
            
            return self._rtdeC.moveL(pathWithParams)
        except Exception as e:
            print(f"Error executing path: {e}")
            return False
    
    def stop(self, deceleration: float = 10.0) -> bool:
        """
        Stop robot movement.
        
        Args:
            deceleration: Deceleration rate in m/s² (default: 10)
        
        Returns:
            True if stop command successful, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            return False
        
        try:
            self._rtdeC.stopL(deceleration)
            return True
        except Exception as e:
            print(f"Error stopping robot: {e}")
            return False
    
    def stopScript(self) -> bool:
        """
        Stop the running RTDE control script.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            return False
        
        try:
            self._rtdeC.stopScript()
            return True
        except Exception as e:
            print(f"Error stopping script: {e}")
            return False
    
    # =========================================================================
    # FREEDRIVE METHODS
    # =========================================================================
    
    def startFreedrive(self) -> bool:
        """
        Enable freedrive mode (allows manual robot movement).
        
        Returns:
            True if freedrive started successfully, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            print("Error: Not connected to robot")
            return False
        
        try:
            self._rtdeC.freedriveMode()
            self._freedriveActive = True
            print("Freedrive mode enabled")
            return True
        except Exception as e:
            print(f"Error enabling freedrive: {e}")
            return False
    
    def stopFreedrive(self) -> bool:
        """
        Disable freedrive mode.
        
        Returns:
            True if freedrive stopped successfully, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            return False
        
        try:
            self._rtdeC.endFreedriveMode()
            self._freedriveActive = False
            print("Freedrive mode disabled")
            return True
        except Exception as e:
            print(f"Error disabling freedrive: {e}")
            return False
    
    def toggleFreedrive(self) -> bool:
        """
        Toggle freedrive mode on/off.
        
        Returns:
            True if operation successful, False otherwise
        """
        if self._freedriveActive:
            return self.stopFreedrive()
        else:
            return self.startFreedrive()
    
    def isFreedriveActive(self) -> bool:
        """Check if freedrive mode is active."""
        return self._freedriveActive
    
    # =========================================================================
    # FORCE/TORQUE METHODS
    # =========================================================================
    
    def zeroFtSensor(self) -> bool:
        """
        Zero the force/torque sensor.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.isConnected() or self._rtdeC is None:
            return False
        
        try:
            self._rtdeC.zeroFtSensor()
            return True
        except Exception as e:
            print(f"Error zeroing FT sensor: {e}")
            return False
    
    def getTcpForceInTcpFrame(self) -> Optional[List[float]]:
        """
        Get TCP force transformed to TCP frame.
        
        Returns:
            Force [Fx, Fy, Fz, Mx, My, Mz] in TCP frame or None
        """
        force = self.getTcpForce()
        pose = self.getTcpPose()
        
        if force is None or pose is None:
            return None
        
        # Transform force from base frame to TCP frame
        RTcp = axis_angle_to_rotation_matrix(pose[3], pose[4], pose[5])
        RTcpInv = RTcp.T
        
        forceTcp = RTcpInv @ np.array(force[:3])
        momentTcp = RTcpInv @ np.array(force[3:6])
        
        return [
            float(forceTcp[0]), float(forceTcp[1]), float(forceTcp[2]),
            float(momentTcp[0]), float(momentTcp[1]), float(momentTcp[2])
        ]
    
    # =========================================================================
    # RTDE INTERFACE ACCESS (for advanced use)
    # =========================================================================
    
    @property
    def rtdeControl(self):
        """Get the RTDE control interface (for advanced use)."""
        return self._rtdeC
    
    @property
    def rtdeReceive(self):
        """Get the RTDE receive interface (for advanced use)."""
        return self._rtdeR
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def addToRenderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Add robot visualization to a VTK renderer.
        
        Args:
            renderer: VTK renderer to add the robot actors to
        """
        self._actor.addToRenderer(renderer)
    
    def removeFromRenderer(self) -> None:
        """Remove robot visualization from the current renderer."""
        self._actor.removeFromRenderer()
    
    def setVisibility(self, visible: bool) -> None:
        """
        Set visibility of the robot visualization.
        
        Args:
            visible: True to show, False to hide
        """
        self._actor.setVisibility(visible)
    
    def resetVisualization(self) -> None:
        """Reset the robot visualization to initial state."""
        self._actor.reset()
    
    @property
    def actor(self) -> UniversalRobotActor:
        """Get the UniversalRobotActor for direct access."""
        return self._actor
    
    # =========================================================================
    # POSE PROPERTIES (convenient access)
    # =========================================================================
    
    @property
    def tcpPose(self) -> Optional[List[float]]:
        """Current TCP pose [x, y, z, rx, ry, rz]."""
        return self._tcpPose.copy() if self._tcpPose else None
    
    @property
    def flangePose(self) -> Optional[List[float]]:
        """Current flange pose [x, y, z, rx, ry, rz]."""
        return self._flangePose.copy() if self._flangePose else None
    
    @property
    def tcpOffset(self) -> Optional[List[float]]:
        """Current TCP offset [x, y, z, rx, ry, rz]."""
        return self._tcpOffset.copy() if self._tcpOffset else None
    
    @property
    def jointPositions(self) -> Optional[List[float]]:
        """Current joint positions [q0, q1, q2, q3, q4, q5]."""
        return self._jointPositions.copy() if self._jointPositions else None
    
    @property
    def tcpForce(self) -> Optional[List[float]]:
        """Current TCP force/torque [Fx, Fy, Fz, Tx, Ty, Tz]."""
        return self._tcpForce.copy() if self._tcpForce else None


def _isEqualPoses(pose1: List[float], pose2: List[float], position_tolerance: float = None, orientation_tolerance: float = None) -> bool:
        """Check if two poses are equal within tolerance."""
        if pose1 is None or pose2 is None:
            return False
        
        # Use config defaults if not specified
        if position_tolerance is None:
            position_tolerance = CONFIG.tolerance.position
        if orientation_tolerance is None:
            orientation_tolerance = CONFIG.tolerance.orientation
        
        # Compare position (first 3 elements)
        pos_diff = math.sqrt(
            (pose1[0] - pose2[0])**2 +
            (pose1[1] - pose2[1])**2 +
            (pose1[2] - pose2[2])**2
        )
        
        # Compare orientation (last 3 elements) - use axis-angle magnitude
        orient_diff = math.sqrt(
            (pose1[3] - pose2[3])**2 +
            (pose1[4] - pose2[4])**2 +
            (pose1[5] - pose2[5])**2
        )
        
        return pos_diff < position_tolerance and orient_diff < orientation_tolerance

import time
from PyQt6.QtCore import QThread, pyqtSignal

class RobotUpdateThread(QThread):
    """Background thread for polling robot state via RTDE.
    
    Signals:
        pose_updated(tcp_pose, flange_pose): Emitted when new pose data is available
        error_occurred(message): Emitted on protective stop, emergency stop, or errors
    """
    pose_updated = pyqtSignal(tuple, tuple)  # tcp_pose, flange_pose
    error_occurred = pyqtSignal(str)
    
    def __init__(self, robot, update_interval_ms: int = 50):
        super().__init__()
        self.robot = robot
        self.running = True
        self.update_interval_s = update_interval_ms / 1000.0
    
    def run(self):
        while self.running:
            try:
                # Check connection first
                if not self.robot.isConnected():
                    break
                
                if self.robot.isProtectiveStopped():
                    self.error_occurred.emit("Protective Stop")
                    break  # Stop loop after emitting error
                elif self.robot.isEmergencyStopped():
                    self.error_occurred.emit("Emergency Stop")
                    break  # Stop loop after emitting error
                else:
                    poses = self.robot.update()
                    if poses:
                        self.pose_updated.emit(tuple(poses[0]), tuple(poses[1]))
            except Exception as e:
                self.error_occurred.emit(str(e))
                break  # Stop loop after error
            
            time.sleep(self.update_interval_s)
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()