"""Universal Robot actor class for 3D visualization."""

from typing import List, Optional
import numpy as np
import vtk
from .base_actor import BaseActor
from .axes_actor import AxesActor
from .reference_frame import ReferenceFrame
from .line_actor import LineActor
from src.utils import axis_angle_to_rotation_matrix


class UniversalRobotActor(BaseActor):
    """
    Universal Robot actor containing base frame, flange frame, TCP frame, and connecting lines.
    
    This class is a pure visualization class that encapsulates:
    - Base reference frame (axes at origin)
    - Flange reference frame (robot flange pose)
    - TCP reference frame (tool center point pose)
    - Line from base to flange
    - Line from flange to TCP
    
    All calculations should be done externally (e.g., in UniversalRobot class).
    This class only handles visualization updates.
    """
    
    def __init__(self, baseScale: float = 0.5, flangeScale: float = 0.2,
                 tcpScale: float = 0.25, refFrameScale: float = 0.04,
                 tcpSphereRadius: float = 0.005,
                 tcpSphereColor: List[float] = [1, 1, 0],
                 refFrameSphereColor: List[float] = [1, 1, 1],
                 baseToFlangeColor: List[float] = [0.8, 0.8, 0.8],
                 flangeToTcpColor: List[float] = [0.5, 0.5, 0.5],
                 lineWidth: int = 2):
        """
        Create a Universal Robot actor.
        
        Args:
            baseScale: Scale factor for base axes length
            flangeScale: Scale factor for flange axes length
            tcpScale: Scale factor for TCP axes length
            refFrameScale: Scale factor for reference frame axes length
            tcpSphereRadius: Radius of TCP position sphere
            tcpSphereColor: RGB color [r, g, b] (0.0-1.0) for TCP sphere
            refFrameSphereColor: RGB color [r, g, b] (0.0-1.0) for ref frame sphere
            baseToFlangeColor: RGB color for line from base to flange
            flangeToTcpColor: RGB color for line from flange to TCP
            lineWidth: Width of connecting lines
        """
        # Base frame at origin with identity rotation
        baseOrigin = [0, 0, 0]
        baseRotation = np.eye(3)
        self._baseAxes = AxesActor(baseOrigin, baseRotation, scale=baseScale)
        
        # Flange frame (initially at origin, will be updated)
        self._flange = ReferenceFrame(
            origin=[0, 0, 0],
            rotationMatrix=np.eye(3),
            sphereRadius=0.005,
            axesScale=flangeScale,
            sphereColor=[0.1, 0.1, 0.1],
            sphereResolution=20,
            cylinderRadius=0.06,
            coneRadius=0.55
        )
        self._flange.setOpacity(0.1)
        self._flange.setSphereOpacity(1.0)
        
        # TCP frame (initially at origin, will be updated)
        self._tcp = ReferenceFrame(
            origin=[0, 0, 0],
            rotationMatrix=np.eye(3),
            sphereRadius=tcpSphereRadius,
            axesScale=tcpScale,
            sphereColor=tcpSphereColor,
            sphereResolution=20,
            cylinderRadius=0.03,
            coneRadius=0.5
        )
        
        # Reference frame (relative to TCP, no connecting line)
        self._refFrame = ReferenceFrame(
            origin=[0, 0, 0],
            rotationMatrix=np.eye(3),
            sphereRadius=0.024,
            axesScale=refFrameScale,
            sphereColor=refFrameSphereColor,
            sphereResolution=20,
            cylinderRadius=0.03,
            coneRadius=0.5
        )
        self._refFrame.setSphereOpacity(0.15)
        self._refFrame.setVisibility(False)
        
        # Lines connecting the frames
        self._baseToFlangeLine = LineActor(
            [0, 0, 0], [0, 0, 0],
            baseToFlangeColor,
            lineWidth=lineWidth
        )
        
        self._flangeToTcpLine = LineActor(
            [0, 0, 0], [0, 0, 0],
            flangeToTcpColor,
            lineWidth=lineWidth
        )
        
        # Store current poses (for reference only, no calculations)
        self._tcpPose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz] in base frame
        self._flangePose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz] in base frame
        self._refFramePose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz] in base frame
        
        # Use base_axes actor as the main actor for BaseActor compatibility
        super().__init__(self._baseAxes.getActor())
    
    def addToRenderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Add all robot components to a renderer.
        
        Args:
            renderer: VTK renderer to add the actors to
        """
        if self._renderer is not None:
            self.removeFromRenderer()
        
        # Add all components to renderer
        self._baseAxes.addToRenderer(renderer)
        self._flange.addToRenderer(renderer)
        self._tcp.addToRenderer(renderer)
        self._refFrame.addToRenderer(renderer)
        self._baseToFlangeLine.addToRenderer(renderer)
        self._flangeToTcpLine.addToRenderer(renderer)
        
        self._renderer = renderer
    
    def removeFromRenderer(self) -> None:
        """Remove all robot components from the current renderer."""
        if self._renderer is not None:
            self._baseAxes.removeFromRenderer()
            self._flange.removeFromRenderer()
            self._tcp.removeFromRenderer()
            self._refFrame.removeFromRenderer()
            self._baseToFlangeLine.removeFromRenderer()
            self._flangeToTcpLine.removeFromRenderer()
            self._renderer = None
    
    def updateTcpPose(self, tcpPose: List[float]) -> None:
        """
        Update TCP frame visualization.
        
        Args:
            tcpPose: TCP pose [x, y, z, rx, ry, rz] in base frame
        """
        self._tcpPose = tcpPose.copy()
        
        # Update TCP frame visualization
        tcpPos = tcpPose[:3]
        tcpRot = axis_angle_to_rotation_matrix(tcpPose[3], tcpPose[4], tcpPose[5])
        self._tcp.updatePose(tcpPos, tcpRot)
        self._tcp.setVisibility(True)
        
        # Update flange to TCP line if flange pose is available
        if self._flangePose is not None:
            flangePos = self._flangePose[:3]
            self._flangeToTcpLine.updateEndpoints(flangePos, tcpPos)
            self._flangeToTcpLine.setVisibility(True)
    
    def updateFlangePose(self, flangePose: List[float]) -> None:
        """
        Update flange frame visualization.
        
        Args:
            flangePose: Flange pose [x, y, z, rx, ry, rz] in base frame
        """
        self._flangePose = flangePose.copy()
        
        # Update flange frame visualization
        flangePos = flangePose[:3]
        flangeRot = axis_angle_to_rotation_matrix(flangePose[3], flangePose[4], flangePose[5])
        self._flange.updatePose(flangePos, flangeRot)
        self._flange.setVisibility(True)
        
        # Update base to flange line
        self._baseToFlangeLine.updateEndpoints([0, 0, 0], flangePos)
        self._baseToFlangeLine.setVisibility(True)
        
        # Update flange to TCP line if TCP pose is available
        if self._tcpPose is not None:
            tcpPos = self._tcpPose[:3]
            self._flangeToTcpLine.updateEndpoints(flangePos, tcpPos)
            self._flangeToTcpLine.setVisibility(True)
    
    def updatePoses(self, tcpPose: List[float], flangePose: Optional[List[float]] = None) -> None:
        """
        Update both TCP and flange poses at once.
        
        Args:
            tcpPose: TCP pose [x, y, z, rx, ry, rz] in base frame
            flangePose: Optional flange pose [x, y, z, rx, ry, rz] in base frame
        """
        self._tcpPose = tcpPose.copy()
        
        # Update TCP frame
        tcpPos = tcpPose[:3]
        tcpRot = axis_angle_to_rotation_matrix(tcpPose[3], tcpPose[4], tcpPose[5])
        self._tcp.updatePose(tcpPos, tcpRot)
        self._tcp.setVisibility(True)
        
        # Update flange frame if provided
        if flangePose is not None:
            self._flangePose = flangePose.copy()
            flangePos = flangePose[:3]
            flangeRot = axis_angle_to_rotation_matrix(flangePose[3], flangePose[4], flangePose[5])
            self._flange.updatePose(flangePos, flangeRot)
            self._flange.setVisibility(True)
            
            # Update lines
            self._baseToFlangeLine.updateEndpoints([0, 0, 0], flangePos)
            self._baseToFlangeLine.setVisibility(True)
            self._flangeToTcpLine.updateEndpoints(flangePos, tcpPos)
            self._flangeToTcpLine.setVisibility(True)
        else:
            # No flange pose, hide flange and lines
            self._flange.setVisibility(False)
            self._baseToFlangeLine.setVisibility(False)
            self._flangeToTcpLine.setVisibility(False)
    
    def updateRefFramePose(self, refFramePose: List[float]) -> None:
        """
        Update reference frame visualization.
        
        Args:
            refFramePose: Ref frame pose [x, y, z, rx, ry, rz] in base frame
        """
        self._refFramePose = refFramePose.copy()
        refPos = refFramePose[:3]
        refRot = axis_angle_to_rotation_matrix(refFramePose[3], refFramePose[4], refFramePose[5])
        self._refFrame.updatePose(refPos, refRot)
        self._refFrame.setVisibility(True)
    
    def hideRefFrame(self) -> None:
        """Hide the reference frame."""
        self._refFramePose = None
        self._refFrame.setVisibility(False)
    
    def reset(self) -> None:
        """Reset all components to origin and hide them."""
        self._tcpPose = None
        self._flangePose = None
        self._refFramePose = None
        
        # Reset TCP, flange, and ref frame to origin
        self._tcp.reset()
        self._flange.reset()
        self._refFrame.reset()
        
        # Reset lines
        self._baseToFlangeLine.reset()
        self._flangeToTcpLine.reset()
    
    def setVisibility(self, visible: bool) -> None:
        """
        Set the visibility of all robot components.
        
        Args:
            visible: True to make visible, False to hide
        """
        self._baseAxes.setVisibility(visible)
        self._flange.setVisibility(visible)
        self._tcp.setVisibility(visible)
        if self._refFramePose is not None:
            self._refFrame.setVisibility(visible)
        self._baseToFlangeLine.setVisibility(visible)
        self._flangeToTcpLine.setVisibility(visible)
    
    # =========================================================================
    # COMPONENT ACCESSORS
    # =========================================================================
    
    def getBaseAxes(self) -> AxesActor:
        """Get the base axes actor."""
        return self._baseAxes
    
    def getTcp(self) -> ReferenceFrame:
        """Get the TCP reference frame actor."""
        return self._tcp
    
    def getFlange(self) -> ReferenceFrame:
        """Get the flange reference frame actor."""
        return self._flange
    
    def getRefFrame(self) -> ReferenceFrame:
        """Get the reference frame actor."""
        return self._refFrame
    
    def getBaseToFlangeLine(self) -> LineActor:
        """Get the line actor from base to flange."""
        return self._baseToFlangeLine
    
    def getFlangeToTcpLine(self) -> LineActor:
        """Get the line actor from flange to TCP."""
        return self._flangeToTcpLine
    
    # =========================================================================
    # POSE ACCESSORS (read-only, for reference)
    # =========================================================================
    
    @property
    def tcpPose(self) -> Optional[List[float]]:
        """Get the current TCP pose."""
        return self._tcpPose.copy() if self._tcpPose is not None else None
    
    @property
    def flangePose(self) -> Optional[List[float]]:
        """Get the current flange pose."""
        return self._flangePose.copy() if self._flangePose is not None else None
    
    @property
    def refFramePose(self) -> Optional[List[float]]:
        """Get the current reference frame pose."""
        return self._refFramePose.copy() if self._refFramePose is not None else None
