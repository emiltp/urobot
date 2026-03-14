"""Reference frame actor combining sphere and axes for 3D visualization."""

from typing import List, Optional
import numpy as np
import vtk
from .base_actor import BaseActor
from .sphere_actor import SphereActor
from .axes_actor import AxesActor


class ReferenceFrame(BaseActor):
    """Reference frame actor combining a sphere (position) and axes (orientation)."""
    
    def __init__(self, origin: List[float], rotationMatrix: np.ndarray,
                 sphereRadius: float = 0.005, axesScale: float = 0.25,
                 sphereColor: List[float] = [1, 1, 0], 
                 sphereResolution: int = 20, **kwargs):
        """
        Create a reference frame actor.
        
        Args:
            origin: Origin position [x, y, z]
            rotationMatrix: 3x3 rotation matrix for axes orientation
            sphereRadius: Radius of the position sphere
            axesScale: Scale factor for axes length
            sphereColor: RGB color [r, g, b] (0.0-1.0) for the sphere
            sphereResolution: Sphere resolution (theta and phi)
            kwargs: Additional keyword arguments for the axes actor
        """
        # Create sphere actor for position
        self._sphere = SphereActor(origin, sphereRadius, sphereColor, sphereResolution)
        
        # Create axes actor for orientation
        self._axes = AxesActor(origin, rotationMatrix, scale=axesScale, **kwargs)
        
        # Use the axes actor as the main actor for BaseActor compatibility
        # (we'll override add/remove methods to handle both actors)
        super().__init__(self._axes.getActor())
        
        # Store origin and rotation
        self._origin = origin.copy()
        self._rotationMatrix = rotationMatrix.copy()
        self._sphereRadius = sphereRadius
        self._axesScale = axesScale
    
    def addToRenderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Add this reference frame to a renderer.
        
        Args:
            renderer: VTK renderer to add the actors to
        """
        if self._renderer is not None:
            self.removeFromRenderer()
        
        # Add both sphere and axes to renderer
        self._sphere.addToRenderer(renderer)
        self._axes.addToRenderer(renderer)
        self._renderer = renderer
    
    def removeFromRenderer(self) -> None:
        """Remove this reference frame from its current renderer."""
        if self._renderer is not None:
            self._sphere.removeFromRenderer()
            self._axes.removeFromRenderer()
            self._renderer = None
    
    def setVisibility(self, visible: bool) -> None:
        """
        Set the visibility of the reference frame.
        
        Args:
            visible: True to make visible, False to hide
        """
        self._sphere.setVisibility(visible)
        self._axes.setVisibility(visible)
    
    def isVisible(self) -> bool:
        """Check if the reference frame is currently visible."""
        return self._sphere.isVisible() or self._axes.isVisible()
    
    def updatePose(self, origin: List[float], rotationMatrix: np.ndarray) -> None:
        """
        Update the reference frame position and orientation.
        
        Args:
            origin: New origin position [x, y, z]
            rotationMatrix: New 3x3 rotation matrix
        """
        self._origin = origin.copy()
        self._rotationMatrix = rotationMatrix.copy()
        
        # Update sphere position
        self._sphere.updatePosition(origin)
        
        # Update axes pose
        self._axes.updatePose(origin, rotationMatrix)
    
    def updatePosition(self, origin: List[float]) -> None:
        """
        Update only the reference frame position (keeps current orientation).
        
        Args:
            origin: New origin position [x, y, z]
        """
        self._origin = origin.copy()
        
        # Update sphere position
        self._sphere.updatePosition(origin)
        
        # Update axes position (keeps current rotation)
        self._axes.updatePosition(origin)
    
    def updateOrientation(self, rotationMatrix: np.ndarray) -> None:
        """
        Update only the reference frame orientation (keeps current position).
        
        Args:
            rotationMatrix: New 3x3 rotation matrix
        """
        self._rotationMatrix = rotationMatrix.copy()
        
        # Update axes orientation
        self._axes.updatePose(self._origin, rotationMatrix)
    
    def getOrigin(self) -> List[float]:
        """Get the current origin position."""
        return self._origin.copy()
    
    def getRotationMatrix(self) -> np.ndarray:
        """Get the current rotation matrix."""
        return self._rotationMatrix.copy()
    
    def setSphereColor(self, r: float, g: float, b: float) -> None:
        """
        Set the color of the sphere.
        
        Args:
            r: Red component (0.0-1.0)
            g: Green component (0.0-1.0)
            b: Blue component (0.0-1.0)
        """
        self._sphere.setColor(r, g, b)
    
    def setSphereRadius(self, radius: float) -> None:
        """
        Set the radius of the sphere.
        
        Args:
            radius: New radius
        """
        self._sphereRadius = radius
        self._sphere.updateRadius(radius)
    
    def setOpacity(self, opacity: float) -> None:
        """Set opacity for both the sphere and axes (0.0 = transparent, 1.0 = opaque)."""
        self._sphere.getActor().GetProperty().SetOpacity(opacity)
        self._axes.setOpacity(opacity)
    
    def setSphereOpacity(self, opacity: float) -> None:
        """Set opacity for the sphere only."""
        self._sphere.getActor().GetProperty().SetOpacity(opacity)
    
    def setAxesScale(self, scale: float) -> None:
        """
        Set the scale of the axes.
        
        Args:
            scale: New scale factor
        """
        self._axesScale = scale
        # Note: AxesActor doesn't have a direct scale update method,
        # so we need to recreate it or add that functionality
        # For now, we'll update the pose which will use the existing scale
        self._axes.updatePose(self._origin, self._rotationMatrix)

    def reset(self) -> None:
        """Reset the reference frame to the origin."""
        # Reset to origin with identity rotation
        self.updatePose([0, 0, 0], np.eye(3))
        # Hide reference frame
        self.setVisibility(False)
