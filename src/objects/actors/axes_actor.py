"""Axes actor class for 3D visualization."""

from typing import List
import numpy as np
import vtk
from .base_actor import BaseActor


class AxesActor(BaseActor):
    """Axes actor that can update its position and orientation efficiently."""
    
    def __init__(self, origin: List[float], rotationMatrix: np.ndarray, scale: float = 1.0, **kwargs):
        """
        Create an axes actor.
        
        Args:
            origin: Origin position [x, y, z]
            rotationMatrix: 3x3 rotation matrix for axes orientation
            scale: Scale factor for axes length
        """
        cylinderRadius = kwargs.get('cylinderRadius', 0.05 * scale)
        coneRadius = kwargs.get('coneRadius', 0.8 * scale)
        ntl = kwargs.get('normalizedTipLength', 0.2)
        nsl = kwargs.get('normalizedShaftLength', 0.8)

        self._axesActor = vtk.vtkAxesActor()
        self._axesActor.SetXAxisLabelText("")
        self._axesActor.SetYAxisLabelText("")
        self._axesActor.SetZAxisLabelText("")
        self._axesActor.SetTotalLength(scale, scale, scale)
        self._axesActor.SetShaftTypeToCylinder()
        self._axesActor.SetCylinderRadius(cylinderRadius)
        self._axesActor.SetConeRadius(coneRadius)
        self._axesActor.SetNormalizedTipLength(ntl, ntl, ntl)
        self._axesActor.SetNormalizedShaftLength(nsl, nsl, nsl)
        
        self._transform = vtk.vtkTransform()
        self._updateTransform(origin, rotationMatrix)
        self._axesActor.SetUserTransform(self._transform)
        
        super().__init__(self._axesActor)
    
    def _updateTransform(self, origin: List[float], rotationMatrix: np.ndarray) -> None:
        """Update the transform matrix."""
        vtkMatrix = vtk.vtkMatrix4x4()
        
        # Set rotation part (3x3 upper-left)
        for i in range(3):
            for j in range(3):
                vtkMatrix.SetElement(i, j, rotationMatrix[i, j])
        
        # Set translation part
        vtkMatrix.SetElement(0, 3, origin[0])
        vtkMatrix.SetElement(1, 3, origin[1])
        vtkMatrix.SetElement(2, 3, origin[2])
        
        # Set bottom row
        vtkMatrix.SetElement(3, 0, 0.0)
        vtkMatrix.SetElement(3, 1, 0.0)
        vtkMatrix.SetElement(3, 2, 0.0)
        vtkMatrix.SetElement(3, 3, 1.0)
        
        self._transform.SetMatrix(vtkMatrix)
    
    def updatePose(self, origin: List[float], rotationMatrix: np.ndarray) -> None:
        """
        Update the axes position and orientation.
        
        Args:
            origin: New origin position [x, y, z]
            rotationMatrix: New 3x3 rotation matrix
        """
        self._updateTransform(origin, rotationMatrix)
    
    def updatePosition(self, origin: List[float]) -> None:
        """
        Update only the axes position (keeps current orientation).
        
        Args:
            origin: New origin position [x, y, z]
        """
        # Get current rotation from transform matrix
        currentMatrix = self._transform.GetMatrix()
        rotationMatrix = np.eye(3)
        for i in range(3):
            for j in range(3):
                rotationMatrix[i, j] = currentMatrix.GetElement(i, j)
        
        self._updateTransform(origin, rotationMatrix)
    
    def setOpacity(self, opacity: float) -> None:
        """Set opacity for all axis components (0.0 = transparent, 1.0 = opaque)."""
        axes = self._axesActor
        for getter in (axes.GetXAxisShaftProperty, axes.GetYAxisShaftProperty, axes.GetZAxisShaftProperty,
                        axes.GetXAxisTipProperty, axes.GetYAxisTipProperty, axes.GetZAxisTipProperty):
            getter().SetOpacity(opacity)
    
    def getOrigin(self) -> List[float]:
        """Get the current origin position."""
        matrix = self._transform.GetMatrix()
        return [
            matrix.GetElement(0, 3),
            matrix.GetElement(1, 3),
            matrix.GetElement(2, 3)
        ]
