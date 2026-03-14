"""Sphere actor classes for 3D visualization."""

from typing import List
import vtk
from .base_actor import BaseActor


class SphereActor(BaseActor):
    """Sphere actor that can update its position efficiently."""
    
    def __init__(self, center: List[float], radius: float, color: List[float], 
                 resolution: int = 20):
        """
        Create a sphere actor.
        
        Args:
            center: Center position [x, y, z]
            radius: Sphere radius
            color: RGB color [r, g, b] (0.0-1.0)
            resolution: Sphere resolution (theta and phi)
        """
        self._sphereSource = vtk.vtkSphereSource()
        self._sphereSource.SetCenter(center[0], center[1], center[2])
        self._sphereSource.SetRadius(radius)
        self._sphereSource.SetThetaResolution(resolution)
        self._sphereSource.SetPhiResolution(resolution)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._sphereSource.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        
        super().__init__(actor)
    
    def updatePosition(self, center: List[float]) -> None:
        """
        Update the sphere center position.
        
        Args:
            center: New center position [x, y, z]
        """
        self._sphereSource.SetCenter(center[0], center[1], center[2])
        self._sphereSource.Modified()
    
    def updateRadius(self, radius: float) -> None:
        """
        Update the sphere radius.
        
        Args:
            radius: New radius
        """
        self._sphereSource.SetRadius(radius)
        self._sphereSource.Modified()
    
    def getCenter(self) -> List[float]:
        """Get the current center position."""
        center = self._sphereSource.GetCenter()
        return [center[0], center[1], center[2]]
    
    def getRadius(self) -> float:
        """Get the current radius."""
        return self._sphereSource.GetRadius()


class WireframeSphereActor(BaseActor):
    """Wireframe sphere actor that can update its position efficiently."""
    
    def __init__(self, center: List[float], radius: float, color: List[float], resolution: int = 20, linewidth: int = 2):
        """
        Create a wireframe sphere actor.
        
        Args:
            center: Center position [x, y, z]
            radius: Sphere radius
            color: RGB color [r, g, b] (0.0-1.0)
            resolution: Sphere resolution (theta and phi)
            linewidth: Line width for wireframe
        """
        self._sphereSource = vtk.vtkSphereSource()
        self._sphereSource.SetCenter(center[0], center[1], center[2])
        self._sphereSource.SetRadius(radius)
        self._sphereSource.SetThetaResolution(resolution)
        self._sphereSource.SetPhiResolution(resolution)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._sphereSource.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(linewidth)
        
        super().__init__(actor)
    
    def updatePosition(self, center: List[float]) -> None:
        """
        Update the sphere center position.
        
        Args:
            center: New center position [x, y, z]
        """
        self._sphereSource.SetCenter(center[0], center[1], center[2])
        self._sphereSource.Modified()
    
    def updateRadius(self, radius: float) -> None:
        """
        Update the sphere radius.
        
        Args:
            radius: New radius
        """
        self._sphereSource.SetRadius(radius)
        self._sphereSource.Modified()
    
    def getCenter(self) -> List[float]:
        """Get the current center position."""
        center = self._sphereSource.GetCenter()
        return [center[0], center[1], center[2]]
    
    def getRadius(self) -> float:
        """Get the current radius."""
        return self._sphereSource.GetRadius()
