"""Base class for VTK actors with common functionality."""

from typing import Optional
import vtk


class BaseActor:
    """Base class for all VTK actors with common update methods."""
    
    def __init__(self, actor: vtk.vtkActor):
        """
        Initialize the base actor.
        
        Args:
            actor: VTK actor object
        """
        self._actor = actor
        self._renderer: Optional[vtk.vtkRenderer] = None
    
    def getActor(self) -> vtk.vtkActor:
        """Get the underlying VTK actor."""
        return self._actor
    
    def addToRenderer(self, renderer: vtk.vtkRenderer) -> None:
        """
        Add this actor to a renderer.
        
        Args:
            renderer: VTK renderer to add the actor to
        """
        if self._renderer is not None:
            self.removeFromRenderer()
        
        renderer.AddActor(self._actor)
        self._renderer = renderer
    
    def removeFromRenderer(self) -> None:
        """Remove this actor from its current renderer."""
        if self._renderer is not None:
            self._renderer.RemoveActor(self._actor)
            self._renderer = None
    
    def setVisibility(self, visible: bool) -> None:
        """
        Set the visibility of the actor.
        
        Args:
            visible: True to make visible, False to hide
        """
        self._actor.SetVisibility(1 if visible else 0)
    
    def isVisible(self) -> bool:
        """Check if the actor is currently visible."""
        return self._actor.GetVisibility() > 0
    
    def setColor(self, r: float, g: float, b: float) -> None:
        """
        Set the color of the actor.
        
        Args:
            r: Red component (0.0-1.0)
            g: Green component (0.0-1.0)
            b: Blue component (0.0-1.0)
        """
        self._actor.GetProperty().SetColor(r, g, b)
    
    def getColor(self) -> tuple:
        """Get the current color of the actor as (r, g, b)."""
        return self._actor.GetProperty().GetColor()
