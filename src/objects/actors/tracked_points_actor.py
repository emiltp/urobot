"""Tracked points actor using VTK glyphs for efficient rendering."""

from typing import List, Optional, Union
import numpy as np
import vtk
from .base_actor import BaseActor


class TrackedPointsActor(BaseActor):
    """Actor for efficiently rendering multiple tracked points using glyphs."""
    
    def __init__(self, color: List[float] = [0, 1, 1], radius: float = 0.005, resolution: int = 20, opacity: float = 1.0):
        """
        Create a tracked points actor.
        
        Args:
            color: RGB color [r, g, b] (0.0-1.0) for the points
            radius: Radius of each point sphere
            resolution: Sphere resolution (theta and phi)
        """
        self._color = color
        self._radius = radius
        self._resolution = resolution
        self._opacity = opacity
        # Create point set
        self._points = vtk.vtkPoints()
        self._polyData = vtk.vtkPolyData()
        self._polyData.SetPoints(self._points)
        
        # Create vertex cells (one per point) - required for glyphs
        self._vertices = vtk.vtkCellArray()
        self._polyData.SetVerts(self._vertices)
        
        # Initialize with at least one dummy point to avoid VTK issues with empty poly data
        # This will be removed when first real point is added
        self._hasDummyPoint = True
        dummyId = self._points.InsertNextPoint(0, 0, 0)
        self._vertices.InsertNextCell(1)
        self._vertices.InsertCellPoint(dummyId)
        self._pointCount = 0  # Track real points separately
        
        # Create glyph source (sphere)
        self._glyphSource = vtk.vtkSphereSource()
        self._glyphSource.SetRadius(radius)
        self._glyphSource.SetThetaResolution(resolution)
        self._glyphSource.SetPhiResolution(resolution)
        # Create glyph filter
        self._glyph = vtk.vtkGlyph3D()
        self._glyph.SetInputData(self._polyData)
        self._glyph.SetSourceConnection(self._glyphSource.GetOutputPort())
        # Set scale mode to use constant scale (not by scalar or vector)
        # Use the numeric constant: 0 = SCALE_BY_SCALAR, 1 = SCALE_BY_VECTOR, 2 = SCALE_BY_CONSTANT
        self._glyph.SetScaleMode(2)  # SCALE_BY_CONSTANT
        self._glyph.SetScaleFactor(1.0)
        self._glyph.OrientOff()  # Don't orient glyphs, just place them at points
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._glyph.GetOutputPort())
        self._mapper = mapper  # Keep reference
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        super().__init__(actor)
        
        # Track number of points
        self._pointCount = 0
        
        # Hide actor initially if it only has dummy point
        if self._hasDummyPoint:
            actor.SetVisibility(False)
    
    def addPoint(self, point: List[float]) -> None:
        """
        Add a single point to the collection.
        
        Args:
            point: Point position [x, y, z]
        """
        # Remove dummy point if this is the first real point
        if self._hasDummyPoint and self._pointCount == 0:
            self._points.Reset()
            self._vertices.Reset()
            self._hasDummyPoint = False
            # Make actor visible now that we have real points
            self._actor.SetVisibility(True)
        
        pointId = self._points.InsertNextPoint(point[0], point[1], point[2])
        
        # Create vertex cell for this point (required for glyphs)
        # InsertNextCell(1) means 1 point in the cell
        self._vertices.InsertNextCell(1)
        self._vertices.InsertCellPoint(pointId)
        
        self._pointCount += 1
        
        # Update the poly data (VTK will automatically update the pipeline)
        self._polyData.Modified()
        self._points.Modified()
        self._vertices.Modified()
    
    def addPoints(self, points: Union[List[List[float]], np.ndarray]) -> None:
        """
        Add multiple points to the collection.
        
        Args:
            points: Point positions as list [[x, y, z], ...] or ndarray (N, 3)
        """
        for point in points:
            self.addPoint(point)
    
    def setPoints(self, points: Union[List[List[float]], np.ndarray]) -> None:
        """
        Replace all points with a new set.
        
        Args:
            points: Point positions as list [[x, y, z], ...] or ndarray (N, 3)
        """
        self.clearPoints()
        self.addPoints(points)
    
    def clearPoints(self) -> None:
        """Clear all points from the collection."""
        self._points.Reset()
        self._vertices.Reset()
        self._pointCount = 0
        self._hasDummyPoint = True
        
        # Add dummy point to avoid VTK issues with empty poly data
        dummyId = self._points.InsertNextPoint(0, 0, 0)
        self._vertices.InsertNextCell(1)
        self._vertices.InsertCellPoint(dummyId)
        
        # Update the poly data (VTK will automatically update the pipeline)
        self._polyData.Modified()
        self._points.Modified()
        self._vertices.Modified()
    
    def getPointCount(self) -> int:
        """Get the number of points currently in the collection."""
        return self._pointCount
    
    def getPoints(self) -> List[List[float]]:
        """
        Get all points as a list.
        
        Returns:
            List of point positions [[x, y, z], ...]
        """
        points = []
        for i in range(self._pointCount):
            point = self._points.GetPoint(i)
            points.append([point[0], point[1], point[2]])
        return points
    
    def setVisibility(self, visible: bool) -> None:
        """
        Set the visibility of the actor.
        Only show if we have real points (not just dummy point).
        
        Args:
            visible: True to make visible, False to hide
        """
        # Only show if we have real points (not just dummy point)
        if visible and self._pointCount == 0 and self._hasDummyPoint:
            visible = False
        self._actor.SetVisibility(1 if visible else 0)
    
    def setColor(self, r: float, g: float, b: float) -> None:
        """
        Set the color of all points.
        
        Args:
            r: Red component (0.0-1.0)
            g: Green component (0.0-1.0)
            b: Blue component (0.0-1.0)
        """
        self._color = [r, g, b]
        self._actor.GetProperty().SetColor(r, g, b)
    
    def setRadius(self, radius: float) -> None:
        """
        Set the radius of all point spheres.
        
        Args:
            radius: New radius
        """
        self._radius = radius
        self._glyphSource.SetRadius(radius)
        self._glyphSource.Modified()
    
    def getRadius(self) -> float:
        """Get the current radius."""
        return self._radius
