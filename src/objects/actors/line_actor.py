"""Line actor class for 3D visualization."""

from typing import List
import vtk
from .base_actor import BaseActor


class LineActor(BaseActor):
    """Line actor that can update its endpoints efficiently."""
    
    def __init__(self, p1: List[float], p2: List[float], color: List[float],
                 lineWidth: int = 2):
        """
        Create a line actor.
        
        Args:
            p1: First endpoint [x, y, z]
            p2: Second endpoint [x, y, z]
            color: RGB color [r, g, b] (0.0-1.0)
            lineWidth: Line width
        """
        self._points = vtk.vtkPoints()
        self._points.InsertNextPoint(p1[0], p1[1], p1[2])
        self._points.InsertNextPoint(p2[0], p2[1], p2[2])
        
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        
        self._polyData = vtk.vtkPolyData()
        self._polyData.SetPoints(self._points)
        self._polyData.SetLines(cells)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._polyData)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetLineWidth(lineWidth)
        
        super().__init__(actor)
    
    def updateEndpoints(self, p1: List[float], p2: List[float]) -> None:
        """
        Update both line endpoints.
        
        Args:
            p1: New first endpoint [x, y, z]
            p2: New second endpoint [x, y, z]
        """
        self._points.SetPoint(0, p1[0], p1[1], p1[2])
        self._points.SetPoint(1, p2[0], p2[1], p2[2])
        self._points.Modified()
    
    def updateStartPoint(self, p1: List[float]) -> None:
        """
        Update the first endpoint.
        
        Args:
            p1: New first endpoint [x, y, z]
        """
        self._points.SetPoint(0, p1[0], p1[1], p1[2])
        self._points.Modified()
    
    def updateEndPoint(self, p2: List[float]) -> None:
        """
        Update the second endpoint.
        
        Args:
            p2: New second endpoint [x, y, z]
        """
        self._points.SetPoint(1, p2[0], p2[1], p2[2])
        self._points.Modified()
    
    def getStartPoint(self) -> List[float]:
        """Get the first endpoint."""
        point = self._points.GetPoint(0)
        return [point[0], point[1], point[2]]
    
    def getEndPoint(self) -> List[float]:
        """Get the second endpoint."""
        point = self._points.GetPoint(1)
        return [point[0], point[1], point[2]]

    def reset(self) -> None:
        """Reset the line to the origin."""
        self.updateEndpoints([0, 0, 0], [0, 0, 0])
        self.setVisibility(False)
