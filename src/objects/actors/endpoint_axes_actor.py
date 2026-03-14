"""Endpoint axes actor: white shafts, colored tips (RGB), yellow transparent origin."""

from typing import List
import numpy as np
import vtk
from .base_actor import BaseActor
from .sphere_actor import SphereActor


class EndpointAxesActor(BaseActor):
    """Axes visualization with white shafts, RGB-colored tips, and a yellow origin sphere.

    Visual parameters are matched to the Ref-frame actor defaults
    (sphereRadius=0.024, axesScale=0.04, sphereOpacity=0.15).
    """

    def __init__(self, origin: List[float], rotationMatrix: np.ndarray,
                 axesScale: float = 0.03,
                 sphereRadius: float = 0.024,
                 sphereColor: List[float] = None,
                 sphereOpacity: float = 0.15,
                 shaftColor: List[float] = None,
                 tipColors: List[List[float]] = None,
                 **kwargs):
        """
        Args:
            origin: [x, y, z]
            rotationMatrix: 3x3 rotation matrix
            axesScale: length of each axis shaft+tip
            sphereRadius: origin sphere radius
            sphereColor: RGB for the sphere (default yellow)
            sphereOpacity: sphere opacity
            shaftColor: RGB for all three cylinder shafts (default white)
            tipColors: list of three RGB for cone tips [X, Y, Z] (default R, G, B)
            axesOpacity: opacity for shafts and tips (default 0.1)
        """
        if sphereColor is None:
            sphereColor = [1.0, 1.0, 0.0]
        if shaftColor is None:
            shaftColor = [1.0, 1.0, 1.0]
        if tipColors is None:
            tipColors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self._axesScale = axesScale
        self._sphereRadius = sphereRadius
        self._sphereOpacity = sphereOpacity
        self._shaftColor = shaftColor
        self._tipColors = tipColors
        self._axesOpacity = kwargs.get('axesOpacity', 0.1)

        # Proportions matching vtkAxesActor with cylinderRadius=0.03, coneRadius=0.5
        normalizedTipFraction = 0.2
        normalizedShaftFraction = 0.8
        shaftHeight = normalizedShaftFraction * axesScale
        coneHeight = normalizedTipFraction * axesScale
        shaftRadius = 0.03 * axesScale
        coneRadius = 0.5 * coneHeight  # fraction of tip length, not total length

        self._assembly = vtk.vtkAssembly()

        directions = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]

        for i, direction in enumerate(directions):
            cyl = vtk.vtkCylinderSource()
            cyl.SetRadius(shaftRadius)
            cyl.SetHeight(shaftHeight)
            cyl.SetResolution(16)

            cylMapper = vtk.vtkPolyDataMapper()
            cylMapper.SetInputConnection(cyl.GetOutputPort())
            cylActor = vtk.vtkActor()
            cylActor.SetMapper(cylMapper)
            cylActor.GetProperty().SetColor(shaftColor)
            cylActor.GetProperty().SetOpacity(self._axesOpacity)

            cylTransform = vtk.vtkTransform()
            cylTransform.PostMultiply()
            if i == 0:    # X
                cylTransform.RotateZ(-90)
            elif i == 2:  # Z
                cylTransform.RotateX(90)
            cylTransform.Translate(direction * shaftHeight / 2)
            cylActor.SetUserTransform(cylTransform)
            self._assembly.AddPart(cylActor)

            cone = vtk.vtkConeSource()
            cone.SetRadius(coneRadius)
            cone.SetHeight(coneHeight)
            cone.SetResolution(16)
            cone.SetDirection(float(direction[0]), float(direction[1]), float(direction[2]))

            coneMapper = vtk.vtkPolyDataMapper()
            coneMapper.SetInputConnection(cone.GetOutputPort())
            coneActor = vtk.vtkActor()
            coneActor.SetMapper(coneMapper)
            coneActor.GetProperty().SetColor(tipColors[i])
            coneActor.GetProperty().SetOpacity(self._axesOpacity)

            coneTransform = vtk.vtkTransform()
            coneTransform.Translate(direction * (shaftHeight + coneHeight / 2))
            coneActor.SetUserTransform(coneTransform)
            self._assembly.AddPart(coneActor)

        # Origin sphere
        self._sphere = SphereActor(origin, sphereRadius, sphereColor, resolution=20)
        self._sphere.getActor().GetProperty().SetOpacity(sphereOpacity)

        # Top-level transform that positions + orients the whole assembly
        self._transform = vtk.vtkTransform()
        self._updateTransform(origin, rotationMatrix)
        self._assembly.SetUserTransform(self._transform)

        super().__init__(self._assembly)

    # ------------------------------------------------------------------
    # transform helpers
    # ------------------------------------------------------------------

    def _updateTransform(self, origin: List[float], rotationMatrix: np.ndarray) -> None:
        m = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                m.SetElement(i, j, rotationMatrix[i, j])
        m.SetElement(0, 3, origin[0])
        m.SetElement(1, 3, origin[1])
        m.SetElement(2, 3, origin[2])
        m.SetElement(3, 0, 0.0)
        m.SetElement(3, 1, 0.0)
        m.SetElement(3, 2, 0.0)
        m.SetElement(3, 3, 1.0)
        self._transform.SetMatrix(m)

    def updatePose(self, origin: List[float], rotationMatrix: np.ndarray) -> None:
        self._updateTransform(origin, rotationMatrix)
        self._sphere.updatePosition(origin)

    # ------------------------------------------------------------------
    # renderer management (handles both assembly + sphere)
    # ------------------------------------------------------------------

    def addToRenderer(self, renderer: vtk.vtkRenderer) -> None:
        if self._renderer is not None:
            self.removeFromRenderer()
        renderer.AddActor(self._assembly)
        self._sphere.addToRenderer(renderer)
        self._renderer = renderer

    def removeFromRenderer(self) -> None:
        if self._renderer is not None:
            self._renderer.RemoveActor(self._assembly)
            self._sphere.removeFromRenderer()
            self._renderer = None

    def setVisibility(self, visible: bool) -> None:
        self._assembly.SetVisibility(1 if visible else 0)
        self._sphere.setVisibility(visible)

    def isVisible(self) -> bool:
        return self._assembly.GetVisibility() > 0
