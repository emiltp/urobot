import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QButtonGroup
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPainter, QColor, QPainterPath, QPolygonF
from PyQt6.QtCore import QPointF
import qdarkstyle


class ArrowButton(QPushButton):
    """Custom arrow-shaped toggle button."""
    
    def __init__(self, direction="left", parent=None):
        super().__init__(parent)
        self.direction = direction  # "left" or "right"
        self.setFixedSize(120, 80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)  # Make it a toggle button
        
        # QDarkStyle colors
        self._base_color = QColor("#455364")  # QDarkStyle highlight/selection color
        self._hover_color = QColor("#54687A")
        self._pressed_color = QColor("#37414F")
        self._checked_color = QColor("#1464A0")  # QDarkStyle selection/highlight blue
        self._checked_hover_color = QColor("#1a7bc4")
        
    def _get_current_color(self):
        """Get the appropriate color based on button state."""
        if self.isChecked():
            if self.underMouse():
                return self._checked_hover_color
            return self._checked_color
        else:
            if self.underMouse():
                return self._hover_color
            return self._base_color
        
    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)
        
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Arrow dimensions
        arrow_head_width = 0.45 * w
        shaft_height = 0.4 * h
        
        if self.direction == "left":
            # Left-pointing arrow
            points = [
                QPointF(0, h / 2),                          # Arrow tip (left)
                QPointF(arrow_head_width, 0),               # Top of arrow head
                QPointF(arrow_head_width, (h - shaft_height) / 2),  # Top inner corner
                QPointF(w, (h - shaft_height) / 2),         # Top right of shaft
                QPointF(w, (h + shaft_height) / 2),         # Bottom right of shaft
                QPointF(arrow_head_width, (h + shaft_height) / 2),  # Bottom inner corner
                QPointF(arrow_head_width, h),               # Bottom of arrow head
            ]
        else:
            # Right-pointing arrow
            points = [
                QPointF(w, h / 2),                          # Arrow tip (right)
                QPointF(w - arrow_head_width, 0),           # Top of arrow head
                QPointF(w - arrow_head_width, (h - shaft_height) / 2),  # Top inner corner
                QPointF(0, (h - shaft_height) / 2),         # Top left of shaft
                QPointF(0, (h + shaft_height) / 2),         # Bottom left of shaft
                QPointF(w - arrow_head_width, (h + shaft_height) / 2),  # Bottom inner corner
                QPointF(w - arrow_head_width, h),           # Bottom of arrow head
            ]
        
        polygon = QPolygonF(points)
        
        painter.setBrush(self._get_current_color())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(polygon)


class CircleWidget(QWidget):
    """Static circle with BACK/FRONT labels."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 140)
        self._color = QColor("#455364")  # QDarkStyle color
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        
        # Draw "BACK" label
        painter.setPen(QColor("#455364"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(0, 0, w, 25, Qt.AlignmentFlag.AlignCenter, "BACK")
        
        # Draw circle
        circle_diameter = 80
        circle_x = (w - circle_diameter) // 2
        circle_y = 28
        
        painter.setBrush(self._color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(circle_x, circle_y, circle_diameter, circle_diameter)
        
        # Draw "FRONT" label
        painter.setPen(QColor("#455364"))
        painter.drawText(0, circle_y + circle_diameter + 5, w, 25, Qt.AlignmentFlag.AlignCenter, "FRONT")


class TestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arrow Button Test")
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Left arrow button (toggle)
        self.left_arrow = ArrowButton("left")
        self.left_arrow.toggled.connect(lambda checked: print(f"Left arrow toggled: {checked}"))
        
        # Circle with labels
        self.circle = CircleWidget()
        
        # Right arrow button (toggle)
        self.right_arrow = ArrowButton("right")
        self.right_arrow.toggled.connect(lambda checked: print(f"Right arrow toggled: {checked}"))
        
        # Button group for mutual exclusivity (only one can be toggled at a time)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.button_group.addButton(self.left_arrow)
        self.button_group.addButton(self.right_arrow)
        
        layout.addWidget(self.left_arrow, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.circle, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.right_arrow, alignment=Qt.AlignmentFlag.AlignVCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    
    window = TestWindow()
    window.show()
    
    sys.exit(app.exec())

