import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QButtonGroup, QGroupBox, QFrame,
    QSizePolicy, QToolButton
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QParallelAnimationGroup, QAbstractAnimation
from PyQt6.QtGui import QPainter, QColor, QPainterPath, QPolygonF
from PyQt6.QtCore import QPointF
import qdarkstyle


class CollapsibleGroupBox(QWidget):
    """A group box that can be collapsed/expanded."""
    
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        
        self._expanded = expanded
        self._title = title
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header button - compact style
        self._toggle_btn = QToolButton()
        self._toggle_btn.setStyleSheet("""
            QToolButton {
                background-color: #32414B;
                border: none;
                border-radius: 2px;
                padding: 2px 4px;
                text-align: left;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #3A4D5C;
            }
        """)
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_btn.setText(title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(expanded)
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle_btn.setFixedHeight(20)
        self._toggle_btn.clicked.connect(self._toggle)
        main_layout.addWidget(self._toggle_btn)
        
        # Content frame - compact style
        self._content_frame = QFrame()
        self._content_frame.setStyleSheet("""
            QFrame {
                border: none;
                background-color: transparent;
            }
        """)
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self._content_layout.setSpacing(4)
        main_layout.addWidget(self._content_frame)
        
        # Set initial visibility
        self._content_frame.setVisible(expanded)
    
    def _toggle(self, checked):
        """Toggle the collapsed/expanded state."""
        self._expanded = checked
        self._toggle_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._content_frame.setVisible(checked)
    
    def addWidget(self, widget):
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)
    
    def setExpanded(self, expanded: bool):
        """Set the expanded state."""
        self._toggle_btn.setChecked(expanded)
        self._toggle(expanded)
    
    def isExpanded(self) -> bool:
        """Check if the group is expanded."""
        return self._expanded
    
    def contentLayout(self):
        """Return the content layout for direct access."""
        return self._content_layout
    
    def setTitle(self, title: str):
        """Set the group box title."""
        self._title = title
        self._toggle_btn.setText(title)


class ArrowButton(QPushButton):
    """Custom arrow-shaped toggle button. Optional label drawn inside (e.g. FRONT, LEFT)."""
    
    def __init__(self, direction="left", parent=None, label=None):
        super().__init__(parent)
        self.direction = direction  # "left" or "right"
        self._label = label  # None = no text (flexion/rotation)
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
        
        # Optional label inside arrow (e.g. FRONT, BACK, LEFT, RIGHT) — UI background color
        if self._label:
            painter.setPen(QColor("#19232D"))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, self._label)


class CircleWidget(QWidget):
    """Static circle with configurable top/bottom labels. Default BACK/FRONT (flexion/rotation)."""
    
    def __init__(self, parent=None, top_label="BACK", bottom_label="FRONT"):
        super().__init__(parent)
        self.setFixedSize(50, 70)
        self._color = QColor("#455364")  # QDarkStyle color
        self._top_label = top_label
        self._bottom_label = bottom_label
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        
        # Top label (above knob)
        painter.setPen(QColor("#455364"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(0, 0, w, 14, Qt.AlignmentFlag.AlignCenter, self._top_label)
        
        # Draw circle
        circle_diameter = 40
        circle_x = (w - circle_diameter) // 2
        circle_y = 15
        
        painter.setBrush(self._color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(circle_x, circle_y, circle_diameter, circle_diameter)
        
        # Bottom label (below knob)
        painter.setPen(QColor("#455364"))
        painter.drawText(0, circle_y + circle_diameter + 2, w, 14, Qt.AlignmentFlag.AlignCenter, self._bottom_label)

