"""
MotionWidget protocol for movement modules.

All motion widgets implement this interface so TCPVisualizer can handle them
generically. Use hasattr() for optional methods (freemove has no direction_group,
axial_rotation has no saved paths, etc.).
"""

from typing import Protocol, Optional, Any


class MotionWidgetProtocol(Protocol):
    """Protocol for motion widgets embedded in the main window.

    Required:
        - _hide_path_visualization(): Hide path visualization when switching motions
        - _on_direction_changed(button, checked): Handle direction selection (no-op for freemove)

    Optional (use hasattr before calling):
        - direction_group: QButtonGroup or None - for motions with left/right (None for freemove)
        - _get_current_method_module(): Module with get_path_filename(direction), path_exists(direction)
        - _remove_endpoint_visualization(): For new_x, new_y, new_z
        - _update_button_states(): For freemove
        - _update_status_from_saved_path(): For freemove after path delete
    """

    def _hide_path_visualization(self) -> None:
        """Hide the path visualization for this motion."""
        ...

    def _on_direction_changed(self, button: Optional[Any], checked: bool) -> None:
        """Handle direction selection change. No-op for freemove."""
        ...
