"""Arc (y) movement module — moveL around ref Y axis, Fz=0, Fx/My limits, null sensing."""
import os
from config import get_path_filename as _get_path_filename


def get_path_filename(direction: str) -> str:
    """Return path file for arc_y motion (direction: 'left' or 'right')."""
    return _get_path_filename('arc_y', direction, method='movel')


def path_exists(direction: str) -> bool:
    """Return True if a path file exists for the given direction."""
    return os.path.exists(get_path_filename(direction))


from .widget import ArcYWidget
