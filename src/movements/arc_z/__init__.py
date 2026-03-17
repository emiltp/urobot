"""Arc (z) movement module — moveL around ref Z axis, Fx/Fy/Fz compliant, Mz limit only."""
import os
from config import get_path_filename as _get_path_filename


def get_path_filename(direction: str) -> str:
    """Return path file for arc_z motion (direction: 'left' or 'right')."""
    return _get_path_filename('arc_z', direction, method='movel')


def path_exists(direction: str) -> bool:
    """Return True if a path file exists for the given direction."""
    return os.path.exists(get_path_filename(direction))


from .widget import ArcZWidget
