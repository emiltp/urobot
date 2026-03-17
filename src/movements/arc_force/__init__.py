"""Arc (Fz-constrained) movement module."""
import os
from config import get_path_filename as _get_path_filename


def get_path_filename(direction: str) -> str:
    """Return path file for arc_force motion (direction: 'left' or 'right')."""
    return _get_path_filename('arc_force', direction, method='arc')


def path_exists(direction: str) -> bool:
    """Return True if a path file exists for the given direction."""
    return os.path.exists(get_path_filename(direction))


from .widget import ArcForceWidget
