"""Arc (z) - moveL around ref Z axis, Fx/Fy/Fz compliant, Mz limit only."""
import os
from config import get_path_filename as _get_path_filename

from .main import execute


def get_path_filename(direction: str) -> str:
    return _get_path_filename('arc_z', direction, method='movel')


def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))
