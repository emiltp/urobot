"""Arc (y) - moveL around ref Y axis, Fz=0, Fx/My limits, null sensing."""
import os
from config import get_path_filename as _get_path_filename

from .main import execute


def get_path_filename(direction: str) -> str:
    return _get_path_filename('arc_y', direction, method='movel')


def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))
