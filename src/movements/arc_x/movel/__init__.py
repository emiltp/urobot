"""Arc (x) - moveL around ref X axis, Fz=0, null sensing.
Execution is delegated to arc_force.movel with axis='x' and force_baseline='null'.
"""
import os
from config import get_path_filename as _get_path_filename


def get_path_filename(direction: str) -> str:
    return _get_path_filename('arc_x', direction, method='movel')


def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))
