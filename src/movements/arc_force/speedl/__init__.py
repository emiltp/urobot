"""Arc force - speedL method."""
import os
from config import get_path_filename as _get_path_filename
from .main import execute


def get_path_filename(direction: str) -> str:
    return _get_path_filename('arc_force', direction, method='arc')


def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))
