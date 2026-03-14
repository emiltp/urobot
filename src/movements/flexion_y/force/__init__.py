"""Flexion Y force method."""
import os
from config import get_path_filename as _get_path_filename

def get_path_filename(direction: str) -> str:
    return _get_path_filename('flexion_y_force', direction)

def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))

from .main import execute
