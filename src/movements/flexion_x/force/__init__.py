"""Flexion X force method - force mode + speedL control loop."""
import os
from config import get_path_filename as _get_path_filename

def get_path_filename(direction: str) -> str:
    return _get_path_filename('flexion_x_force', direction)

def path_exists(direction: str) -> bool:
    return os.path.exists(get_path_filename(direction))

from .main import execute
