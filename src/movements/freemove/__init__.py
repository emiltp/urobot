"""Freemove movement module.

This module allows waypoint collection during freedrive mode.
The user manually moves the robot while waypoints are collected,
then the recorded path can be traversed programmatically.
"""
import os
from config import get_path_filename as _get_path_filename


def get_path_filename() -> str:
    """Get the filename for the freemove path.
    
    Returns:
        Full path to the freemove path file
    """
    return _get_path_filename('freemove')


def path_exists() -> bool:
    """Check if a freemove path exists.
    
    Returns:
        True if path file exists, False otherwise
    """
    return os.path.exists(get_path_filename())


# Import submodules after functions are defined to avoid circular import
from .widget import FreemoveWidget

