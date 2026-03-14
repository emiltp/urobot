"""
Movement modules for UR Robot Controller.

Registry of motions displayed in the main window dropdown.
Home is not in the registry; import separately: from src.movements import home
"""

import importlib

MOTIONS = [
    ("new_x", "New (x)"),
    ("new_y", "New (y)"),
    ("new_z", "New (z)"),
    ("freemove", "Free Move"),
    ("flexion_x", "Flexion (x) "),
    ("flexion_y", "Flexion (y)"),
    ("axial_rotation", "Axial Rotation"),
]

# motion_id -> import path segment (when different from motion_id)
MOTION_MODULE_MAP = {
    "axial_rotation": "rotation",
}


def get_registered_motions():
    """Yield (motion_id, display_name, widget_class) for each motion in the dropdown."""
    for motion_id, display_name in MOTIONS:
        mod_name = MOTION_MODULE_MAP.get(motion_id, motion_id)
        mod_path = f"src.movements.{mod_name}.widget"
        mod = importlib.import_module(mod_path)
        widget_cls = getattr(mod, "MotionWidget")
        yield motion_id, display_name, widget_cls
