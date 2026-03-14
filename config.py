"""
Configuration loader for UR Robot Controller.

Loads all settings from data/config.yaml (the single source of truth).
Exposes a Defaults object that supports dotted-attribute access so every
consumer module can keep using:

    from config import defaults, runtime_home_position, runtime_tcp_offset
    CONFIG = defaults
    speed = CONFIG.movement.speed
"""

import os
import sys
from ruamel.yaml import YAML

# =============================================================================
# YAML engine (comment-preserving via ruamel.yaml)
# =============================================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_YAML_PATH = os.path.join(_BASE_DIR, 'data', 'config.yaml')
_DATA_DIR = os.path.join(_BASE_DIR, 'data')
_LOGS_DIR = os.path.join(_BASE_DIR, 'logs')

# Path patterns for non-movement data (DEFAULT_WAYPOINTS_FILE, offset, etc.)
_PATH_PATTERNS = {
    'path_filename': 'last.path.npz',
    'offset_filename': 'offset.npz',
}


def get_movements_data_dir() -> str:
    """Return the writable base path for movements data.

    - Development: project/data/movements/
    - Frozen: platform-specific user data dir + movements/
    """
    if getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS'):
        # Frozen app (PyInstaller, cx_Freeze, etc.)
        if sys.platform == 'darwin':
            base = os.path.expanduser('~/Library/Application Support/URRobot')
        elif sys.platform == 'win32':
            base = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'URRobot')
        else:
            base = os.path.expanduser('~/.urobot')
        return os.path.join(base, 'movements')
    return os.path.join(_DATA_DIR, 'movements')


def _migrate_from_data_subfolder():
    """One-time migration: data/movements/X/data/*.path.npz -> data/movements/X/*.path.npz"""
    base = get_movements_data_dir()
    if not os.path.isdir(base):
        return
    for motion_name in os.listdir(base):
        motion_path = os.path.join(base, motion_name)
        old_data_dir = os.path.join(motion_path, 'data')
        if not os.path.isdir(old_data_dir):
            continue
        for name in os.listdir(old_data_dir):
            if name.endswith('.path.npz'):
                src = os.path.join(old_data_dir, name)
                dst = os.path.join(motion_path, name)
                if not os.path.exists(dst):
                    try:
                        os.rename(src, dst)
                    except OSError:
                        pass
        try:
            os.rmdir(old_data_dir)
        except OSError:
            pass


# Legacy motion_method -> (motion, method) for migration to motion/method structure
_LEGACY_TO_MOTION_METHOD = {
    'flexion_x': ('flexion_x', 'original'),
    'flexion_x_hybrid': ('flexion_x', 'hybrid'),
    'flexion_x_force': ('flexion_x', 'force'),
    'flexion_y': ('flexion_y', 'original'),
    'flexion_y_hybrid': ('flexion_y', 'hybrid'),
    'flexion_y_force': ('flexion_y', 'force'),
    'new_x': ('new_x', 'original'),
    'new_x_hybrid': ('new_x', 'hybrid'),
    'new_x_force': ('new_x', 'force'),
    'new_y': ('new_y', 'original'),
    'new_y_hybrid': ('new_y', 'hybrid'),
    'new_y_force': ('new_y', 'force'),
    'new_z': ('new_z', 'original'),
    'new_z_hybrid': ('new_z', 'hybrid'),
    'new_z_force': ('new_z', 'force'),
    'rotation_hybrid': ('rotation', 'hybrid'),
    'rotation_force': ('rotation', 'force'),
    'rotation': ('rotation', 'direct'),  # rotation/main.py -> rotation/direct
}


def _migrate_to_motion_method_structure():
    """Migrate movements/{legacy}/*.path.npz -> movements/{motion}/{method}/*.path.npz"""
    base = get_movements_data_dir()
    if not os.path.isdir(base):
        return
    for legacy_name in list(os.listdir(base)):
        if legacy_name not in _LEGACY_TO_MOTION_METHOD:
            continue
        motion, method = _LEGACY_TO_MOTION_METHOD[legacy_name]
        src_dir = os.path.join(base, legacy_name)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(base, motion, method)
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            if name.endswith('.path.npz'):
                src = os.path.join(src_dir, name)
                dst = os.path.join(dst_dir, name)
                if not os.path.exists(dst):
                    try:
                        os.rename(src, dst)
                    except OSError:
                        pass
        try:
            os.rmdir(src_dir)
        except OSError:
            pass


def get_path_filename(motion_name: str, direction: str = None, method: str = None) -> str:
    """Return the full path for a motion's path file.

    Structure:
      - With method: movements/{motion}/{method}/{direction}.path.npz
      - Freemove: movements/freemove/path.npz (method and direction ignored)

    Args:
        motion_name: Base motion (e.g. 'flexion_x', 'freemove') or legacy (e.g. 'flexion_x_hybrid')
        direction: 'left' or 'right', or None for freemove
        method: 'original', 'hybrid', 'force', 'direct' (rotation) - if None, motion_name may be legacy

    Returns:
        Full path; creates parent directories if needed
    """
    base = get_movements_data_dir()
    if motion_name == 'freemove':
        motion_dir = os.path.join(base, 'freemove')
        os.makedirs(motion_dir, exist_ok=True)
        return os.path.join(motion_dir, 'path.npz')
    # Resolve motion and method (support legacy motion_name like flexion_x_hybrid)
    if method is not None:
        motion, resolved_method = motion_name, method
    elif motion_name in _LEGACY_TO_MOTION_METHOD:
        motion, resolved_method = _LEGACY_TO_MOTION_METHOD[motion_name]
    else:
        motion, resolved_method = motion_name, 'original'
    motion_dir = os.path.join(base, motion, resolved_method)
    os.makedirs(motion_dir, exist_ok=True)
    if direction:
        return os.path.join(motion_dir, f'{direction}.path.npz')
    return os.path.join(motion_dir, 'path.npz')


_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = None


def _load_yaml_raw():
    """Load config.yaml as a ruamel ordered dict (preserves comments)."""
    if os.path.exists(_CONFIG_YAML_PATH):
        with open(_CONFIG_YAML_PATH, 'r') as f:
            return _yaml.load(f) or {}
    raise FileNotFoundError(
        f"Configuration file not found: {_CONFIG_YAML_PATH}\n"
        "This file is required — it contains all application defaults."
    )


def _save_yaml_raw(data) -> None:
    """Write the ruamel dict back to config.yaml, preserving comments."""
    os.makedirs(os.path.dirname(_CONFIG_YAML_PATH), exist_ok=True)
    with open(_CONFIG_YAML_PATH, 'w') as f:
        _yaml.dump(data, f)


# =============================================================================
# Defaults — recursive attribute-access config loaded from YAML
# =============================================================================

class Defaults(dict):
    """Dict subclass that exposes keys as attributes (recursively).

    Built from config.yaml via Defaults.from_yaml(), which loads the YAML
    file, converts every nested mapping into a Defaults instance, and
    injects the runtime-computed data_dir / logs_dir paths.
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(
                f"Config has no key '{name}'. "
                f"Available keys: {', '.join(self.keys())}"
            )
        if isinstance(value, dict) and not isinstance(value, Defaults):
            value = Defaults(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    @classmethod
    def _convert(cls, data):
        """Recursively convert a (possibly ruamel) mapping into Defaults."""
        result = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = cls._convert(value)
            elif isinstance(value, list):
                result[key] = [
                    float(x) if hasattr(x, '__float__') and not isinstance(x, bool) else x
                    for x in value
                ]
            else:
                # ruamel.yaml returns ScalarFloat/ScalarInt for numbers; convert to native
                # Python types so PyQt setValue() and other consumers work correctly
                if isinstance(value, (bool, str, type(None))):
                    result[key] = value
                elif hasattr(value, '__float__'):
                    f = float(value)
                    result[key] = int(f) if not isinstance(value, bool) and f == int(f) else f
                else:
                    result[key] = value
        return result

    @classmethod
    def from_yaml(cls):
        """Load config.yaml and return a fully-populated Defaults instance."""
        raw = _load_yaml_raw()
        cfg = cls._convert(raw)

        # Paths are hardcoded in config.py (not user-editable)
        cfg['paths'] = cls._convert({
            **_PATH_PATTERNS,
            'data_dir': _DATA_DIR,
            'logs_dir': _LOGS_DIR,
            'movements_data_dir': get_movements_data_dir(),
        })

        return cfg


# =============================================================================
# Global instances
# =============================================================================

defaults = Defaults.from_yaml()

# One-time migrations
_migrate_from_data_subfolder()
_migrate_to_motion_method_structure()

runtime_home_position = list(defaults.robot.home_position)
runtime_tcp_offset = list(defaults.robot.tcp_offset)


# =============================================================================
# Per-motion hybrid config lookup (for waypoint_collector force-compliant paths)
# =============================================================================


class ConfigKeyNotFoundError(Exception):
    """Raised when no config section exists for the given path file."""
    pass


def _get_hybrid_force_params(hybrid_cfg):
    """Extract force_mode_z_limit, force_mode_damping, force_mode_gain_scaling from hybrid config.
    Supports both flat config (flexion, etc.) and nested z_limit/xy_limit (new_z_hybrid)."""
    if hasattr(hybrid_cfg, 'z_limit'):
        return (
            hybrid_cfg.z_limit.force_mode_z_limit,
            (hybrid_cfg.z_limit.force_mode_damping + hybrid_cfg.xy_limit.force_mode_damping) / 2,
            (hybrid_cfg.z_limit.force_mode_gain_scaling + hybrid_cfg.xy_limit.force_mode_gain_scaling) / 2,
        )
    return (
        getattr(hybrid_cfg, 'force_mode_z_limit', 0.05),
        getattr(hybrid_cfg, 'force_mode_damping', 0.1),
        getattr(hybrid_cfg, 'force_mode_gain_scaling', 1.0),
    )


def get_hybrid_config_for_path(filepath: str):
    """Return the hybrid config section for a path file based on its path or filename.
    
    Used when executing force-compliant traverse (forceHybrid or forceCompliantBackwardTraverse).
    
    Infers motion from path structure: .../movements/{motion_name}/{file}.path.npz
    Fallback: parse legacy filename, e.g. "left.flexion_x_hybrid.path.npz" -> flexion_x_hybrid.
    
    Args:
        filepath: Full path to the .npz file
        
    Returns:
        Config section with force_mode_damping, force_mode_gain_scaling, force_mode_z_limit
        (or nested z_limit/xy_limit for new_z_hybrid). Use _get_hybrid_force_params() for
        normalized extraction.
    """
    filepath = filepath or ""
    normalized = os.path.normpath(filepath)
    parts = normalized.split(os.sep)

    # Try new structure: .../movements/{motion}/{method}/{file}.path.npz
    # or .../movements/{motion}/{file}.path.npz (freemove)
    motion_key = None
    if 'movements' in parts:
        idx = parts.index('movements')
        if idx + 2 < len(parts):
            motion, method = parts[idx + 1], parts[idx + 2]
            if method and not method.endswith('.path.npz'):
                config_key = f"{motion}_{method}"
                if hasattr(defaults, config_key):
                    return getattr(defaults, config_key)
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate and not candidate.endswith('.path.npz'):
                motion_key = candidate

    # Fallback: motion_key from path (single dir under movements, e.g. freemove) or legacy filename
    if not motion_key:
        basename = os.path.basename(filepath)
        stem = basename[:-9] if basename.endswith(".path.npz") else (basename[:-4] if basename.endswith(".npz") else basename)
        filename_parts = stem.split(".")
        motion_key = filename_parts[1] if len(filename_parts) >= 2 else (filename_parts[0] if filename_parts else "freemove")

    if not motion_key.endswith(("_hybrid", "_force")):
        config_key = motion_key + "_hybrid"
    else:
        config_key = motion_key

    if not hasattr(defaults, config_key):
        raise ConfigKeyNotFoundError(
            f"No config section '{config_key}' for path file. "
            f"Cannot determine force-mode parameters for: {filepath or '(unknown)'}"
        )
    return getattr(defaults, config_key)


# =============================================================================
# Write-back helpers (update YAML on disk, preserving comments)
# =============================================================================

def _update_yaml_field(section: str, key: str, value) -> None:
    """Update a single field in config.yaml and save, preserving comments."""
    raw = _load_yaml_raw()
    if section not in raw:
        raw[section] = {}
    raw[section][key] = value
    _save_yaml_raw(raw)


def update_home_position_in_config(new_position: list) -> None:
    """Persist a new home_position to config.yaml, Defaults, and the runtime list.

    Args:
        new_position: List of 6 floats [x, y, z, rx, ry, rz]
    """
    if len(new_position) != 6:
        raise ValueError("Home position must have 6 elements [x, y, z, rx, ry, rz]")

    rounded = [round(float(x), 4) for x in new_position]
    _update_yaml_field('robot', 'home_position', rounded)

    defaults.robot['home_position'] = [float(x) for x in new_position]
    runtime_home_position[:] = [float(x) for x in new_position]


def update_tcp_offset_in_config(new_offset: list) -> None:
    """Persist a new tcp_offset to config.yaml, Defaults, and the runtime list.

    Args:
        new_offset: List of 6 floats [x, y, z, rx, ry, rz]
    """
    if len(new_offset) != 6:
        raise ValueError("TCP offset must have 6 elements [x, y, z, rx, ry, rz]")

    rounded = [round(float(x), 4) for x in new_offset]
    _update_yaml_field('robot', 'tcp_offset', rounded)

    defaults.robot['tcp_offset'] = [float(x) for x in new_offset]
    runtime_tcp_offset[:] = [float(x) for x in new_offset]
