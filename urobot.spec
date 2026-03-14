# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for UR Robot Controller.

Build with:
    ./build.sh
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root – use SPECPATH if available, otherwise fall back to cwd
try:
    PROJECT_ROOT = os.path.abspath(SPECPATH)
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

# Read version from version.py (single source of truth)
sys.path.insert(0, PROJECT_ROOT)
from version import __version__ as APP_VERSION

ICON_PATH = os.path.join(PROJECT_ROOT, 'src', 'urlogo.icns')
if not os.path.exists(ICON_PATH):
    ICON_PATH = None

# ─── Hidden imports ──────────────────────────────────────────────────────────
# Packages that PyInstaller may not detect automatically
hidden_imports = [
    # ur_rtde native extensions
    'rtde_control',
    'rtde_receive',
    'rtde_io',
    'dashboard_client',
    'script_client',
    # Qt / VTK
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    'vtkmodules',
    'vtkmodules.all',
    'vtkmodules.qt.QVTKRenderWindowInteractor',
    'vtkmodules.util',
    'vtkmodules.util.numpy_support',
    'vtkmodules.numpy_interface',
    'vtkmodules.numpy_interface.dataset_adapter',
    # Scientific
    'numpy',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.transform',
    'scipy.linalg',
    # Plotting / styling
    'pyqtgraph',
    'qdarkstyle',
    'qdarkstyle.dark',
    'qdarkstyle.light',
    # YAML config support
    'yaml',
    # Project modules
    'version',
    'config',
    'src',
    'src.utils',
    'src.ui',
    'src.sphere',
    'src.terminalstream',
    'src.motion_logger',
    'src.graphs_widget',
    'src.dialogboxes',
    'src.dialogboxes.tcpoffset',
    'src.objects',
    'src.objects.universal_robot',
    'src.objects.actors',
    'src.objects.actors.base_actor',
    'src.objects.actors.sphere_actor',
    'src.objects.actors.axes_actor',
    'src.objects.actors.line_actor',
    'src.objects.actors.tracked_points_actor',
    'src.objects.actors.reference_frame',
    'src.objects.actors.universal_robot_actor',
    'src.movements',
    'src.movements.home',
    'src.movements.async_motion_runner',
    'src.movements.waypoint_collector',
    'src.movements.flexion_x',
    'src.movements.flexion_x.widget',
    'src.movements.flexion_x.original',
    'src.movements.flexion_x.hybrid',
    'src.movements.flexion_x.force',
    'src.movements.flexion_y',
    'src.movements.flexion_y.widget',
    'src.movements.flexion_y.original',
    'src.movements.flexion_y.hybrid',
    'src.movements.flexion_y.force',
    'src.movements.new_x',
    'src.movements.new_x.widget',
    'src.movements.new_x.original',
    'src.movements.new_x.hybrid',
    'src.movements.new_x.force',
    'src.movements.new_y',
    'src.movements.new_y.widget',
    'src.movements.new_y.original',
    'src.movements.new_y.hybrid',
    'src.movements.new_y.force',
    'src.movements.new_z',
    'src.movements.new_z.widget',
    'src.movements.new_z.original',
    'src.movements.new_z.hybrid',
    'src.movements.new_z.force',
    'src.movements.rotation',
    'src.movements.rotation.widget',
    'src.movements.rotation.direct',
    'src.movements.rotation.hybrid',
    'src.movements.rotation.force',
    'src.movements.freemove',
    'src.movements.freemove.widget',
]

# Collect all vtkmodules submodules (VTK has many internal modules)
hidden_imports += collect_submodules('vtkmodules')

# ─── Data files ──────────────────────────────────────────────────────────────
# qdarkstyle ships stylesheets, SVGs, and PNGs that must be bundled
datas = collect_data_files('qdarkstyle')

# Bundle the project's data/ (including config.yaml), logs/, and version.py
datas += [
    (os.path.join(PROJECT_ROOT, 'data'), 'data'),
    (os.path.join(PROJECT_ROOT, 'logs'), 'logs'),
    (os.path.join(PROJECT_ROOT, 'version.py'), '.'),
]

# ─── Native binaries ────────────────────────────────────────────────────────
# ur_rtde .so extensions live at the top level of site-packages
import rtde_control as _rc
site_packages_dir = os.path.dirname(_rc.__file__)

binaries = []
for mod_name in ['rtde_control', 'rtde_receive', 'rtde_io', 'dashboard_client', 'script_client']:
    so_path = os.path.join(site_packages_dir, f'{mod_name}.cpython-310-darwin.so')
    if os.path.exists(so_path):
        binaries.append((so_path, '.'))

# ─── Analysis ────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(PROJECT_ROOT, 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ─── Executable (onedir mode for fast launch) ────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],                     # binaries/datas go into COLLECT, not EXE
    exclude_binaries=True,
    name='URobotController',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,          # GUI app – no terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,    # macOS: allow drag-and-drop / Finder launch
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH,
)

# Collect all files into a directory alongside the executable
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='URobotController',
)

# macOS .app bundle (wraps the collected directory)
app = BUNDLE(
    coll,
    name='URobotController.app',
    icon=ICON_PATH,
    bundle_identifier='com.urobot.controller',
    info_plist={
        'CFBundleDisplayName': 'UR Robot Controller',
        'CFBundleShortVersionString': APP_VERSION,
        'NSHighResolutionCapable': True,
    },
)
