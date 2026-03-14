# URobot - Universal Robot Control & Visualization

A Python application for controlling and visualizing Universal Robots with real-time 3D visualization, motion control, and force-aware movement capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)
![VTK](https://img.shields.io/badge/VTK-3D%20Visualization-orange.svg)

## Features

- **Real-time 3D Visualization**: VTK-based rendering of robot base frame, flange, and TCP positions
- **RTDE Communication**: Direct control via Universal Robots RTDE protocol
- **Motion Control**: Multiple movement types including flexion, rotation, and freemove
- **Force-Aware Movements**: Hardware-level force mode for compliant motion
- **Waypoint Collection & Traversal**: Record paths and replay with various methods
- **Motion Logging**: Record force, position, and velocity data during movements
- **Dark Theme UI**: Modern qdarkstyle interface

## Requirements

### System Dependencies
- Python 3.10+
- Universal Robot with RTDE enabled (tested with UR5e, UR10e)

### Python Packages
```bash
pip install vtk pyqt6 numpy scipy qdarkstyle
pip install ur-rtde  # For robot communication
```

### Conda Environment
The project uses a conda environment named `ur-test`:
```bash
conda activate ur-test
python app.py
```

## Project Structure

```
urobot/
├── app.py                      # Main application entry point
├── config.py                   # Configuration dataclasses
├── README.md                   # This file
├── data/                       # Saved path files (.npz)
├── logs/                       # Motion log files
├── src/
│   ├── objects/
│   │   ├── universal_robot.py  # Robot control class
│   │   └── actors/             # VTK visualization actors
│   ├── movements/
│   │   ├── waypoint_collector.py   # Path recording & traversal
│   │   ├── async_motion_runner.py  # Async movement execution
│   │   ├── home.py                 # Home position movements
│   │   ├── flexion_x/              # Flexion X movement
│   │   ├── flexion_x_hybrid/       # Flexion X with force mode
│   │   ├── flexion_x_force/        # Flexion X force-only mode
│   │   ├── flexion_y/              # Flexion Y movement
│   │   ├── flexion_y_hybrid/       # Flexion Y with force mode
│   │   ├── flexion_y_force/        # Flexion Y force-only mode
│   │   ├── rotation/               # Axial rotation movement
│   │   └── freemove/               # Freedrive waypoint collection
│   ├── dialogboxes/            # UI dialogs
│   ├── graphs_widget.py        # Real-time force/position graphs
│   ├── motion_logger.py        # Motion data logging
│   ├── ui.py                   # Custom UI components
│   └── utils.py                # Utility functions
└── ur_rtde_source/             # ur_rtde library source
```

## Usage

### Starting the Application
```bash
conda activate ur-test
python app.py
```

### Connecting to Robot
1. Enter the robot's IP address in the connection field
2. Click **Connect**
3. The 3D visualization will show real-time TCP and flange positions

### Motion Control Widgets

Select motion type from the dropdown menu:

#### Flexion (X/Y)
Rotational movement around the X or Y axis with force limiting.

**Collection Methods:**
| Method | Description |
|--------|-------------|
| **Original** | Stop-adjust-resume approach for Fz control |
| **Hybrid** | Hardware force mode (Fz=0) + moveL trajectory |
| **Force** | Hardware force mode (Fz=0) + speedL control |

**Parameters:**
- **Angle**: Target rotation in degrees
- **Speed/Accel**: Movement speed and acceleration
- **Max TCP Fy/Fx**: Force limit for stopping

#### Axial Rotation
Pure rotation around the TCP Z-axis with torque monitoring.

#### Freemove
Manual waypoint collection during freedrive mode.

1. Click **Set Path** to enable freedrive and start recording
2. Manually move the robot to record waypoints
3. Click **Stop Collection** to save the path
4. Click **Run** to traverse the recorded path

### Traverse Methods

When running recorded paths, select a traverse method:

| Method | Description | Use Case |
|--------|-------------|----------|
| **moveLPath** | Smooth path with blend radius | General smooth traversal |
| **servoPath** | Precise timing with servoL | Consistent speed |
| **movePath** | Single moveL to end | Quick straight-line |
| **forceHybrid** | Fz=0 compliance + moveLPath | Surface contact traversal |
| **forceSpeedL** | Fz=0 compliance + speedL | Velocity-controlled surface following |

### Path Processing

After recording a path, you can process it:
- **Smooth (TCP/Flange)**: Apply smoothing filter
- **Interpolate (TCP/Flange)**: Resample to uniform distance intervals

### Motion Logging

Enter a filename in the **Log** field to record motion data:
- TCP position and orientation
- Force/torque readings
- Timestamps

Log files are saved to the `logs/` directory as `.log` files.

## Configuration

Configuration is managed via dataclasses in `config.py`:

### Key Configuration Groups

```python
# Robot defaults
RobotDefaults:
    ip: "192.168.1.100"
    home_position: (x, y, z, rx, ry, rz)
    tcp_offset: (x, y, z, rx, ry, rz)

# Movement defaults
MovementDefaults:
    speed: 0.1          # m/s
    acceleration: 0.2   # m/s²

# Flexion parameters
FlexionCommonDefaults:
    angle: 90.0         # degrees
    speed: 0.010        # m/s
    force_limit_y: 10.0 # N
    force_limit_x: 10.0 # N

# Traverse parameters
TraverseMoveLPathDefaults:
    speed: 0.03         # m/s
    acceleration: 0.5   # m/s²
    blend: 0.01         # m

TraverseServoPathDefaults:
    speed: 0.03         # m/s
    dt: 0.008           # s (control loop period)
    lookahead_time: 0.2 # s
    gain: 250
```

## Architecture

### UniversalRobot Class
Central class that encapsulates:
- RTDE connection management
- Robot state reading (poses, forces, joint positions)
- Movement commands (moveL, moveJ, freedrive)
- TCP offset management
- VTK visualization

### AsyncMotionRunner
QThread-based worker for asynchronous movement execution:
- **Collect Mode**: Execute movement while recording waypoints
- **Traverse Mode**: Replay recorded paths with optional force control

### WaypointCollector
Handles path recording and traversal:
- Waypoint collection during movements
- Path saving/loading (.npz format)
- Multiple traversal methods
- Force-compliant traversal options
- Statistics tracking (distance, velocity, time)

## Force Mode

The application supports UR's hardware force mode for compliant motion:

### Hybrid Method (forceHybrid)
```
Force Mode: Fz = 0 (compliant in Z)
Motion: moveLPath (smooth path following)
```

### Force SpeedL Method (forceSpeedL)
```
Force Mode: Fz = 0 (compliant in Z)
Motion: speedL (velocity control loop)
```

### Parameters
- **force_mode_z_limit**: Max Z velocity for compliance (m/s)
- **force_mode_damping**: Force mode damping (0-1)
- **force_mode_gain_scaling**: Gain scaling (0-2)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Emergency stop |
| **F** | Toggle freedrive mode |

## Console Output

During traversal, the application prints detailed statistics:
```
=== Forward Traverse ===
Method: moveLPath
Waypoints: 150
Speed: 0.03 m/s
Acceleration: 0.5 m/s²
Blend radius: 0.01 m

=== Forward traversal Statistics ===
Samples: 847
  TCP:
    Distance: 125.34 mm
    Time: 4.23 s
    Velocity: min=15.20, max=32.45, mean=29.64 mm/s
  Flange:
    Distance: 123.87 mm
    Time: 4.23 s
    Velocity: min=14.89, max=31.92, mean=29.31 mm/s
```

## Data Files

### Path Files (.npz)
Saved in `data/` directory:
```python
{
    'poses': np.array([[x, y, z, rx, ry, rz], ...]),
    'timestamps': np.array([t0, t1, ...])
}
```

### Log Files (.log)
Saved in `logs/` directory with motion data for analysis.

## Troubleshooting

### Connection Issues
- Verify robot IP address
- Ensure RTDE is enabled on the robot
- Check network connectivity

### RTDE Import Error
```bash
pip install ur-rtde
```
Or build from source in `ur_rtde_source/`.

### VTK Import Error
```bash
pip install vtk
```

### Force Mode Not Working
- Ensure robot has force/torque sensor
- Check force mode parameters in config
- Verify selection vector is correct

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

