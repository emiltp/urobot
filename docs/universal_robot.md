# UniversalRobot Class

Documentation for the `UniversalRobot` class in `src/objects/universal_robot.py`, which provides a unified interface for controlling and reading from a Universal Robot via RTDE.

## Overview

The `UniversalRobot` class encapsulates:

- **RTDE connection**: Control and receive interfaces (`rtde_control`, `rtde_receive`)
- **Robot control**: Movement (moveL, moveJ, moveLPath), freedrive, stop
- **State reading**: TCP pose, flange pose, joint positions, force/torque
- **Pose calculations**: TCP offset, flange pose from TCP, reference frame pose
- **Visualization**: 3D rendering via `UniversalRobotActor` (base, flange, TCP, ref frame axes)

It serves as the central abstraction between the application UI/movements and the robot hardware.

---

## Coordinate Frames

| Frame | Description |
|-------|-------------|
| **Base** | Robot base frame (origin at robot base) |
| **Flange** | Tool flange at end of last joint |
| **TCP** | Tool center point; flange + TCP offset |
| **Ref frame** | User-defined frame, offset relative to TCP or flange |

---

## Pose Conventions

- **Format**: `[x, y, z, rx, ry, rz]`
  - Position: `x, y, z` in metres (base frame)
  - Orientation: `rx, ry, rz` in **axis-angle** (radians), magnitude = rotation angle
- **UR/RTDE**: Same convention; UR controller uses axis-angle for orientations

---

## Calculation Formulas

### TCP ↔ Flange (TCP offset)

UR defines the TCP relative to the flange. The homogeneous transform:

```
TCP = Flange @ Offset
```

- `_calculateFlangePoseFromTcp(tcpPose, tcpOffset)` → flange pose
  - `R_flange_to_base = R_tcp_to_base @ R_offset^T`
  - `flange_pos = tcp_pos - R_flange_to_base @ offset_pos`

- `_calculateTcpOffsetFromPoses(tcpPose, flangePose)` → TCP offset in flange frame
  - Inverse of above: offset position and rotation from flange to TCP

**Round-trip invariant**: For any valid `tcp`, `offset`:
```
flange = _calculateFlangePoseFromTcp(tcp, offset)
recovered = _calculateTcpOffsetFromPoses(tcp, flange)
→ recovered ≈ offset (within numerical precision)
```

### Reference Frame

The reference frame is defined relative to a parent (TCP or flange):

```
RefFrame = Parent @ Offset
```

- `_calculateRefFramePose(parentPose, refFrameOffset)` → ref frame pose in base
  - `ref_pos = parent_pos + R_parent @ offset_pos`
  - `ref_rot = R_parent @ R_offset`

Parent is chosen via `setRefFrameRelativeTo("tcp")` or `"flange"`.

### Wrench Transformations

UR reports force/torque at the **flange** in the **base** frame: `[Fx, Fy, Fz, Mx, My, Mz]`.

| Method | Description |
|--------|-------------|
| `getTcpForceInTcpFrame()` | Wrench at TCP in TCP frame |
| | 1. Translate moment: flange → TCP (`M_tcp = M_flange - (tcp - flange) × F`) |
| | 2. Rotate force and moment into TCP frame |
| `getFlangeForceInFlangeFrame()` | Wrench at flange in flange frame |
| | Rotate force and moment into flange frame (no position translation) |
| `getRefFrameForceInRefFrame(relative_to)` | Wrench at ref frame in ref frame |
| | Translate moment flange → ref; rotate to ref frame |

---

## API Reference

### Connection

| Method | Description |
|--------|-------------|
| `connect(ip, tcpOffset)` | Connect via RTDE; optionally set TCP offset |
| `disconnect()` | Disconnect; ends freedrive if active |
| `reconnect()` | Disconnect then connect with stored IP |
| `isConnected()` | True if RTDE receive interface is connected |
| `ip` | Property: current IP address |

### State Reading

| Method | Description |
|--------|-------------|
| `update()` | Update state from robot; refresh visualization; return `(tcpPose, flangePose)` |
| `getTcpPose()` | TCP pose from RTDE |
| `getFlangePose()` | Flange pose computed from TCP and offset |
| `getJointPositions()` | Joint angles `[q0..q5]` in radians |
| `getTcpForce()` | Force/torque at flange in base frame |
| `isProtectiveStopped()` | True if protective stop |
| `isEmergencyStopped()` | True if emergency stop |
| `isEqualToTcpPose(pose, pos_tol, orient_tol)` | Compare current TCP to given pose |

### TCP Offset

| Method | Description |
|--------|-------------|
| `getTcpOffset()` | Current TCP offset or None |
| `setTcpOffset(offset)` | Set TCP offset (6 elements); syncs to robot when connected |
| `calculateTcpOffsetForAlignment(targetCenter, targetAxis)` | Compute TCP offset so TCP aligns with target point; keeps current orientation offset (simplified) |

### Reference Frame

| Method | Description |
|--------|-------------|
| `getRefFrameOffset()` | Ref frame offset `[x,y,z,rx,ry,rz]` |
| `setRefFrameOffset(offset)` | Set ref frame offset |
| `setRefFrameRelativeTo("tcp"|"flange")` | Parent for ref frame offset |
| `getRefFrameRelativeTo()` | Current parent |
| `clearRefFrameOffset()` | Reset offset to zero |

### Control

| Method | Description |
|--------|-------------|
| `moveL(pose, speed, acceleration, asynchronous)` | Linear move |
| `moveJ(joints, speed, acceleration, asynchronous)` | Joint move |
| `moveLPath(path, speed, acceleration, blend)` | Path of linear moves |
| `stop(deceleration)` | Stop linear motion |
| `stopScript()` | Stop RTDE control script |
| `startFreedrive()` / `stopFreedrive()` / `toggleFreedrive()` | Freedrive mode |
| `zeroFtSensor()` | Zero force/torque sensor |

### Force / Torque

| Method | Description |
|--------|-------------|
| `getTcpForceInTcpFrame()` | Wrench at TCP in TCP frame |
| `getFlangeForceInFlangeFrame()` | Wrench at flange in flange frame |
| `getRefFrameForceInRefFrame(relative_to)` | Wrench at ref frame in ref frame |

### Visualization

| Method | Description |
|--------|-------------|
| `addToRenderer(renderer)` | Add robot actors to VTK renderer |
| `removeFromRenderer()` | Remove from renderer |
| `setVisibility(visible)` | Show/hide visualization |
| `resetVisualization()` | Reset actor state |
| `actor` | Property: `UniversalRobotActor` instance |

### Properties (convenience)

| Property | Returns |
|----------|---------|
| `tcpPose` | Current TCP pose (copy) or None |
| `flangePose` | Current flange pose (copy) or None |
| `tcpOffset` | Current TCP offset (copy) or None |
| `jointPositions` | Joint positions (copy) or None |
| `tcpForce` | Force/torque (copy) or None |
| `refFrameOffset` | Ref frame offset (copy) |

---

## Dependencies

- `rtde_control`, `rtde_receive` (from `ur-rtde`)
- `src.utils`: `axis_angle_to_rotation_matrix`, `rotation_matrix_to_axis_angle`
- `config.defaults`: tolerance values for pose comparison
- `UniversalRobotActor` (from `src.objects.actors.universal_robot_actor`)

---

## Related

- **transform_wrench** (`src.utils`): Used for baseline/differential wrenches (e.g. arc movements); see `docs/transform_wrench_replacement_list.md`.
- **RobotUpdateThread**: QThread that polls `robot.update()` and emits `pose_updated` / `error_occurred` for UI.

---

## Testing

Run tests with the project conda environment:

```bash
conda activate ur-test
python -m pytest tests/ -v
```

Tests cover calculation methods (round-trips, ref frame), connection/state with mocked RTDE, movement control, and wrench transformations.
