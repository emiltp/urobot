# transform_wrench() Replacement Guide

Use `robot.getTcpForceInTcpFrame()` where only the **current** wrench is needed.  
Use `transform_wrench()` where **baseline** or **differential** wrenches are transformed.

---

## Replace with robot.getTcpForceInTcpFrame()

**Reason:** These use only the current wrench from RTDE. `getTcpForceInTcpFrame()` returns the wrench at TCP in TCP frame (moment translation + rotation) and is the correct source.

| File | Line | Current Code |
|------|------|--------------|
| `src/movements/flexion_x/force/main.py` | 108-110 | `tcpWrenchInBase = self.rtde_r.getActualTCPForce()`<br>`tcpForce = transform_wrench(currentPose, tcpWrenchInBase)` |
| `src/movements/flexion_x/hybrid/main.py` | 105-107 | same pattern |
| `src/movements/flexion_x/original/main.py` | 62-65 | same pattern |
| `src/movements/flexion_y/force/main.py` | 108-111 | same pattern |
| `src/movements/flexion_y/hybrid/main.py` | 95-98 | same pattern |
| `src/movements/flexion_y/original/main.py` | 62-65 | same pattern |
| `src/movements/new_x/force/main.py` | 97-100 | same pattern |
| `src/movements/new_x/hybrid/main.py` | 76-79 | same pattern |
| `src/movements/new_x/original/main.py` | 66-69 | same pattern |
| `src/movements/new_y/force/main.py` | 95-98 | same pattern |
| `src/movements/new_y/hybrid/main.py` | 76-79 | same pattern |
| `src/movements/new_y/original/main.py` | 66-69 | same pattern |
| `src/movements/new_z/force/main.py` | 102-105 | same pattern |
| `src/movements/new_z/hybrid/main.py` | 83-86 | same pattern |
| `src/movements/new_z/original/main.py` | 67-70 | same pattern |
| `src/movements/rotation/force/main.py` | 90-93 | same pattern |
| `src/movements/rotation/direct/main.py` | 82-85 | same pattern |
| `src/movements/rotation/hybrid/main.py` | 76-79 | same pattern |
| `src/movements/waypoint_collector.py` | 1515-1520 | `tcpWrench = self.rtde_r.getActualTCPForce()`<br>`tcpPose = list(self.rtde_r.getActualTCPPose())`<br>`tcpForce = transform_wrench(tcpPose, tcpWrench)` |

**Replacement:**
```python
tcpForce = self.robot.getTcpForceInTcpFrame()
if tcpForce is None:
    tcpForce = [0.0] * 6  # fallback if robot unavailable
```

**Note:** Remove the `transform_wrench` import from these files if it is no longer used.

---

## Keep transform_wrench() – Cannot Replace

**Reason:** These transform **baseline** wrenches (captured at start) or **differential** wrenches (current − initial), not the live wrench. `getTcpForceInTcpFrame()` only returns the current wrench.

### Arc modules: baseline + differential

| File | Line | Usage | Argument |
|------|------|-------|----------|
| **arc_force/movel/main.py** | 72-74 | Target wrench from baseline | `transform_wrench(initial_pose, initial_wrench_base)` |
| | 104-106 | Target wrench from baseline | `transform_wrench(current_pose, initial_wrench_base)` |
| | 115-118 | Monitor differential | `wrench_base = [c-i for c,i in zip(wrench_base, initial_wrench_base)]`<br>`wrench_tcp = transform_wrench(current_pose, wrench_base)` |
| **arc_force/servol/main.py** | 86-88 | Target wrench from baseline | `transform_wrench(current_pose, initial_wrench_base)` |
| | 100-102 | Target wrench from baseline | same |
| | 116-119 | Monitor differential | same pattern |
| **arc_force/speedl/main.py** | 89-91 | Target wrench from baseline | same |
| | 103-106 | Target wrench from baseline | same |
| | 114-117 | Monitor differential | same pattern |
| **arc_y/movel/main.py** | 68-70 | Target wrench from baseline | `transform_wrench(initial_pose, initial_wrench_base)` |
| | 94-96 | Target wrench from baseline | `transform_wrench(current_pose, initial_wrench_base)` |
| | 105-108 | Monitor differential | same pattern |
| **arc_z/movel/main.py** | 74-76 | Target wrench from baseline | same |
| | 100-103 | Target wrench from baseline | same |
| | 111-114 | Monitor differential | same pattern |

**Why keep transform_wrench():**
- **Baseline:** `initial_wrench_base` is saved at movement start (e.g. after zeroing). It is not the current wrench.
- **Differential:** `wrench_base = current - initial` removes the initial bias. Neither current nor initial can be replaced by `getTcpForceInTcpFrame()` here.

**Optional improvement:** Pass `flange_pose` where available so moment translation is correct:
```python
flange_pose = None
try:
    if hasattr(self.rtde_c, 'getActualToolFlangePose'):
        flange_pose = list(self.rtde_c.getActualToolFlangePose())
except Exception:
    pass
wrench_tcp = transform_wrench(current_pose, wrench_base, flange_pose)
```
Apply the same pattern for baseline target wrenches if you want moment translation for those as well.

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Replace with getTcpForceInTcpFrame | 19 call sites in 19 files | Use `robot.getTcpForceInTcpFrame()` |
| Keep transform_wrench | 15 call sites in 5 arc modules | Keep; optionally add `flange_pose` for moment translation |
