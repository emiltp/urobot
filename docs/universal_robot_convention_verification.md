# Convention Verification for UniversalRobot Class

This document cites external references to verify the conventions used in `UniversalRobot` and related code. It covers **all** pose and offset calculations for flange, TCP, and ref frame.

Section **"Pure Math and General Robotics"** below cites only non-UR references (axis-angle, rigid-body transforms, moment translation). UR-specific references are in the individual sections.

---

## Pure Math and General Robotics References

These references verify the **calculations and conventions** without any UR-specific documentation.

| Topic | Reference | What It Verifies |
|-------|-----------|------------------|
| **Axis-angle / rotation vector** | [SciPy `Rotation.from_rotvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html) | Direction = rotation axis, magnitude = angle (rad). Code uses `Rotation.from_rotvec(axis * angle)` and `as_rotvec()`, matching UR’s (rx, ry, rz). |
| **Axis-angle** | [Wikipedia – Axis-angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) | Rotation vector = axis × angle; direction = axis, magnitude = θ. |
| **Moment translation** | [Engineering LibreTexts – Moment about a point](https://eng.libretexts.org/Bookshelves/Mechanical_Engineering/Mechanics_Map/03:_Static_Equilibrium/3.05:_Moment_about_a_Point) | `M_B = M_A + r_AB × F` with r_AB from A to B. Code: flange (A) → TCP (B), so `M_TCP = M_flange + (p_flange − p_tcp) × F`. |
| **Moment translation** | [Mechanics Map – Moment vector](https://mechanicsmap.psu.edu/websites/3_equilibrium_rigid_body/3-4-moment_vector/momentvector.html) | M = r × F; right-hand rule; same convention. |
| **Wrench / power invariance** | [Modern Robotics – 3.4 Wrenches](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/) | Wrench transforms via adjoint transpose; power invariance implies correct moment translation when changing reference point. |
| **Rigid body transform** | Standard robotics (Craig, Siciliano) | Child = Parent @ Offset: `p_child = p_parent + R_parent @ offset_pos`, `R_child = R_parent @ R_offset`. |
| **Vector frame change** | Standard linear algebra | To express vector v from frame A in frame B: `v_B = R_A_to_B @ v_A`. With `R_base_tcp` = base from TCP: `v_tcp = R_base_tcp.T @ v_base`. |

---

## Overview: Offsets and Poses

| Frame | Pose (base) | Offset | Applied To | Formula |
|-------|-------------|--------|------------|---------|
| **Flange** | From FK or computed | — | — | Base reference |
| **TCP** | From RTDE | TCP offset [x,y,z,rx,ry,rz] | Flange | TCP = Flange @ TCP_offset |
| **Ref** | Computed | Ref offset [x,y,z,rx,ry,rz] | TCP or flange | Ref = Parent @ Ref_offset |

All offsets use the same rigid-transform composition: `Child = Parent @ Offset` (position: `p_child = p_parent + R_parent @ offset_pos`; rotation: `R_child = R_parent @ R_offset`).

**Code methods:**
- `_calculateFlangePoseFromTcp(tcpPose, tcpOffset)` — flange from TCP
- `_calculateTcpOffsetFromPoses(tcpPose, flangePose)` — TCP offset from poses
- `_calculateRefFramePose(parentPose, refFrameOffset)` — ref from parent (TCP or flange)
- `getTcpForceInTcpFrame()` — wrench at TCP (moment flange→TCP, rotate to TCP)
- `getFlangeForceInFlangeFrame()` — wrench at flange (rotate to flange)
- `getRefFrameForceInRefFrame(relative_to)` — wrench at ref (moment flange→ref, rotate to ref)

---

## 1. TCP Offset (Flange → TCP)

**Convention in code:** TCP offset is the transformation from flange to TCP: `TCP = Flange @ Offset`

**Reference:** [Universal Robots – get_tcp_offset()](https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-scriptmanual/all_scripts/get_tcp_offset.htm)

> Gets the active tcp offset, i.e. **the transformation from the output flange coordinate system to the TCP as a pose**.

**Reference:** ur_rtde API (setTcp, getTCPOffset)

> Sets/gets the active tcp offset, i.e. **the transformation from the output flange coordinate system to the TCP as a pose**.

**Conclusion:** Code convention matches the UR specification.

---

## 2. Pose Format (x, y, z, rx, ry, rz)

**Convention in code:** Position in metres; orientation as axis-angle in radians (rx, ry, rz; magnitude = angle).

**Reference:** [Universal Robots – Axis-angle representation](https://www.universal-robots.com/articles/ur/programming/axis-angle-representation/)

> Axis-angle representation for a rotation… The direction of the vector defines the rotation axis; **the magnitude (length) of the vector equals the rotation angle θ in radians**.

**Reference:** [UR RTDE – actual_TCP_pose](https://www.universal-robots.com/manuals/EN/HTML/SW10_10/Content/Prod-RTDE/Real_Time_Data_Exchange_RTDE.htm)

> actual_TCP_pose | VECTOR6D | Actual Cartesian coordinates of the tool: **(x,y,z,rx,ry,rz)**, where **rx, ry and rz is a rotation vector representation** of the tool orientation

**Reference:** ur_rtde – getActualTCPPose()

> Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where rx, ry and rz is a **rotation vector representation** of the tool orientation

**Conclusion:** Code convention matches UR pose format.

---

## 3. Force/Torque Measurement Frame

**Convention in code:** `getActualTCPForce()` returns wrench **at the flange** in **base frame**.

**Reference:** [Universal Robots – get_tcp_force()](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-scriptmanual/G5/get_tcp_force.htm)

> Returns the force/torque vector **at the tool flange**.
>
> The function returns p[Fx(N), Fy(N), Fz(N), TRx(Nm), TRy(Nm), TRz(Nm)] where the forces: Fx, Fy, and Fz in Newtons and the torques: TRx, TRy and TRz in Newtonmeters are all **measured at the tool flange** with the **orientation of the robot base coordinate system**.

**Reference:** [UR RTDE – actual_TCP_force](https://www.universal-robots.com/manuals/EN/HTML/SW10_10/Content/Prod-RTDE/Real_Time_Data_Exchange_RTDE.htm)

> actual_TCP_force | VECTOR6D | Generalized forces in the TCP

(Interpreted via get_tcp_force script manual as flange measurement in base orientation.)

**Reference:** [Universal_Robots_ROS2_Driver #235](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/issues/235)

> Confirms UR reports wrench at flange in base frame; ROS driver transforms to TCP frame for users.

**Conclusion:** UR reports wrench at the flange and expressed in base frame. Code is aligned with this.

---

## 4. Moment Translation Between Points

**Convention in code:** To move moment from flange (A) to TCP (B):  
`M_TCP = M_flange - (p_tcp - p_flange) × F`  
(equivalent to `M_TCP = M_flange + (p_flange - p_tcp) × F`)

**Reference:** [Modern Robotics – 3.4 Wrenches](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/)

> Wrench transformation uses the adjoint transpose; power invariance implies moment translation when changing reference point.

**Reference:** Standard mechanics

> For a force F at point A, moment about point B:  
> `M_B = M_A + r_BA × F` where `r_BA = p_A - p_B` (from B to A).  
> Here A = flange, B = TCP, so `r = p_flange - p_tcp`, hence  
> `M_TCP = M_flange + (p_flange - p_tcp) × F = M_flange - (p_tcp - p_flange) × F`.

**Reference:** [UR Script manual – get_tcp_force example](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-scriptmanual/G5/get_tcp_force.htm)

> Example uses `pose_trans` and `wrench_trans(get_tcp_offset(), …)` to move wrench from flange to TCP, matching the same physical transformation.

**Conclusion:** Code moment-translation formula matches standard mechanics and UR-style transforms.

---

## 5. Ref Frame Transform (Parent @ Offset)

**Convention in code:** `RefFrame = Parent @ Offset`  
`ref_pos = parent_pos + R_parent @ offset_pos`  
`ref_rot = R_parent @ R_offset`

The ref offset is applied to a **parent** frame chosen by the user:
- **Parent = TCP**: `Ref = TCP @ Ref_offset` — offset defined in TCP frame
- **Parent = flange**: `Ref = Flange @ Ref_offset` — offset defined in flange frame

Same composition rule as TCP offset (Flange @ TCP_offset), just with a different parent.

**Reference:** [ur_rtde – poseTrans](https://sdurobotics.gitlab.io/ur_rtde/api/api.html)

> T_world->to = T_world->from * T_from->to  
> (Same composition rule as used for ref frame.)

**Conclusion:** Code ref-frame computation follows standard rigid-transform composition. Ref offset applied to TCP or flange uses the same convention.

---

## 6. Tool Flange Pose (Base Frame)

**Reference:** ur_rtde – getActualToolFlangePose()

> Returns the 6d pose representing the **tool flange position and orientation specified in the base frame**, without the Tool Center Point offset.

**Conclusion:** Flange pose is given in base frame, matching what the code expects.

---

## 7. Inverse: Flange from TCP (Applied Offset)

**Convention in code:** From `TCP = Flange @ TCP_offset` →  
`Flange = TCP @ inv(TCP_offset)`  
- `R_flange = R_tcp @ R_offset^T`  
- `p_flange = p_tcp - R_flange @ offset_pos`

**Reference:** Standard homogeneous transform inversion. Same as Section 1 (TCP offset) — just the inverse operation.

**Conclusion:** `_calculateFlangePoseFromTcp` correctly inverts the TCP offset.

---

## 8. Wrench at Ref Frame (Flange → Ref Moment Translation)

**Convention in code:** Same moment-translation rule as flange→TCP:  
`M_ref = M_flange + (p_flange - p_ref) × F`

Used in `getRefFrameForceInRefFrame(relative_to)` when ref offset is set. Parent (TCP or flange) determines where ref is; moment is always translated from flange (where UR measures) to ref.

**Conclusion:** Same formula as Section 4; applies to ref as well as TCP.

---

## Summary

| Convention            | Applied To | Code Assumption                    | Verified By                                                          |
|-----------------------|------------|------------------------------------|----------------------------------------------------------------------|
| TCP offset            | Flange     | TCP = Flange @ Offset              | UR get_tcp_offset, ur_rtde setTcp/getTCPOffset                       |
| Ref offset            | TCP or flange | Ref = Parent @ Offset          | ur_rtde poseTrans, same composition as TCP offset                    |
| Flange from TCP       | Inverse    | Flange = TCP @ inv(Offset)        | Standard transform inversion                                         |
| Pose format           | All        | [x,y,z,rx,ry,rz], axis-angle rad  | UR axis-angle docs, RTDE actual_TCP_pose                            |
| Force/torque origin   | Flange     | Measured at flange, base frame    | UR get_tcp_force manual                                              |
| Moment flange→TCP     | TCP        | M_tcp = M_flange - (tcp-flange)×F | Mechanics, Modern Robotics, UR wrench example                        |
| Moment flange→ref     | Ref        | M_ref = M_flange + (flange-ref)×F | Same formula as flange→TCP                                           |
