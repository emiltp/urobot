"""Parameter tooltips explaining what each parameter does and higher/lower effects."""

PARAMETER_TOOLTIPS = {
    "speed": (
        "Linear velocity of the TCP during movement or path traversal.\n"
        "Higher: faster movement, quicker collection.\n"
        "Lower: slower, more precise."
    ),
    "acceleration": (
        "Rate at which the robot reaches target speed from rest.\n"
        "Higher: quicker reach to target speed.\n"
        "Lower: gentler start/stop."
    ),
    "force_limit": (
        "Maximum allowed force (Fx or Fy) in the TCP frame before the motion stops for safety.\n"
        "Higher: allows more force before stop.\n"
        "Lower: stricter safety limit."
    ),
    "force_control_gain": (
        "How much Z position adjusts per unit of force error (m/N). Controls response to Fz deviation from target.\n"
        "Higher: more responsive to force error, faster Fz correction.\n"
        "Lower: smoother, less aggressive."
    ),
    "force_deadband": (
        "Force threshold (N) below which no Z adjustment is made. Reduces reaction to noise.\n"
        "Higher: ignores small forces, less jitter.\n"
        "Lower: more sensitive, reacts to small forces."
    ),
    "max_adj_per_step": (
        "Maximum Z displacement per correction step in the Original method.\n"
        "Higher: larger Z adjustments per step, faster correction.\n"
        "Lower: finer steps, smoother."
    ),
    "min_adj_interval": (
        "Minimum time (s) between Z adjustments in the Original method.\n"
        "Higher: less frequent adjustments, more stable.\n"
        "Lower: more frequent, more reactive."
    ),
    "force_mode_z_limit": (
        "Maximum Z velocity (m/s) allowed during force-mode compliance. Limits how fast the robot yields in Z.\n"
        "Higher: faster Z compliance velocity.\n"
        "Lower: slower, more controlled."
    ),
    "force_mode_xy_limit": (
        "Maximum XY velocity (m/s) during force-mode compliance for lateral yielding.\n"
        "Higher: faster XY compliance velocity.\n"
        "Lower: slower, more controlled."
    ),
    "force_mode_damping": (
        "Reduces oscillation in force-mode control. 0 = no damping, 1 = maximum.\n"
        "Higher: less oscillation, more stable.\n"
        "Lower: more responsive, may oscillate."
    ),
    "force_mode_gain_scaling": (
        "Scales the force-mode controller gain. Affects how strongly the robot responds to force error.\n"
        "Higher: stronger force response.\n"
        "Lower: weaker, softer response."
    ),
    "control_loop_dt": (
        "Time step (s) of the control loop. Smaller = more frequent updates.\n"
        "Higher: lower CPU usage, less precise.\n"
        "Lower: higher precision, more CPU."
    ),
    "rotation_speed_factor": (
        "Converts linear speed to angular speed (rad/s) for rotational motion.\n"
        "Higher: faster angular rotation.\n"
        "Lower: slower rotation."
    ),
    "blend": (
        "Blend radius (m) at waypoints. Path rounds corners instead of stopping.\n"
        "Higher: smoother corners between waypoints.\n"
        "Lower: sharper corners (0 = sharp)."
    ),
    "servo_dt": (
        "Control loop period (s) for servoPath. Time between servoL commands.\n"
        "Higher: less CPU.\n"
        "Lower: finer control loop."
    ),
    "lookahead": (
        "Lookahead time (s) for servoL. How far ahead the path is evaluated.\n"
        "Higher: smoother path following, more lag.\n"
        "Lower: more responsive, less smooth."
    ),
    "servo_gain": (
        "Stiffness of path tracking for servoPath. Higher = follow path more tightly.\n"
        "Higher: stiffer path tracking.\n"
        "Lower: softer, more compliant."
    ),
    "ramp_up": (
        "Time (s) to linearly ramp from zero to full speed at traverse start.\n"
        "Higher: slower speed ramp at start.\n"
        "Lower: quicker to full speed."
    ),
    "angle": (
        "Target rotation angle (deg) around the movement axis.\n"
        "Higher: larger rotation angle.\n"
        "Lower: smaller rotation."
    ),
    "max_moment": (
        "Maximum allowed moment (Nm) in the TCP frame before the motion stops for safety.\n"
        "Higher: allows more moment before stop.\n"
        "Lower: stricter limit."
    ),
}
