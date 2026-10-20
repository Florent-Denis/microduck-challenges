"""The Tug of War's training environment: the Arena's pulling bench.

The duck trains on the Arena's tug: a lane cut by a pit, the west duck's start, a string tied
at the base of its neck and run to a winch that pulls it towards the pit. The HF job installs
no Arena, so the event's numbers and the course's arithmetic (arena/course.py) are copied here;
tests/test_tug_of_war.py pins each copy against the Arena's own model.
"""

import mujoco

# Copied from the Arena (the HF job cannot import it); tests/test_tug_of_war.py pins each one.
LENGTH_M, WIDTH_M, WALL_HEIGHT_M = 4.0, 0.8, 0.15          # arena/events/tug-of-war.toml
PIT_M, PIT_DEPTH_M, STRING_M = 0.30, 0.30, 1.00
COMMAND_VX = -0.30
TIME_LIMIT_S, SETTLE_S = 30.0, 0.5
JOINT_PERTURBATION_RAD, YAW_PERTURBATION_DEG = 0.01, 2.0
FALL_GRAVITY_Z, FALL_HOLD_S = -0.5, 0.3                    # [termination]: a fall loses
BENCH_RAMP_N_PER_S, BENCH_TOP_N, BENCH_JERK_N, BENCH_JERK_HZ = 0.25, 7.0, 0.4, (0.7, 1.9)
SPAWN_HEIGHT_M, WALL_THICKNESS_M, PLATFORM_OVERRUN_M = 0.12, 0.05, 20.0   # arena/course.py
STRING_OFFSET = (0.026, 0.0145, 0.0324215)                 # string_offset("legs")
PIT_DROP_M = 0.05                                          # arena/tug.py
STRING_SITE, STRING, WINCH_SITE, WINCH = "string_end", "string", "winch_drum", "winch"

ROBOT = "robot/"  # the prefix mjlab attaches the robot entity under


def course_boxes() -> list[tuple[str, tuple, tuple]]:
    """The tug's ground and walls, `(name, pos, half_extents)`: the Arena's
    `pit_boxes(course) + tug_walls(course)`, same order, same arithmetic. Two platforms with
    their top face at z = 0 either side of the pit, the pit's floor at z = -PIT_DEPTH_M, and
    the lane's two walls; no plane, so the pit is a hole."""
    half_t = WALL_THICKNESS_M / 2
    half_y = WIDTH_M / 2 + WALL_THICKNESS_M + PLATFORM_OVERRUN_M
    x_out = LENGTH_M / 2 + PLATFORM_OVERRUN_M
    half_pit = PIT_M / 2
    slab = PIT_DEPTH_M + WALL_THICKNESS_M
    platform = ((x_out - half_pit) / 2, half_y, slab / 2)
    centre = (x_out + half_pit) / 2
    half_h = WALL_HEIGHT_M / 2
    y_off = WIDTH_M / 2 + half_t
    return [
        ("platform_west", (-centre, 0.0, -slab / 2), platform),
        ("platform_east", (centre, 0.0, -slab / 2), platform),
        ("pit_floor", (0.0, 0.0, -PIT_DEPTH_M - WALL_THICKNESS_M / 2), (half_pit, half_y, WALL_THICKNESS_M / 2)),
    ] + [
        (name, (0.0, y, half_h), (LENGTH_M / 2, half_t, half_h))
        for name, y in (("wall_left", y_off), ("wall_right", -y_off))
    ]


def west_start() -> tuple[float, float]:
    """The bench duck's trunk (x, y), facing east: its string end on y = 0, half the string
    west of the pit's centre."""
    ox, oy, _ = STRING_OFFSET
    half = STRING_M / 2
    return -half - ox, -oy


def winch_position() -> tuple[float, float, float]:
    """The drum: where a duck facing west would hold the other end, so the string starts taut."""
    return (STRING_M / 2, 0.0, SPAWN_HEIGHT_M + STRING_OFFSET[2])


def tug_scene(spec: mujoco.MjSpec) -> None:
    """mjlab's `SceneCfg.spec_fn`: the Arena's bench, built on the scene once the robot is in.

    The course is a static body named `terrain`, as mjlab's foot contact sensor expects. The
    string runs from the base of the neck to the drum; the winch is a motor on it whose ctrl is
    the pull in newtons (gear -1: a positive ctrl shortens the string). It is the scene's last
    actuator, so the robot's ctrl addresses stay, and the scene keyframe takes its 0.0.
    """
    terrain = spec.worldbody.add_body(name="terrain")
    for name, pos, size in course_boxes():
        terrain.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
    spec.body(ROBOT + "trunk_base").add_site(name=ROBOT + STRING_SITE, pos=STRING_OFFSET)
    spec.worldbody.add_site(name=WINCH_SITE, pos=winch_position())
    string = spec.add_tendon(name=STRING)
    string.wrap_site(ROBOT + STRING_SITE)
    string.wrap_site(WINCH_SITE)
    winch = spec.add_actuator(name=WINCH, target=STRING, trntype=mujoco.mjtTrn.mjTRN_TENDON)
    winch.set_to_motor()
    winch.gear = [-1.0, 0, 0, 0, 0, 0]
    key = spec.key("init_state")
    key.ctrl = [*key.ctrl, 0.0]
