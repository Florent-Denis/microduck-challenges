"""The Tug of War trains in the Arena's tug: the same course, string, winch, pull, start and command."""
import subprocess, sys
from types import SimpleNamespace
import mujoco, numpy as np, pytest, torch
from mjlab.scene import Scene, SceneCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

import microduck_challenges  # noqa: F401
from arena import course as arena_course, tug as arena_tug
from arena.constants import DEFAULT_POSE, JOINT_NAMES
from arena.manifest import load_event
from microduck_challenges.tug_of_war import env as tug

EVENT = load_event("tug-of-war")
CONTACT = ("geom_type", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
           "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap")
TENDON = ("tendon_limited", "tendon_stiffness", "tendon_damping", "tendon_frictionloss", "tendon_armature")
ACTUATOR = ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_gear",
            "actuator_gainprm", "actuator_biasprm", "actuator_ctrllimited", "actuator_ctrlrange",
            "actuator_forcelimited", "actuator_forcerange")


@pytest.fixture(scope="module")
def arena_model():
    return arena_course.compile_tug(EVENT, 1)


@pytest.fixture(scope="module")
def training_model():
    scene = SceneCfg(num_envs=1, entities={"robot": MICRODUCK_STANDUP_ROBOT_CFG}, spec_fn=tug.tug_scene)
    return Scene(scene, device="cpu").compile()


def test_the_training_never_imports_the_arena():
    """HF Jobs install the default groups only: no Arena there. The package's __init__ imports every tasks.py."""
    code = "import sys; sys.modules['arena'] = None; import microduck_challenges.tug_of_war.env"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_constants_are_the_arenas():
    c, b, s = EVENT.course, EVENT.bench, EVENT.start
    assert (tug.LENGTH_M, tug.WIDTH_M, tug.WALL_HEIGHT_M, tug.PIT_M, tug.PIT_DEPTH_M, tug.STRING_M) == (
        c.length_m, c.width_m, c.wall_height_m, c.pit_m, c.pit_depth_m, c.string_m)
    assert (tug.BENCH_RAMP_N_PER_S, tug.BENCH_TOP_N, tug.BENCH_JERK_N, tug.BENCH_JERK_HZ) == (
        b.ramp_n_per_s, b.top_n, b.jerk_n, b.jerk_hz)
    assert tuple(EVENT.command) == (tug.COMMAND_VX,) + (0.0,) * 12
    assert (tug.TIME_LIMIT_S, tug.SETTLE_S, tug.JOINT_PERTURBATION_RAD, tug.YAW_PERTURBATION_DEG) == (
        EVENT.time_limit_s, s.settle_s, s.joint_rad, s.yaw_deg)
    assert (tug.FALL_GRAVITY_Z, tug.FALL_HOLD_S) == (EVENT.termination.fall_gravity_z, EVENT.termination.fall_hold_s)
    assert tug.STRING_OFFSET == pytest.approx(arena_course.string_offset("legs"), abs=1e-12)
    assert (tug.SPAWN_HEIGHT_M, tug.WALL_THICKNESS_M, tug.PLATFORM_OVERRUN_M, tug.PIT_DROP_M) == (
        arena_course.SPAWN_HEIGHT_M, arena_course.WALL_THICKNESS_M, arena_course.PLATFORM_OVERRUN_M, arena_tug.PIT_DROP_M)
    assert (tug.STRING_SITE, tug.STRING, tug.WINCH_SITE, tug.WINCH) == (
        arena_course.STRING_SITE, arena_course.STRING, arena_course.WINCH_SITE, arena_course.WINCH)


def _static_colliders(model):
    return {model.geom(g).name for g in range(model.ngeom)
            if model.body_weldid[model.geom_bodyid[g]] == 0 and (model.geom_contype[g] or model.geom_conaffinity[g])}


def test_the_course_compiles_like_the_arenas(arena_model, training_model):
    names = {n for n, _, _ in tug.course_boxes()}
    assert _static_colliders(training_model) == names == _static_colliders(arena_model)  # no floor plane left
    datas = [mujoco.MjData(m) for m in (arena_model, training_model)]
    for m, d in zip((arena_model, training_model), datas):
        mujoco.mj_forward(m, d)
    for name in names:
        a, t = arena_model.geom(name).id, training_model.geom(name).id
        np.testing.assert_allclose(datas[1].geom_xpos[t], datas[0].geom_xpos[a], atol=1e-12, err_msg=name)
        for field in CONTACT:
            np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[a], err_msg=f"{name} {field}")


def test_the_string_and_winch_are_built_like_the_arenas(arena_model, training_model):
    for fields, kind, name in ((TENDON, "tendon", tug.STRING), (ACTUATOR, "actuator", tug.WINCH)):
        a, t = getattr(arena_model, kind)(name).id, getattr(training_model, kind)(name).id
        for field in fields:
            np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[a], err_msg=field)


def _pull(model, prefix, yaw_deg, pitch_deg, ctrl):
    """String end, drum, string length, and the winch's force on the trunk's free joint at the bench's start."""
    data = mujoco.MjData(model)
    trunk = model.body(prefix + "trunk_base").id
    free, dof = model.jnt_qposadr[model.body_jntadr[trunk]], model.jnt_dofadr[model.body_jntadr[trunk]]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians([0.0, pitch_deg, yaw_deg]), "xyz")
    west = arena_course.tug_layout(EVENT.course, 1)[0]
    data.qpos[free:free + 7] = [west.x, west.y, arena_course.SPAWN_HEIGHT_M, *quat]
    for joint, angle in zip(JOINT_NAMES, DEFAULT_POSE):
        data.qpos[model.joint(prefix + joint).qposadr[0]] = angle
    winch, forces = model.actuator(tug.WINCH).id, []
    for value in (0.0, ctrl):
        data.ctrl[winch] = value
        mujoco.mj_forward(model, data)
        forces.append(data.qfrc_actuator[dof:dof + 6].copy())
    return (data.site(prefix + tug.STRING_SITE).xpos.copy(), data.site(tug.WINCH_SITE).xpos.copy(),
            data.ten_length[model.tendon(tug.STRING).id], forces[1] - forces[0])


@pytest.mark.parametrize("yaw_deg, pitch_deg", [(0.0, 0.0), (12.0, -25.0)])
def test_the_winch_pulls_like_the_arenas(arena_model, training_model, yaw_deg, pitch_deg):
    theirs = _pull(arena_model, "", yaw_deg, pitch_deg, 3.0)
    ours = _pull(training_model, "robot/", yaw_deg, pitch_deg, 3.0)
    for t, a in zip(ours, theirs):
        np.testing.assert_allclose(t, a, atol=1e-9)
    force = ours[3][:3]
    assert np.linalg.norm(force) == pytest.approx(3.0, rel=1e-6) and force[0] > 0  # 3 N, towards the drum
