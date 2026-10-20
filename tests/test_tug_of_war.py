"""The Tug of War trains in the Arena's tug: the same course, string, winch, pull, start and command."""
import subprocess, sys
from types import SimpleNamespace
import mujoco, numpy as np, pytest, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab_microduck import challenge as ch
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

import microduck_challenges  # noqa: F401
from arena import course as arena_course, tug as arena_tug
from arena.constants import DEFAULT_POSE, JOINT_NAMES
from arena.manifest import load_event
from microduck_challenges.tug_of_war import env as tug, rewards

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


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_pull_is_the_benchs(seed):
    profile = arena_tug.bench_profile(EVENT.bench, EVENT.id, seed)
    row = [profile["ramp_n_per_s"], profile["top_n"], *[v for jerk in profile["jerks"] for v in jerk]]
    t = torch.linspace(0.0, EVENT.time_limit_s, 1501, dtype=torch.float64)
    ours = tug.bench_force(torch.tensor([row] * len(t), dtype=torch.float64), t)
    assert ours.tolist() == pytest.approx([arena_tug.bench_force(profile, s) for s in t.tolist()], abs=1e-9)
    settle = torch.tensor([-tug.SETTLE_S, -0.02], dtype=torch.float64)
    assert tug.bench_force(torch.tensor([row] * 2, dtype=torch.float64), settle).tolist() == [0.0, 0.0]


def test_training_pulls_around_the_bench():
    cfg = tug.WinchCommandCfg()
    assert cfg.ramp_n_per_s[0] <= tug.BENCH_RAMP_N_PER_S <= cfg.ramp_n_per_s[1]
    assert cfg.jerk_n[0] <= tug.BENCH_JERK_N <= cfg.jerk_n[1]
    assert cfg.hz_scale[0] <= 1.0 <= cfg.hz_scale[1] and cfg.jerk_hz == tug.BENCH_JERK_HZ


def test_the_winch_pulls_its_profile_on_the_arenas_clock(training_model):
    n = 64
    ctrl = torch.zeros(n, training_model.nu, dtype=torch.float64)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, episode_length_buf=torch.zeros(n, dtype=torch.long),
                          sim=SimpleNamespace(mj_model=training_model, data=SimpleNamespace(ctrl=ctrl)))
    term = tug.WinchCommandCfg().build(env)
    term.reset(torch.arange(n))
    p = term.profile
    assert ((p[:, 0] >= 0.125) & (p[:, 0] <= 0.75) & (p[:, 1] >= 0.5) & (p[:, 1] <= 2.0)).all()
    winch = training_model.actuator(tug.WINCH).id
    for k in (0, 24, 25, 26, 500, 1524):
        env.episode_length_buf[:] = k
        term.compute(0.02)
        expected = tug.bench_force(p, torch.full((n,), k * 0.02 - tug.SETTLE_S, dtype=p.dtype))
        assert torch.allclose(ctrl[:, winch], expected.to(ctrl.dtype))
        assert torch.allclose(term.command[:, 0], (expected / tug.BENCH_TOP_N).to(term.command.dtype))
    assert not ctrl[:, [i for i in range(training_model.nu) if i != winch]].any()  # servos untouched


def test_the_winch_top_widens_stage_by_stage():
    term = SimpleNamespace(cfg=tug.WinchCommandCfg())
    env = SimpleNamespace(common_step_counter=0, command_manager=SimpleNamespace(get_term={"winch": term}.get))
    stages = [{"step": 0, "top_n": (0.5, 2.0)}, {"step": 100, "top_n": (0.5, 4.0)}]
    for step, top in ((0, (0.5, 2.0)), (99, (0.5, 2.0)), (100, (0.5, 4.0))):
        env.common_step_counter = step
        tug.winch_top_curriculum(env, None, stages)
        assert term.cfg.top_n == top


@pytest.fixture(scope="module")
def cfg():
    return tug.make_env_cfg()


def test_the_scene_and_solver_are_the_tugs(cfg, arena_model):
    assert cfg.scene.terrain is None and cfg.scene.spec_fn is tug.tug_scene
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG  # collisions pinned by test_got_a_talent
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (arena_model.opt.iterations, arena_model.opt.ls_iterations)
    assert cfg.sim.mujoco.timestep == pytest.approx(arena_model.opt.timestep) and cfg.decimation == 4
    assert cfg.sim.mujoco.integrator == "implicitfast" and arena_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_the_actor_reads_the_events_command(cfg):
    ranges = cfg.commands["twist"].ranges
    assert ranges.lin_vel_x == (EVENT.command[0],) * 2 and ranges.lin_vel_y == (0.0, 0.0) and ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.commands["twist"].rel_standing_envs == 0.0 and "standing_envs" not in cfg.curriculum
    for name in ("head_pose", "body_pose"):
        assert all(lo <= 0.0 <= hi for lo, hi in cfg.commands[name].ranges)  # the Arena sends zeros
    actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    privileged = {"winch_pull", "pit_margin", "heading_error"}
    assert privileged <= set(critic) and not privileged & set(actor)


def test_no_env_is_sent_forward(cfg):
    """mjlab's template sends a fifth of the envs forward (vx made |vx| ≥ 0.3): the Arena never does."""
    assert cfg.commands["twist"].rel_forward_envs == 0.0


def test_the_start_is_the_arenas(cfg):
    pose, west = cfg.events["reset_base"].params["pose_range"], arena_course.tug_layout(EVENT.course, 1)[0]
    assert sum(pose["x"]) / 2 == pytest.approx(west.x) and sum(pose["y"]) / 2 == pytest.approx(west.y)
    assert pose["z"][0] <= arena_course.SPAWN_HEIGHT_M <= pose["z"][1]
    yaw = np.radians(EVENT.start.yaw_deg)
    assert pose["yaw"][0] <= -yaw and yaw <= pose["yaw"][1] <= 0.1  # not the template's ±π
    joints = cfg.events["reset_robot_joints"].params["position_range"]
    assert joints[0] <= -EVENT.start.joint_rad and EVENT.start.joint_rad <= joints[1]
    assert cfg.episode_length_s == pytest.approx(EVENT.start.settle_s + EVENT.time_limit_s)


def test_the_curriculum_pulls_past_the_benchs_top(cfg):
    last = cfg.curriculum["winch_top"].params["top_stages"][-1]["top_n"]
    assert last[0] < tug.BENCH_TOP_N < last[1]


def test_the_winch_top_widens_on_training_iterations(cfg):
    """winch_top_curriculum finds the command term named `winch`, and its stages count env steps:
    num_steps_per_env per training iteration."""
    assert isinstance(cfg.commands["winch"], tug.WinchCommandCfg)
    stages = cfg.curriculum["winch_top"].params["top_stages"]
    assert [s["step"] for s in stages] == [i * tug.RL_CFG.num_steps_per_env for i in (0, 500, 1000, 1500)]


def test_the_pit_and_a_fall_end_an_episode_early(cfg):
    assert set(cfg.terminations) == {"time_out", "nan_state", "in_pit", "fell"}
    assert cfg.terminations["in_pit"].params["minimum_height"] == -arena_tug.PIT_DROP_M
    assert cfg.terminations["fell"].func is rewards.ArenaFall
    assert cfg.terminations["fell"].params == {"fall_gravity_z": EVENT.termination.fall_gravity_z,
                                               "hold_steps": round(EVENT.termination.fall_hold_s / 0.02)}
    assert cfg.rewards["out"].weight == -50.0


def test_a_fall_is_the_arenas_rule(cfg):
    """Gravity z above fall_gravity_z for the hold's straight control steps, as arena.stage.fell_at counts it."""
    params = cfg.terminations["fell"].params
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])   # env 0 upright, env 1 on its side
    env = SimpleNamespace(num_envs=2, device="cpu",
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))})
    term = rewards.ArenaFall(cfg.terminations["fell"], env)
    seen = [term(env, **params).tolist() for _ in range(params["hold_steps"])]
    assert seen[-2:] == [[False, False], [False, True]]   # out on the step that completes the hold
    term.reset(torch.tensor([1]))
    assert term(env, **params).tolist() == [False, False]


def test_hold_pays_the_pull_only_on_its_feet():
    """The winch's pull while upright and not out; nothing once tipped past the fall rule's line, or out."""
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))},
                          command_manager=SimpleNamespace(get_command={"winch": torch.full((3, 1), 0.4)}.get),
                          termination_manager=SimpleNamespace(terminated=torch.tensor([False, False, True])))
    assert rewards.hold(env, tug.FALL_GRAVITY_Z).tolist() == pytest.approx([0.4, 0.0, 0.0])


def test_the_string_end_is_where_the_arena_judges_it():
    pos, quat = torch.tensor([[-0.53, -0.01, 0.11]]), torch.tensor([[0.97, 0.05, -0.2, 0.1]])
    quat = quat / quat.norm()
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos, root_link_quat_w=quat))})
    margin = rewards.pit_margin(env, tug.STRING_OFFSET, -tug.PIT_M / 2)
    frame = np.zeros((1, 1, 21))
    frame[0, 0, :3], frame[0, 0, 3:7] = pos[0].numpy(), quat[0].numpy()
    end = arena_tug.string_ends(frame, arena_course.string_offset("legs"))[0, 0]
    assert margin.item() == pytest.approx(-tug.PIT_M / 2 - end[0], abs=1e-6)


def test_the_manifest_says_what_the_policy_reads():
    assert "-0.30" in ch.for_task("Mjlab-TugOfWar-MicroDuck").command["twist"]


@pytest.fixture(scope="module")
def live():
    cfg = tug.make_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def test_the_env_pulls_on_time(live):
    term, winch = live.command_manager.get_term("winch"), live.sim.mj_model.actuator(tug.WINCH).id
    for _ in range(40):
        obs, *_ = live.step(torch.zeros(2, 14))
        t = live.episode_length_buf.double() * live.step_dt - tug.SETTLE_S
        assert torch.allclose(live.sim.data.ctrl[:, winch], tug.bench_force(term.profile.double(), t).float(), atol=1e-5)
    assert obs["actor"].shape == (2, 61) and torch.isfinite(obs["critic"]).all()


def test_a_trunk_in_the_pit_ends_the_episode(live):
    robot = live.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[0, 2] = -0.2
    robot.write_root_link_pose_to_sim(pose)
    live.sim.forward()
    _, _, terminated, _, _ = live.step(torch.zeros(2, 14))
    assert terminated.tolist() == [True, False]
