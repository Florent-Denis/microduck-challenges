"""The Tug of War trains in the Arena's tug: the same course, string end, start and command."""
import hashlib, itertools, math, subprocess, sys, tomllib
from pathlib import Path
from types import SimpleNamespace
import mujoco, numpy as np, onnx, onnxruntime, pytest, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab_microduck import challenge as ch
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

import microduck_challenges
from arena import course as arena_course, runner as arena_runner, tug as arena_tug
from arena.assets import ASSET_ROOT
from arena.constants import ACTION_SCALE, DEFAULT_POSE, JOINT_NAMES
from arena.manifest import load_event
from microduck_challenges.tug_of_war import env as tug, opponent, rewards

EVENT = load_event("tug-of-war")
T7_250_SHA256 = "0c4e8b0abc9223c51bc4a32bc78591a4dc72c92ff873b6de7cd894bfad4bc1a2"
CONTACT = ("geom_type", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
           "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap")
TENDON = ("tendon_limited", "tendon_range", "tendon_solref_lim", "tendon_solimp_lim", "tendon_margin",
          "tendon_stiffness", "tendon_damping", "tendon_frictionloss")


@pytest.fixture(scope="module")
def arena_model():
    return arena_course.compile_tug(EVENT)


@pytest.fixture(scope="module")
def training_model():
    scene = SceneCfg(num_envs=1, entities={"robot": MICRODUCK_STANDUP_ROBOT_CFG, "opponent": MICRODUCK_STANDUP_ROBOT_CFG},
                     spec_fn=tug.tug_scene)
    return Scene(scene, device="cpu").compile()


def test_the_training_never_imports_the_arena():
    """HF Jobs install the default groups only: no Arena there. The package's __init__ imports every tasks.py."""
    code = ("import sys; sys.modules['arena'] = None; "
            "import microduck_challenges.tug_of_war.env, microduck_challenges.tug_of_war.opponent")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_constants_are_the_arenas():
    c, s = EVENT.course, EVENT.start
    assert (tug.LENGTH_M, tug.WIDTH_M, tug.WALL_HEIGHT_M, tug.PIT_M, tug.PIT_DEPTH_M, tug.STRING_M) == (
        c.length_m, c.width_m, c.wall_height_m, c.pit_m, c.pit_depth_m, c.string_m)
    assert tuple(EVENT.command) == tug.COMMAND == (tug.COMMAND_VX,) + (0.0,) * 12
    assert (tug.BOUT_S, tug.SETTLE_S, tug.JOINT_PERTURBATION_RAD, tug.YAW_PERTURBATION_DEG) == (
        EVENT.bout.time_limit_s, s.settle_s, s.joint_rad, s.yaw_deg)
    # The Arena's fall rule, summaries only there: the training ends an episode on it.
    assert (tug.FALL_GRAVITY_Z, tug.FALL_HOLD_S) == (EVENT.termination.fall_gravity_z, EVENT.termination.fall_hold_s)
    assert tug.STRING_OFFSET == pytest.approx(arena_course.string_offset("legs"), abs=1e-12)
    assert (tug.WALL_THICKNESS_M, tug.PLATFORM_OVERRUN_M, tug.PIT_DROP_M) == (
        arena_course.WALL_THICKNESS_M, arena_course.PLATFORM_OVERRUN_M, arena_tug.PIT_DROP_M)
    assert tug.STRING_SITE == arena_course.STRING_SITE


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
    privileged = {"pit_margin", "heading_error"}
    assert privileged <= set(critic) and not privileged & set(actor)


def test_no_env_is_sent_forward(cfg):
    """mjlab's template sends a fifth of the envs forward (vx made |vx| ≥ 0.3): the Arena never does."""
    assert cfg.commands["twist"].rel_forward_envs == 0.0


def test_the_start_is_the_arenas(cfg):
    pose, west = cfg.events["reset_base"].params["pose_range"], arena_course.tug_layout(EVENT.course)[0]
    assert pose["x"][0] == pytest.approx(west.x) and sum(pose["y"]) / 2 == pytest.approx(west.y)
    assert pose["z"][0] <= arena_course.SPAWN_HEIGHT_M <= pose["z"][1]
    yaw = np.radians(EVENT.start.yaw_deg)
    assert pose["yaw"][0] <= -yaw and yaw <= pose["yaw"][1] <= 0.1  # not the template's ±π
    joints = cfg.events["reset_robot_joints"].params["position_range"]
    assert joints[0] <= -EVENT.start.joint_rad and EVENT.start.joint_rad <= joints[1]
    assert cfg.episode_length_s == pytest.approx(tug.SETTLE_S + EVENT.bout.time_limit_s)


def test_backing_away_pays_far_more_than_standing_still(cfg):
    """t1–t5 never discovered the walk back: standing still kept 0.41 of track_linear_velocity's 1.0.
    Walk backwards defaults to 3.0 at std √0.05: at the event's -0.30 m/s the duck earns 3.0, standing
    still 3·e^-1.8 = 0.50."""
    knob = tomllib.loads((Path(tug.__file__).parent / "challenge.toml").read_text())["params"]["back_reward"]
    assert knob["value"] == 3.0 and knob["min"] <= 3.0 <= knob["max"]
    term = cfg.rewards["track_linear_velocity"]
    assert term.weight == 3.0 and term.params["std"] == pytest.approx(math.sqrt(0.05))

    def paid(vx):
        data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]))
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                              command_manager=SimpleNamespace(get_command={"twist": torch.tensor([[tug.COMMAND_VX, 0.0, 0.0]])}.get))
        return term.weight * term.func(env, **term.params).item()

    assert paid(tug.COMMAND_VX) == pytest.approx(3.0)
    assert paid(0.0) == pytest.approx(3.0 * math.exp(-1.8), abs=1e-6)   # 0.50, a sixth of walking back


def test_the_pit_and_a_training_fall_end_an_episode(cfg):
    """The pit, as on the Arena; and a fall, training only (the Arena judges only the pit), so the duck
    keeps learning to stand and pull instead of lying on its side for the rest of the episode."""
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


def _retreat_run(cfg, path, reset_at=None):
    """What retreat pays each step for string ends at `path[k]` (x per env, m), the trunk upright,
    the env reset just before step `reset_at`."""
    params = cfg.rewards["retreat"].params
    data = SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(path[0])))
    env = SimpleNamespace(num_envs=len(path[0]), device="cpu", step_dt=0.02, scene={"robot": SimpleNamespace(data=data)})
    term = rewards.Retreat(cfg.rewards["retreat"], env)
    paid = []
    for k, xs in enumerate(path):
        if k == reset_at:
            term.reset(torch.tensor([0]))
        data.root_link_pos_w = torch.tensor([[x - tug.STRING_OFFSET[0], 0.0, 0.11] for x in xs])
        paid.append(term(env, **params).tolist())
    return paid


def test_retreat_pays_new_ground(cfg):
    """Each step, the ground the string end gains west of the westmost point it reached this episode,
    at the scale's pace (0.2 m/s) tanh(1): a steady walk back pays it every step, being dragged east
    nothing, a sway with no net gain only its first new ground."""
    assert cfg.rewards["retreat"].func is rewards.Retreat and cfg.rewards["retreat"].weight == 1.0
    assert cfg.rewards["retreat"].params == {"offset": tug.STRING_OFFSET, "scale": 0.2}
    step = 0.2 * 0.02   # 0.2 m/s over a control step
    sway = [-0.5, -0.5 - step, -0.5 + step] * 4
    paid = _retreat_run(cfg, [(-0.5 - k * step, -0.5 + k * step, sway[k]) for k in range(12)])
    walk, dragged, swayed = zip(*paid)
    assert walk[0] == 0.0 and walk[1:] == pytest.approx([math.tanh(1.0)] * 11, abs=1e-4)
    assert dragged == (0.0,) * 12
    assert swayed[1] == pytest.approx(math.tanh(1.0), abs=1e-4) and swayed[:1] + swayed[2:] == (0.0,) * 11


def test_a_reset_clears_the_retreat_record(cfg):
    """Back at the start after a reset, the ground the last episode won is new again."""
    step = 0.2 * 0.02
    paid = _retreat_run(cfg, [(-0.5 - k * step,) for k in range(10)] + [(-0.5 - k * step,) for k in range(3)], reset_at=10)
    assert [p[0] for p in paid[10:]] == pytest.approx([0.0, math.tanh(1.0), math.tanh(1.0)], abs=1e-4)


def test_the_recipe_knows_no_winch(cfg, training_model):
    assert "hold" not in cfg.rewards
    assert "winch" not in cfg.commands and "winch_top" not in cfg.curriculum
    assert not {"winch_pull", "winch_top"} & set(cfg.observations["critic"].terms)


def _bout_data(model, prefixes, gap):
    """Both ducks in DEFAULT_POSE at their bout starts, each moved gap / 2 away from the pit, mj_forward."""
    data = mujoco.MjData(model)
    for prefix, start in zip(prefixes, arena_course.tug_layout(EVENT.course)):
        x = start.x + math.copysign(1.0, start.x) * gap / 2
        trunk = model.body(prefix + "trunk_base")
        free = model.jnt_qposadr[trunk.jntadr[0]]
        half = math.radians(start.yaw_deg) / 2
        data.qpos[free:free + 7] = [x, start.y, arena_course.SPAWN_HEIGHT_M, math.cos(half), 0, 0, math.sin(half)]
        for name, angle in zip(JOINT_NAMES, DEFAULT_POSE):
            data.qpos[model.joint(prefix + name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return data


def _ends_apart(west_x, west_y, west_yaw, east_x, east_y, east_yaw):
    """The string ends' distance for trunks at these (x, y, yaw): STRING_OFFSET turned by each yaw."""
    ends = [(x + tug.STRING_OFFSET[0] * math.cos(yaw) - tug.STRING_OFFSET[1] * math.sin(yaw),
             y + tug.STRING_OFFSET[0] * math.sin(yaw) + tug.STRING_OFFSET[1] * math.cos(yaw))
            for x, y, yaw in ((west_x, west_y, west_yaw), (east_x, east_y, east_yaw))]
    return math.dist(*ends)


def test_the_string_is_tied_like_the_arenas(arena_model, training_model):
    a, t = arena_model.tendon(tug.STRING).id, training_model.tendon(tug.STRING).id
    for field in TENDON:
        np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[a], err_msg=field)
    for ours, theirs in (("robot/", ""), ("opponent/", "duck1/")):
        site, arena_site = training_model.site(ours + tug.STRING_SITE), arena_model.site(theirs + tug.STRING_SITE)
        assert training_model.body(site.bodyid[0]).name == ours + "trunk_base"
        np.testing.assert_allclose(site.pos, arena_site.pos, atol=1e-12)
    assert training_model.nu == arena_model.nu == 28   # two ducks' servos, nothing else


@pytest.mark.parametrize("gap, held", [(-0.01, False), (0.01, True)])
def test_the_string_holds_at_its_length(arena_model, training_model, gap, held):
    for model, prefixes in ((arena_model, ("", "duck1/")), (training_model, ("robot/", "opponent/"))):
        data = _bout_data(model, prefixes, gap)
        assert data.ten_length[model.tendon(tug.STRING).id] == pytest.approx(tug.STRING_M + gap, abs=1e-6)
        assert bool((data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON).any()) is held


def test_the_bout_is_the_same_from_either_side():
    boxes = {(tuple(round(v, 9) for v in pos), tuple(size)) for _, pos, size in tug.course_boxes()}
    assert {((round(-x, 9), round(-y, 9), z), size) for (x, y, z), size in boxes} == boxes
    west, east = arena_course.tug_layout(EVENT.course)
    assert (east.x, east.y, east.yaw_deg) == pytest.approx((-west.x, -west.y, 180.0))
    assert tug.east_start() == pytest.approx((east.x, east.y)) and tug.west_start() == pytest.approx((west.x, west.y))


def test_no_start_stretches_the_string(cfg):
    """No corner of the start ranges puts the ends more than 5 mm past the string (the Arena's ±2° alone gives ~1 mm)."""
    west, east = cfg.events["reset_base"].params["pose_range"], cfg.events["reset_opponent_base"].params["pose_range"]
    corners = itertools.product(west["x"], west["y"], west["yaw"], east["x"], east["y"], east["yaw"])
    assert max(_ends_apart(*corner) for corner in corners) <= tug.STRING_M + 0.005   # STRING_OFFSET turned by each yaw
    assert west["x"][0] == pytest.approx(tug.west_start()[0]) and east["x"][1] == pytest.approx(tug.east_start()[0])
    yaw = math.radians(EVENT.start.yaw_deg)
    assert east["yaw"][0] <= math.pi - yaw and math.pi + yaw <= east["yaw"][1]


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


def test_the_env_runs_a_bout_on_the_actors_61_numbers(live):
    for _ in range(10):
        obs, *_ = live.step(torch.zeros(2, 14))
    assert obs["actor"].shape == (2, 61) and torch.isfinite(obs["critic"]).all()


def test_a_trunk_in_the_pit_ends_the_episode(live):
    robot = live.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[0, 2] = -0.2
    robot.write_root_link_pose_to_sim(pose)
    live.sim.forward()
    _, _, terminated, _, _ = live.step(torch.zeros(2, 14))
    assert terminated.tolist() == [True, False]


def _relu_actor(tmp_path):
    """An actor the loader does not know: Sub, Div, Gemm, Relu, Gemm, 61 → 14."""
    rng, helper = np.random.default_rng(0), onnx.helper
    shapes = {"mean": (1, 61), "std": (1, 61), "w0": (32, 61), "b0": (32,), "w1": (14, 32), "b1": (14,)}
    weights = [onnx.numpy_helper.from_array(rng.normal(size=shape).astype(np.float32), name) for name, shape in shapes.items()]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["centred"]), helper.make_node("Div", ["centred", "std"], ["scaled"]),
             helper.make_node("Gemm", ["scaled", "w0", "b0"], ["hidden"], transB=1), helper.make_node("Relu", ["hidden"], ["relu"]),
             helper.make_node("Gemm", ["relu", "w1", "b1"], ["actions"], transB=1)]
    graph = helper.make_graph(nodes, "actor", [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 61])],
                              [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 14])], weights)
    path = tmp_path / "relu.onnx"
    onnx.save(helper.make_model(graph), path)
    return path


def test_the_opponent_net_is_its_onnx():
    obs = np.random.default_rng(0).normal(0.0, 3.0, (1000, 61)).astype(np.float32)
    session = onnxruntime.InferenceSession(str(opponent.REFERENCE), providers=["CPUExecutionProvider"])
    theirs = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs])
    with torch.no_grad():
        ours = opponent.load_actor(opponent.REFERENCE)(torch.from_numpy(obs)).numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


def test_a_graph_the_loader_does_not_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Relu"):
        opponent.load_actor(_relu_actor(tmp_path))


def test_the_reference_ships_with_the_challenge():
    assert hashlib.sha256(opponent.REFERENCE.read_bytes()).hexdigest() == T7_250_SHA256
    root = Path(microduck_challenges.__file__).parents[2]
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.split()
    assert "src/microduck_challenges/tug_of_war/reference.onnx" in listed   # hf_jobs.py tars exactly this list


def test_the_opponent_acts_on_the_arenas_joints(live):
    term, duck = live.action_manager.get_term("opponent"), live.scene["opponent"]
    assert term.target_names == list(JOINT_NAMES) and live.action_manager.total_action_dim == 14
    home = duck.data.default_joint_pos[:, term.target_ids]
    # mjlab_microduck's HOME_FRAME is rounded to 4 decimals: the learner carries the same gap.
    assert home[0].tolist() == pytest.approx(list(DEFAULT_POSE), abs=5e-5)
    act = torch.linspace(-0.2, 0.2, 14).expand(2, 14)
    policy, term.policy = term.policy, lambda obs: act
    try:
        term.process_actions(torch.zeros(2, 0))
        term.apply_actions()
    finally:
        term.policy = policy
    expected = home + act * ACTION_SCALE   # runner._Duck.act, on the robot's own default
    assert torch.allclose(duck.data.joint_pos_target[:, term.target_ids], expected, atol=1e-6)
    assert torch.equal(term.last_action, act)   # the policy's raw output, as runner._Duck keeps it


def _duck_addresses(model, prefix):
    """The duck's qpos and qvel addresses: its trunk's free joint, then its joints in JOINT_NAMES order."""
    trunk, joints = model.body(prefix + "trunk_base").jntadr[0], [model.joint(prefix + name) for name in JOINT_NAMES]
    qpos = [*range(model.jnt_qposadr[trunk], model.jnt_qposadr[trunk] + 7), *(j.qposadr[0] for j in joints)]
    qvel = [*range(model.jnt_dofadr[trunk], model.jnt_dofadr[trunk] + 6), *(j.dofadr[0] for j in joints)]
    return qpos, qvel


def test_the_opponent_observes_as_the_arena_does(live, arena_model, monkeypatch):
    """Env 0's opponent pitched 20°, yaw 170°, joints ± U(0.2) off home, velocities U(±2), forward; known memory.
    MjData(arena_model) gets env 0's raw qpos/qvel: trunk free joint by body, joints "opponent/x" → "duck1/x".
    `_observe` is fed the robot's own default pose: the same formula (the HOME_FRAME gap is pinned above)."""
    rng, term = np.random.default_rng(1), live.action_manager.get_term("opponent")
    home = live.scene["opponent"].data.default_joint_pos[0, term.target_ids].cpu().numpy().astype(np.float64)
    monkeypatch.setattr(arena_runner, "DEFAULT_POSE", home)
    (qpos, qvel), (their_qpos, their_qvel) = _duck_addresses(live.sim.mj_model, "opponent/"), _duck_addresses(arena_model, "duck1/")
    state = live.sim.data.qpos[0, qpos].cpu().numpy().astype(np.float64)
    mujoco.mju_mulQuat(state[3:7], [math.cos(math.radians(85.0)), 0.0, 0.0, math.sin(math.radians(85.0))],
                       [math.cos(math.radians(10.0)), 0.0, math.sin(math.radians(10.0)), 0.0])
    state[7:] = DEFAULT_POSE + rng.uniform(-0.2, 0.2, 14)
    live.sim.data.qpos[0, qpos] = torch.tensor(state, dtype=torch.float32)
    live.sim.data.qvel[0, qvel] = torch.tensor(rng.uniform(-2.0, 2.0, 20), dtype=torch.float32)
    live.sim.forward()
    last_action, prev_joint_vel = (rng.uniform(-1.0, 1.0, 14).astype(np.float32) for _ in range(2))
    term.last_action[0], term.prev_joint_vel[0] = torch.from_numpy(last_action), torch.from_numpy(prev_joint_vel)

    data = mujoco.MjData(arena_model)
    data.qpos[their_qpos], data.qvel[their_qvel] = live.sim.data.qpos[0, qpos].cpu().numpy(), live.sim.data.qvel[0, qvel].cpu().numpy()
    mujoco.mj_forward(arena_model, data)
    adr = arena_runner._Addresses.of(arena_model, EVENT.start.keyframe, "duck1/")
    theirs = arena_runner._observe(arena_model, data, adr, last_action, np.asarray(EVENT.command, dtype=np.float64), prev_joint_vel)
    np.testing.assert_allclose(term.observe()[0].numpy(), theirs, atol=1e-4)


def test_the_opponent_remembers_as_the_arena_does(live):
    term, duck = live.action_manager.get_term("opponent"), live.scene["opponent"]
    live.reset()
    live.step(torch.zeros(2, 14))   # moving: a reset's velocities are zeros
    before = duck.data.joint_vel[:, term.target_ids].clone()
    live.step(torch.zeros(2, 14))
    obs, now = term.observe(), duck.data.joint_vel[:, term.target_ids]
    assert torch.equal(obs[:, 20:34], before) and not torch.equal(obs[:, 20:34], now)   # a step old
    assert torch.equal(obs[:, 34:48], term.last_action) and term.last_action.any()


def test_a_reset_clears_the_opponents_memory(live):
    term = live.action_manager.get_term("opponent")
    live.step(torch.zeros(2, 14))
    live.reset(env_ids=torch.tensor([1]))
    assert not term.last_action[1].any() and not term.prev_joint_vel[1].any() and term.last_action[0].any()


def _bout(seed, loser, how, time_s):
    return arena_tug.BoutOutcome(seed, loser, how, time_s, np.zeros((2, 0, 21)), np.zeros((2, 1, 21)))


def test_the_summary_counts_who_lost_and_how():
    from microduck_challenges.tug_of_war import check
    results = [("west", _bout(1, 0, "pit", 6.2)), ("east", _bout(2, 0, "pit", 3.0)),
               ("west", _bout(3, 1, "marker", 9.0)), ("east", _bout(4, None, "draw", 1.0))]
    assert check.summary(results) == [
        "bout 1 · seed 1 · A west · A loses by pit at 6.20 s",
        "bout 2 · seed 2 · A east · B loses by pit at 3.00 s",
        "bout 3 · seed 3 · A west · B loses by marker at 9.00 s",
        "bout 4 · seed 4 · A east · drawn at 1.00 s",
        "A 2 · B 1 · draws 1 · by pit 2 · by marker 1 · median decision 4.6 s"]


def test_the_check_swaps_sides_on_seeds_one_up(capsys):
    from microduck_challenges.tug_of_war import check
    walker = str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")
    assert check.main([walker, walker, "--bouts", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("bout 1 · seed 1 · A west") and lines[1].startswith("bout 2 · seed 2 · A east")
    assert lines[2].startswith("A ") and "median decision" in lines[2]


def test_an_unreadable_policy_is_reported(tmp_path, capsys):
    from microduck_challenges.tug_of_war import check
    missing = str(tmp_path / "gone.onnx")
    assert check.main([missing, missing]) == 1
    assert missing in capsys.readouterr().err
