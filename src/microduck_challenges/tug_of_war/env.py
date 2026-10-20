"""The Tug of War's training environment: the Arena's pulling bench.

The duck trains on the Arena's tug: a lane cut by a pit, the west duck's start, a string tied
at the base of its neck and run to a winch that pulls it towards the pit. The HF job installs
no Arena, so the event's numbers and the course's arithmetic (arena/course.py) are copied here;
tests/test_tug_of_war.py pins each copy against the Arena's own model. The winch pulls, each
episode, a profile drawn around the bench's, on the Arena's clock (`WinchCommand`).
"""

import math
from dataclasses import dataclass

import mujoco
import torch
from mjlab.managers import CommandTerm, CommandTermCfg

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


def bench_force(profile: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The winch's pull in newtons per row: arena.tug.bench_force on `profile`, (n, 2 + 3k) as
    ramp, top, then (amp, hz, phase) per jerk, at `t`, (n,) seconds into the timed run; 0 before
    it starts, through the settle."""
    amp, hz, phase = profile[:, 2:].unflatten(1, (-1, 3)).unbind(-1)
    pull = profile[:, 0] * t + (amp * torch.sin(2 * math.pi * hz * t[:, None] + phase)).sum(1)
    return torch.where(t < 0, 0.0, torch.minimum(profile[:, 1], pull.clamp(min=0.0)))


@dataclass(kw_only=True)
class WinchCommandCfg(CommandTermCfg):
    """The ranges each episode's profile is drawn from: they hold the bench's ramp, jerks and
    frequencies, and pull harder, so the policy meets the bench's pull and learns to lose late."""

    resampling_time_range: tuple[float, float] = (1e9, 1e9)   # one profile per episode, drawn at reset
    ramp_n_per_s: tuple[float, float] = (0.5 * BENCH_RAMP_N_PER_S, 3.0 * BENCH_RAMP_N_PER_S)
    top_n: tuple[float, float] = (0.5, 2.0)                    # widened by the winch_top curriculum
    jerk_n: tuple[float, float] = (0.0, 2.0 * BENCH_JERK_N)
    jerk_hz: tuple[float, ...] = BENCH_JERK_HZ
    hz_scale: tuple[float, float] = (0.7, 1.3)
    settle_s: float = SETTLE_S

    def build(self, env) -> "WinchCommand":
        return WinchCommand(self, env)


class WinchCommand(CommandTerm):
    """The winch: draws the episode's profile at reset and, before every control step, writes
    the pull the Arena's bench applies at that time to the winch's ctrl (0 through the settle).
    The winch is no robot actuator, so mjlab never writes its ctrl: the pull holds through the
    step's physics, as the Arena sets it before each timed step."""

    cfg: WinchCommandCfg

    def __init__(self, cfg: WinchCommandCfg, env):
        super().__init__(cfg, env)
        self._winch = env.sim.mj_model.actuator(WINCH).id
        # float64, as the Arena computes the pull: a float32 step · 0.02 − 0.5 is off by up to 1.6 µs.
        self.profile = torch.zeros(self.num_envs, 2 + 3 * len(cfg.jerk_hz), dtype=torch.float64, device=self.device)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 1): the pull now over the bench's top."""
        return self._command

    def _update_metrics(self) -> None:
        pass  # nothing to log; mjlab requires the method

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        c, k = self.cfg, len(self.cfg.jerk_hz)

        def draw(range_: tuple[float, float], width: int) -> torch.Tensor:
            return self.profile.new_empty(len(env_ids), width).uniform_(*range_)

        hz = self.profile.new_tensor(c.jerk_hz) * draw(c.hz_scale, k)
        jerks = torch.stack([draw(c.jerk_n, k), hz, draw((0.0, 2 * math.pi), k)], dim=-1).flatten(1)
        self.profile[env_ids] = torch.cat([draw(c.ramp_n_per_s, 1), draw(c.top_n, 1), jerks], dim=1)

    def _update_command(self) -> None:
        t = self._env.episode_length_buf.to(self.profile.dtype) * self._env.step_dt - self.cfg.settle_s
        pull = bench_force(self.profile, t)
        self._env.sim.data.ctrl[:, self._winch] = pull
        self._command[:, 0] = pull / BENCH_TOP_N


def winch_top_curriculum(env, env_ids, top_stages: list[dict]) -> dict:
    """Widens the winch's top: `top_stages` is a list of `{"step", "top_n": (lo, hi)}`; the last
    stage whose step the training has reached sets the range the command term `winch` draws its
    next profiles from (microduck_mdp.reward_weight's stage pattern)."""
    del env_ids
    cfg = env.command_manager.get_term("winch").cfg
    for stage in top_stages:
        if stage["step"] <= env.common_step_counter:
            cfg.top_n = stage["top_n"]
    return {"top_n_low": cfg.top_n[0], "top_n_high": cfg.top_n[1]}
