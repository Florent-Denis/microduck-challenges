"""The Tug of War's own terms: where the learner's string end and the marker are, the new ground the
marker wins towards the learner, the opponent as the critic reads it, which end of a bout an episode
met, and the Arena's fall rule as a training-only termination.

The string is the Arena's rope (env.add_string): its points are the sites p0 (the west end, tied to
the learner) to p16 (the east end, tied to the opponent), where `arena.tug` records them. The marker
is the middle point, p{MIDDLE}, as `arena.tug.marker_x` reads it: the Arena rates a bout by it. A
string end, as pit_margin reads it, is where the Arena ties the string, the base of the neck: the
trunk's position plus the offset turned by the trunk's quaternion.
"""

import torch
from mjlab.managers import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply

STRING = "string/"  # the string's names, as arena.course.STRING
MIDDLE = 8  # the point the marker is, as arena.tug.MIDDLE


def _string_end(data, offset: tuple) -> torch.Tensor:
    """(n, 3): the string end in the world."""
    pos, quat = data.root_link_pos_w, data.root_link_quat_w
    return pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos))


def _point(env, k: int) -> torch.Tensor:
    """(n, 3): the string's point p{k} in the world."""
    return env.sim.data.site_xpos[:, env.sim.mj_model.site(f"{STRING}p{k}").id]


def pit_margin(env, offset: tuple, pit_edge_x: float) -> torch.Tensor:
    """(n, 1): how far west of the pit's edge the string end is, in m; negative once over the pit."""
    return (pit_edge_x - _string_end(env.scene["robot"].data, offset)[:, 0]).unsqueeze(-1)


def pit_margin_reward(env, offset: tuple, pit_edge_x: float, scale: float) -> torch.Tensor:
    """(n,): tanh(max(margin, 0) / scale), rising with the ground kept between the string and the pit."""
    return torch.tanh(pit_margin(env, offset, pit_edge_x).squeeze(-1).clamp(min=0.0) / scale)


def marker_x(env) -> torch.Tensor:
    """(n, 1): the marker's x, the string's middle point, in m; negative on the learner's side."""
    return _point(env, MIDDLE)[:, :1]


def string_length(env) -> torch.Tensor:
    """(n, 1): the distance between the string's two ends, in m: short of its 1 m, it sags."""
    return (_point(env, 2 * MIDDLE) - _point(env, 0)).norm(dim=-1, keepdim=True)


def opponent_state(env) -> torch.Tensor:
    """(n, 12): the opponent's trunk, for the critic: its position from the learner's, its projected
    gravity, its world linear velocity and its own-frame angular velocity."""
    learner, opponent = env.scene["robot"].data, env.scene["opponent"].data
    return torch.cat([opponent.root_link_pos_w - learner.root_link_pos_w, opponent.projected_gravity_b,
                      opponent.root_link_lin_vel_w, opponent.root_link_ang_vel_b], dim=-1)


class MarkerGround(ManagerTermBase):
    """(n,): the new ground the marker wins towards the learner (west, -x) each control step:
    tanh(max(best - x, 0) / (scale * step_dt)), `best` the westmost x it reached this episode, then
    best = min(best, x). A marker won at a steady `scale` m/s pays tanh(1) a step; a sway with no
    net gain pays only its first new ground, and being dragged east pays nothing (pit_margin_reward
    prices that). The record starts on an episode's first step: mjlab resets reward terms before it
    refreshes the kinematics, so the reset pose cannot be read in `reset`."""

    def __init__(self, cfg, env):
        del cfg  # the scale comes with each call
        super().__init__(env)
        self._best = torch.full((env.num_envs,), float("nan"), device=env.device)

    def __call__(self, env, scale: float) -> torch.Tensor:
        x = marker_x(env).squeeze(-1)
        best = torch.where(self._best.isnan(), x, self._best)
        self._best = torch.minimum(best, x)
        return torch.tanh((best - x).clamp(min=0.0) / (scale * env.step_dt))

    def reset(self, env_ids) -> None:
        self._best[env_ids] = float("nan")


def ended_by(env, terms: tuple[str, ...]) -> torch.Tensor:
    """(n,): 1.0 where one of the named terminations fired this step, else 0.0."""
    fired = torch.stack([env.termination_manager.get_term(name) for name in terms]).any(dim=0)
    return fired.float()


class ArenaFall(ManagerTermBase):
    """The Arena's fall rule (arena.stage.fell_at) as a termination, TRAINING ONLY: projected
    gravity's z above `fall_gravity_z` for `hold_steps` straight control steps, out on the step
    that completes the hold. A fall ends a training episode so the duck keeps learning to stand and
    pull; the Arena judges only the pit. mjlab builds it with (cfg, env) and resets each env's count."""

    def __init__(self, cfg, env):
        del cfg  # the rule's numbers come with each call
        super().__init__(env)
        self._tipped_for = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, fall_gravity_z: float, hold_steps: int) -> torch.Tensor:
        tipped = env.scene["robot"].data.projected_gravity_b[:, 2] > fall_gravity_z
        self._tipped_for.add_(1).mul_(tipped)
        return self._tipped_for >= hold_steps

    def reset(self, env_ids) -> None:
        self._tipped_for[env_ids] = 0
