"""The Tug of War's own terms: how far the string end is from the pit, the new ground it wins
away from it, and the pull held.

The string end is where the Arena ties the string, the base of the neck: the trunk's position
plus the offset turned by the trunk's quaternion, as `arena.tug.string_ends` reads a trace.
"""

import torch
from mjlab.managers import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply


def _string_end(data, offset: tuple) -> torch.Tensor:
    """(n, 3): the string end in the world."""
    pos, quat = data.root_link_pos_w, data.root_link_quat_w
    return pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos))


def pit_margin(env, offset: tuple, pit_edge_x: float) -> torch.Tensor:
    """(n, 1): how far west of the pit's edge the string end is, in m; negative once over the pit."""
    return (pit_edge_x - _string_end(env.scene["robot"].data, offset)[:, 0]).unsqueeze(-1)


def pit_margin_reward(env, offset: tuple, pit_edge_x: float, scale: float) -> torch.Tensor:
    """(n,): tanh(max(margin, 0) / scale), rising with the ground kept between the string and the pit."""
    return torch.tanh(pit_margin(env, offset, pit_edge_x).squeeze(-1).clamp(min=0.0) / scale)


class Retreat(ManagerTermBase):
    """(n,): the new ground the string end wins away from the pit (west, -x) each control step:
    tanh(max(best - x, 0) / (scale * step_dt)), `best` the westmost x it reached this episode, then
    best = min(best, x). A steady walk back at `scale` m/s pays tanh(1) a step; a sway with no net
    gain pays only its first new ground, and being dragged east pays nothing (pit_margin_reward
    prices that). The record starts on an episode's first step: mjlab resets reward terms before it
    refreshes the kinematics, so the reset pose cannot be read in `reset`."""

    def __init__(self, cfg, env):
        del cfg  # the scale and offset come with each call
        super().__init__(env)
        self._best = torch.full((env.num_envs,), float("nan"), device=env.device)

    def __call__(self, env, offset: tuple, scale: float) -> torch.Tensor:
        x = _string_end(env.scene["robot"].data, offset)[:, 0]
        best = torch.where(self._best.isnan(), x, self._best)
        self._best = torch.minimum(best, x)
        return torch.tanh((best - x).clamp(min=0.0) / (scale * env.step_dt))

    def reset(self, env_ids) -> None:
        self._best[env_ids] = float("nan")


def hold(env) -> torch.Tensor:
    """(n,): the winch's pull over the bench's top while the duck is not out, whatever its posture."""
    return env.command_manager.get_command("winch")[:, 0] * ~env.termination_manager.terminated
