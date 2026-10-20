"""The Tug of War's own terms: how far the string end is from the pit, the pull held on its feet,
and the Arena's fall rule as a termination.

The string end is where the Arena ties the string, the base of the neck: the trunk's position
plus the offset turned by the trunk's quaternion, as `arena.tug.string_ends` reads a trace.
"""

import torch
from mjlab.managers import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply


def pit_margin(env, offset: tuple, pit_edge_x: float) -> torch.Tensor:
    """(n, 1): how far west of the pit's edge the string end is, in m; negative once over the pit."""
    data = env.scene["robot"].data
    pos, quat = data.root_link_pos_w, data.root_link_quat_w
    end = pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos))
    return (pit_edge_x - end[:, 0]).unsqueeze(-1)


def pit_margin_reward(env, offset: tuple, pit_edge_x: float, scale: float) -> torch.Tensor:
    """(n,): tanh(max(margin, 0) / scale), rising with the ground kept between the string and the pit."""
    return torch.tanh(pit_margin(env, offset, pit_edge_x).squeeze(-1).clamp(min=0.0) / scale)


def hold(env, fall_gravity_z: float) -> torch.Tensor:
    """(n,): the winch's pull over the bench's top while the duck is upright (gravity z at or
    below the fall rule's line) and not out: a duck earns by pulling on its feet."""
    upright = env.scene["robot"].data.projected_gravity_b[:, 2] <= fall_gravity_z
    pulling = upright & ~env.termination_manager.terminated
    return env.command_manager.get_command("winch")[:, 0] * pulling


class ArenaFall(ManagerTermBase):
    """The Arena's fall rule (arena.stage.fell_at, runner.run_once) as a termination: projected
    gravity's z above `fall_gravity_z` for `hold_steps` straight control steps, out on the step
    that completes the hold. mjlab builds it with (cfg, env) and resets each env's count."""

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
