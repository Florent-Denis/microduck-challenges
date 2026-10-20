"""The Tug of War's own terms: how far the string end is from the pit, how fast it backs away
from it, and the pull held.

The string end is where the Arena ties the string, the base of the neck: the trunk's position
plus the offset turned by the trunk's quaternion, as `arena.tug.string_ends` reads a trace.
"""

import torch
from mjlab.utils.lab_api.math import quat_apply


def _arm(data, offset: tuple) -> torch.Tensor:
    """(n, 3): from the trunk's origin to the string end, in the world."""
    quat = data.root_link_quat_w
    return quat_apply(quat, quat.new_tensor(offset).expand(len(quat), 3))


def pit_margin(env, offset: tuple, pit_edge_x: float) -> torch.Tensor:
    """(n, 1): how far west of the pit's edge the string end is, in m; negative once over the pit."""
    data = env.scene["robot"].data
    end = data.root_link_pos_w + _arm(data, offset)
    return (pit_edge_x - end[:, 0]).unsqueeze(-1)


def pit_margin_reward(env, offset: tuple, pit_edge_x: float, scale: float) -> torch.Tensor:
    """(n,): tanh(max(margin, 0) / scale), rising with the ground kept between the string and the pit."""
    return torch.tanh(pit_margin(env, offset, pit_edge_x).squeeze(-1).clamp(min=0.0) / scale)


def retreat(env, offset: tuple, scale: float) -> torch.Tensor:
    """(n,): tanh(max(-v_x, 0) / scale), v_x the string end's x velocity in the world (the trunk
    origin's plus its spin about it): the west duck backing away from the pit, along -x."""
    data = env.scene["robot"].data
    v = data.root_link_lin_vel_w + torch.cross(data.root_link_ang_vel_w, _arm(data, offset), dim=-1)
    return torch.tanh((-v[:, 0]).clamp(min=0.0) / scale)


def hold(env) -> torch.Tensor:
    """(n,): the winch's pull over the bench's top while the duck is not out, whatever its posture."""
    return env.command_manager.get_command("winch")[:, 0] * ~env.termination_manager.terminated
