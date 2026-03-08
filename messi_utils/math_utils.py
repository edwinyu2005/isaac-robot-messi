"""
High-performance batched math utilities for coordinate transformations.
Optimized for PyTorch tensors on GPU.
"""

import torch


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector v by the inverse of quaternion q (world to local).
    q: [N, 4] (w, x, y, z)
    v: [N, 3] (x, y, z)
    """
    q_w = q[:, 0]
    q_vec = q[:, 1:]

    # v_rotated = v + 2 * cross(q_vec, cross(q_vec, v) - q_w * v)
    # For inverse rotation (world -> local), we use the conjugate of q
    a = torch.cross(q_vec, v, dim=-1) - q_w.unsqueeze(-1) * v
    b = torch.cross(q_vec, a, dim=-1)

    return v + 2.0 * b


@torch.jit.script
def get_relative_pos(pos_a: torch.Tensor, pos_b: torch.Tensor, quat_a: torch.Tensor) -> torch.Tensor:
    """
    Calculate the position of B relative to A's local frame.
    pos_a/pos_b: [N, 3]
    quat_a: [N, 4] (A's rotation in world frame)
    """
    # 1. Get world-space displacement vector
    displacement = pos_b - pos_a

    # 2. Rotate displacement into A's local frame
    return quat_rotate_inverse(quat_a, displacement)
