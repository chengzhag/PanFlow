import torch
import numpy as np
from einops import einsum, rearrange, repeat
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def unproject_erp(
    coordinates: torch.Tensor,
    depth: torch.Tensor | float | None = None,
):
    # Apply the inverse intrinsics to the coordinates.
    phi_theta = coordinates * coordinates.new_tensor([2 * np.pi, np.pi]) - coordinates.new_tensor([np.pi, np.pi / 2])
    ray_directions = torch.stack([
        torch.cos(phi_theta[..., 1]) * torch.sin(phi_theta[..., 0]),
        torch.sin(phi_theta[..., 1]),
        torch.cos(phi_theta[..., 1]) * torch.cos(phi_theta[..., 0]),
    ], dim=-1)

    # Apply the supplied depth values.
    if depth is None:
        depth = 1.
    if isinstance(depth, torch.Tensor):
        depth = depth[..., None]
    return ray_directions * depth


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,  # float coordinates (xy indexing)
    torch.Tensor,  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def project_erp(
    ray_directions: torch.Tensor,
) -> torch.Tensor:
    dirs = ray_directions / torch.linalg.norm(
        ray_directions, dim=-1, keepdim=True).clamp(min=1e-8)
    x, y, z = dirs.unbind(dim=-1)
    phi   = torch.atan2(x, z)        # atan2(sinφ·cosθ, cosφ·cosθ) → φ
    theta = torch.asin(y)            # asin(sinθ) → θ
    u = (phi + np.pi) / (2 * np.pi)
    v = (theta + np.pi/2) / np.pi
    return torch.stack((u, v), dim=-1)


def homogenize_points(
    vectors: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([vectors, torch.ones_like(vectors[..., :1])], dim=-1)


def transformation_to_flow(
    M: torch.Tensor,
    shape: tuple[int, int],
    depth: torch.Tensor | float | None = None,
) -> torch.Tensor:
    origins, _ = sample_image_grid(shape)
    rays = unproject_erp(origins, depth=depth)
    rays = homogenize_points(rays)
    rays = einsum(M, rays, "n i j, h w j -> n h w i")
    rays = rays[..., :3]
    destinations = project_erp(rays)
    flow = destinations - origins
    flow[..., 0] = (flow[..., 0] + 0.5) % 1. - 0.5
    flow = flow * flow.new_tensor(shape[::-1])
    flow = rearrange(flow, "n h w i -> n i h w")
    return flow


def equilib_rotation(
    rotations: torch.Tensor,
):
    rotations = rotations.numpy()

    # Calculate the conversion matrix M to transform from the source coordinate system 
    # (x-right, y-down, z-forward) to the equi2equi required right-handed coordinate system 
    # (x = forward, y = left, z = up)
    M = np.array([
        [0,  0,  1],   # Target x axis = source z axis (forward)
        [-1, 0,  0],   # Target y axis = - source x axis (left)
        [0, -1,  0]    # Target z axis = - source y axis (up)
    ])
    rotations = M[None] @ rotations @ (M.T)[None]

    rots = []
    for rotation in rotations:
        r_obj = R.from_matrix(rotation)
        roll, pitch, yaw = r_obj.as_euler('xyz', degrees=False)
        rot = {
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        }
        rots.append(rot)

    return rots
