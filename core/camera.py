import torch
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CameraConfig:
    position: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_degrees: float = 45.0
    near: float = 0.1
    far: float = 1000.0
    scale: float = 1.0 # For 2D / orthogonal adjustments

CAMERA_PRESETS = {
    "Three-Quarter-Prospect": CameraConfig(
        position=(15.0, 10.0, 15.0),
        look_at=(0.0, 0.0, 0.0),
        fov_degrees=45.0
    ),
    "Side Profile": CameraConfig(
        position=(20.0, 0.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        fov_degrees=40.0
    ),
    "Macro Detail": CameraConfig(
        position=(5.0, 5.0, 5.0),
        look_at=(2.0, 2.0, 2.0),
        fov_degrees=30.0
    ),
    "Default 2D": CameraConfig(
        position=(0.0, 0.0, 10.0),
        look_at=(0.0, 0.0, 0.0),
        fov_degrees=45.0,
        scale=5.0
    )
}

def look_at_matrix(eye, target, up, device):
    """
    Creates a view matrix.
    eye: (3,) tensor
    target: (3,) tensor
    up: (3,) tensor
    """
    z_axis = eye - target
    z_axis = z_axis / torch.linalg.norm(z_axis)
    
    x_axis = torch.linalg.cross(up, z_axis)
    x_axis = x_axis / torch.linalg.norm(x_axis)
    
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / torch.linalg.norm(y_axis)
    
    view_matrix = torch.eye(4, device=device)
    view_matrix[0, :3] = x_axis
    view_matrix[1, :3] = y_axis
    view_matrix[2, :3] = z_axis
    
    view_matrix[0, 3] = -torch.dot(x_axis, eye)
    view_matrix[1, 3] = -torch.dot(y_axis, eye)
    view_matrix[2, 3] = -torch.dot(z_axis, eye)
    
    return view_matrix

def perspective_matrix(fov_degrees, aspect_ratio, near, far, device):
    """
    Creates a perspective projection matrix.
    """
    fov_rad = math.radians(fov_degrees)
    f = 1.0 / math.tan(fov_rad / 2.0)
    
    proj_matrix = torch.zeros((4, 4), device=device)
    proj_matrix[0, 0] = f / aspect_ratio
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    return proj_matrix

def project_points(points_3d, camera_config: CameraConfig, width: int, height: int):
    """
    Applies 3D perspective projection to a batch of points.
    points_3d: Tensor of shape (N, 3)
    Returns: Tensor of shape (N, 2) containing pixel coordinates.
    """
    device = points_3d.device
    num_points = points_3d.shape[0]
    
    eye = torch.tensor(camera_config.position, device=device, dtype=torch.float32)
    target = torch.tensor(camera_config.look_at, device=device, dtype=torch.float32)
    up = torch.tensor(camera_config.up, device=device, dtype=torch.float32)
    
    view_mat = look_at_matrix(eye, target, up, device)
    aspect_ratio = width / float(height)
    proj_mat = perspective_matrix(camera_config.fov_degrees, aspect_ratio, camera_config.near, camera_config.far, device)
    
    # Combined View-Projection matrix
    vp_mat = proj_mat @ view_mat
    
    # Homogeneous coordinates
    ones = torch.ones((num_points, 1), device=device, dtype=torch.float32)
    points_4d = torch.cat([points_3d, ones], dim=1)
    
    # Project
    clip_space = points_4d @ vp_mat.T
    
    # Perspective divide
    w = clip_space[:, 3:4]
    w = torch.where(w == 0, torch.tensor(1e-6, device=device), w)
    ndc_space = clip_space[:, :3] / w
    
    # Screen coordinates (NDC is [-1, 1], map to [0, width] and [0, height])
    # Flip Y axis for screen space
    x_pixel = (ndc_space[:, 0] + 1.0) * 0.5 * width
    y_pixel = (1.0 - ndc_space[:, 1]) * 0.5 * height
    
    pixel_coords = torch.stack([x_pixel, y_pixel], dim=1)
    return pixel_coords
