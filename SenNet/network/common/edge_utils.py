from __future__ import annotations

import torch
import torch.nn.functional as F


def _laplace_kernel_3d(device: torch.device, dtype: torch.dtype, channels: int) -> torch.Tensor:
    kernel = torch.zeros((channels, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel[:, 0, 1, 1, 1] = 6.0
    kernel[:, 0, 0, 1, 1] = -1.0
    kernel[:, 0, 2, 1, 1] = -1.0
    kernel[:, 0, 1, 0, 1] = -1.0
    kernel[:, 0, 1, 2, 1] = -1.0
    kernel[:, 0, 1, 1, 0] = -1.0
    kernel[:, 0, 1, 1, 2] = -1.0
    return kernel


def laplace_response_3d(x: torch.Tensor) -> torch.Tensor:
    kernel = _laplace_kernel_3d(x.device, x.dtype, x.shape[1])
    return F.conv3d(x, kernel, padding=1, groups=x.shape[1])


def high_frequency_map(x: torch.Tensor) -> torch.Tensor:
    smooth = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x - smooth


def probability_boundary_map(prob: torch.Tensor) -> torch.Tensor:
    """用概率图近似可导边界。输入 shape: B,C,H,W,D。"""
    max_prob, _ = torch.max(prob, dim=1, keepdim=True)
    return torch.abs(laplace_response_3d(max_prob))
