from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

_MAMBA_IMPORT_ERROR = None
from mamba_ssm import Mamba, Mamba2



def _build_mamba_module(dim: int, d_state: int, d_conv: int, expand: int, use_v2: bool):
    if Mamba is None:
        raise ImportError('mamba_ssm.Mamba could not be imported.') from _MAMBA_IMPORT_ERROR
    if use_v2 and Mamba2 is not None:
        return Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=8,
        )
    return Mamba(
        d_model=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


class MambaLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_v2=False):
        super().__init__()
        self.dim = dim
        self.norm = MambaLayerNorm(dim)
        self.mamba = _build_mamba_module(dim, d_state, d_conv, expand, use_v2)

    def forward(self, x):
        batch_size, channels = x.shape[:2]
        x_skip = x
        assert channels == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(batch_size, channels, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(batch_size, channels, *img_dims)
        return out + x_skip


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False, use_v2=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = _build_mamba_module(dim, d_state, d_conv, expand, use_v2)
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        batch_size, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(batch_size, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.transpose(-1, -2).reshape(batch_size, d_model, *img_dims)

    def forward_channel_token(self, x):
        batch_size, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f'd_model: {d_model}, self.dim: {self.dim}'
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f'x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}'
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.reshape(batch_size, n_tokens, *img_dims)

    @autocast('cuda', enabled=False)
    def forward(self, x):
        input_dtype = x.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out.to(dtype=input_dtype)


class ToMamba(nn.Module):
    """SegMamba-style tokenized Mamba extractor without residual reconstruction."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.input_norm = nn.LayerNorm(dim)
        self.mamba = _build_mamba_module(dim, d_state, d_conv, expand, use_v2=True)
        self.num_slices = num_slices

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        batch_size, channels = x.shape[:2]
        assert channels == self.dim, f'channels: {channels}, self.dim: {self.dim}'
        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        x_tokens = x.reshape(batch_size, channels, n_tokens).transpose(-1, -2)
        x_tokens = self.input_norm(x_tokens)
        x_tokens = self.mamba(x_tokens)
        out = x_tokens.transpose(-1, -2).reshape(batch_size, channels, *spatial_shape)
        return out.to(dtype=input_dtype)