from __future__ import annotations

import os
from typing import Optional, Sequence, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common.FDM_blocks import ConvNormAct, HybridEncoderStage, match_tensor_to_reference
from SenNet.network.common.helper import maybe_convert_scalar_to_list

_TOMAMBA_IMPORT_ERROR = None
try:
    from SenNet.network.net.mamba_blocks import ToMamba as _ToMamba
except Exception as exc:
    _ToMamba = None
    _TOMAMBA_IMPORT_ERROR = exc


def _apply_token_layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    input_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x_float = x.float()
        batch_size, channels = x_float.shape[:2]
        spatial_shape = x_float.shape[2:]
        tokens = x_float.reshape(batch_size, channels, -1).transpose(-1, -2)
        tokens = norm(tokens)
        x_float = tokens.transpose(-1, -2).reshape(batch_size, channels, *spatial_shape)
    return x_float.to(dtype=input_dtype)


class FDTMContextBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        allow_fallback = os.environ.get('SENNET_ALLOW_MAMBA_FALLBACK', '0').lower() in ('1', 'true', 'yes')
        if _ToMamba is None:
            if not allow_fallback:
                raise ImportError(
                    'ToMamba could not be imported from SenNet.network.net.mamba_blocks. '
                    'Install mamba_ssm or set SENNET_ALLOW_MAMBA_FALLBACK=1 to use the convolution fallback.'
                ) from _TOMAMBA_IMPORT_ERROR
            self.state_extractor = ConvNormAct(
                channels,
                channels,
                conv_op,
                kernel_size=3,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        else:
            self.state_extractor = _ToMamba(channels)

        self.output_norm = nn.LayerNorm(channels)
        self.reconstruct = ConvNormAct(
            channels,
            channels,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.state_extractor(x)
        x = _apply_token_layer_norm(x, self.output_norm)
        x = self.reconstruct(x)
        x = match_tensor_to_reference(x, residual)
        return x + residual


class FDTMChannelMLP(nn.Module):
    def __init__(self, channels: int, conv_op: Type[_ConvNd], hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels * 2
        self.fc1 = conv_op(channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = conv_op(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FDTMGSCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.proj = ConvNormAct(
            channels,
            channels,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.proj2 = ConvNormAct(
            channels,
            channels,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.proj3 = ConvNormAct(
            channels,
            channels,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.proj4 = ConvNormAct(
            channels,
            channels,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.proj(x)
        x1 = self.proj2(x1)
        x2 = self.proj3(x)
        x = self.proj4(x1 + x2)
        return x + residual


class FDTMSegMambaEncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        stride: Union[int, Sequence[int]],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        num_mamba_blocks: int = 1,
    ) -> None:
        super().__init__()
        stride_list = maybe_convert_scalar_to_list(conv_op, stride)
        if any(s != 1 for s in stride_list):
            kernel_size = stride_list
            self.downsample = ConvNormAct(
                in_channels,
                out_channels,
                conv_op,
                kernel_size=kernel_size,
                stride=stride_list,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=None,
                nonlin_kwargs=nonlin_kwargs,
            )
        else:
            self.downsample = ConvNormAct(
                in_channels,
                out_channels,
                conv_op,
                kernel_size=3,
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )

        self.gsc = FDTMGSCBlock(
            out_channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.mamba_blocks = nn.Sequential(
            *[
                FDTMContextBlock(
                    out_channels,
                    conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(max(1, int(num_mamba_blocks)))
            ]
        )

        if norm_op is not None:
            self.out_norm = norm_op(out_channels, **(norm_op_kwargs or {}))
        else:
            self.out_norm = nn.InstanceNorm3d(out_channels)
        self.out_mlp = FDTMChannelMLP(out_channels, conv_op, hidden_channels=out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.gsc(x)
        x = self.mamba_blocks(x)
        x = self.out_norm(x)
        x = self.out_mlp(x)
        return x


class FDTMHybridEncoderStage(HybridEncoderStage):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        stride: Union[int, Sequence[int]],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        use_mamba: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            conv_op,
            stride=stride,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            use_mamba=False,
        )
        if use_mamba:
            self.context = FDTMContextBlock(
                out_channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )