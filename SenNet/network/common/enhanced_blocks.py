from __future__ import annotations

from typing import Optional, Sequence, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
import torch.nn.functional as F
from SenNet.network.common.helper import get_matching_convtransp, maybe_convert_scalar_to_list


def match_tensor_to_reference(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    将 x 的空间尺寸对齐到 ref。
    先 pad，不够再 center crop。
    适用于 3D tensor: [B, C, D, H, W]
    """
    xd, xh, xw = x.shape[2:]
    rd, rh, rw = ref.shape[2:]

    # 先 pad 到不小于 ref
    pd = max(rd - xd, 0)
    ph = max(rh - xh, 0)
    pw = max(rw - xw, 0)

    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(
            x,
            [
                pw // 2, pw - pw // 2,   # W
                ph // 2, ph - ph // 2,   # H
                pd // 2, pd - pd // 2    # D
            ]
        )

    # 再中心裁剪到 ref 大小
    xd, xh, xw = x.shape[2:]
    sd = max((xd - rd) // 2, 0)
    sh = max((xh - rh) // 2, 0)
    sw = max((xw - rw) // 2, 0)

    x = x[:, :, sd:sd + rd, sh:sh + rh, sw:sw + rw]
    return x

class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        padding = [(k - 1) // 2 for k in kernel_size]
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        ops = [
            conv_op(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=conv_bias,
            )
        ]
        if norm_op is not None:
            ops.append(norm_op(out_channels, **norm_op_kwargs))
        if nonlin is not None:
            ops.append(nonlin(**nonlin_kwargs))
        self.block = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            conv_op,
            kernel_size=kernel_size,
            stride=stride,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.conv2 = ConvNormAct(
            out_channels,
            out_channels,
            conv_op,
            kernel_size=kernel_size,
            stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=None,
            nonlin_kwargs=nonlin_kwargs,
        )
        if in_channels != out_channels or (isinstance(stride, int) and stride != 1):
            self.shortcut = ConvNormAct(
                in_channels,
                out_channels,
                conv_op,
                kernel_size=1,
                stride=stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=None,
                nonlin_kwargs=nonlin_kwargs,
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nonlin(**(nonlin_kwargs or {})) if nonlin is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if residual.shape[2:] != x.shape[2:]:
            residual = torch.nn.functional.interpolate(
                residual, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        return self.act(x + residual)


class MultiScaleInputStem(nn.Module):
    """浅层多尺度块，适合作为增强和分割编码器入口。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        branch_channels = max(out_channels // 3, 8)
        self.branch1 = ConvNormAct(
            in_channels,
            branch_channels,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.branch3 = ConvNormAct(
            in_channels,
            branch_channels,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.branch5 = nn.Sequential(
            ConvNormAct(
                in_channels,
                branch_channels,
                conv_op,
                kernel_size=3,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            ConvNormAct(
                branch_channels,
                branch_channels,
                conv_op,
                kernel_size=3,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
        )
        self.fuse = ConvNormAct(
            branch_channels * 3,
            out_channels,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)
        return self.fuse(x)


class EdgeResponse(nn.Module):
    """用固定卷积近似边缘响应，便于增强模块做边界保持。"""

    def __init__(self, channels: int, dimension: int = 3) -> None:
        super().__init__()
        if dimension != 3:
            raise ValueError("当前版本只实现 3D edge response")
        kernel = torch.zeros(1, 1, 3, 3, 3)
        kernel[0, 0, 1, 1, 1] = 6.0
        kernel[0, 0, 0, 1, 1] = -1.0
        kernel[0, 0, 2, 1, 1] = -1.0
        kernel[0, 0, 1, 0, 1] = -1.0
        kernel[0, 0, 1, 2, 1] = -1.0
        kernel[0, 0, 1, 1, 0] = -1.0
        kernel[0, 0, 1, 1, 2] = -1.0
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1, 1))
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv3d(x, self.kernel, padding=1, groups=self.groups)


class EdgeGuidedSkip(nn.Module):
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
        self.edge = EdgeResponse(channels)
        self.proj = ConvNormAct(
            channels * 2,
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
        edge = self.edge(x)
        return self.proj(torch.cat([x, edge], dim=1))


class GatedFusionBlock(nn.Module):
    """raw/enhanced 双分支门控融合。"""

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
        self.gate = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
                conv_op,
                kernel_size=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            conv_op(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = ConvNormAct(
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

    def forward(self, raw_feat: torch.Tensor, enh_feat: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([raw_feat, enh_feat], dim=1))
        fused = alpha * raw_feat + (1.0 - alpha) * enh_feat
        return self.out(fused)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        transp = get_matching_convtransp(conv_op=conv_op)
        self.up = transp(in_channels, out_channels, kernel_size=2, stride=2, bias=conv_bias)
        self.conv = ResidualConvBlock(
            out_channels * 2,
            out_channels,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = match_tensor_to_reference(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

