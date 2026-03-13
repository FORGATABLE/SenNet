from __future__ import annotations

from typing import Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common.helper import get_matching_convtransp, maybe_convert_scalar_to_list
import os
_MAMBA_IMPORT_ERROR = None
try:
    from SenNet.network.net.mamba_blocks import MambaLayer as _MambaLayer
except Exception as exc:
    _MambaLayer = None
    _MAMBA_IMPORT_ERROR = exc


def match_tensor_to_reference(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    spatial_dims = ref.shape[2:]
    if x.shape[2:] == spatial_dims:
        return x

    paddings = []
    for x_dim, ref_dim in zip(reversed(x.shape[2:]), reversed(spatial_dims)):
        pad = max(ref_dim - x_dim, 0)
        paddings.extend([pad // 2, pad - pad // 2])
    if any(paddings):
        x = F.pad(x, paddings)

    slices = [slice(None), slice(None)]
    for x_dim, ref_dim in zip(x.shape[2:], spatial_dims):
        start = max((x_dim - ref_dim) // 2, 0)
        slices.append(slice(start, start + ref_dim))
    return x[tuple(slices)]


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        dilation = maybe_convert_scalar_to_list(conv_op, dilation)
        padding = [((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation)]
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        layers = [
            conv_op(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=conv_bias,
            )
        ]
        if norm_op is not None:
            layers.append(norm_op(out_channels, **norm_op_kwargs))
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs))
        self.block = nn.Sequential(*layers)

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
        needs_projection = in_channels != out_channels or any(
            s != 1 for s in maybe_convert_scalar_to_list(conv_op, stride)
        )
        if needs_projection:
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
        residual = match_tensor_to_reference(residual, x)
        return self.act(x + residual)


class FourierUnit3D(nn.Module):
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
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}
        layers = [conv_op(channels * 2, channels * 2, kernel_size=1, bias=conv_bias)]
        if norm_op is not None:
            layers.append(norm_op(channels * 2, **norm_op_kwargs))
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs))
        self.freq_proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            spatial_shape = x.shape[2:]
            fft_x = torch.fft.rfftn(x, dim=(2, 3, 4), norm="ortho")
            freq = torch.cat([fft_x.real, fft_x.imag], dim=1)
            freq = self.freq_proj(freq)
            real, imag = torch.chunk(freq, 2, dim=1)
            fft_out = torch.complex(real, imag)
            out = torch.fft.irfftn(fft_out, s=spatial_shape, dim=(2, 3, 4), norm="ortho")
        return out.to(dtype=input_dtype)


class FastFourierConvBlock(nn.Module):
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
        self.local = ConvNormAct(
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
        self.global_unit = FourierUnit3D(
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.gate = nn.Sequential(
            conv_op(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = ResidualConvBlock(
            channels,
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local(x)
        global_feat = self.global_unit(x)
        gate = self.gate(torch.cat([local_feat, global_feat], dim=1))
        fused = local_feat * gate + global_feat * (1.0 - gate)
        return self.out(x + fused)


class DynamicConvBranch(nn.Module):
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
        dilations: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    out_channels,
                    conv_op,
                    kernel_size=3,
                    dilation=dilation,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for dilation in dilations
            ]
        )
        hidden = max(in_channels // 2, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, len(dilations), kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(self.pool(x)).flatten(1)
        weights = torch.softmax(weights, dim=1)
        outputs = [branch(x) for branch in self.branches]
        fused = outputs[0].new_zeros(outputs[0].shape)
        for idx, branch_out in enumerate(outputs):
            fused = fused + branch_out * weights[:, idx].view(-1, 1, 1, 1, 1)
        return fused


class DynamicMultiScaleInputBlock(nn.Module):
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
        self.gray_branch = ConvNormAct(
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
        self.context_branch = ConvNormAct(
            in_channels,
            branch_channels,
            conv_op,
            kernel_size=3,
            dilation=2,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.dynamic_branch = DynamicConvBranch(
            in_channels,
            branch_channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.fuse = ResidualConvBlock(
            branch_channels * 3,
            out_channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.gray_branch(x), self.context_branch(x), self.dynamic_branch(x)],
            dim=1,
        )
        return self.fuse(features)


class MambaContextBlock(nn.Module):
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
        allow_fallback = os.environ.get("SENNET_ALLOW_MAMBA_FALLBACK", "0").lower() in ("1", "true", "yes")
        if _MambaLayer is None:
            if not allow_fallback:
                raise ImportError(
                    "MambaLayer could not be imported from SenNet.network.net.mamba_blocks. "
                    "Install mamba_ssm or set SENNET_ALLOW_MAMBA_FALLBACK=1 to use the convolution fallback."
                ) from _MAMBA_IMPORT_ERROR
            self.context = ResidualConvBlock(
                channels,
                channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        else:
            self.context = _MambaLayer(channels, channel_token=False, use_v2=False)
        self.refine = ResidualConvBlock(
            channels,
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context(x)
        return self.refine(x)


class HybridEncoderStage(nn.Module):
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
        super().__init__()
        self.local = ResidualConvBlock(
            in_channels,
            out_channels,
            conv_op,
            stride=stride,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.context = (
            MambaContextBlock(
                out_channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            if use_mamba
            else ResidualConvBlock(
                out_channels,
                out_channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        return self.context(x)


class CrossStateSpaceFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.raw_proj = ResidualConvBlock(
            channels,
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.enh_proj = ResidualConvBlock(
            channels,
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.selection_gate = nn.Sequential(
            conv_op(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.condition_proj = ConvNormAct(
            channels * 2,
            channels * 2,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.enh_adapter = ConvNormAct(
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
        self.context = (
            MambaContextBlock(
                channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
            if use_mamba
            else ResidualConvBlock(
                channels,
                channels,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        )
        self.update_gate = nn.Sequential(
            conv_op(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = ResidualConvBlock(
            channels * 2,
            channels,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

    def forward(self, raw_feat: torch.Tensor, enh_feat: torch.Tensor) -> torch.Tensor:
        raw_feat = self.raw_proj(raw_feat)
        enh_feat = self.enh_proj(enh_feat)

        selection = self.selection_gate(torch.cat([raw_feat, enh_feat], dim=1))
        cond_params = self.condition_proj(torch.cat([raw_feat * selection, enh_feat], dim=1))
        gamma, beta = torch.chunk(cond_params, 2, dim=1)
        gamma = torch.tanh(gamma)

        conditioned_raw = raw_feat * (1.0 + gamma) + beta
        conditioned_raw = conditioned_raw + selection * self.enh_adapter(enh_feat)

        scanned_raw = self.context(conditioned_raw)
        update = self.update_gate(torch.cat([scanned_raw, enh_feat], dim=1))
        guided_enh = update * enh_feat
        return self.out(torch.cat([scanned_raw, guided_enh], dim=1))


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
        self.refine = ResidualConvBlock(
            out_channels * 2,
            out_channels,
            conv_op,
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
        return self.refine(x)


class FrequencyEnhancementBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Encoder - transform the raw CBCT volume into a stable latent space before spectral processing.
        self.input_proj = ConvNormAct(
            in_channels,
            c1,
            conv_op,
            kernel_size=3,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.shallow_encoder = ResidualConvBlock(
            c1,
            c1,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.down1 = ResidualConvBlock(
            c1,
            c2,
            conv_op,
            stride=2,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.down2 = ResidualConvBlock(
            c2,
            c3,
            conv_op,
            stride=2,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        # Transform - low-level and bottleneck spectral modeling explicitly separate the frequency stage.
        self.low_freq_transform = FastFourierConvBlock(
            c2,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.bottleneck_transform = nn.Sequential(
            FastFourierConvBlock(
                c3,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            MambaContextBlock(
                c3,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            FastFourierConvBlock(
                c3,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
        )

        # Reconstruction - decode the transformed representation back to the image space.
        self.skip_shallow = ConvNormAct(
            c1,
            c1,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.skip_low = ConvNormAct(
            c2,
            c2,
            conv_op,
            kernel_size=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.up1 = UpBlock(
            c3,
            c2,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.up2 = UpBlock(
            c2,
            c1,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.reconstruct = nn.Sequential(
            ResidualConvBlock(
                c1,
                c1,
                conv_op,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            ConvNormAct(
                c1,
                c1,
                conv_op,
                kernel_size=3,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
        )
        self.out_proj = conv_op(c1, in_channels, kernel_size=1, bias=True)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shallow = self.shallow_encoder(self.input_proj(x))
        low_level = self.low_freq_transform(self.down1(shallow))
        bottleneck = self.bottleneck_transform(self.down2(low_level))

        x = self.up1(bottleneck, self.skip_low(low_level))
        x = self.up2(x, self.skip_shallow(shallow))
        x = self.reconstruct(x)

        residual = self.activation(self.out_proj(x))
        return x + residual, residual