from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common.enhanced_blocks import (
    ConvNormAct,
    EdgeGuidedSkip,
    GatedFusionBlock,
    MultiScaleInputStem,
    ResidualConvBlock,
    UpBlock,
)
from SenNet.network.common.helper import InitWeights_He


class ResidualEnhancer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = MultiScaleInputStem(
            in_channels,
            c1,
            conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.down1 = ResidualConvBlock(c1, c2, conv_op, stride=2, conv_bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.down2 = ResidualConvBlock(c2, c3, conv_op, stride=2, conv_bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.mid = ResidualConvBlock(c3, c3, conv_op, conv_bias=conv_bias,
                                     norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                     nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.skip1 = EdgeGuidedSkip(c1, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.skip2 = EdgeGuidedSkip(c2, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.up1 = UpBlock(c3, c2, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.up2 = UpBlock(c2, c1, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
        self.out = conv_op(c1, in_channels, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        x3 = self.down2(s2)
        x3 = self.mid(x3)
        x = self.up1(x3, self.skip2(s2))
        x = self.up2(x, self.skip1(s1))
        residual = self.tanh(self.out(x))
        return x.new_tensor(1.0) * residual


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd],
        stride: Union[int, Sequence[int]],
        conv_bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        use_stem: bool = False,
    ) -> None:
        super().__init__()
        if use_stem:
            self.block = nn.Sequential(
                MultiScaleInputStem(in_channels, out_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs),
                ResidualConvBlock(out_channels, out_channels, conv_op, conv_bias=conv_bias,
                                  norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                  nonlin=nonlin, nonlin_kwargs=nonlin_kwargs),
            )
        else:
            self.block = ResidualConvBlock(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EnhancedSegNet(nn.Module):
    """CBCT 结构保持增强 + 双分支协同分割。"""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        conv_op: Type[_ConvNd],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_conv_per_stage: Sequence[int],
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.n_stages = n_stages
        self.features_per_stage = list(features_per_stage)
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.conv_op = conv_op

        base_channels = max(features_per_stage[0] // 2, 16)
        self.enhancer = ResidualEnhancer(
            input_channels,
            base_channels,
            conv_op,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
        )

        self.raw_encoder = nn.ModuleList()
        self.enh_encoder = nn.ModuleList()
        self.fusion = nn.ModuleList()

        prev_channels = input_channels
        prev_channels_enh = input_channels
        for stage_idx in range(n_stages):
            out_channels = features_per_stage[stage_idx]
            stride = 1 if stage_idx == 0 else strides[stage_idx]
            use_stem = stage_idx == 0
            self.raw_encoder.append(
                EncoderStage(prev_channels, out_channels, conv_op, stride, conv_bias,
                             norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, use_stem=use_stem)
            )
            self.enh_encoder.append(
                EncoderStage(prev_channels_enh, out_channels, conv_op, stride, conv_bias,
                             norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, use_stem=use_stem)
            )
            self.fusion.append(
                GatedFusionBlock(out_channels, conv_op, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs)
            )
            prev_channels = out_channels
            prev_channels_enh = out_channels

        self.decoder = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        for i in range(n_stages - 1, 0, -1):
            self.decoder.append(
                UpBlock(
                    features_per_stage[i],
                    features_per_stage[i - 1],
                    conv_op,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                )
            )
            self.seg_heads.append(conv_op(features_per_stage[i - 1], num_classes, kernel_size=1, bias=True))

        self.raw_aux_head = conv_op(features_per_stage[-1], num_classes, kernel_size=1, bias=True)
        self.enh_aux_head = conv_op(features_per_stage[-1], num_classes, kernel_size=1, bias=True)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def _encode(self, x_raw: torch.Tensor, x_enh: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        fused_skips: List[torch.Tensor] = []
        raw_feats: List[torch.Tensor] = []
        enh_feats: List[torch.Tensor] = []
        for raw_stage, enh_stage, fuse in zip(self.raw_encoder, self.enh_encoder, self.fusion):
            x_raw = raw_stage(x_raw)
            x_enh = enh_stage(x_enh)
            raw_feats.append(x_raw)
            enh_feats.append(x_enh)
            fused_skips.append(fuse(x_raw, x_enh))
        return fused_skips[:-1], raw_feats[-1], enh_feats[-1]

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        residual = self.enhancer(x)
        enhanced = x + residual

        skips, raw_bottom, enh_bottom = self._encode(x, enhanced)
        x_dec = self.fusion[-1](raw_bottom, enh_bottom)

        seg_outputs: List[torch.Tensor] = []
        for i, dec in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x_dec = dec(x_dec, skip)
            seg_outputs.append(self.seg_heads[i](x_dec))

        seg_outputs = seg_outputs[::-1]
        seg = seg_outputs if self.deep_supervision else seg_outputs[0]

        if not return_aux:
            return seg

        return {
            "seg": seg,
            "enhanced": enhanced,
            "residual": residual,
            "raw_aux": self.raw_aux_head(raw_bottom),
            "enh_aux": self.enh_aux_head(enh_bottom),
        }
