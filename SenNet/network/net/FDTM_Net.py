from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common.FDM_blocks import (
    CrossStateSpaceFusion,
    DynamicMultiScaleInputBlock,
    FrequencyEnhancementBackbone,
    HybridEncoderStage,
    UpBlock,
)
from SenNet.network.common.FDTM_blocks import FDTMSegMambaEncoderStage
from SenNet.network.common.helper import InitWeights_He


class FDTMNet(nn.Module):
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
        self.n_stages = n_stages
        self.features_per_stage = list(features_per_stage)
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        base_channels = max(features_per_stage[0] // 2, 16)
        self.enhancer = FrequencyEnhancementBackbone(
            in_channels=input_channels,
            base_channels=base_channels,
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.raw_encoder = nn.ModuleList()
        self.enh_encoder = nn.ModuleList()
        self.fusion = nn.ModuleList()

        prev_raw = input_channels
        prev_enh = input_channels
        mamba_from_stage = max(2, n_stages - 2)
        for stage_idx in range(n_stages):
            out_channels = features_per_stage[stage_idx]
            use_mamba = stage_idx >= mamba_from_stage
            if stage_idx == 0:
                raw_stage = DynamicMultiScaleInputBlock(
                    prev_raw,
                    out_channels,
                    conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                enh_stage = DynamicMultiScaleInputBlock(
                    prev_enh,
                    out_channels,
                    conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            elif stage_idx < mamba_from_stage:
                raw_stage = HybridEncoderStage(
                    prev_raw,
                    out_channels,
                    conv_op,
                    stride=strides[stage_idx],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    use_mamba=False,
                )
                enh_stage = HybridEncoderStage(
                    prev_enh,
                    out_channels,
                    conv_op,
                    stride=strides[stage_idx],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    use_mamba=False,
                )
            else:
                num_blocks = n_conv_per_stage[stage_idx] if stage_idx < len(n_conv_per_stage) else 1
                raw_stage = FDTMSegMambaEncoderStage(
                    prev_raw,
                    out_channels,
                    conv_op,
                    stride=strides[stage_idx],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    num_mamba_blocks=num_blocks,
                )
                enh_stage = FDTMSegMambaEncoderStage(
                    prev_enh,
                    out_channels,
                    conv_op,
                    stride=strides[stage_idx],
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    num_mamba_blocks=num_blocks,
                )

            self.raw_encoder.append(raw_stage)
            self.enh_encoder.append(enh_stage)
            self.fusion.append(
                CrossStateSpaceFusion(
                    out_channels,
                    conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    use_mamba=use_mamba,
                )
            )
            prev_raw = out_channels
            prev_enh = out_channels

        self.decoder = nn.ModuleList()
        self.seg_heads = nn.ModuleList()
        for stage_idx in range(n_stages - 1, 0, -1):
            self.decoder.append(
                UpBlock(
                    features_per_stage[stage_idx],
                    features_per_stage[stage_idx - 1],
                    conv_op,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            self.seg_heads.append(
                conv_op(features_per_stage[stage_idx - 1], num_classes, kernel_size=1, bias=True)
            )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def _encode(self, raw_x: torch.Tensor, enh_x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        skips: List[torch.Tensor] = []
        fused = raw_x
        for stage_idx, (raw_stage, enh_stage, fuse_stage) in enumerate(
            zip(self.raw_encoder, self.enh_encoder, self.fusion)
        ):
            raw_x = raw_stage(raw_x)
            enh_x = enh_stage(enh_x)
            fused = fuse_stage(raw_x, enh_x)
            if stage_idx < self.n_stages - 1:
                skips.append(fused)
        return skips, fused

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        enhanced, residual = self.enhancer(x)
        skips, bottleneck = self._encode(x, residual)

        decoder_outputs: List[torch.Tensor] = []
        x_dec = bottleneck
        for idx, decoder in enumerate(self.decoder):
            x_dec = decoder(x_dec, skips[-(idx + 1)])
            decoder_outputs.append(self.seg_heads[idx](x_dec))

        decoder_outputs = decoder_outputs[::-1]
        seg = decoder_outputs if self.deep_supervision else decoder_outputs[0]

        if not return_aux:
            return seg

        return {
            'seg': seg,
            'enhanced': enhanced,
            'residual': residual,
        }