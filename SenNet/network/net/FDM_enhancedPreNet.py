from __future__ import annotations

from typing import Optional, Sequence, Type

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common.FDM_blocks import FrequencyEnhancementBackbone
from SenNet.network.common.helper import InitWeights_He


class FDMEnhancedPreNet(nn.Module):
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
        deep_supervision: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
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

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def forward(self, x: torch.Tensor, return_dict: bool = True):
        enhanced, residual = self.enhancer(x)
        if not return_dict:
            return enhanced
        return {
            "enhanced": enhanced,
            "residual": residual,
        }
