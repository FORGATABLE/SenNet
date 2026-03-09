from typing import Type, Union, List

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from SenNet.network.common import helper
from SenNet.network.common.kan import KANLinear
from SenNet.network.net.conv_blocks import BasicConvBlock
# from SenNet.network.net.kan_blocks import (KANBlock)
from SenNet.network.net.mamba_blocks import MambaLayer


class SelectiveFusionBlock(nn.Module):
    """
    Selective Fusion Block used in SIB-UNet and SMM-UNet.
    """

    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        proj_type: str = 'mlp',  # 'conv', 'mlp' or 'kan'
        use_max: bool = False,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.projection = None
        if proj_type == 'mlp':
            self.projection = nn.Linear(channels * 2, channels)
        elif proj_type == 'kan':
            self.projection = KANLinear(channels * 2, channels)
        elif proj_type == 'conv':
            self.projection = BasicConvBlock(channels * 2, channels, conv_op, 1, 1)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        self.nonlin = nn.Softmax(dim=1)
        self.use_max = use_max

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        reduce_dims = tuple(range(2, x.dim()))

        mean_x = torch.mean(x, dim=reduce_dims, keepdim=True)
        if self.use_max:
            max_x = torch.amax(x, dim=reduce_dims, keepdim=True)
            mean_x += max_x
        x = mean_x

        if self.proj_type == 'conv':
            # Conv expects channels-first tensor.
            x = self.projection(x)
        else:
            # Linear/KAN expects channels-last tensor.
            permute_to_last = [0] + list(range(2, x.dim())) + [1]
            x = x.permute(*permute_to_last)
            x = self.projection(x)
            permute_back = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
            x = x.permute(*permute_back)

        x = self.nonlin(x)
        y = 1 - x
        x1 = x1 * x
        x2 = x2 * y
        return x1 + x2


class MultiScaleFusionBlock(nn.Module):

    def __init__(self, features: list, conv_op: Type[_ConvNd],
                 norm_op: Union[None, Type[nn.Module]] = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 norm_layer=nn.LayerNorm,
                 use_mamba_v2=False,
                 tri_orientation=False,
                 fusion_count=5
                 ):
        super().__init__()
        self.fusion_count = fusion_count
        self.features = features[:fusion_count]
        self.tri_orientation = tri_orientation
        feature_sum = sum(features)
        mid = (fusion_count - 1) / 2.
        dis = fusion_count // 2
        self.samples = nn.ModuleList()
        for i in range(fusion_count):
            if i < mid:
                stride = 2**(dis-i)
                self.samples.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
            elif i > mid:
                scale = 2**(i-dis)
                self.samples.append(nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True))
            else:
                self.samples.append(None)
        self.mamba = MambaLayer(dim=feature_sum, channel_token=False, use_v2=use_mamba_v2)
        if self.tri_orientation:
            self.mamba2 = MambaLayer(dim=feature_sum, channel_token=False, use_v2=use_mamba_v2)
            self.mamba3 = MambaLayer(dim=feature_sum, channel_token=False, use_v2=use_mamba_v2)
        # self.mlp = nn.Sequential(
        #     nn.Linear(feature_sum, feature_sum),
        #     nn.ReLU(),
        #     nn.Linear(feature_sum, feature_sum)
        # )
        self.mlp = None
        self.layer_norm = norm_layer(feature_sum)

    def forward(self, skips):
        resmaple_skips = skips[:self.fusion_count]
        for i, skip in enumerate(resmaple_skips):
            if self.samples[i] is not None:
                skip = self.samples[i](skip)

            resmaple_skips[i] = skip

        concat = torch.cat(resmaple_skips, dim=1)
        x = self.mamba(concat)
        if self.tri_orientation:
            x2 = self.mamba2(concat)
            x3 = self.mamba3(concat)
            x = x + x2 + x3
            x = helper.channel_to_the_last(x)
            x = self.layer_norm(x)
            x = helper.channel_to_the_second(x)

        if self.mlp is not None:
            x = helper.channel_to_the_last(x)
            x = self.mlp(x)
            x = helper.channel_to_the_second(x)

        # Restore x back to skip list.
        feature_sum = 0
        for i, feature in enumerate(self.features):
            skip = x[:, feature_sum:feature_sum+feature]
            if self.samples[-i-1] is not None:
                skip = self.samples[-i-1](skip)
            feature_sum += feature
            skips[i] = skip + skips[i]
        return skips


if __name__ == '__main__':
    block = MultiScaleFusionBlock([32, 32, 32, 32, 32], nn.Conv3d).cuda()
    x0 = torch.randn(2, 32, 32, 32, 32).cuda()
    x1 = torch.randn(2, 32, 16, 16, 16).cuda()
    x2 = torch.randn(2, 32, 8, 8, 8).cuda()
    x3 = torch.randn(2, 32, 4, 4, 4).cuda()
    x4 = torch.randn(2, 32, 2, 2, 2).cuda()
    skips = [x0, x1, x2, x3, x4]
    block(skips)
    for i, skip in enumerate(skips):
        print(f"Skip {i}: {skip.size()}")
    # x = torch.arange(0, 8).reshape(2, 2, 2)
    # print(x)
    # x1 = x
    # x2 = x.permute(1, 2, 0)
    # x3 = x.permute(2, 0, 1)
    # print(x1.flatten(0))
    # print(x2.flatten(0))
    # print(x3.flatten(0))
