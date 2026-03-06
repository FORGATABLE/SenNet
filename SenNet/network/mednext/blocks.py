from SenNet.network.common.helper import get_matching_convtransp
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_op: Type[_ConvNd],
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 grn=False
                 ):

        super().__init__()

        self.do_res = do_res

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv_op(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.grn = grn
        if grn:
            self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, conv_op: Type[_ConvNd], conv_kernel_size, stride=2, conv_bias=True,
                 exp_r=4, kernel_size=7, do_res=False, norm_type='group', grn=False):

        super().__init__(in_channels, out_channels, conv_op, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type,
                         grn=grn)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv_op(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )

        self.conv1 = conv_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in conv_kernel_size],
            groups=in_channels,
            bias=conv_bias,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, conv_op: Type[_ConvNd], stride=2, conv_bias=True,
                 exp_r=4, kernel_size=7, do_res=False, norm_type='group', grn=False):
        super().__init__(in_channels, out_channels, conv_op, exp_r, kernel_size,
                         do_res=False, norm_type=norm_type, grn=grn)

        self.resample_do_res = do_res
        conv_op = get_matching_convtransp(conv_op=conv_op)

        if do_res:
            self.res_conv = conv_op(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )

        self.conv1 = conv_op(
            in_channels,
            in_channels,
            stride,
            stride,
            bias=conv_bias,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape

        # x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        # if self.resample_do_res:
        #     res = self.res_conv(x)
        #     res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
        #     x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, conv_op: Type[_ConvNd]):
        super().__init__()

        conv = get_matching_convtransp(conv_op=conv_op)
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


if __name__ == "__main__":
    # network = nnUNeXtBlock(in_channels=12, out_channels=12, do_res=False).cuda()

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 8, 8, 8)).cuda()
    #     print(network(x).shape)

    # network = DownsampleBlock(in_channels=12, out_channels=24, do_res=False)

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 128, 128, 128))
    #     print(network(x).shape)

    network = MedNeXtBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 12, 64, 64, 64)).cuda()
        print(network(x).shape)