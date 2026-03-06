
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from SenNet.network.common.helper import *
import torch
from torch import nn
from typing import Union, Type, List, Tuple, Optional

class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class Net(nn.Module):
    def __init__(self,
                 n_stages: int,
                 enable_skip_layer: bool = True,
                 skip_merge_type: Optional[str] = 'concat',  # 可以是'concat'或者'add'或者None。
                 deep_supervision: bool = False,
                 down_sample_first: bool = True,
                 **invalid_args
                 ):
        """
        Args:
            n_stages: 网络总共的深度（编码器和解码器的数量+1=n_stages）。
            enable_skip: 如果是True，那么网络会有skip connection。
            skip_merge_type: 可以是'concat'或者'add'。如果是'concat'，那么skip connection会被concatenate到decoder的输出上。
            deep_supervision: 如果是True，那么返回的是一个list，里面包含了每个stage的输出。
        """
        super().__init__()
        self.n_stages = n_stages
        self.enable_skip_layer = enable_skip_layer
        self.skip_merge_type = skip_merge_type
        self.deep_supervision = deep_supervision
        self.down_sample_first = down_sample_first

        # 编码器层，一共有n_stages-1个编码器层。
        self.encoder_layers = nn.ModuleList()
        # 降采样，改变空间大小，一共有n_stages-1个降采样层。
        self.down_sample_blocks = nn.ModuleList()
        # 跳跃连接层的处理。
        self.skip_layers = nn.ModuleList()
        # 瓶颈层，中间层。
        self.bottle_neck: Optional[nn.Module] = None
        # 上采样，还原空间大小，一共有n_stages-1个上采样层。
        self.up_sample_blocks = nn.ModuleList()
        # 解码器层，一共有n_stages-1个解码器层。
        self.decoder_layers = nn.ModuleList()
        # 分割层，输出分割结果，一共有n_stages-1个分割层（可深度监督）。
        self.seg_layers = nn.ModuleList()
        # 跳跃连接层的合并，当skip_merge_type是None时，需要这个层。
        self.skip_merge_blocks = nn.ModuleList()
        self.skip_fusion_block = None


    # def forward(self, x):
    #     skips = []
    #     for i, encoder_layer in enumerate(self.encoder_layers):
    #         x = encoder_layer(x)
    #         t = x
    #         x = self.down_sample_blocks[i](x)
    #         if self.enable_skip_layer and len(self.skip_layers) > i and self.skip_layers[i] is not None:
    #             t = self.skip_layers[i](t)
    #
    #         skips.append(t)
    #
    #     x = self.bottle_neck(x)
    #     # for i, skip in enumerate(skips):
    #     #     print("skip {}: {}".format(i, skip.size()))
    #
    #     seg_outputs = []
    #     for i in range(1, self.n_stages):
    #         x = self.up_sample_blocks[-i](x)
    #
    #         if self.enable_skip_layer:
    #             if self.skip_merge_type is None:
    #                 x = self.skip_merge_blocks[- i](x, skips[- i])
    #             elif self.skip_merge_type == 'concat':
    #                 x = torch.cat([x, skips[- i]], dim=1)
    #             elif self.skip_merge_type == 'add':
    #                 x = x + skips[- i]
    #             else:
    #                 raise ValueError("skip_merge_type should be 'concat' or 'add'")
    #
    #         x = self.decoder_layers[-i](x)
    #         if self.deep_supervision:
    #             seg_outputs.append(self.seg_layers[- i](x))
    #
    #     if not self.deep_supervision:
    #         seg_outputs.append(self.seg_layers[0](x))
    #
    #     seg_outputs = seg_outputs[::-1]
    #     if self.deep_supervision:
    #         return seg_outputs
    #     else:
    #         return seg_outputs[-1]

    def forward(self, x):
        skips = []
        for i, encoder_layer in enumerate(self.encoder_layers):
            if self.down_sample_first:
                x = self.down_sample_blocks[i](x)
                x = encoder_layer(x)
            else:
                x = encoder_layer(x)
                x = self.down_sample_blocks[i](x)
            t = x
            # 降采样之后才会有 skip！并且瓶颈层没有 skip！
            if i < self.n_stages - 1:
                if self.enable_skip_layer and len(self.skip_layers) > i and self.skip_layers[i] is not None:
                    t = self.skip_layers[i](t)
                skips.append(t)

        if self.bottle_neck is not None:
            x = self.bottle_neck(x)

        if self.skip_fusion_block is not None:
            skips = self.skip_fusion_block(skips)

        seg_outputs = []
        for i in range(1, self.n_stages):
            x = self.up_sample_blocks[-i](x)
            x = self.connect_skip(i, x, skips[-i])
            x = self.decoder_layers[-i](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[- i](x))

        if not self.deep_supervision:
            seg_outputs.append(self.seg_layers[0](x))

        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[-1]

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def connect_skip(self, i, x, skip):
        if self.enable_skip_layer:
            if self.skip_merge_type is None:
                x = self.skip_merge_blocks[- i](x, skip)
            elif self.skip_merge_type == 'concat':
                x = torch.cat([x, skip], dim=1)
            elif self.skip_merge_type == 'add':
                x = x + skip
            else:
                raise ValueError("skip_merge_type should be 'concat' or 'add'")
        return x

