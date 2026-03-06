from typing import Union, List, Tuple

from torch.nn.modules.conv import _ConvNd, Conv3d

from SenNet.network.mednext.blocks import *

class MyMedNeXt(nn.Module):

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 conv_bias: bool = False,
                 n_channels: int = 32,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 grn=False
                 ):

        super().__init__()

        self.n_stages = n_stages
        self.do_ds = deep_supervision

        self.stem = Conv3d(input_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(self.n_stages)]

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        cur_channel = n_channels
        for i in range(self.n_stages - 1):
            self.enc_blocks.append(nn.Sequential(*[MedNeXtBlock(
                in_channels=cur_channel,
                out_channels=cur_channel,
                exp_r=exp_r[i],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                conv_op=conv_op,
                grn=grn
            ) for j in range(block_counts[i])]))
            self.down_blocks.append(MedNeXtDownBlock(
                in_channels=cur_channel,
                out_channels=cur_channel * 2,
                exp_r=exp_r[i],
                kernel_size=kernel_size,
                conv_kernel_size=kernel_sizes[i],
                stride=strides[i],
                conv_bias=conv_bias,
                do_res=do_res_up_down,
                norm_type=norm_type,
                conv_op=conv_op,
            ))
            self.dec_blocks.append(nn.Sequential(*[MedNeXtBlock(
                in_channels=cur_channel * 2,
                out_channels=cur_channel * 2,
                exp_r=exp_r[i],
                kernel_size=kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                conv_op=conv_op,
                grn=grn
            ) for j in range(block_counts[i])]))
            self.up_blocks.append(MedNeXtUpBlock(
                in_channels=cur_channel * 4,
                out_channels=cur_channel * 2,
                exp_r=exp_r[i],
                kernel_size=kernel_size,
                stride=strides[i + 1],
                conv_bias=conv_bias,
                do_res=do_res_up_down,
                norm_type=norm_type,
                conv_op=conv_op,
                grn=grn
            ))
            self.out_blocks.append(OutBlock(in_channels=cur_channel * 2, n_classes=num_classes, conv_op=conv_op))
            cur_channel *= 2

        # 最底层。
        self.bottleneck = MedNeXtDownBlock(
                in_channels=cur_channel,
                out_channels=cur_channel * 2,
                exp_r=exp_r[-1],
                kernel_size=kernel_size,
                conv_kernel_size=kernel_sizes[-1],
                stride=strides[-1],
                conv_bias=conv_bias,
                do_res=do_res_up_down,
                norm_type=norm_type,
                conv_op=conv_op,
            )


    def forward(self, x):
        x = self.stem(x)
        down_res_list = []
        for i in range(self.n_stages - 1):
            x = self.enc_blocks[i](x)
            x = self.down_blocks[i](x)
            down_res_list.append(x)

        x = self.bottleneck(x)

        seg_outputs = []

        for i in range(self.n_stages - 2, -1, -1):
            up_res = self.up_blocks[i](x)
            res = up_res + down_res_list[i]
            x = self.dec_blocks[i](res)
            seg_outputs.append(self.out_blocks[i](x))
            del down_res_list[i], up_res

        seg_outputs = seg_outputs[::-1]

        if self.do_ds:
            return seg_outputs
        else:
            return seg_outputs[0]


if __name__ == "__main__":
    network = MyMedNeXt(
        input_channels=1,
        n_stages=6,
        n_channels=8,
        num_classes=2,
        kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        conv_bias=True,
        conv_op=nn.Conv3d,
        exp_r = 2,
        kernel_size=5,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        block_counts = [2,2,2,2,2,2,2],
        grn=False
    ).cuda()

    x = torch.zeros((2, 1, 20, 320, 256), requires_grad=False).cuda()

    # with torch.autocast(device_type='cuda', enabled=True):
    #     print(x.device)
    #     pred = network(x)
    #     for y in pred:
    #         print(y.size())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("count_parameters: ", count_parameters(network))
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table
    # x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x).total() / 1e9
    params = count_parameters(network) / 1e6
    print(f"GFLOPS: {flops:.2f}, 参数量: {params:.2f}M")

