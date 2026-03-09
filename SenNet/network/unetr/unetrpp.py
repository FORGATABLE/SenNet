import einops
from monai.networks.blocks import UnetResBlock, UnetOutBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from SenNet.network import cfg
from SenNet.network.common.segnetwork import SegmentationNetwork
from SenNet.network.network_analyzer import NetworkAnalyzer
from SenNet.network.net.norm_blocks import LayerNorm
from SenNet.network.net.transformer_blocks import TransformerBlock

from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer


class UnetrPPEncoder(nn.Module):
    def __init__(
        self,
        input_size=None,
        dims=None,
        proj_size=None,
        depths=None,
        num_heads=4,
        spatial_dims=3,
        in_channels=4,
        dropout=0.0,
        transformer_dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__()
        if input_size is None:
            input_size = [32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4]
        if dims is None:
            dims = [32, 64, 128, 256]
        if proj_size is None:
            proj_size = [64, 64, 64, 32]
        if depths is None:
            depths = [3, 3, 3, 3]

        self.downsample_layers = nn.ModuleList()
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=(4, 4, 4),
                stride=(4, 4, 4),
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        proj_size=proj_size[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=True,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.decoder_block = nn.ModuleList()
        if conv_decoder:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.1,
                        pos_embed=True,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    Reference: https://github.com/Amshaker/unetr_plus_plus
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        img_size: Union[Sequence[int], int] = (128, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds=False,
        **kwargs,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        if dims is None:
            dims = [32, 64, 128, 256]
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        img_size = tuple(int(i) for i in img_size)
        if len(img_size) != 3:
            raise ValueError(f"img_size must be 3D, got {img_size}")
        if any(i % 32 != 0 for i in img_size):
            raise ValueError(f"img_size must be divisible by 32 for UNETR_PP, got {img_size}")

        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = num_classes
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        stage_sizes = [
            tuple(i // 4 for i in img_size),
            tuple(i // 8 for i in img_size),
            tuple(i // 16 for i in img_size),
            tuple(i // 32 for i in img_size),
        ]
        self.feat_size = stage_sizes[-1]
        self.hidden_size = hidden_size

        encoder_input_size = [int(s[0] * s[1] * s[2]) for s in stage_sizes]
        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=encoder_input_size,
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            in_channels=input_channels,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=stage_sizes[2][0] * stage_sizes[2][1] * stage_sizes[2][2],
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=stage_sizes[1][0] * stage_sizes[1][1] * stage_sizes[1][2],
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=stage_sizes[0][0] * stage_sizes[0][1] * stage_sizes[0][2],
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=img_size[0] * img_size[1] * img_size[2],
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=num_classes)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=num_classes)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=num_classes)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


if __name__ == '__main__':
    model = UNETR_PP(**cfg.stage5_network_args).cuda()
    NetworkAnalyzer(model, print_flops=True, test_backward=True).analyze()
