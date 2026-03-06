import torch
from mamba_ssm import Mamba, Mamba2
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class MambaLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
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


class MambaBlock(nn.Module):
    """
    SegMamba 使用的模块，包含 Mamba 模块和 LayerNorm。实现多尺度和全局特征建模，同时在训练和推理过程中保持高效率。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_v2=False):
        super().__init__()
        self.dim = dim
        self.norm = MambaLayerNorm(dim)
        if use_v2:
            self.mamba = Mamba2(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
            )
        else:
            self.mamba = Mamba(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
            )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out


class MambaLayer(nn.Module):
    """
    UMamba 使用的模块，添加v2版本的Mamba模块，支持更多的参数。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False, use_v2=False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        if use_v2:
            self.mamba = Mamba2(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
                headdim=8  # dim * expand / headdim 必须是 8 的倍数！
            )
        else:
            self.mamba = Mamba(
                d_model=dim,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
            )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out