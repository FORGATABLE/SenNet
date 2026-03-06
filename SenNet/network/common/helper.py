from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm


def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == nn.Conv1d:
        return 1
    elif conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))


def get_matching_pool_op(conv_op: Type[_ConvNd] = None,
                         dimension: int = None,
                         adaptive=False,
                         pool_type: str = 'avg') -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if dimension == 1:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool1d
            else:
                return nn.AvgPool1d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool1d
            else:
                return nn.MaxPool1d
    elif dimension == 2:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool2d
            else:
                return nn.AvgPool2d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool2d
            else:
                return nn.MaxPool2d
    elif dimension == 3:
        if pool_type == 'avg':
            if adaptive:
                return nn.AdaptiveAvgPool3d
            else:
                return nn.AvgPool3d
        elif pool_type == 'max':
            if adaptive:
                return nn.AdaptiveMaxPool3d
            else:
                return nn.MaxPool3d


def get_matching_interpolate(dimension: int = None) -> str:
    """
    :param dimension:
    """
    if dimension == 2:
        return 'bilinear'
    elif dimension == 3:
        return 'trilinear'
    raise ValueError("Only 2d and 3d supported")


def get_matching_instancenorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_InstanceNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    if dimension is not None:
        assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.InstanceNorm1d
    elif dimension == 2:
        return nn.InstanceNorm2d
    elif dimension == 3:
        return nn.InstanceNorm3d


def get_matching_convtransp(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.ConvTranspose1d
    elif dimension == 2:
        return nn.ConvTranspose2d
    elif dimension == 3:
        return nn.ConvTranspose3d


def get_matching_batchnorm(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_BatchNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return nn.BatchNorm1d
    elif dimension == 2:
        return nn.BatchNorm2d
    elif dimension == 3:
        return nn.BatchNorm3d


def get_matching_dropout(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_DropoutNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    '''
    PangTeen: 这是人写出来的逻辑？不要conv_op 直说好吗？
    '''
    if dimension == 1:
        return nn.Dropout
    elif dimension == 2:
        return nn.Dropout2d
    elif dimension == 3:
        return nn.Dropout3d


def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar


def get_default_network_config(dimension: int = 2,
                               nonlin: str = "ReLU",
                               norm_type: str = "bn") -> dict:
    """
    Use this to get a standard configuration. A network configuration looks like this:

    config = {'conv_op': torch.nn.modules.conv.Conv2d,
              'dropout_op': torch.nn.modules.dropout.Dropout2d,
              'norm_op': torch.nn.modules.batchnorm.BatchNorm2d,
              'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
              'nonlin': torch.nn.modules.activation.ReLU,
              'nonlin_kwargs': {'inplace': True}}

    There is no need to use get_default_network_config. You can create your own. Network configs are a convenient way of
    setting dimensionality, normalization and nonlinearity.

    :param dimension: integer denoting the dimension of the data. 1, 2 and 3 are accepted
    :param nonlin: string (ReLU or LeakyReLU)
    :param norm_type: string (bn=batch norm, in=instance norm)
    torch.nn.Module
    :return: dict
    """
    config = {}
    config['conv_op'] = convert_dim_to_conv_op(dimension)
    config['dropout_op'] = get_matching_dropout(dimension=dimension)
    if norm_type == "bn":
        config['norm_op'] = get_matching_batchnorm(dimension=dimension)
    elif norm_type == "in":
        config['norm_op'] = get_matching_instancenorm(dimension=dimension)

    config['norm_op_kwargs'] = None  # this will use defaults

    if nonlin == "LeakyReLU":
        config['nonlin'] = nn.LeakyReLU
        config['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        config['nonlin'] = nn.ReLU
        config['nonlin_kwargs'] = {'inplace': True}
    else:
        raise NotImplementedError('Unknown nonlin %s. Only "LeakyReLU" and "ReLU" are supported for now' % nonlin)

    return config


def channel_to_the_last(t):
    """
    调整张量的维度排列，适配二维 (B, C, H, W) 和三维 (B, C, H, W, D) 数据。
    对二维数据，变换为 (B, H, W, C)。
    对三维数据，变换为 (B, H, W, D, C)。

    Args:
        t (torch.Tensor): 输入张量，形状为 (B, C, H, W) 或 (B, C, H, W, D)。

    Returns:
        torch.Tensor: 调整维度后的张量。
    """
    if t.ndimension() == 4:  # 二维数据: B, C, H, W
        t = t.permute(0, 2, 3, 1).contiguous()  # 变换为 B, H, W, C
    elif t.ndimension() == 5:  # 三维数据: B, C, H, W, D
        t = t.permute(0, 2, 3, 4, 1).contiguous()  # 变换为 B, H, W, D, C
    else:
        raise ValueError(f"Unsupported tensor dimension: {t.ndimension()}D. Expected 4D or 5D tensor.")

    return t


def channel_to_the_second(t):
    """
    恢复张量的维度顺序，适配二维和三维数据。
    从 (B, H, W, C) 恢复为 (B, C, H, W)。
    从 (B, H, W, D, C) 恢复为 (B, C, H, W, D)。

    Args:
        t (torch.Tensor): 输入张量，形状为 (B, H, W, C) 或 (B, H, W, D, C)。

    Returns:
        torch.Tensor: 恢复维度后的张量。
    """
    if t.ndimension() == 4:  # 二维数据: B, H, W, C
        t = t.permute(0, 3, 1, 2).contiguous()  # 变换回 B, C, H, W
    elif t.ndimension() == 5:  # 三维数据: B, H, W, D, C
        t = t.permute(0, 4, 1, 2, 3).contiguous()  # 变换回 B, C, H, W, D
    else:
        raise ValueError(f"Unsupported tensor dimension: {t.ndimension()}D. Expected 4D or 5D tensor.")

    return t


def to_token(x):
    # B C X Y Z -> B C X*Y*Z
    size = x.size()[2:]
    return x.view(x.size(0), x.size(1), -1), size


def to_patch(x, size):
    # B C X*Y*Z -> B C X Y Z
    return x.view(x.size(0), x.size(1), *size)


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data
import torch.nn.functional as F
softmax_helper = lambda x: F.softmax(x, 1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
