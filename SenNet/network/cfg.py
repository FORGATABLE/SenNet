from torch import nn

from SenNet.network.common import helper

default_network_args = {
    'input_channels': 1,
    'n_stages': 6,
    'features_per_stage': [32, 64, 128, 256, 320, 320],
    'conv_op': nn.Conv3d,
    'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
    'num_classes': 4,
    'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
    'conv_bias': True,
    'norm_op': helper.get_matching_instancenorm(dimension=3),
    'norm_op_kwargs': {},
    'nonlin': nn.LeakyReLU,
    'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
    'deep_supervision': True,
}

def cut_stage(args, n_stages, cut_from_last=True):
    # 复制一份参数，然后修改 n_stages 的值。
    network_args = args.copy()
    network_args['n_stages'] = n_stages
    if cut_from_last:
        network_args['features_per_stage'] = network_args['features_per_stage'][:n_stages]
        network_args['kernel_sizes'] = network_args['kernel_sizes'][:n_stages]
        network_args['strides'] = network_args['strides'][:n_stages]
        network_args['n_conv_per_stage'] = network_args['n_conv_per_stage'][:n_stages]
        network_args['n_conv_per_stage_decoder'] = network_args['n_conv_per_stage_decoder'][:n_stages - 1]
    else:
        network_args['features_per_stage'] = network_args['features_per_stage'][-n_stages:]
        network_args['kernel_sizes'] = network_args['kernel_sizes'][-n_stages:]
        network_args['strides'] = network_args['strides'][-n_stages:]
        network_args['n_conv_per_stage'] = network_args['n_conv_per_stage'][-n_stages:]
        network_args['n_conv_per_stage_decoder'] = network_args['n_conv_per_stage_decoder'][-n_stages:]
    return network_args