from os.path import join, isfile
from typing import Union, List, Tuple, Optional
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from SenNet import config
from SenNet.network import cfg
from SenNet.network.common.helper import get_matching_dropout


class SenTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


    @staticmethod
    def update_network_args(arch_init_kwargs: dict,
                            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                            num_input_channels: int,
                            num_output_channels: int,
                            enable_deep_supervision: bool,
                            invalid_args: list[str] = None,
                            n_stages: Optional[int] = None,
                            cut_from_last: bool = True,
                            print_args: bool = False) -> dict:
        """
        构建模型的通用部分。
        """
        from pydoc import locate
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision

        architecture_kwargs['input_channels'] = num_input_channels
        architecture_kwargs['num_classes'] = num_output_channels

        if invalid_args:
            for i in invalid_args:
                architecture_kwargs.pop(i)

        if n_stages:
            architecture_kwargs = cfg.cut_stage(architecture_kwargs, n_stages, cut_from_last)

        if config.drop_out_rate:
            architecture_kwargs['dropout_op'] = get_matching_dropout(dimension=3)
            architecture_kwargs['dropout_op_kwargs'] = {'p': config.drop_out_rate}

        if print_args:
            print("Look ! It's architecture args : {}", architecture_kwargs)

        return architecture_kwargs

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        让模型选择变成继承制的，不用每次去改 Plan。
        """
        architecture_kwargs = SenTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision)

        network = PlainConvUNet(**architecture_kwargs)

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

