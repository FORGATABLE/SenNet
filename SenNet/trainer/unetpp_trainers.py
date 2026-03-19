import pydoc
from typing import Union, List, Tuple

import torch
from torch import nn

from SenNet.network.unetpp.unetpp import BasicUNetPlusPlus
from SenNet.trainer.trainers import SenTrainer

class UNetPPTrainer(SenTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_epochs = 300
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = SenTrainer.update_network_args(arch_init_kwargs, arch_init_kwargs_req_import,
                                                            num_input_channels, num_output_channels,
                                                            enable_deep_supervision,
                                                            print_args=True)

        # if 'n_stages' in architecture_kwargs:
        #     architecture_kwargs.pop('n_stages')
        # architecture_kwargs.pop('features_per_stage')
        # architecture_kwargs.pop('n_conv_per_stage')
        # architecture_kwargs.pop('n_conv_per_stage_decoder')
        # architecture_kwargs.pop('norm_op')
        # architecture_kwargs.pop('norm_op_kwargs')
        # architecture_kwargs.pop('dropout_op')
        # architecture_kwargs.pop('dropout_op_kwargs')
        # architecture_kwargs.pop('nonlin')
        # architecture_kwargs.pop('nonlin_kwargs')
        network = BasicUNetPlusPlus(
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
