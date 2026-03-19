import pydoc
from typing import Union, List, Tuple

import torch
from torch import nn

from SenNet import config
from SenNet.network.swin_unetr.swin_unetr import SwinUNETR
from SenNet.trainer.trainers import SenTrainer


class SwinUNETRTrainer(SenTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p default -tr SwinUNETRTrainer -num_gpus 1 > main.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False

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

        network = SwinUNETR(
            img_size=config.patch_size,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class SwinUNETRV2Trainer(SenTrainer):
    """
    CUDA_VISIBLE_DEVICES=2 nohup python -u pangteen/train.py 201 3d_fullres 3 -p default -tr SwinUNETRV2Trainer -num_gpus 1 > main.out 2>&1 &
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

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

        network = SwinUNETR(
            img_size=config.patch_size,
            use_v2=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network
