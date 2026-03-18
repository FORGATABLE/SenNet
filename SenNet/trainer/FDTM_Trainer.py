from __future__ import annotations

from typing import List, Tuple, Union

from torch import nn

from SenNet.network.net.FDTM_Net import FDTMNet
from SenNet.trainer.FDM_Trainer import FDMTrainer


class FDTMTrainer(FDMTrainer):
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if FDTMNet is None:
            raise ImportError('FDTMNet is not available. Please ensure SenNet/network/net/FDTM_Net.py exists.')
        architecture_kwargs = FDMTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = FDTMNet(**architecture_kwargs)
        if hasattr(network, 'initialize'):
            network.apply(network.initialize)
        return network