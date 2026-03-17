from __future__ import annotations

import os
import sys
from time import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch._dynamo
from torch import nn
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.utilities.helpers import empty_cache

from SenNet.network.losses.FDM_hybridLoss import FDMEnhancementLoss
from SenNet.trainer.trainers import SenTrainer

try:
    from SenNet.network.net.FDM_enhancedPreNet import FDMEnhancedPreNet
except Exception:
    FDMEnhancedPreNet = None


torch._dynamo.config.suppress_errors = True


class FDMEnhancedPreTrainer(SenTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._best_val_loss = None
        self.num_epochs = 50
        self.initial_lr = 5e-3
        self.enable_deep_supervision = False
        self.enhancement_loss = FDMEnhancementLoss(
            lambda_rec=1.0,
            lambda_edge=0.2,
            lambda_freq=0.1,
            lambda_res=0.01,
        )

    def _do_i_compile(self):
        return False

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if FDMEnhancedPreNet is None:
            raise ImportError("FDMEnhancedPreNet is not available. Please ensure SenNet/network/net/FDM_enhancedPreNet.py exists.")
        architecture_kwargs = SenTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = FDMEnhancedPreNet(**architecture_kwargs)
        if hasattr(network, "initialize"):
            network.apply(network.initialize)
        return network

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        data = batch["data"]
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        data = data.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=(self.device.type == "cuda")):
            outputs = self.network(data, return_dict=True)
            loss_dict = self.enhancement_loss(outputs, data)
            loss = loss_dict["total"]

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "rec": float(loss_dict["rec"].detach().cpu()),
            "edge": float(loss_dict["edge"].detach().cpu()),
            "freq": float(loss_dict["freq"].detach().cpu()),
            "residual_reg": float(loss_dict["residual_reg"].detach().cpu()),
        }

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        data = batch["data"]
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.network(data, return_dict=True)
                loss_dict = self.enhancement_loss(outputs, data)
                loss = loss_dict["total"]

        return {
            "loss": float(loss.detach().cpu()),
            "rec": float(loss_dict["rec"].detach().cpu()),
            "edge": float(loss_dict["edge"].detach().cpu()),
            "freq": float(loss_dict["freq"].detach().cpu()),
            "residual_reg": float(loss_dict["residual_reg"].detach().cpu()),
        }

    def on_validation_epoch_end(self, val_outputs: List[Dict[str, float]]) -> None:
        if len(val_outputs) == 0:
            self.logger.log("val_losses", np.nan, self.current_epoch)
            self.print_to_log_file("Validation outputs empty!")
            return

        keys = val_outputs[0].keys()
        outputs_collated = {k: [o[k] for o in val_outputs] for k in keys}

        mean_loss = float(np.mean(outputs_collated["loss"]))
        mean_rec = float(np.mean(outputs_collated["rec"]))
        mean_edge = float(np.mean(outputs_collated["edge"]))
        mean_freq = float(np.mean(outputs_collated["freq"]))
        mean_residual_reg = float(np.mean(outputs_collated["residual_reg"]))

        self.logger.log("val_losses", mean_loss, self.current_epoch)
        self.print_to_log_file(
            f"Validation loss: {mean_loss:.6f} | "
            f"rec: {mean_rec:.6f} | "
            f"edge: {mean_edge:.6f} | "
            f"freq: {mean_freq:.6f} | "
            f"residual_reg: {mean_residual_reg:.6f}"
        )

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        train_loss = self.logger.my_fantastic_logging["train_losses"][-1]
        val_loss = self.logger.my_fantastic_logging["val_losses"][-1]

        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
        self.print_to_log_file("val_loss", np.round(val_loss, decimals=4))
        epoch_time = (
            self.logger.my_fantastic_logging["epoch_end_timestamps"][-1]
            - self.logger.my_fantastic_logging["epoch_start_timestamps"][-1]
        )
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        if self._best_val_loss is None or val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self.print_to_log_file(f"Yayy! New best validation loss: {np.round(self._best_val_loss, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def on_train_end(self):
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        old_stdout = sys.stdout
        with open(os.devnull, "w") as sink:
            sys.stdout = sink
            if self.dataloader_train is not None and isinstance(
                self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)
            ):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and isinstance(
                self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)
            ):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")
