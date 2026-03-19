from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch._dynamo
from torch import nn
from SenNet.network.net.SFFDTM_Net import SFFDTMNet
from SenNet.network.losses.FDM_hybridLoss import FDMHybridLoss
from SenNet.trainer.trainers import SenTrainer
from SenNet.network.net.FDM_Net import FDMNet
from SenNet.network.net.SF_FDMNet import SFFDMNet
from SenNet.network.net.FDTM_Net import FDTMNet
torch._dynamo.config.suppress_errors = True


class FDMTrainer(SenTrainer):
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
        # self.enable_deep_supervision = os.environ.get("SENNET_FDM_DEEP_SUPERVISION", "0").lower() in ("1", "true", "yes")
        self.enable_deep_supervision = True
        self.initial_lr = 1e-4
        self.num_epochs = 150
        self.pretrained_enhancer_ckpt = os.environ.get(
            "SENNET_FDM_ENHANCER_CKPT",
            os.environ.get("SENNET_ENHANCER_CKPT", None),
        )
        try:
            self.freeze_enhancer_epochs = int(
                os.environ.get("SENNET_FDM_FREEZE_EPOCHS", os.environ.get("SENNET_FREEZE_ENHANCER_EPOCHS", "20"))
            )
        except Exception:
            self.freeze_enhancer_epochs = 10
        self._enhancer_ckpt_loaded = False
        self._enhancer_is_frozen = None

    def _do_i_compile(self):
        return False

    def _get_network_module(self) -> nn.Module:
        network = self.network.module if self.is_ddp else self.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        if os.environ.get("SENNET_FDM_DEEP_SUPERVISION", "0").lower() not in ("1", "true", "yes"):
            enabled = False
        super().set_deep_supervision_enabled(enabled)
        network = self._get_network_module()
        if hasattr(network, "deep_supervision"):
            network.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if FDMNet is None:
            raise ImportError("FDMNet is not available. Please ensure SenNet/network/net/FDM_Net.py exists.")
        architecture_kwargs = SenTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = FDMNet(**architecture_kwargs)
        if hasattr(network, "initialize"):
            network.apply(network.initialize)
        return network

    def _build_loss(self):
        deep_supervision_weights = None
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
            if len(weights) > 0:
                if self.is_ddp and not self._do_i_compile():
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0.0
                weights = weights / weights.sum()
                deep_supervision_weights = weights.tolist()

        boundary_mode = os.environ.get("SENNET_FDM_BOUNDARY_MODE", "traditional")
        lambda_boundary = float(os.environ.get("SENNET_FDM_LAMBDA_BOUNDARY", "0.2"))
        lambda_anatomy = float(os.environ.get("SENNET_FDM_LAMBDA_ANATOMY", "0.05"))

        boundary_class_weights = None
        raw_class_weights = os.environ.get("SENNET_FDM_BOUNDARY_CLASS_WEIGHTS", "").strip()
        if raw_class_weights:
            boundary_class_weights = [float(item.strip()) for item in raw_class_weights.split(",") if item.strip()]

        return FDMHybridLoss(
            deep_supervision=self.enable_deep_supervision,
            deep_supervision_weights=deep_supervision_weights,
            lambda_seg=1.0,
            lambda_boundary=lambda_boundary,
            lambda_anatomy=lambda_anatomy,
            boundary_mode=boundary_mode,
            boundary_class_weights=boundary_class_weights,
        )

    def initialize(self):
        super().initialize()
        if self.pretrained_enhancer_ckpt is not None and not self._enhancer_ckpt_loaded:
            self._load_pretrained_enhancer(self.pretrained_enhancer_ckpt)
            self._enhancer_ckpt_loaded = True

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in ["network_weights", "state_dict", "network", "model", "model_state_dict"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    def _load_pretrained_enhancer(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            self.print_to_log_file(f"[WARN] enhancer checkpoint not found: {ckpt_path}")
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[7:]
            if new_key.startswith("network."):
                new_key = new_key[8:]
            cleaned[new_key] = value

        network = self._get_network_module()
        target_module = getattr(network, "enhancer", None)
        if target_module is None:
            self.print_to_log_file("[WARN] no enhancer module found in FDMNet")
            return

        target_keys = set(target_module.state_dict().keys())
        sub_state_dict = {}
        for key, value in cleaned.items():
            if key.startswith("enhancer."):
                sub_key = key[len("enhancer.") :]
                if sub_key in target_keys:
                    sub_state_dict[sub_key] = value
            elif key in target_keys:
                sub_state_dict[key] = value

        incompatible = target_module.load_state_dict(sub_state_dict, strict=False)
        self.print_to_log_file(
            f"[INFO] loaded FDM enhancer weights from {ckpt_path} | "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )

    def _set_enhancer_trainability(self, requires_grad: bool):
        network = self._get_network_module()
        target_module = getattr(network, "enhancer", None)
        if target_module is None:
            return
        for parameter in target_module.parameters():
            parameter.requires_grad = requires_grad

    def _maybe_update_enhancer_freeze_state(self):
        should_freeze = self.current_epoch < self.freeze_enhancer_epochs
        if self._enhancer_is_frozen is None or self._enhancer_is_frozen != should_freeze:
            self._set_enhancer_trainability(not should_freeze)
            self._enhancer_is_frozen = should_freeze
            self.print_to_log_file(f"[INFO] enhancer frozen={should_freeze} at epoch={self.current_epoch}")

    def _move_target_to_device(self, target):
        if isinstance(target, list):
            return [item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in target]
        if isinstance(target, tuple):
            return tuple(item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in target)
        if torch.is_tensor(target):
            return target.to(self.device, non_blocking=True)
        return target

    def _get_main_target(self, target):
        if isinstance(target, (list, tuple)):
            return target[0]
        return target

    def _compute_hard_stats(
        self, seg_logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            pred = torch.argmax(seg_logits, dim=1)
            if target.ndim == pred.ndim + 1 and target.shape[1] == 1:
                target = target[:, 0]

            num_classes = seg_logits.shape[1]
            tp, fp, fn = [], [], []
            for class_idx in range(1, num_classes):
                pred_mask = pred == class_idx
                gt_mask = target == class_idx
                tp.append(torch.sum(pred_mask & gt_mask).detach().cpu().item())
                fp.append(torch.sum(pred_mask & (~gt_mask)).detach().cpu().item())
                fn.append(torch.sum((~pred_mask) & gt_mask).detach().cpu().item())
            return np.array(tp, dtype=np.float64), np.array(fp, dtype=np.float64), np.array(fn, dtype=np.float64)

    def train_step(self, batch: dict) -> dict:
        self._maybe_update_enhancer_freeze_state()

        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target_to_device(batch["target"])

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.network(data, return_aux=True)
            losses = self.loss(outputs, target, data)
            total_loss = losses["total"]

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        result = {"loss": float(total_loss.detach().cpu())}
        for key in ["seg", "bd", "anat"]:
            if key in losses:
                result[key] = float(losses[key].detach().cpu())
        return result

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target_to_device(batch["target"])

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.network(data, return_aux=True)
                losses = self.loss(outputs, target, data)

            seg_logits = outputs["seg"]
            if isinstance(seg_logits, list):
                seg_logits = seg_logits[0]
            main_target = self._get_main_target(target)
            tp_hard, fp_hard, fn_hard = self._compute_hard_stats(seg_logits, main_target)

        result = {
            "loss": float(losses["total"].detach().cpu()),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
        for key in ["seg", "bd", "anat"]:
            if key in losses:
                result[key] = float(losses[key].detach().cpu())
        return result
class SFFDMTrainer(FDMTrainer):
    """"
    璺宠穬杩炴帴鍔犲叆鐗瑰緛閫夋嫨妯″潡
    """
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
        self.initial_lr = 0.0005
        self.num_epochs = 200
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if SFFDMNet is None:
            raise ImportError('SFFDMNet is not available. Please ensure SenNet/network/net/SF_FDMNet.py exists.')
        architecture_kwargs = FDMTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = SFFDMNet(**architecture_kwargs)
        if hasattr(network, 'initialize'):
            network.apply(network.initialize)
        return network

class FDTMTrainer(FDMTrainer):

    """
        淇敼mamba妯″潡鍔犲叆GSC鍜孡ayerNorm
    """
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

class SFFDTMTrainer(FDTMTrainer):
    """
    FDTM + 鐗瑰緛閫夋嫨妯″潡
    """

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
        self.initial_lr = 0.0005
        self.num_epochs = 200
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if SFFDTMNet is None:
            raise ImportError('SFFDTMNet is not available. Please ensure SenNet/network/net/SFFDTM_Net.py exists.')
        architecture_kwargs = FDTMTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = SFFDTMNet(**architecture_kwargs)
        if hasattr(network, 'initialize'):
            network.apply(network.initialize)
        return network

if __name__ == "__main__":
    # trainer = FDMTrainer(plans={}, configuration="FDM", fold=0, dataset_json={})
    # trainer.initialize()
    """
    export SENNET_FDM_ENHANCER_CKPT=/path/to/checkpoint_best.pth
    export SENNET_FDM_FREEZE_EPOCHS=20
    nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr FDMTrainer
    """