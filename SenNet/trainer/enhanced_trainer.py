from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch._dynamo
from torch import nn

from SenNet.network.losses.enhanced_losses import EnhancedHybridLoss
from SenNet.network.net.enhanced_net import EnhancedSegNet
from SenNet.trainer.trainers import SenTrainer


torch._dynamo.config.suppress_errors = True
os.environ["SENNET_FREEZE_ENHANCER_EPOCHS"] = "20"
os.environ["SENNET_ENHANCER_CKPT"] = ("/mnt/data4/zr/nnUNet_Datasets/nnUNet_results/Dataset112_MaskedFullAndLocalBLL_171/"
                                      "EnhancementPretrainTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
class EnhancedTrainer(SenTrainer):
    """
    第四步：增强模块 + 分割模块联合训练
    环境变量:
      SENNET_ENHANCER_CKPT=/path/to/step3_checkpoint_best.pth
      SENNET_FREEZE_ENHANCER_EPOCHS=20
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda:1"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True
        self.num_epochs = 150
        self.pretrained_enhancer_ckpt = os.environ.get("SENNET_ENHANCER_CKPT", None)
        try:
            self.freeze_enhancer_epochs = int(os.environ.get("SENNET_FREEZE_ENHANCER_EPOCHS", "0"))
        except Exception:
            self.freeze_enhancer_epochs = 0
        self._enhancer_ckpt_loaded = False
        self._enhancer_is_frozen = None

    def _do_i_compile(self):
        return False
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        nnUNetTrainer 默认仅为 U-Net 风格网络设置 `decoder.deep_supervision`。
        EnhancedSegNet 直接读取 `self.deep_supervision`，因此这里需要同步两者，
        否则推理阶段可能仍输出 deep supervision 列表。
        """
        super().set_deep_supervision_enabled(enabled)
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        if hasattr(mod, "deep_supervision"):
            mod.deep_supervision = enabled

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        nnUNetTrainer 默认仅为 U-Net 风格网络设置 `decoder.deep_supervision`。
        EnhancedSegNet 直接读取 `self.deep_supervision`，因此这里需要同步两者，
        否则推理阶段可能仍输出 deep supervision 列表。
        """
        super().set_deep_supervision_enabled(enabled)
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        if hasattr(mod, "deep_supervision"):
            mod.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        architecture_kwargs = SenTrainer.update_network_args(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            print_args=True,
        )
        network = EnhancedSegNet(**architecture_kwargs)
        if hasattr(network, "initialize"):
            network.apply(network.initialize)
        return network

    def _build_loss(self):
        return EnhancedHybridLoss(deep_supervision=self.enable_deep_supervision)

    def initialize(self):
        super().initialize()
        if self.pretrained_enhancer_ckpt is not None and (not self._enhancer_ckpt_loaded):
            self._load_pretrained_enhancer(self.pretrained_enhancer_ckpt)
            self._enhancer_ckpt_loaded = True

    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for k in ["network_weights", "state_dict", "network", "model", "model_state_dict"]:
                if k in checkpoint and isinstance(checkpoint[k], dict):
                    return checkpoint[k]
        return checkpoint

    def _load_pretrained_enhancer(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            self.print_to_log_file(f"[WARN] enhancer ckpt not found: {ckpt_path}")
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = self._extract_state_dict(checkpoint)

        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[7:]
            if nk.startswith("network."):
                nk = nk[8:]
            cleaned[nk] = v

        target_module = None
        if hasattr(self.network, "enhancer"):
            target_module = self.network.enhancer
        elif hasattr(self.network, "enhancement_module"):
            target_module = self.network.enhancement_module

        if target_module is None:
            self.print_to_log_file("[WARN] no enhancer module found in EnhancedSegNet, skip loading enhancer ckpt")
            return

        target_keys = set(target_module.state_dict().keys())
        sub_cleaned = {}
        for k, v in cleaned.items():
            if k.startswith("enhancer."):
                kk = k[len("enhancer."):]
                if kk in target_keys:
                    sub_cleaned[kk] = v
            elif k in target_keys:
                sub_cleaned[k] = v

        incompatible = target_module.load_state_dict(sub_cleaned, strict=False)
        self.print_to_log_file(
            f"[INFO] loaded pretrained enhancer from {ckpt_path} | "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    def _set_enhancer_trainability(self, requires_grad: bool):
        target_module = None
        if hasattr(self.network, "enhancer"):
            target_module = self.network.enhancer
        elif hasattr(self.network, "enhancement_module"):
            target_module = self.network.enhancement_module
        if target_module is None:
            return
        for p in target_module.parameters():
            p.requires_grad = requires_grad

    def _maybe_update_enhancer_freeze_state(self):
        should_freeze = self.current_epoch < self.freeze_enhancer_epochs
        if self._enhancer_is_frozen is None or self._enhancer_is_frozen != should_freeze:
            self._set_enhancer_trainability(not should_freeze)
            self._enhancer_is_frozen = should_freeze
            self.print_to_log_file(f"[INFO] enhancer frozen={should_freeze} at epoch={self.current_epoch}")

    def _move_target_to_device(self, target):
        if isinstance(target, list):
            return [t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t for t in target]
        if isinstance(target, tuple):
            return tuple(t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t for t in target)
        if torch.is_tensor(target):
            return target.to(self.device, non_blocking=True)
        return target

    def _get_main_target(self, target):
        if isinstance(target, (list, tuple)):
            return target[0]
        return target

    def _compute_hard_stats(self, seg_logits: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            pred = torch.argmax(seg_logits, dim=1)
            if target.ndim == pred.ndim + 1 and target.shape[1] == 1:
                target = target[:, 0]

            num_classes = seg_logits.shape[1]
            tp, fp, fn = [], [], []
            for c in range(1, num_classes):
                pred_c = pred == c
                gt_c = target == c
                tp.append(torch.sum(pred_c & gt_c).detach().cpu().item())
                fp.append(torch.sum(pred_c & (~gt_c)).detach().cpu().item())
                fn.append(torch.sum((~pred_c) & gt_c).detach().cpu().item())
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
        for k in ["seg", "rec", "img_edge", "freq", "edge_cons", "bd", "anat"]:
            if k in losses:
                result[k] = float(losses[k].detach().cpu())
        return result

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._move_target_to_device(batch["target"])

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.network(data, return_aux=True)
                losses = self.loss(outputs, target, data)

            seg_logits = outputs["seg"] if isinstance(outputs, dict) else outputs
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
        for k in ["seg", "rec", "img_edge", "freq", "edge_cons", "bd", "anat"]:
            if k in losses:
                result[k] = float(losses[k].detach().cpu())
        return result
