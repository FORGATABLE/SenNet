from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import nn

from SenNet.network.common.edge_utils import high_frequency_map, laplace_response_3d


def _to_main_target(target: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(target, (list, tuple)):
        return target[0]
    return target


def _prepare_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        target = target[:, 0]
    return target.long()


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def _boundary_map(x: torch.Tensor) -> torch.Tensor:
    max_map = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    min_map = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
    return (max_map - min_map).clamp_min(0.0)


def _signed_distance_map(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    target_np = target.detach().cpu().numpy()
    sdf = np.zeros((target_np.shape[0], num_classes, *target_np.shape[1:]), dtype=np.float32)
    for batch_idx in range(target_np.shape[0]):
        for class_idx in range(1, num_classes):
            fg_mask = target_np[batch_idx] == class_idx
            if not np.any(fg_mask):
                continue
            sdf[batch_idx, class_idx] = distance_transform_edt(~fg_mask) - distance_transform_edt(fg_mask)
    return torch.from_numpy(sdf).to(device=target.device, dtype=torch.float32)


class FDMEnhancementLoss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_edge: float = 0.2,
        lambda_freq: float = 0.1,
        lambda_res: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq
        self.lambda_res = lambda_res

    def forward(self, model_output: Dict[str, torch.Tensor], raw_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        enhanced = model_output["enhanced"]
        residual = model_output["residual"]

        rec = F.l1_loss(enhanced, raw_img)
        edge = F.l1_loss(torch.abs(laplace_response_3d(enhanced)), torch.abs(laplace_response_3d(raw_img)))
        freq = F.l1_loss(high_frequency_map(enhanced), high_frequency_map(raw_img))
        residual_reg = torch.mean(torch.abs(residual))

        total = (
            self.lambda_rec * rec
            + self.lambda_edge * edge
            + self.lambda_freq * freq
            + self.lambda_res * residual_reg
        )
        return {
            "rec": rec,
            "edge": edge,
            "freq": freq,
            "residual_reg": residual_reg,
            "total": total,
            "loss": total,
        }


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        probs = torch.softmax(logits, dim=1)
        target_onehot = _one_hot(target, logits.shape[1])
        dims = tuple(range(2, logits.ndim))
        intersection = torch.sum(probs * target_onehot, dim=dims)
        denominator = torch.sum(probs + target_onehot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        ce = F.cross_entropy(logits, target)
        dice = self.dice(logits, target)
        return self.dice_weight * dice + self.ce_weight * ce


class BoundaryLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        probs = torch.softmax(logits, dim=1)
        target_onehot = _one_hot(target, logits.shape[1])

        pred_boundary = _boundary_map(probs[:, 1:])
        target_boundary = _boundary_map(target_onehot[:, 1:])
        return F.binary_cross_entropy(pred_boundary.clamp(1e-6, 1 - 1e-6), target_boundary)


class AnatomicalDistanceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _prepare_target(target, logits)
        probs = torch.softmax(logits, dim=1)
        target_onehot = _one_hot(target, logits.shape[1])
        sdf = _signed_distance_map(target, logits.shape[1])[:, 1:]
        sdf = sdf.abs()
        if sdf.numel() > 0:
            max_per_map = torch.amax(sdf, dim=tuple(range(2, sdf.ndim)), keepdim=True)
            sdf = sdf / (max_per_map + 1e-6)
        return torch.mean(torch.abs(probs[:, 1:] - target_onehot[:, 1:]) * (1.0 + sdf))


class FDMHybridLoss(nn.Module):
    def __init__(
        self,
        deep_supervision: bool = True,
        deep_supervision_weights: Sequence[float] | None = None,
        lambda_seg: float = 1.0,
        lambda_boundary: float = 0.2,
        lambda_anatomy: float = 0.1,
    ) -> None:
        super().__init__()
        self.seg_loss = DiceCELoss()
        self.boundary_loss = BoundaryLoss()
        self.anatomy_loss = AnatomicalDistanceLoss()
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = list(deep_supervision_weights) if deep_supervision_weights is not None else None
        self.lambda_seg = lambda_seg
        self.lambda_boundary = lambda_boundary
        self.lambda_anatomy = lambda_anatomy

    def _target_for_level(self, target: Union[List[torch.Tensor], torch.Tensor], level: int) -> torch.Tensor:
        if isinstance(target, (list, tuple)):
            return target[level]
        return target

    def _segmentation_loss(
        self,
        seg_output: Union[List[torch.Tensor], torch.Tensor],
        target: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(seg_output, list):
            return self.seg_loss(seg_output, self._target_for_level(target, 0))

        if self.deep_supervision_weights is None:
            weights = [1 / (2 ** i) for i in range(len(seg_output))]
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]
        else:
            weights = self.deep_supervision_weights

        total = seg_output[0].new_tensor(0.0)
        for level, pred in enumerate(seg_output):
            total = total + weights[level] * self.seg_loss(pred, self._target_for_level(target, level))
        return total

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Union[List[torch.Tensor], torch.Tensor],
        raw_img: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        seg_output = model_output["seg"]
        main_logits = seg_output[0] if isinstance(seg_output, list) else seg_output
        main_target = _to_main_target(target)

        seg = self._segmentation_loss(seg_output, target)
        bd = self.boundary_loss(main_logits, main_target)
        anat = self.anatomy_loss(main_logits, main_target)

        total = self.lambda_seg * seg + self.lambda_boundary * bd + self.lambda_anatomy * anat
        return {
            "seg": seg,
            "bd": bd,
            "anat": anat,
            "total": total,
            "loss": total,
        }
