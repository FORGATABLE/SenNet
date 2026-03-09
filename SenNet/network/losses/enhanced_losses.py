from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from SenNet.network.common.edge_utils import high_frequency_map, laplace_response_3d, probability_boundary_map


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()
        prob = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        dims = tuple(range(2, prob.ndim))
        intersection = torch.sum(prob * target_onehot, dim=dims)
        denom = torch.sum(prob + target_onehot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()
        ce = F.cross_entropy(logits, target)
        dice = self.dice(logits, target)
        return self.dice_weight * dice + self.ce_weight * ce


class BoundaryConsistencyLoss(nn.Module):
    def forward(self, raw_logits: torch.Tensor, enh_logits: torch.Tensor) -> torch.Tensor:
        raw_prob = torch.softmax(raw_logits, dim=1)
        enh_prob = torch.softmax(enh_logits, dim=1)
        raw_boundary = probability_boundary_map(raw_prob)
        enh_boundary = probability_boundary_map(enh_prob)
        return F.l1_loss(raw_boundary, enh_boundary)


class ImageEnhancementConsistencyLoss(nn.Module):
    def forward(self, raw_img: torch.Tensor, enhanced_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        rec = F.l1_loss(enhanced_img, raw_img)
        edge = F.l1_loss(torch.abs(laplace_response_3d(enhanced_img)), torch.abs(laplace_response_3d(raw_img)))
        freq = F.l1_loss(high_frequency_map(enhanced_img), high_frequency_map(raw_img))
        return {"rec": rec, "img_edge": edge, "freq": freq}


class EnhancedHybridLoss(nn.Module):
    def __init__(
        self,
        deep_supervision: bool = True,
        deep_supervision_weights: Sequence[float] | None = None,
        lambda_seg: float = 1.0,
        lambda_edge_cons: float = 0.1,
        lambda_rec: float = 0.2,
        lambda_img_edge: float = 0.05,
        lambda_freq: float = 0.05,
        lambda_bd: float = 0.0,
        lambda_anat: float = 0.0,
    ) -> None:
        super().__init__()
        self.seg_loss = DiceCELoss()
        self.boundary_cons = BoundaryConsistencyLoss()
        self.image_cons = ImageEnhancementConsistencyLoss()
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = list(deep_supervision_weights) if deep_supervision_weights is not None else None
        self.lambda_seg = lambda_seg
        self.lambda_edge_cons = lambda_edge_cons
        self.lambda_rec = lambda_rec
        self.lambda_img_edge = lambda_img_edge
        self.lambda_freq = lambda_freq
        self.lambda_bd = lambda_bd
        self.lambda_anat = lambda_anat

    def _target_for_level(self, target: Union[List[torch.Tensor], torch.Tensor], i: int) -> torch.Tensor:
        if isinstance(target, list):
            return target[i]
        return target

    def _seg_loss(self, seg_output: Union[List[torch.Tensor], torch.Tensor], target: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(seg_output, list):
            if self.deep_supervision_weights is None:
                weights = [1 / (2 ** i) for i in range(len(seg_output))]
                s = sum(weights)
                weights = [w / s for w in weights]
            else:
                weights = list(self.deep_supervision_weights)
            total = seg_output[0].new_tensor(0.0)
            for i, pred in enumerate(seg_output):
                total = total + weights[i] * self.seg_loss(pred, self._target_for_level(target, i))
            return total
        return self.seg_loss(seg_output, self._target_for_level(target, 0))

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Union[List[torch.Tensor], torch.Tensor],
        raw_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        losses["seg"] = self._seg_loss(model_output["seg"], target)

        if "raw_aux" in model_output and "enh_aux" in model_output:
            losses["edge_cons"] = self.boundary_cons(model_output["raw_aux"], model_output["enh_aux"])
        else:
            losses["edge_cons"] = losses["seg"].new_tensor(0.0)

        img_losses = self.image_cons(raw_img, model_output["enhanced"])
        losses["rec"] = img_losses["rec"]
        losses["img_edge"] = img_losses["img_edge"]
        losses["freq"] = img_losses["freq"]

        losses["bd"] = losses["seg"].new_tensor(0.0)
        losses["anat"] = losses["seg"].new_tensor(0.0)

        losses["total"] = (
            self.lambda_seg * losses["seg"]
            + self.lambda_edge_cons * losses["edge_cons"]
            + self.lambda_rec * losses["rec"]
            + self.lambda_img_edge * losses["img_edge"]
            + self.lambda_freq * losses["freq"]
            + self.lambda_bd * losses["bd"]
            + self.lambda_anat * losses["anat"]
        )
        losses["loss"] = losses["total"]
        return losses
