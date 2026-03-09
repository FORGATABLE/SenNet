from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from SenNet.network.common.edge_utils import high_frequency_map, laplace_response_3d


class EnhancementPretrainLoss(nn.Module):
    """
    第三阶段增强模块预训练损失。

    目标：
    1. 输出增强图不能偏离原图过远；
    2. 保持主要边界位置稳定；
    3. 保持高频细节，避免过度平滑；
    4. 通过残差正则避免网络输出过大的无意义修正。
    """

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
