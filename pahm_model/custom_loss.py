"""
================================================================================
MÓDULO: pahm_model/custom_loss.py
FUNCIÓN: Implementación corregida de la función de pérdida regularizada.
================================================================================
"""

from typing import Tuple
import torch
import torch.nn.functional as F


class PAHMProjectLoss(torch.nn.Module):
    """Función de pérdida compuesta para la estimación de perturbaciones físicas."""

    def __init__(self, lambda_1: float, lambda_2: float) -> None:
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(
        self,
        theta_obs: torch.Tensor,
        theta_sim: torch.Tensor,
        tau_w_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_rec = F.mse_loss(theta_sim, theta_obs)
        loss_pars = torch.mean(tau_w_history ** 2)

        delta_tau_w = tau_w_history[:, 1:] - tau_w_history[:, :-1]
        loss_smooth = torch.mean(delta_tau_w ** 2)

        loss_total = loss_rec + (self.lambda_1 * loss_pars) + (self.lambda_2 * loss_smooth)
        return loss_total, loss_rec, loss_pars, loss_smooth