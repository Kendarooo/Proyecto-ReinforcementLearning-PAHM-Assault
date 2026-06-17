"""
================================================================================
MÓDULO: pahm_model/custom_loss.py
FUNCIÓN: Implementación de la función de pérdida regularizada de tres términos
         según la Ecuación 13 del enunciado para el entrenamiento del estimador.
VERSIÓN: 1.0.2
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn


class PAHMProjectLoss(nn.Module):
    """Función de pérdida compuesta para la estimación de perturbaciones físicas.

    Cumple con el principio de Responsabilidad Única (SRP) al centralizar
    exclusivamente la lógica matemática y los criterios de optimización del viento.
    """

    def __init__(self, lambda_1: float, lambda_2: float) -> None:
        """Inicializa los coeficientes de regularización.

        Args:
            lambda_1: Coeficiente de parsimonia (Penaliza la norma del viento).
            lambda_2: Coeficiente de suavidad temporal (Penaliza cambios bruscos).
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mse = nn.MSELoss()

    def forward(
        self,
        theta_obs: torch.Tensor,
        theta_sim: torch.Tensor,
        tau_w_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcula la pérdida total combinada (Ecuación 13).

        Args:
            theta_obs: Ángulo real observado de dimensiones (Batch, 1).
            theta_sim: Ángulo simulado integrado por el RK4 de dimensiones (Batch, 1).
            tau_w_history: Historial de torques estimados en la ventana temporal
                           de dimensiones (Batch, Sequence_Length).

        Returns:
            Tuple conteniendo:
                - loss_total: Pérdida agregada final para el backward pass.
                - loss_rec: Término de reconstrucción cinemática pura.
                - loss_pars: Término de parsimonia pura.
                - loss_smooth: Término de suavidad temporal pura.
        """
        # 1. Error de reconstrucción entre la observación real y la simulación del RK4
        loss_rec = self.mse(theta_sim, theta_obs)

        # 2. Término de parsimonia: Penaliza la norma L2 de la última estimación
        current_tau_w = tau_w_history[:, -1]
        loss_pars = torch.mean(current_tau_w ** 2)

        # 3. Término de suavidad temporal: Diferencia consecutiva vectorizada
        # delta_tau_w = tau(t) - tau(t-1)
        delta_tau_w = tau_w_history[:, 1:] - tau_w_history[:, :-1]
        loss_smooth = torch.mean(delta_tau_w ** 2)

        # Agregación ponderada final basada en los parámetros externos de config.json
        loss_total = loss_rec + (self.lambda_1 * loss_pars) + (self.lambda_2 * loss_smooth)

        return loss_total, loss_rec, loss_pars, loss_smooth