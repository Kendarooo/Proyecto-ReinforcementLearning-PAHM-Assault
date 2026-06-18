"""
================================================================================
MÓDULO: pahm_model/sequence_estimator.py
FUNCIÓN: Red neuronal recurrente basada en GRU para inferir el torque latente
         del viento a partir de ventanas de historia cinemática.
         La salida usa tanh escalado para limitar τ_w a un rango físicamente
         plausible y evitar torques explosivos durante el entrenamiento.
VERSIÓN: 3.0.0
================================================================================
"""

import torch
import torch.nn as nn


class WindSequenceEstimator(nn.Module):
    """Estimador de secuencias recurrente para perturbaciones aditivas (CON-2).

    La salida está acotada por tanh(x) * tau_max para garantizar que el
    torque estimado sea físicamente plausible y no explote durante el
    entrenamiento. tau_max se calibra según el rango de los residuos
    observados en FR-3 (~0.94 en escala normalizada).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        tau_max: float = 2.0
    ) -> None:
        """Inicializa la arquitectura GRU con salida acotada.

        Args:
            input_dim:  Número de features de entrada (fijo en 4: sin θ, cos θ, dθ/dt, u).
            hidden_dim: Dimensión del estado oculto de la GRU.
            num_layers: Número de capas GRU apiladas.
            tau_max:    Límite absoluto del torque estimado. La salida se acota
                        en [-tau_max, +tau_max] via tanh escalado.
        """
        super().__init__()
        self.input_dim = input_dim
        self.tau_max   = tau_max

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Infiere el torque escalar del viento acotado en [-tau_max, tau_max].

        Args:
            x: Tensor (batch, seq_len, 4) con [sin θ, cos θ, dθ/dt, u].

        Returns:
            Tensor (batch, 1) con τ̂_w ∈ [-tau_max, tau_max].
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Dimensión de entrada incorrecta: esperado {self.input_dim}, "
                f"recibido {x.shape[-1]}. Verificar que la ventana contenga "
                f"(sin_theta, cos_theta, theta_dot, u)."
            )

        out, _ = self.gru(x)                    # (batch, seq_len, hidden_dim)
        last_step = out[:, -1, :]               # (batch, hidden_dim)
        raw = self.fc(last_step)                # (batch, 1) — sin acotamiento
        return torch.tanh(raw) * self.tau_max   # (batch, 1) ∈ [-tau_max, tau_max]