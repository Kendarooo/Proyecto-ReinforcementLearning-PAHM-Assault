"""
================================================================================
MÓDULO: pahm_model/sequence_estimator.py
FUNCIÓN: Implementación de la arquitectura de red neuronal recurrente (GRU)
         para la inferencia del torque de perturbación del viento a partir de
         ventanas de historial cinemático y de control.
VERSIÓN: 1.0.0
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

import torch
import torch.nn as nn


class WindSequenceEstimator(nn.Module):
    """Estimador de secuencias basado en GRU para inferencia de viento latente.

    Esta clase implementa una arquitectura recurrente limpia que respeta de
    forma estricta la restricción CON-2: solo procesa variables de estado y
    control directas, impidiendo el acceso al residuo calculado r(t).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        """Inicializa las capas recurrentes y de proyección lineal.

        Args:
            input_dim: Número de variables físicas de entrada por paso (fijo en 4).
            hidden_dim: Dimensión del estado oculto de la celda GRU.
            num_layers: Cantidad de capas GRU acopladas verticalmente.
        """
        super().__init__()
        
        # Capa recurrente principal. Procesa lotes con formato (Batch, Seq, Features)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Capa lineal de salida para estimar el escalar del torque adicional
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Realiza la propagación hacia adelante del estimador.

        Args:
            x: Tensor con dimensiones (Batch, Sequence_Length, Input_Dim).

        Returns:
            Tensor de dimensiones (Batch, 1) que representa el torque tau_w^hat.
        """
        # out: (Batch, Sequence_Length, Hidden_Dim)
        out, _ = self.gru(x)
        
        # Se extrae únicamente el estado del último paso temporal de la secuencia (L-1)
        last_step_output = out[:, -1, :]
        
        # Proyección final al escalar del torque estimado
        estimated_torque: torch.Tensor = self.regressor(last_step_output)
        
        return estimated_torque