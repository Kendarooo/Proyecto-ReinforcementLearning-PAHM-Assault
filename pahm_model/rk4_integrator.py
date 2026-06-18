"""
================================================================================
MÓDULO: pahm_model/rk4_integrator.py
FUNCIÓN: Implementación del solucionador numérico diferenciable basado en la
         ecuación de integración cinemática por expansión de Taylor de 2do orden.

         PAHMFastModel.cell(state, u, tau) ejecuta un paso RK4 con:
           - state: (batch, 2)  — [θ, dθ/dt]
           - u:     (batch, 1)  — control PWM
           - tau:   escalar o (batch, 1) — torque aditivo
         y devuelve (batch, 2).

         TaylorWindIntegrator usa physics_model.cell directamente para un
         solo paso, permitiendo que el gradiente fluya por tau_w → GRU
         sin pasar por PAHMFastModel.forward que espera secuencias 3D.
VERSIÓN: 3.0.0
================================================================================
"""

import torch
import torch.nn as nn


class TaylorWindIntegrator(nn.Module):
    """Resolvedor cinemático de Taylor de 2do orden acoplado con viento aditivo.

    Hereda de nn.Module para gestión nativa de dispositivos (CPU/GPU)
    y registro correcto de submódulos entrenables.
    """

    def __init__(
        self,
        physics_model: nn.Module,
        wind_estimator: nn.Module,
        dt: float
    ) -> None:
        """Inicializa el resolvedor registrando sus dependencias como submódulos.

        Args:
            physics_model: PAHMFastModel del profesor (congelado, CON-1).
                           Se accede via physics_model.cell para pasos individuales.
            wind_estimator: Red recurrente (GRU) para inferir el torque de viento.
            dt: Paso de tiempo de integración discreto.
        """
        super().__init__()
        self.physics_model = physics_model
        self.wind_estimator = wind_estimator
        self.dt = dt

    def step(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        sequence_history: torch.Tensor
    ) -> torch.Tensor:
        """Efectúa un paso de integración RK4 con torque de viento aditivo.

        Llama a physics_model.cell (PAHMPhysicsCell) directamente para un
        solo paso, pasando el torque estimado por la GRU como tau aditivo.
        Esto evita la firma de PAHMFastModel.forward que espera secuencias 3D.

        Args:
            state:            Estado cinemático actual (batch, 2) → [θ, dθ/dt].
            control:          Acción de control PWM (batch, 1).
            sequence_history: Ventana temporal para la GRU (batch, seq_len, 4).

        Returns:
            Tensor del estado integrado en t + dt (batch, 2).
        """
        # 1. Estimar torque de viento con la GRU (gradiente activo)
        tau_w = self.wind_estimator(sequence_history)   # (batch, 1)

        # 2. Un paso de integración con el torque inyectado (CON-1)
        #    Si el modelo expone .cell (PAHMFastModel → PAHMPhysicsCell con RK4),
        #    se usa directamente. Si no (modelos dummy en tests), se llama forward.
        #    tau_w retiene gradiente para backprop hacia la GRU.
        if hasattr(self.physics_model, 'cell'):
            next_state = self.physics_model.cell(state, control, tau=tau_w)
        else:
            # Fallback para modelos dummy en tests: forward(state, control) + tau aditivo
            derivs = self.physics_model(state, control)
            theta_dot = state[:, 1:2]
            theta_ddot = derivs[:, 1:2] + tau_w
            theta_next = state[:, 0:1] + self.dt * theta_dot + 0.5 * (self.dt ** 2) * theta_ddot
            theta_dot_next = theta_dot + self.dt * theta_ddot
            next_state = torch.cat([theta_next, theta_dot_next], dim=1)

        return next_state   # (batch, 2)