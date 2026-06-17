"""
================================================================================
MÓDULO: pahm_model/rk4_integrator.py
FUNCIÓN: Implementación de un solucionador numérico Runge-Kutta de 4to Orden (RK4)
         diferenciable, encargado de fusionar la física base del PAHM con la
         perturbación aditiva de viento estimada por la red neuronal.
VERSIÓN: 1.0.2
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

import torch
import torch.nn as nn


class RK4WindIntegrator:
    """Integrador numérico RK4 acoplado con estimación dinámica de viento.

    Aplica el principio de Inyección de Dependencias al requerir los modelos
    físicos y neuronales en su constructor.
    """

    def __init__(self, physics_model: nn.Module, wind_estimator: nn.Module, dt: float) -> None:
        """Inicializa el integrador con sus dependencias físicas y dinámicas.

        Args:
            physics_model: Modelo base preentrenado (Cumple CON-1).
            wind_estimator: Red recurrente (GRU/LSTM) para inferir el viento.
            dt: Paso de tiempo de integración discreto (Delta t).
        """
        self.physics_model = physics_model
        self.wind_estimator = wind_estimator
        self.dt = dt

    def _ode_dynamics(self, state: torch.Tensor, u: torch.Tensor, tau_w: torch.Tensor) -> torch.Tensor:
        """Calcula la derivada temporal total combinando física conocida y viento.

        Args:
            state: Tensor de estados (Batch, 2) -> [theta, theta_dot]
            u: Señal de control normalizada (Batch, 1) -> [PWM]
            tau_w: Torque de perturbación estimado aditivo (Batch, 1)

        Returns:
            Tensor de derivadas respecto al tiempo (Batch, 2)
        """
        with torch.set_grad_enabled(False):
            base_derivatives = self.physics_model(state, u)
            
        theta_dot_base = base_derivatives[:, 0:1]
        theta_ddot_base = base_derivatives[:, 1:2]

        # Inyección aditiva del torque de viento estimado (conserva gradiente)
        theta_ddot_corrected = theta_ddot_base + tau_w

        return torch.cat([theta_dot_base, theta_ddot_corrected], dim=1)

    def step(self, state: torch.Tensor, control: torch.Tensor, sequence_history: torch.Tensor) -> torch.Tensor:
        """Efectúa un paso completo de integración Runge-Kutta de 4to Orden.

        Args:
            state: Estado cinemático actual de dimensiones (Batch, 2).
            control: Acción de control PWM aplicada de dimensiones (Batch, 1).
            sequence_history: Ventana temporal requerida por la GRU (Batch, Seq, 4).

        Returns:
            Tensor del estado integrado en t + dt de dimensiones (Batch, 2).
        """
        tau_w = self.wind_estimator(sequence_history)

        k1 = self._ode_dynamics(state, control, tau_w)
        k2 = self._ode_dynamics(state + 0.5 * self.dt * k1, control, tau_w)
        k3 = self._ode_dynamics(state + 0.5 * self.dt * k2, control, tau_w)
        k4 = self._ode_dynamics(state + self.dt * k3, control, tau_w)

        next_state: torch.Tensor = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state