"""
================================================================================
MÓDULO: test/test_rk4_integrator.py
FUNCIÓN: Suite de pruebas unitarias para validar dimensiones, diferenciabilidad
         y el comportamiento físico de consistencia de la ODE (NFR-6b).
VERSIÓN: 2.1.0 (Auditado por Senior - Coherencia Física Inyectada)
================================================================================
"""

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pahm_model.rk4_integrator import TaylorWindIntegrator  # noqa: E402


class DummyModel(torch.nn.Module):
    """Modelo físico base dummy con parámetros fijos."""
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 1.5

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]
        d_theta = theta_dot
        d_theta_dot = self.alpha * (u ** 2) - self.beta * theta_dot - self.gamma * torch.sin(theta)
        return torch.cat([d_theta, d_theta_dot], dim=1)


class DummyEstimator(torch.nn.Module):
    """Estimador recurrente dummy."""
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([0.2]))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]
        return self.param.expand(batch_size, 1)


class ZeroWindEstimator(torch.nn.Module):
    """Estimador nulo para validación de consistencia cinemática pura."""
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]
        return torch.zeros((batch_size, 1), device=history.device)


def test_rk4_step_dimensions_and_differentiability() -> None:
    """Verifica que el paso de integración de Taylor conserve dimensiones y gradientes."""
    batch_size = 2
    dt = 0.02
    state = torch.tensor([[0.5, 0.1], [-0.5, -0.1]], dtype=torch.float32)
    u = torch.tensor([[0.8], [0.4]], dtype=torch.float32)
    dummy_history = torch.zeros((batch_size, 10, 4), dtype=torch.float32)

    model = DummyModel()
    estimator = DummyEstimator()
    integrator = TaylorWindIntegrator(physics_model=model, wind_estimator=estimator, dt=dt)

    next_state = integrator.step(state=state, control=u, sequence_history=dummy_history)

    assert next_state.shape == (batch_size, 2)

    loss = next_state.sum()
    loss.backward()

    assert estimator.param.grad is not None
    assert estimator.param.grad.item() != 0.0


def test_taylor_integration_physics_consistency_nfr6b() -> None:
    """Valida (NFR-6b) que con tau_w = 0 el resultado coincida con la ecuación pura."""
    dt = 0.02
    state = torch.tensor([[0.5, 0.1]], dtype=torch.float32)
    u = torch.tensor([[0.8]], dtype=torch.float32)
    dummy_history = torch.zeros((1, 10, 4), dtype=torch.float32)

    model = DummyModel()
    zero_estimator = ZeroWindEstimator()
    integrator = TaylorWindIntegrator(physics_model=model, wind_estimator=zero_estimator, dt=dt)

    # Simulación a través de la clase modular
    next_state_sim = integrator.step(state=state, control=u, sequence_history=dummy_history)

    # Cálculo analítico directo de Taylor esperado en el test
    derivs = model(state, u)
    t_dot = state[:, 1:2]
    t_ddot = derivs[:, 1:2]
    
    expected_theta = state[:, 0:1] + dt * t_dot + 0.5 * (dt ** 2) * t_ddot
    expected_theta_dot = t_dot + dt * t_ddot

    # Afirmar consistencia matemática exacta del comportamiento físico
    assert torch.allclose(next_state_sim[:, 0:1], expected_theta)
    assert torch.allclose(next_state_sim[:, 1:2], expected_theta_dot)
