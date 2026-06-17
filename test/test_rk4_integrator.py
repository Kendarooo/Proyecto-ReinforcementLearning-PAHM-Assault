"""
================================================================================
MÓDULO: test/test_rk4_integrator.py
FUNCIÓN: Suite de pruebas unitarias (TDD) para validar el comportamiento físico
         y la diferenciabilidad del integrador numérico RK4 utilizando inyección
         robusta de rutas en sys.path.
VERSIÓN: 1.0.3
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

import os
import sys
import pytest
import torch

# EXPLICACIÓN TÉCNICA: Calculamos dinámicamente la raíz del proyecto para meterla
# en el path de búsqueda, permitiendo resolver "pahm_model.rk4_integrator" sin importar
# desde dónde se invoque el comando pytest.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class DummyModel(torch.nn.Module):
    """Modelo físico base dummy con parámetros fijos."""
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 1.5

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Calcula la derivada del estado sin viento."""
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]
        d_theta = theta_dot
        d_theta_dot = self.alpha * (u ** 2) - self.beta * theta_dot - self.gamma * torch.sin(theta)
        return torch.cat([d_theta, d_theta_dot], dim=1)


class DummyEstimator(torch.nn.Module):
    """Estimador recurrente dummy que retorna un valor con gradiente requerido."""
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([0.2]))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Retorna un torque proporcional a un parámetro entrenable."""
        batch_size = history.shape[0]
        return self.param.expand(batch_size, 1)


def test_rk4_step_dimensions_and_differentiability() -> None:
    """Verifica que el paso de integración RK4 conserve dimensiones y gradientes."""
    # Ahora la resolución por paquete absoluto funcionará de forma garantizada
    from pahm_model.rk4_integrator import RK4WindIntegrator

    # Arreglar (Arrange)
    batch_size = 2
    dt = 0.02
    state = torch.tensor([[0.5, 0.1], [-0.5, -0.1]], dtype=torch.float32)
    u = torch.tensor([[0.8], [0.4]], dtype=torch.float32)
    dummy_history = torch.zeros((batch_size, 10, 4), dtype=torch.float32)

    model = DummyModel()
    estimator = DummyEstimator()
    integrator = RK4WindIntegrator(physics_model=model, wind_estimator=estimator, dt=dt)

    # Actuar (Act)
    next_state = integrator.step(state=state, control=u, sequence_history=dummy_history)

    # Afirmar (Assert)
    assert next_state.shape == (batch_size, 2)

    loss = next_state.sum()
    loss.backward()

    assert estimator.param.grad is not None
    assert estimator.param.grad.item() != 0.0