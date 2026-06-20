"""
================================================================================
MÓDULO: test/test_verification.py
FUNCIÓN: Suite TDD. Recuperación sintética estricta (NFR-6f) y esqueleto
         de evaluación real en lazo abierto (FR-7).
         Corregido para formato real del dataloader:
           - sequences_padded → PWM (batch, T, 1)
           - targets_padded   → ángulo θ (batch, T, 1)
           - get_dataloaders devuelve 4 valores: train, val, test, dataset
VERSIÓN: 4.0.0
================================================================================
"""

import os
import sys
import json
import pytest
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
pahm_model_dir = os.path.join(project_root, "pahm_model")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if pahm_model_dir not in sys.path:
    sys.path.insert(0, pahm_model_dir)

from sequence_estimator import WindSequenceEstimator  # noqa: E402
from rk4_integrator import TaylorWindIntegrator  # noqa: E402
from dataloader import get_dataloaders  # noqa: E402
from pahm_fast import PAHMFastModel  # noqa: E402


class MockPerfectPhysics(torch.nn.Module):
    """Modelo físico ideal con parámetros conocidos para pruebas sintéticas."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = 1.0
        self.beta = 0.1
        self.gamma = 1.0

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]
        # u puede llegar como (batch,1) o (batch,1) — reshape seguro
        u_flat = u.reshape(state.shape[0], -1)[:, 0:1]
        d_theta_dot = (
            self.alpha * (u_flat ** 2)
            - self.beta * theta_dot
            - self.gamma * torch.sin(theta)
        )
        return torch.cat([theta_dot, d_theta_dot], dim=1)


class TargetConstantWindEstimator(torch.nn.Module):
    """Estimador dummy que devuelve un torque de viento constante conocido."""

    def __init__(self, target_wind: float) -> None:
        super().__init__()
        self.wind = target_wind

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (history.shape[0], 1), self.wind, device=history.device
        )


def test_synthetic_wind_recovery_nfr6f() -> None:
    """NFR-6f: el estimador debe recuperar un viento conocido con error < 0.25."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.02
    seq_len = 10
    viento_verdadero = 0.5

    physics = MockPerfectPhysics().to(device)
    estimator_under_test = WindSequenceEstimator(
        input_dim=4, hidden_dim=32, num_layers=2
    ).to(device)

    integrator_train = TaylorWindIntegrator(
        physics_model=physics, wind_estimator=estimator_under_test, dt=dt
    )
    integrator_ideal = TaylorWindIntegrator(
        physics_model=physics,
        wind_estimator=TargetConstantWindEstimator(viento_verdadero),
        dt=dt
    )

    optimizer = torch.optim.Adam(estimator_under_test.parameters(), lr=0.03)

    # Estado inicial y control: shapes (batch=1, features)
    state = torch.tensor([[0.0, 0.0]], device=device)
    u = torch.tensor([[0.6]], device=device)

    # Buffer de ventana inicial con estado estático
    history_buffer = [
        [np.sin(0.0), np.cos(0.0), 0.0, 0.6] for _ in range(seq_len)
    ]
    window_tensor = torch.tensor(
        [history_buffer], dtype=torch.float32, device=device
    )

    # Generar 5 pasos de trayectoria de referencia con viento ideal
    trajectory_states = []
    trajectory_windows = []
    current_state = state.clone()

    for _ in range(5):
        trajectory_windows.append(list(history_buffer))
        trajectory_states.append(current_state.clone())
        with torch.no_grad():
            next_state = integrator_ideal.step(current_state, u, window_tensor)
        current_state = next_state

    # 150 épocas de optimización con LR agresivo para convergencia estricta
    for _ in range(150):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for step_idx in range(5):
            win_tensor = torch.tensor(
                [trajectory_windows[step_idx]], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                st_next_real = integrator_ideal.step(
                    trajectory_states[step_idx], u, win_tensor
                )
            st_next_sim = integrator_train.step(
                trajectory_states[step_idx], u, win_tensor
            )
            loss = loss + torch.mean(
                (st_next_sim[:, 0:1] - st_next_real[:, 0:1]) ** 2
            )
        loss.backward()
        optimizer.step()

    error_estimacion = abs(
        estimator_under_test(window_tensor).item() - viento_verdadero
    )
    assert error_estimacion < 0.25, (
        f"El estimador no convergió: error={error_estimacion:.4f} >= 0.25"
    )

def test_open_loop_baseline_comparison_fr7() -> None:
    """FR-7: compara MSE de ODE pura vs ODE+GRU sobre el conjunto de prueba retenido."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(project_root, "gym_wrapper/config.json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    hparams = config["estimator_hyperparameters"]
    dt = hparams["dt"]
    seq_len = hparams["sequence_length"]

    # get_dataloaders devuelve 4 valores — desempaquetar correctamente
    _, _, test_loader, _ = get_dataloaders(
        root_dir=os.path.join(project_root, "data"),
        batch_size=1,
        verbose=False
    )

    estimator = WindSequenceEstimator(
        input_dim=4,
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"]
    ).to(device)

    checkpoint_path = os.path.join(
        project_root,
        hparams["checkpoint_dir"],
        f"estimator_checkpoint_epoch_{hparams['epochs']}.pth"
    )
    if not os.path.exists(checkpoint_path):
        pytest.skip(
            "Checkpoint de Etapa 1 no disponible para FR-7: "
            f"{checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    estimator.load_state_dict(checkpoint["model_state_dict"])
    estimator.eval()

    physics_model = PAHMFastModel(device=str(device)).to(device)
    # Cargar checkpoint del profesor con parámetros físicos entrenados (CON-1)
    ckpt_physics = os.path.join(pahm_model_dir, "pahm_fast_v2_best.pth")
    if os.path.exists(ckpt_physics):
        physics_model.load_model(ckpt_physics)
    else:
        raise FileNotFoundError(f"Checkpoint del profesor no encontrado: {ckpt_physics}")
    physics_model.eval()
    for param in physics_model.parameters():
        param.requires_grad = False

    integrator_gru = TaylorWindIntegrator(
        physics_model=physics_model, wind_estimator=estimator, dt=dt
    )

    mse_pure_ode = []
    mse_gru = []

    with torch.no_grad():
        for pwm_padded, _, angle_padded in test_loader:
            pwm_padded   = pwm_padded.to(device)
            angle_padded = angle_padded.to(device)
            max_steps = angle_padded.shape[1]

            for t in range(seq_len, max_steps):
                theta_win = angle_padded[:, t - seq_len:t, :]
                sin_t = torch.sin(theta_win)
                cos_t = torch.cos(theta_win)
                delta = torch.zeros_like(theta_win)
                delta[:, 1:, :] = (
                    theta_win[:, 1:, :] - theta_win[:, :-1, :]
                ) / dt
                delta[:, 0, :] = delta[:, 1, :]
                u_win = pwm_padded[:, t - seq_len:t, :]
                window = torch.cat([sin_t, cos_t, delta, u_win], dim=-1)

                theta_t = angle_padded[:, t - 1, :]
                theta_dot_t = (
                    (angle_padded[:, t - 1, :] - angle_padded[:, t - 2, :]) / dt
                    if t >= 2
                    else torch.zeros_like(theta_t)
                )
                st_curr = torch.cat([theta_t, theta_dot_t], dim=1)
                ctrl    = pwm_padded[:, t - 1, :]
                theta_next_real = angle_padded[:, t, :]

                # ODE pura: un paso RK4 directo sin torque de viento
                st_next_pure = physics_model.cell(st_curr, ctrl, tau=0.0)
                mse_pure_ode.append(
                    (st_next_pure[:, 0:1] - theta_next_real).pow(2).mean().item()
                )

                # ODE + GRU estimador
                st_next_gru = integrator_gru.step(st_curr, ctrl, window)
                mse_gru.append(
                    (st_next_gru[:, 0:1] - theta_next_real).pow(2).mean().item()
                )

    final_mse_ode = float(np.mean(mse_pure_ode))
    final_mse_gru = float(np.mean(mse_gru))

    print(f"\n{'='*60}")
    print("  REPORTE FR-7 — Evaluacion en lazo abierto (test set)")
    print(f"{'='*60}")
    print(f"  MSE ODE pura   : {final_mse_ode:.8f}")
    print(f"  MSE ODE + GRU  : {final_mse_gru:.8f}")
    print(f"  Mejora relativa: {100*(final_mse_ode - final_mse_gru)/final_mse_ode:.2f}%")
    print(f"{'='*60}\n")

    assert final_mse_gru <= final_mse_ode, (
        f"El estimador no supera la ODE pura: "
        f"MSE GRU={final_mse_gru:.8f} > MSE ODE={final_mse_ode:.8f}"
    )
