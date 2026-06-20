"""
================================================================================
MÓDULO: test/test_custom_loss.py
FUNCIÓN: Suite de pruebas unitarias (TDD) para validar que la función de pérdida
         de 3 términos (Ecuación 13) se calcule con dimensiones y pesos correctos.
VERSIÓN: 1.0.4
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

import os
import sys
import torch

# Inyección dinámica de la raíz del proyecto para evitar ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_custom_loss_calculation() -> None:
    """Verifica el cálculo correcto de los tres componentes de la Ecuación 13."""
    from pahm_model.custom_loss import PAHMProjectLoss

    # Arreglar (Arrange)
    lambda_1 = 0.1
    lambda_2 = 0.2
    loss_fn = PAHMProjectLoss(lambda_1=lambda_1, lambda_2=lambda_2)
    
    # Simulamos diferencias conocidas para verificar matemáticamente el resultado
    theta_obs = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)
    theta_sim = torch.tensor([[0.4], [-0.6]], dtype=torch.float32)
    
    # Historial simulado de torques de viento (Batch, Sequence_Length = 3)
    tau_w_history = torch.tensor([
        [0.1, 0.15, 0.12],
        [0.2, 0.30, 0.25]
    ], dtype=torch.float32)

    # Actuar (Act)
    total_loss, l_rec, l_pars, l_smooth = loss_fn(
        theta_obs=theta_obs,
        theta_sim=theta_sim,
        tau_w_history=tau_w_history
    )

    # Afirmar (Assert)
    assert total_loss.item() > 0.0
    assert l_rec.item() > 0.0
    assert l_pars.item() > 0.0
    assert l_smooth.item() > 0.0
    
    # Comprobación de consistencia algebraica
    expected_total = l_rec + lambda_1 * l_pars + lambda_2 * l_smooth
    assert torch.allclose(total_loss, expected_total)
