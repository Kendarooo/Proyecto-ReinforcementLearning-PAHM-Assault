"""
================================================================================
MÓDULO: test/test_checkpoint_persistence.py
FUNCIÓN: Suite de pruebas unitarias (TDD) para verificar la persistencia,
         guardado y carga correcta de checkpoints de la GRU y del optimizador.
VERSIÓN: 1.0.0
AUTOR: Gemini (Sénior Software Engineer & ML Expert)
AUDITORÍA: Alexandra, Bryan, Katherine, Kendall
================================================================================
"""

import os
import sys
import shutil
import torch
import torch.optim as optim

# Inyección dinámica de la raíz del proyecto para evitar ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_checkpoint_save_and_load_state() -> None:
    """Valida la serialización y restauración exacta de la red y el optimizador."""
    from pahm_model.sequence_estimator import WindSequenceEstimator

    # Arreglar (Arrange)
    test_dir = os.path.join(project_root, "test_checkpoints_tmp")
    os.makedirs(test_dir, exist_ok=True)
    checkpoint_path = os.path.join(test_dir, "checkpoint_epoch_5.pth")

    # Instanciamos modelo y optimizador iniciales
    model_src = WindSequenceEstimator(input_dim=4, hidden_dim=16, num_layers=2)
    optimizer_src = optim.Adam(model_src.parameters(), lr=0.001)

    # Forzamos un estado en el optimizador haciendo un paso dummy de gradiente
    dummy_input = torch.randn(2, 10, 4)
    dummy_output = model_src(dummy_input)
    loss = dummy_output.sum()
    loss.backward()
    optimizer_src.step()

    # Estructura de guardado exigida por NFR-3
    checkpoint_payload = {
        "epoch": 5,
        "model_state_dict": model_src.state_dict(),
        "optimizer_state_dict": optimizer_src.state_dict(),
        "loss": loss.item()
    }

    # Actuar (Act)
    # 1. Guardar a disco
    torch.save(checkpoint_payload, checkpoint_path)
    assert os.path.exists(checkpoint_path)

    # 2. Instanciar una red nueva (con pesos aleatorios iniciales distintos)
    model_dst = WindSequenceEstimator(input_dim=4, hidden_dim=16, num_layers=2)
    optimizer_dst = optim.Adam(model_dst.parameters(), lr=0.001)

    # 3. Cargar el archivo y restaurar estados
    checkpoint_loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_dst.load_state_dict(checkpoint_loaded["model_state_dict"])
    optimizer_dst.load_state_dict(checkpoint_loaded["optimizer_state_dict"])

    # Afirmar (Assert)
    # Verificar que las épocas coincidan
    assert checkpoint_loaded["epoch"] == 5

    # Verificar numéricamente que los pesos cargados sean idénticos a los originales
    for param_src, param_dst in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.equal(param_src, param_dst)

    # Limpieza de la carpeta temporal de pruebas
    shutil.rmtree(test_dir)
