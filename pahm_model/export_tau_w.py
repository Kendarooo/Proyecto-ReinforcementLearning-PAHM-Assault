"""
================================================================================
MÓDULO: pahm_model/export_tau_w.py
FUNCIÓN: Genera y exporta los torques de viento estimados por la GRU sobre
         todas las trayectorias del dataset (train + val + test).
         Produce los archivos tau_w_<split>_<idx>.npy requeridos por el
         Hito I-1 hacia el Grupo 2, más el checkpoint final copiado como
         estimador_wind.pth.

         El dataloader devuelve:
           - sequences_padded: (batch, T, 1) — PWM
           - targets_padded:   (batch, T, 1) — ángulo θ normalizado
         get_dataloaders devuelve 4 valores: train, val, test, dataset.
VERSIÓN: 2.0.0
================================================================================
"""

import os
import sys
import json
import shutil
import time
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sequence_estimator import WindSequenceEstimator  # noqa: E402
from dataloader import get_dataloaders  # noqa: E402


def build_gru_window(
    pwm: torch.Tensor,
    angle: torch.Tensor,
    t: int,
    seq_len: int,
    dt: float
) -> torch.Tensor:
    """Construye ventana (sin θ, cos θ, dθ/dt, u) para la GRU.

    Args:
        pwm:     (1, T, 1) — señal de control PWM
        angle:   (1, T, 1) — ángulo θ normalizado
        t:       índice del paso actual (ventana cubre [t-seq_len, t))
        seq_len: longitud de la ventana
        dt:      paso de tiempo

    Returns:
        Tensor (1, seq_len, 4)
    """
    theta_win = angle[:, t - seq_len:t, :]          # (1, seq_len, 1)
    sin_theta = torch.sin(theta_win)
    cos_theta = torch.cos(theta_win)

    delta = torch.zeros_like(theta_win)
    delta[:, 1:, :] = (theta_win[:, 1:, :] - theta_win[:, :-1, :]) / dt
    delta[:, 0, :] = delta[:, 1, :]

    u_win = pwm[:, t - seq_len:t, :]                # (1, seq_len, 1)
    return torch.cat([sin_theta, cos_theta, delta, u_win], dim=-1)


def main() -> None:
    """Orquesta la inferencia sobre todo el dataset y exporta los .npy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Exportando torques de viento en hardware: {device}")

    # Cargar configuración
    config_path = os.path.join(project_root, "gym_wrapper/config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    hparams = config["estimator_hyperparameters"]

    seq_len = hparams["sequence_length"]
    dt      = hparams["dt"]

    # Instanciar y cargar el estimador entrenado
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
        print(f"[!] ERROR: No se encontró checkpoint en '{checkpoint_path}'.")
        print("    Asegúrate de haber completado el entrenamiento primero.")
        sys.exit(1)

    print(f"[+] Cargando checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    estimator.load_state_dict(ckpt["model_state_dict"])
    estimator.eval()

    # Directorio de salida
    output_dir = hparams.get(
        "tau_w_output_dir",
        os.path.join(project_root, "outputs/tau_w")
    )
    os.makedirs(output_dir, exist_ok=True)

    # get_dataloaders devuelve 4 valores: train, val, test, dataset
    train_l, val_l, test_l, _ = get_dataloaders(
        root_dir=os.path.join(project_root, "data"),
        batch_size=1,
        verbose=False
    )

    start_time = time.perf_counter()
    tau_all = []
    total_traj = 0

    with torch.no_grad():
        for split_name, loader in [("train", train_l), ("val", val_l), ("test", test_l)]:
            for idx, (pwm_padded, _, angle_padded) in enumerate(loader):
                # pwm_padded:   (1, T, 1) — control PWM
                # angle_padded: (1, T, 1) — ángulo θ normalizado
                pwm_padded   = pwm_padded.to(device)
                angle_padded = angle_padded.to(device)
                max_steps = pwm_padded.shape[1]

                tau_w_traj = []

                # Los primeros seq_len pasos no tienen ventana completa → rellenar con 0
                for _ in range(seq_len):
                    tau_w_traj.append(0.0)

                # Deslizar ventana sobre el resto de la trayectoria
                for t in range(seq_len, max_steps):
                    window = build_gru_window(
                        pwm_padded, angle_padded, t, seq_len, dt
                    )
                    tau_w = estimator(window).item()
                    tau_w_traj.append(tau_w)

                arr = np.array(tau_w_traj, dtype=np.float32)
                out_path = os.path.join(
                    output_dir, f"tau_w_{split_name}_{idx}.npy"
                )
                np.save(out_path, arr)
                tau_all.extend(tau_w_traj[seq_len:])  # excluir los ceros iniciales
                total_traj += 1

    inference_time = time.perf_counter() - start_time
    tau_arr = np.array(tau_all)

    # Copiar checkpoint como estimador_wind.pth (artefacto I-1 para Grupo 2)
    estimador_wind_path = os.path.join(output_dir, "estimador_wind.pth")
    shutil.copy(checkpoint_path, estimador_wind_path)

    print("\n[+] Exportación completada:")
    print(f"    Trayectorias procesadas : {total_traj}")
    print(f"    Directorio de salida    : {output_dir}/")
    print(f"    τ_w — min: {tau_arr.min():.4f} | max: {tau_arr.max():.4f} | std: {tau_arr.std():.4f}")
    print(f"    Tiempo de inferencia    : {inference_time:.2f}s")
    print(f"    Checkpoint I-1          : {estimador_wind_path}")


if __name__ == "__main__":
    main()
