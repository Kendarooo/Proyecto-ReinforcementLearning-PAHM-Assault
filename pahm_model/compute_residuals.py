"""
================================================================================
MÓDULO: pahm_model/compute_residuals.py
FUNCIÓN: Satisface FR-3. Calcula r(t) = theta_obs - theta_ODE integrando la
         ODE base sobre toda la trayectoria de una vez.

         PAHMFastModel.forward(t, u_sequences, initial_state) recibe:
           - t:             ignorado (paso fijo implícito dt)
           - u_sequences:   (batch, seq_len, 1)  — PWM completo
           - initial_state: (batch, 2)            — [θ₀, dθ/dt₀]
         y devuelve (batch, seq_len, 2) con [θ, dθ/dt] en cada paso.

         El dataloader devuelve:
           - sequences_padded: (batch, T, 1) — PWM
           - targets_padded:   (batch, T, 1) — ángulo θ normalizado
         get_dataloaders devuelve 4 valores: train, val, test, dataset.
VERSIÓN: 3.0.0
================================================================================
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataloader import get_dataloaders  # noqa: E402
from pahm_fast import PAHMFastModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula residuos FR-3: r(t) = theta_obs - theta_ODE"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(project_root, "gym_wrapper/config.json")
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(project_root, "data")
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_root, "outputs/residuals")
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"[!] ERROR: La carpeta de datos '{args.data_dir}' no existe.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    hparams = config["estimator_hyperparameters"]

    seed = hparams.get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Calculando residuos FR-3 en hardware: {device}")

    # Modelo físico congelado del profesor (CON-1)
    # Se carga en CPU primero, luego se mueve al device correcto
    model = PAHMFastModel(device=str(device)).to(device)
    # Cargar checkpoint preentrenado del profesor
    checkpoint_path = os.path.join(current_dir, "pahm_fast_v2_best.pth")
    if os.path.exists(checkpoint_path):
        model.load_model(checkpoint_path)
        print(f"[+] Checkpoint del profesor cargado desde: {checkpoint_path}")
    else:
        print(f"[!] ADVERTENCIA: No se encontró {checkpoint_path}. Usando pesos aleatorios.")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # get_dataloaders devuelve 4 valores
    train_l, val_l, test_l, _ = get_dataloaders(
        root_dir=args.data_dir, batch_size=1, verbose=False
    )

    tabla_resultados = []

    with torch.no_grad():
        for split_name, loader in [("train", train_l), ("val", val_l), ("test", test_l)]:
            for idx, (pwm_padded, _, angle_padded) in enumerate(loader):
                # pwm_padded:   (1, T, 1) — señal de control PWM
                # angle_padded: (1, T, 1) — ángulo θ observado (normalizado)
                pwm_padded = pwm_padded.to(device)
                angle_padded = angle_padded.to(device)

                # θ observado a lo largo de la trayectoria: (T,)
                theta_obs = angle_padded[0, :, 0]

                # Estado inicial: [θ₀, dθ/dt₀=0]
                initial_state = torch.tensor(
                    [[theta_obs[0].item(), 0.0]], device=device
                )

                # PAHMFastModel integra toda la trayectoria de una vez:
                # forward(t=None, u_sequences, initial_state) → (1, T, 2)
                trajectory = model(None, pwm_padded, initial_state=initial_state)

                # θ predicho por la ODE base: (T,)
                theta_ode = trajectory[0, :, 0]

                # Residuo: r(t) = θ_obs(t) - θ_ODE(t)
                residuos = (theta_obs - theta_ode).cpu().numpy()
                out_path = os.path.join(
                    args.output_dir, f"residuals_{split_name}_{idx}.npy"
                )
                np.save(out_path, residuos.astype(np.float32))

                mse = float(np.mean(residuos ** 2))
                max_r = float(np.max(np.abs(residuos)))
                std_r = float(np.std(residuos))
                tabla_resultados.append((f"{split_name}_{idx}", mse, max_r, std_r))

    # Tabla de resultados
    print("\n" + "=" * 70)
    print(f"{'Trayectoria':<20} {'MSE ODE pura':>14} {'max|r(t)|':>12} {'std r(t)':>12}")
    print("-" * 70)
    for nombre, mse, max_r, std_r in tabla_resultados:
        print(f"{nombre:<20} {mse:>14.6f} {max_r:>12.6f} {std_r:>12.6f}")
    print("=" * 70)
    print(f"\n[+] Residuos guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
