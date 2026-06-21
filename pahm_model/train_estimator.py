"""
================================================================================
MÓDULO: pahm_model/train_estimator.py
FUNCIÓN: Pipeline definitivo de entrenamiento del estimador GRU de viento.

         PAHMFastModel.cell(state, u, tau) ejecuta un paso RK4 con:
           - state: (batch, 2) — [θ, dθ/dt]
           - u:     (batch, 1) — control PWM
           - tau:   (batch, 1) — torque de viento aditivo (con gradiente)
         y devuelve (batch, 2) con el estado en t+1.

         Estrategia de entrenamiento:
           Truncated BPTT cada TRUNCATE_EVERY pasos para evitar OOM.
           Gradient clipping (max_norm=1.0) para frenar explosión de parsimonia.

         El dataloader devuelve:
           - sequences_padded: (batch, T, 1) — PWM
           - targets_padded:   (batch, T, 1) — ángulo θ normalizado
         get_dataloaders devuelve 4 valores: train, val, test, dataset.
VERSIÓN: 6.0.0
================================================================================
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.optim as optim
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sequence_estimator import WindSequenceEstimator  # noqa: E402
from custom_loss import PAHMProjectLoss  # noqa: E402
from dataloader import get_dataloaders  # noqa: E402
from pahm_fast import PAHMFastModel  # noqa: E402


class PAHMTrainer:
    """Orquestador del entrenamiento del estimador de perturbaciones de viento."""

    def __init__(self, config_path: str, data_dir: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        hparams = self.config["estimator_hyperparameters"]

        # Fijación de semillas antes de instanciar cualquier componente (NFR-4)
        seed = hparams["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.seq_len   = hparams["sequence_length"]
        self.epochs    = hparams["epochs"]
        self.batch_size = hparams["batch_size"]
        self.checkpoint_interval = hparams["checkpoint_interval_epochs"]
        self.checkpoint_dir = hparams["checkpoint_dir"]
        self.dt        = hparams["dt"]
        self.grad_clip = hparams.get("grad_clip_norm", 1.0)
        self.truncate  = hparams.get("truncate_every", 16)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Modelo físico del profesor congelado (CON-1)
        self.physics_model = PAHMFastModel(device=str(self.device)).to(self.device)
        ckpt_physics = os.path.join(current_dir, "pahm_fast_v2_best.pth")
        if os.path.exists(ckpt_physics):
            self.physics_model.load_model(ckpt_physics)
            print(f"[+] Modelo físico cargado desde: {ckpt_physics}")
        self.physics_model.eval()
        for param in self.physics_model.parameters():
            param.requires_grad = False

        # Estimador GRU: input_dim=4 → (sin θ, cos θ, dθ/dt, u)
        self.estimator = WindSequenceEstimator(
            input_dim=4,
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            tau_max=hparams.get("tau_max", 2.0)
        ).to(self.device)

        self.criterion = PAHMProjectLoss(
            lambda_1=hparams["lambda_1_pars"],
            lambda_2=hparams["lambda_2_smooth"]
        )

        self.optimizer = optim.Adam(
            self.estimator.parameters(), lr=hparams["learning_rate"]
        )

        # get_dataloaders devuelve 4 valores: train, val, test, dataset
        train_l, val_l, _, _ = get_dataloaders(
            root_dir=data_dir,
            batch_size=self.batch_size,
            verbose=False
        )
        self.train_loader = train_l
        self.val_loader   = val_l

        wandb.init(project="etapa-1", config=hparams, name="GRU-Wind-Estimator-v6")

    # ------------------------------------------------------------------
    def _build_gru_window(
        self, pwm: torch.Tensor, angle: torch.Tensor, t: int
    ) -> torch.Tensor:
        """Construye ventana (sin θ, cos θ, dθ/dt, u) para la GRU.

        Args:
            pwm:   (batch, T, 1) — señal de control PWM
            angle: (batch, T, 1) — ángulo θ normalizado
            t:     índice del paso actual (ventana cubre [t-seq_len, t))

        Returns:
            Tensor (batch, seq_len, 4)
        """
        theta_win = angle[:, t - self.seq_len:t, :]    # (batch, seq_len, 1)
        sin_theta = torch.sin(theta_win)
        cos_theta = torch.cos(theta_win)

        # dθ/dt por diferencia finita; primer paso replica el segundo
        delta = torch.zeros_like(theta_win)
        delta[:, 1:, :] = (theta_win[:, 1:, :] - theta_win[:, :-1, :]) / self.dt
        delta[:, 0, :]  = delta[:, 1, :]

        u_win = pwm[:, t - self.seq_len:t, :]           # (batch, seq_len, 1)
        return torch.cat([sin_theta, cos_theta, delta, u_win], dim=-1)

    # ------------------------------------------------------------------
    def _predict_next_theta(
        self,
        angle: torch.Tensor,
        pwm: torch.Tensor,
        tau_w: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """Predice θ(t+1) llamando directamente a la celda RK4 del profesor.

        El gradiente fluye por tau_w → GRU. La rama física permanece
        congelada (CON-1: requires_grad=False en todos sus parámetros).

        Args:
            angle: (batch, T, 1) — ángulo θ normalizado
            pwm:   (batch, T, 1) — control PWM
            tau_w: (batch, 1)    — torque estimado por la GRU (con gradiente)
            t:     índice del paso actual

        Returns:
            Tensor (batch, 1) con θ predicho en t+1
        """
        theta_t = angle[:, t - 1, :]
        theta_dot_t = (
            (angle[:, t - 1, :] - angle[:, t - 2, :]) / self.dt
            if t >= 2 else torch.zeros_like(theta_t)
        )
        state  = torch.cat([theta_t, theta_dot_t], dim=1)  # (batch, 2)
        u_step = pwm[:, t - 1, :]                           # (batch, 1)

        # Un paso RK4 con el torque de viento inyectado de forma aditiva
        next_state = self.physics_model.cell(state, u_step, tau=tau_w)
        return next_state[:, 0:1]                           # θ predicho en t+1

    # ------------------------------------------------------------------
    def _step_optimizer(self, loss_acumulada: torch.Tensor) -> None:
        """Backward + gradient clipping + optimizer step."""
        loss_acumulada.backward()
        torch.nn.utils.clip_grad_norm_(
            self.estimator.parameters(), max_norm=self.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    def train_epoch(self) -> dict:
        """Una época con Truncated BPTT y gradient clipping.

        Hace backward cada self.truncate pasos (Truncated BPTT) para
        liberar el grafo computacional y evitar OOM en GPU. El gradient
        clipping limita la norma de los gradientes para estabilizar el
        entrenamiento. El prepadding de ceros del dataloader actua como
        calentamiento suave natural al inicio de cada trayectoria.
        """
        self.estimator.train()
        t_loss, t_rec, t_pars, t_smooth = 0.0, 0.0, 0.0, 0.0
        steps = 0

        for pwm_padded, _, angle_padded in self.train_loader:
            pwm_padded   = pwm_padded.to(self.device)      # (batch, T, 1)
            angle_padded = angle_padded.to(self.device)    # (batch, T, 1)
            batch_size, max_steps, _ = pwm_padded.shape

            if max_steps <= self.seq_len + 1:
                continue

            self.optimizer.zero_grad()
            loss_acumulada = torch.tensor(0.0, device=self.device)
            pasos_en_chunk = 0

            for t in range(self.seq_len, max_steps - 1):
                # Ventana GRU: (batch, seq_len, 4)
                window = self._build_gru_window(pwm_padded, angle_padded, t)

                # Torque estimado con gradiente: (batch, 1)
                tau_w_current = self.estimator(window)

                # θ predicho en t+1 con gradiente fluyendo por tau_w
                theta_next_sim = self._predict_next_theta(
                    angle_padded, pwm_padded, tau_w_current, t
                )

                # θ observado real en t+1
                theta_next_obs = angle_padded[:, t + 1, :]    # (batch, 1)

                # Historial de torques para regularización (sin gradiente)
                tau_w_history = torch.zeros(
                    (batch_size, self.seq_len), device=self.device
                )
                with torch.no_grad():
                    for w_idx in range(self.seq_len - 1):
                        end = t - (self.seq_len - 1 - w_idx)
                        if end - self.seq_len < 0:
                            continue
                        w_slice = self._build_gru_window(
                            pwm_padded, angle_padded, end
                        )
                        tau_w_history[:, w_idx] = (
                            self.estimator(w_slice).squeeze(-1).detach()
                        )
                tau_w_history[:, -1] = tau_w_current.squeeze(-1).detach()

                loss, l_rec, l_pars, l_smooth = self.criterion(
                    theta_next_obs, theta_next_sim, tau_w_history
                )
                loss_acumulada = loss_acumulada + loss
                pasos_en_chunk += 1

                t_loss   += loss.item()
                t_rec    += l_rec.item()
                t_pars   += l_pars.item()
                t_smooth += l_smooth.item()
                steps    += 1

                # Truncated BPTT: liberar grafo cada self.truncate pasos
                if pasos_en_chunk >= self.truncate:
                    self._step_optimizer(loss_acumulada)
                    loss_acumulada = torch.tensor(0.0, device=self.device)
                    pasos_en_chunk = 0

            # Chunk final
            if pasos_en_chunk > 0:
                self._step_optimizer(loss_acumulada)

        if steps == 0:
            return {k: 0.0 for k in [
                "loss_total", "loss_reconstruction",
                "loss_parsimony", "loss_smoothness"
            ]}

        return {
            "loss_total":          t_loss   / steps,
            "loss_reconstruction": t_rec    / steps,
            "loss_parsimony":      t_pars   / steps,
            "loss_smoothness":     t_smooth / steps
        }

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Ciclo completo con telemetría W&B, checkpoints y latencia (NFR-2,3,7)."""
        print(f"[*] Iniciando entrenamiento en hardware: {self.device}")
        for epoch in range(1, self.epochs + 1):
            start_time = time.perf_counter()
            metrics = self.train_epoch()
            epoch_latency = time.perf_counter() - start_time

            metrics["epoch_latency_seconds"] = epoch_latency
            wandb.log(metrics, step=epoch)

            print(
                f"Época {epoch:02d}/{self.epochs} — "
                f"Total: {metrics['loss_total']:.4f} | "
                f"Rec: {metrics['loss_reconstruction']:.6f} | "
                f"Pars: {metrics['loss_parsimony']:.4f} | "
                f"Smooth: {metrics['loss_smoothness']:.4f} | "
                f"{epoch_latency:.1f}s"
            )

            if epoch % self.checkpoint_interval == 0 or epoch == self.epochs:
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"estimator_checkpoint_epoch_{epoch}.pth"
                )
                torch.save({
                    "model_state_dict":     self.estimator.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch":                epoch,
                    "loss":                 metrics["loss_total"]
                }, ckpt_path)
                print(f"[+] Checkpoint guardado: {ckpt_path}")

        wandb.finish()


if __name__ == "__main__":
    runner = PAHMTrainer(
        config_path="gym_wrapper/config.json",
        data_dir="data"
    )
    runner.fit()
