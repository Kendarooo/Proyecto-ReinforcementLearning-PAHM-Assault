"""
train_dqn.py — v1.1
Sistema de entrenamiento Double DQN para ALE/Boxing-v5.

- Headless por defecto (sin renderizado).
- Checkpoints periódicos configurables.
- Telemetría completa en Weights & Biases [NFR-2].
- Toda magnitud calibrable proviene de config_dqn.json [NFR-1].
- Semilla fijable para reproducibilidad [NFR-4].
- Soporte para reanudar entrenamiento desde checkpoint (--resume).

Uso:
    python train_dqn.py                          # entrenamiento desde cero
    python train_dqn.py --resume                 # continúa desde models/best_model.zip
    python train_dqn.py --resume --checkpoint checkpoints/boxing_ddqn_3000000_steps.zip
    python train_dqn.py --debug-render           # activa renderizado (lento)
"""

import argparse
import json
import random
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_config(path: str = "config_dqn.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def set_seeds(seed: int) -> None:
    """Fija semillas en todos los generadores para reproducibilidad [NFR-4]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def build_env(config: dict, render: bool = False):
    """Construye el entorno vectorizado con preprocesamiento Atari estándar."""
    gym.register_envs(ale_py)
    render_mode = "human" if render else None
    env = make_atari_env(
        config["environment"]["env_id"],
        n_envs=config["environment"]["n_envs"],
        seed=config["project"]["seed"],
        env_kwargs={"render_mode": render_mode},
    )
    env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
    return env


def find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    """Busca el checkpoint más reciente en el directorio dado."""
    ckpt_path = Path(checkpoint_dir)
    checkpoints = sorted(ckpt_path.glob("boxing_ddqn_*_steps.zip"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[INFO] Checkpoint más reciente encontrado: {latest}")
        return latest
    return None


# ── Callback de W&B ───────────────────────────────────────────────────────────

class WandbCallback(BaseCallback):
    """
    Registra métricas de entrenamiento en W&B en cada rollout.
    Loguea: recompensa media, longitud de episodio y pérdida de la red.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            wandb.log(
                {
                    "train/mean_reward": np.mean(ep_rewards),
                    "train/mean_ep_length": np.mean(ep_lengths),
                    "train/exploration_rate": self.model.exploration_rate,
                    "train/timesteps": self.num_timesteps,
                },
                step=self.num_timesteps,
            )


# ── Entrenamiento principal ───────────────────────────────────────────────────

def train(config: dict, debug_render: bool = False,
          resume: bool = False, checkpoint_path: str | None = None) -> None:
    seed = config["project"]["seed"]
    set_seeds(seed)

    hp = config["hyperparameters"]
    paths = config["paths"]
    ckpt = config["checkpoints"]

    # Asegurar que los directorios existen
    for key in ("save_dir",):
        Path(ckpt[key]).mkdir(parents=True, exist_ok=True)
    for key in ("log_dir", "model_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)

    # ── Inicializar W&B ───────────────────────────────────────────────────────
    run_name = f"boxing-ddqn-seed{seed}-resume" if resume else f"boxing-ddqn-seed{seed}"
    wandb.init(
        project=config["project"]["wandb_project"],
        entity=config["project"]["wandb_entity"],
        name=run_name,
        config={
            "algorithm": config["model"]["algorithm"],
            "env_id": config["environment"]["env_id"],
            "resumed": resume,
            **hp,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # ── Construir entornos ────────────────────────────────────────────────────
    train_env = build_env(config, render=debug_render)

    eval_env = make_atari_env(
        config["environment"]["env_id"],
        n_envs=1,
        seed=seed + 100,
    )
    eval_env = VecFrameStack(eval_env, n_stack=config["environment"]["frame_stack"])

    # ── Cargar o crear modelo ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Usando dispositivo: {device}")

    if resume:
        # Determinar qué checkpoint cargar
        if checkpoint_path:
            load_path = Path(checkpoint_path)
        else:
            # Intentar el checkpoint más reciente; si no, best_model
            load_path = find_latest_checkpoint(ckpt["save_dir"])
            if load_path is None:
                load_path = Path(paths["model_dir"]) / "best_model.zip"

        if not load_path.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint en '{load_path}'.\n"
                f"Verifica la ruta o usa --checkpoint <ruta>."
            )

        print(f"[INFO] Reanudando desde: {load_path}")
        model = DQN.load(
            str(load_path),
            env=train_env,
            device=device,
            # Actualizar hiperparámetros desde el config actual
            custom_objects={
                "exploration_final_eps": hp["exploration_final_eps"],
                "exploration_fraction": hp["exploration_fraction"],
                "learning_rate": hp["learning_rate"],
            },
        )
        # Forzar epsilon final del config (mejora vs run anterior)
        model.exploration_rate = hp["exploration_final_eps"]
        print(f"[INFO] Epsilon fijado a: {model.exploration_rate}")

    else:
        model = DQN(
            policy=config["model"]["policy"],
            env=train_env,
            learning_rate=hp["learning_rate"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            exploration_fraction=hp["exploration_fraction"],
            exploration_initial_eps=hp["exploration_initial_eps"],
            exploration_final_eps=hp["exploration_final_eps"],
            learning_starts=hp["learning_starts"],
            train_freq=hp["train_freq"],
            target_update_interval=hp["target_update_interval"],
            max_grad_norm=hp["max_grad_norm"],
            optimize_memory_usage=True,
            replay_buffer_kwargs={"handle_timeout_termination": False},
            tensorboard_log=paths["log_dir"],
            verbose=1,
            seed=seed,
            device=device,
        )

    # ── Definir callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(ckpt["save_freq"] // config["environment"]["n_envs"], 1),
        save_path=ckpt["save_dir"],
        name_prefix="boxing_ddqn",
        save_replay_buffer=False,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=paths["model_dir"],
        log_path=paths["log_dir"],
        eval_freq=max(ckpt["save_freq"] // config["environment"]["n_envs"], 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    wandb_cb = WandbCallback()

    # ── Entrenar ──────────────────────────────────────────────────────────────
    mode = "reanudado" if resume else "desde cero"
    print(f"\n[INFO] Entrenamiento {mode} por {hp['total_timesteps']:,} timesteps adicionales...")
    print(f"[INFO] Checkpoints cada {ckpt['save_freq']:,} pasos en '{ckpt['save_dir']}'")
    print("[INFO] Para detener: Ctrl+C (el último checkpoint queda guardado)\n")

    model.learn(
        total_timesteps=hp["total_timesteps"],
        callback=[checkpoint_cb, eval_cb, wandb_cb],
        progress_bar=True,
        reset_num_timesteps=not resume,  # False = continúa el contador global
    )

    # ── Guardar modelo final ──────────────────────────────────────────────────
    suffix = "_resumed" if resume else ""
    final_path = Path(paths["model_dir"]) / f"final_model{suffix}"
    model.save(str(final_path))
    print(f"\n[OK] Modelo final guardado en '{final_path}.zip'")

    train_env.close()
    eval_env.close()
    wandb.finish()
    print("[OK] Entrenamiento completado.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento DQN Boxing")
    parser.add_argument(
        "--debug-render",
        action="store_true",
        help="Activa el renderizado visual (desactiva por defecto según NFR)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda el entrenamiento desde el checkpoint más reciente",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta explícita al checkpoint .zip desde el que reanudar",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_dqn.json",
        help="Ruta al archivo de configuración",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    render = args.debug_render or cfg["environment"]["debug_render"]
    train(cfg, debug_render=render, resume=args.resume, checkpoint_path=args.checkpoint)