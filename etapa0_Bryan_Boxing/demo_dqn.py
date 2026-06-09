"""
demo_dqn.py — v1.0
Sistema de demostración para el agente Double DQN entrenado en ALE/Boxing-v5.

- Carga el modelo desde la ruta configurada en config_dqn.json [NFR-1].
- Renderiza el agente jugando en tiempo real sin aprendizaje adicional [FR-2].
- Muestra estadísticas de la partida al finalizar.

Uso:
    python demo_dqn.py
    python demo_dqn.py --model models/boxing_ddqn_2000000_steps.zip
    python demo_dqn.py --episodes 3
"""

import argparse
import json
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_config(path: str = "config_dqn.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_demo_env(config: dict):
    """Entorno de 1 solo proceso con render human para visualización."""
    gym.register_envs(ale_py)
    env = make_atari_env(
        config["environment"]["env_id"],
        n_envs=1,
        seed=config["project"]["seed"],
        env_kwargs={"render_mode": "human"},
    )
    env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
    return env


# ── Demostración ──────────────────────────────────────────────────────────────

def run_demo(config: dict, model_path: str, n_episodes: int) -> None:
    model_file = Path(model_path)
    if not model_file.exists():
        # Intentar añadir extensión .zip si no existe
        model_file = Path(str(model_path) + ".zip")

    if not model_file.exists():
        print(f"[ERROR] No se encontró el modelo en '{model_path}'")
        print("        Primero entrena con: python train_dqn.py")
        return

    print(f"\n=== Demo — Boxing Double DQN v1.0 ===")
    print(f"Modelo cargado : {model_file}")
    print(f"Episodios      : {n_episodes}")
    print(f"Entorno        : {config['environment']['env_id']}")
    print("=" * 38)

    env = build_demo_env(config)

    # Cargar modelo sin continuar entrenamiento
    model = DQN.load(str(model_file), env=env, device="cpu")
    # Forzar modo determinista: sin exploración aleatoria
    model.exploration_rate = 0.0

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1

        elapsed = time.time() - start_time
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"  Episodio {ep:2d} | "
            f"Recompensa: {total_reward:7.1f} | "
            f"Pasos: {steps:5d} | "
            f"Tiempo: {elapsed:.1f}s"
        )

    env.close()

    print("\n── Resumen ──────────────────────────────")
    print(f"  Recompensa media : {np.mean(episode_rewards):.1f}")
    print(f"  Recompensa máx   : {np.max(episode_rewards):.1f}")
    print(f"  Recompensa mín   : {np.min(episode_rewards):.1f}")
    print(f"  Pasos medios     : {np.mean(episode_lengths):.0f}")
    print("─────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demostración agente DQN Boxing")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al modelo .zip (por defecto usa demo_model_path del config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Número de episodios a demostrar (default: 3)",
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

    # Si no se pasa --model, usar la ruta del config
    model_path = args.model or cfg["paths"]["demo_model_path"]

    run_demo(cfg, model_path=model_path, n_episodes=args.episodes)