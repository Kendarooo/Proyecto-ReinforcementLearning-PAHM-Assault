"""Demo interactiva para la Etapa 0 de Kendall: DQN en Atari Assault."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ale_py
import ale_py.registration as ale_reg
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correr demo de DQN en Assault.")
    parser.add_argument(
        "--model",
        default="etapa0-assault-kendall/models/best_model.zip",
        help="Ruta del modelo .zip entrenado.",
    )
    parser.add_argument(
        "--env-id",
        default="ALE/Assault-v5",
        help="ID del entorno Atari. Alternativa comun: AssaultNoFrameskip-v4.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Cantidad de episodios.")
    parser.add_argument("--seed", type=int, default=0, help="Semilla del entorno.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Corre sin ventana grafica, util para validar carga del modelo.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Pausa entre pasos cuando hay render, para que la demo sea visible.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Factor de escala de la ventana (default: 2 → 320x420). ALE nativo es 160x210.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    if "ALE/Assault-v5" not in gym.envs.registry:
        ale_reg.register_v5_envs()

    render = not args.no_render
    # rgb_array nos da control total del tamaño; ALE nativo es 160×210.
    env_kwargs: dict = {"render_mode": "rgb_array"} if render else {}
    env = make_atari_env(args.env_id, n_envs=1, seed=args.seed, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    # Ventana pygame escalada manualmente.
    screen = None
    if render:
        pygame.init()
        win_w, win_h = 160 * args.scale, 210 * args.scale
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(f"Assault DQN  [{win_w}×{win_h}]")

    try:
        try:
            model = DQN.load(model_path, env=env)
        except ModuleNotFoundError as exc:
            if "numpy._core" in str(exc):
                print(
                    "[ERROR] El modelo fue guardado con numpy>=2.0 pero este entorno tiene numpy<2.\n"
                    "        Solución: desactiva el venv y usa el python3 del sistema:\n"
                    "          deactivate\n"
                    "          python3 etapa0-assault-kendall/demo_assault.py"
                )
                return
            raise

        obs = env.reset()
        episode = 1
        episode_reward = 0.0

        print(f"Corriendo {model_path} en {args.env_id}  [{win_w}×{win_h}]" if render
              else f"Corriendo {model_path} en {args.env_id}")
        while episode <= args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            episode_reward += float(rewards[0])

            if render and screen is not None:
                frame = env.render()  # (H, W, 3) uint8
                if frame is not None:
                    frame = np.asarray(frame)
                    # surfarray espera (W, H, 3); ALE devuelve (H, W, 3)
                    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                    surf = pygame.transform.scale(surf, (win_w, win_h))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        episode = args.episodes + 1
                time.sleep(args.sleep)

            if dones[0]:
                print(f"Episodio {episode}: reward={episode_reward:.2f}")
                episode += 1
                episode_reward = 0.0
                obs = env.reset()
    finally:
        env.close()
        if render:
            pygame.quit()


if __name__ == "__main__":
    main()
