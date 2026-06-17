"""Demo system: loads a trained checkpoint and renders DemonAttack in color.

Usage:
    python demo.py [--checkpoint checkpoints/best_model.pth] [--config config.json] [--episodes 5]
"""

import argparse
import json
import time
import torch

from wrappers import make_env
from dqn_agent import DoubleDQNAgent


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    cfg = {**raw["env"], **raw["agent"], **raw["training"]}
    cfg["seed"] = raw["seed"]
    cfg["wandb"] = raw["wandb"]
    return cfg


def demo(
    checkpoint_path: str,
    config_path: str = "config.json",
    n_episodes: int = 5,
):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Render in color, no reward clipping so we see real scores.
    env = make_env(render_mode="human", config={**cfg, "clip_rewards": False})
    n_actions = env.action_space.n

    agent = DoubleDQNAgent(config=cfg, n_actions=n_actions, device=device)
    episode_loaded = agent.load(checkpoint_path)
    agent.epsilon = 0.0  # pure exploitation during demo
    print(f"Loaded checkpoint from episode {episode_loaded}. Running {n_episodes} demo episodes...")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        step_count = 0
        start_time = time.perf_counter()
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        elapsed = time.perf_counter() - start_time
        print(
            f"Demo episode {ep}/{n_episodes} | "
            f"Score: {total_reward:.0f} | "
            f"Time: {elapsed:.1f}s | "
            f"Steps: {step_count}"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to .pth checkpoint file",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    demo(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        n_episodes=args.episodes,
    )
