"""Headless training script for Double DQN on DemonAttack.

Usage:
    python train.py [--config config.json] [--resume checkpoints/checkpoint_ep500.pth]
"""

import argparse
import json
import random
import shutil
import numpy as np
import torch
import wandb
from pathlib import Path

from wrappers import make_env
from replay_buffer import ReplayBuffer
from dqn_agent import DoubleDQNAgent


SHORT_SWEEP_PROFILES = {
    "base_short": {
        "agent.checkpoint_dir": "checkpoints/sweep_50/base_short",
        "agent.learning_rate": 0.0001,
        "agent.gamma": 0.99,
        "agent.batch_size": 32,
        "agent.epsilon_decay_steps": 300000,
        "agent.epsilon_end": 0.05,
        "agent.target_update_freq": 1000,
        "training.n_episodes": 50,
        "training.eval_every": 10,
        "training.eval_episodes": 5,
        "training.checkpoint_every": 25,
    },
    "fast_learning": {
        "agent.checkpoint_dir": "checkpoints/sweep_50/fast_learning",
        "agent.learning_rate": 0.00035,
        "agent.gamma": 0.99,
        "agent.batch_size": 32,
        "agent.epsilon_decay_steps": 150000,
        "agent.epsilon_end": 0.05,
        "agent.target_update_freq": 500,
        "training.n_episodes": 50,
        "training.eval_every": 10,
        "training.eval_episodes": 5,
        "training.checkpoint_every": 25,
    },
    "more_exploration": {
        "agent.checkpoint_dir": "checkpoints/sweep_50/more_exploration",
        "agent.learning_rate": 0.00018,
        "agent.gamma": 0.995,
        "agent.batch_size": 64,
        "agent.epsilon_decay_steps": 500000,
        "agent.epsilon_end": 0.1,
        "agent.target_update_freq": 1000,
        "training.n_episodes": 50,
        "training.eval_every": 10,
        "training.eval_episodes": 5,
        "training.checkpoint_every": 25,
    },
    "quick_exploitation": {
        "agent.checkpoint_dir": "checkpoints/sweep_50/quick_exploitation",
        "agent.learning_rate": 0.00018,
        "agent.gamma": 0.99,
        "agent.batch_size": 64,
        "agent.epsilon_decay_steps": 100000,
        "agent.epsilon_end": 0.01,
        "agent.target_update_freq": 500,
        "training.n_episodes": 50,
        "training.eval_every": 10,
        "training.eval_episodes": 5,
        "training.checkpoint_every": 25,
    },
    "long_horizon": {
        "agent.checkpoint_dir": "checkpoints/sweep_50/long_horizon",
        "agent.learning_rate": 0.00006,
        "agent.gamma": 0.997,
        "agent.batch_size": 32,
        "agent.epsilon_decay_steps": 250000,
        "agent.epsilon_end": 0.05,
        "agent.target_update_freq": 2500,
        "training.n_episodes": 50,
        "training.eval_every": 10,
        "training.eval_episodes": 5,
        "training.checkpoint_every": 25,
    },
}


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with open(config_path) as f:
        raw = json.load(f)
    # Flatten nested sections for the agent constructor
    cfg = {**raw["env"], **raw["agent"], **raw["training"]}
    cfg["seed"] = raw["seed"]
    cfg["wandb"] = raw["wandb"]
    cfg["config_dir"] = config_path.parent
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        cfg["checkpoint_dir"] = str(config_path.parent / checkpoint_dir)
    return cfg, raw


def configure_wandb_metrics():
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")


def set_config_value(cfg: dict, raw: dict, section: str, name: str, value):
    raw[section][name] = value
    if section in {"env", "agent", "training"}:
        cfg[name] = value

    if section == "agent" and name == "checkpoint_dir":
        checkpoint_dir = Path(value)
        if not checkpoint_dir.is_absolute():
            cfg[name] = str(cfg["config_dir"] / checkpoint_dir)


def apply_sweep_profile(cfg: dict, raw: dict, profile_name: str) -> tuple[dict, dict]:
    if profile_name not in SHORT_SWEEP_PROFILES:
        raise ValueError(
            f"Unknown sweep_profile={profile_name!r}. "
            f"Valid profiles: {sorted(SHORT_SWEEP_PROFILES)}"
        )

    print(f"W&B sweep profile: {profile_name}")
    for key, value in SHORT_SWEEP_PROFILES[profile_name].items():
        section, name = key.split(".", 1)
        if key == "agent.checkpoint_dir":
            value = f"{value}/{wandb.run.id}"
        set_config_value(cfg, raw, section, name, value)
        wandb.config.update({key: value}, allow_val_change=True)
        print(f"W&B profile override: {key}={value}")

    return cfg, raw


def apply_wandb_overrides(cfg: dict, raw: dict) -> tuple[dict, dict]:
    """Apply W&B sweep parameters such as agent.learning_rate to cfg/raw."""
    profile_name = wandb.config.get("sweep_profile")
    if profile_name:
        cfg, raw = apply_sweep_profile(cfg, raw, profile_name)

    for key, value in dict(wandb.config).items():
        if key == "sweep_profile":
            continue

        if isinstance(value, dict) and key in raw:
            for nested_name, nested_value in value.items():
                if nested_name not in raw[key]:
                    continue
                set_config_value(cfg, raw, key, nested_name, nested_value)
            continue

        if "." not in key:
            continue

        section, name = key.split(".", 1)
        if section not in raw or name not in raw[section]:
            continue

        set_config_value(cfg, raw, section, name, value)
        print(f"W&B override: {key}={value}")

    return cfg, raw


def isolate_sweep_checkpoints(cfg: dict, raw: dict) -> tuple[dict, dict]:
    """Keep sweep checkpoints from overwriting the main demo checkpoint."""
    if not getattr(wandb.run, "sweep_id", None):
        return cfg, raw

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    main_checkpoint_dir = cfg["config_dir"] / "checkpoints"
    if checkpoint_dir.resolve() != main_checkpoint_dir.resolve():
        return cfg, raw

    sweep_dir = Path("checkpoints") / "sweeps" / wandb.run.id
    set_config_value(cfg, raw, "agent", "checkpoint_dir", str(sweep_dir))
    wandb.config.update({"agent.checkpoint_dir": str(sweep_dir)}, allow_val_change=True)
    print(f"W&B sweep checkpoint_dir isolated: {cfg['checkpoint_dir']}")
    return cfg, raw


def evaluate(agent: DoubleDQNAgent, cfg: dict, device: torch.device) -> dict:
    eval_env = make_env(render_mode=None, config={**cfg, "clip_rewards": False})
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    lengths = []
    for _ in range(cfg["eval_episodes"]):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_length += 1

        rewards.append(total_reward)
        lengths.append(episode_length)

    agent.epsilon = previous_epsilon
    eval_env.close()

    return {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/best_reward": float(np.max(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/mean_length": float(np.mean(lengths)),
        "eval/epsilon": 0.0,
    }


def train(config_path: str = "config.json", resume_path: str | None = None):
    cfg, raw = load_config(config_path)
    set_seeds(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        group=cfg["wandb"].get("group"),
        job_type=cfg["wandb"].get("job_type", "train"),
        tags=cfg["wandb"].get("tags"),
        config=raw,
        resume="allow",
    )
    configure_wandb_metrics()
    cfg, raw = apply_wandb_overrides(cfg, raw)
    cfg, raw = isolate_sweep_checkpoints(cfg, raw)

    env = make_env(render_mode=None, config=cfg)
    env.action_space.seed(cfg["seed"])
    n_actions = env.action_space.n

    buffer = ReplayBuffer(capacity=cfg["replay_buffer_capacity"], seed=cfg["seed"])
    agent = DoubleDQNAgent(config=cfg, n_actions=n_actions, device=device)
    best_eval_reward = -float("inf")

    start_episode = 0
    if resume_path:
        start_episode = agent.load(resume_path)
        print(f"Resumed from episode {start_episode}")

    # Pre-fill buffer with random experience before training starts
    print(f"Pre-filling replay buffer (min {cfg['min_buffer_size']} transitions)...")
    obs, _ = env.reset(seed=cfg["seed"])
    while not buffer.is_ready(cfg["min_buffer_size"]):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    print("Buffer ready. Starting training...")

    current_episode = start_episode
    interrupted = False
    try:
        for episode in range(start_episode, cfg["n_episodes"]):
            current_episode = episode + 1
            obs, _ = env.reset()
            ep_reward_clipped = 0.0
            ep_reward_raw = 0.0
            ep_loss = []
            episode_length = 0
            done = False

            while not done:
                agent.global_step += 1
                agent.update_epsilon()

                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward_clipped += reward
                ep_reward_raw += info.get("raw_reward", reward)
                episode_length += 1

                loss = agent.learn(buffer)
                if loss is not None:
                    ep_loss.append(loss)

            if (episode + 1) % cfg["log_every"] == 0:
                avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
                print(
                    f"Ep {episode+1}/{cfg['n_episodes']} | "
                    f"Raw: {ep_reward_raw:.1f} | "
                    f"Clipped: {ep_reward_clipped:.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"Step: {agent.global_step}"
                )
                wandb.log({
                    "train/episode": episode + 1,
                    "train/global_step": agent.global_step,
                    "train/reward_raw": ep_reward_raw,
                    "train/reward_clipped": ep_reward_clipped,
                    "train/loss": avg_loss,
                    "train/epsilon": agent.epsilon,
                    "train/exploration_rate": agent.epsilon,
                    "train/exploitation_rate": 1.0 - agent.epsilon,
                    "train/buffer_size": len(buffer),
                    "train/episode_length": episode_length,
                    "train/learn_steps": agent._learn_steps,
                }, step=agent.global_step)

            if (episode + 1) % cfg["eval_every"] == 0:
                eval_metrics = evaluate(agent, cfg, device)
                print(
                    f"Eval ep {episode+1} | "
                    f"Mean reward: {eval_metrics['eval/mean_reward']:.1f} | "
                    f"Best: {eval_metrics['eval/best_reward']:.1f}"
                )
                wandb.log({
                    "train/global_step": agent.global_step,
                    **eval_metrics,
                }, step=agent.global_step)

                if cfg.get("save_best", True) and eval_metrics["eval/mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval/mean_reward"]
                    best_path = agent.save(episode + 1)
                    best_alias = agent.checkpoint_dir / "best_model.pth"
                    shutil.copy2(best_path, best_alias)
                    print(f"Best checkpoint updated: {best_alias}")
                    wandb.save(str(best_alias))

            if (episode + 1) % cfg["checkpoint_every"] == 0:
                path = agent.save(episode + 1)
                print(f"Checkpoint saved: {path}")
                wandb.save(str(path))
    except KeyboardInterrupt:
        interrupted = True
        interrupted_path = agent.save(current_episode)
        print(f"Interrupted. Emergency checkpoint saved: {interrupted_path}")
        wandb.save(str(interrupted_path))
    finally:
        final_episode = current_episode if interrupted else cfg["n_episodes"]
        final_path = agent.save(final_episode)
        print(f"Final/latest checkpoint saved: {final_path}")
        wandb.save(str(final_path))

    env.close()
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(config_path=args.config, resume_path=args.resume)
