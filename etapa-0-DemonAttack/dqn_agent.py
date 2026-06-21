"""Double DQN agent: action selection, learning step, checkpoint I/O."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from network import DQNNetwork
from replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """Double DQN agent with epsilon-greedy exploration and target network."""

    def __init__(self, config: dict, n_actions: int, device: torch.device):
        self.n_actions = n_actions
        self.device = device

        n_frames = config["n_frames"]
        h = config["frame_height"]
        w = config["frame_width"]

        self.online_net = DQNNetwork(n_frames, h, w, n_actions).to(device)
        self.target_net = DQNNetwork(n_frames, h, w, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=config["learning_rate"]
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epsilon = config["epsilon_start"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_schedule = config.get("epsilon_schedule", "linear")
        self.epsilon_decay_steps = max(1, int(config.get("epsilon_decay_steps", 300000)))

        self._learn_steps = 0
        self.global_step = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.online_net(s).argmax(dim=1).item())

    def update_epsilon(self, global_step: int | None = None):
        """Update epsilon from the configured exploration schedule."""
        if global_step is not None:
            self.global_step = global_step

        if self.epsilon_schedule == "linear":
            progress = min(1.0, self.global_step / self.epsilon_decay_steps)
            span = self.epsilon_start - self.epsilon_end
            self.epsilon = self.epsilon_start - span * progress
        else:
            decay = float(getattr(self, "epsilon_decay", 0.9999))
            self.epsilon = self.epsilon_start * (decay ** self.global_step)

        self.epsilon = max(self.epsilon_end, float(self.epsilon))

    def decay_epsilon(self):
        """Backward-compatible one-step epsilon update."""
        self.global_step += 1
        self.update_epsilon()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, buffer: ReplayBuffer) -> float | None:
        if not buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        # Double DQN: online net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, episode: int):
        path = self.checkpoint_dir / f"checkpoint_ep{episode}.pth"
        torch.save({
            "episode": episode,
            "global_step": self.global_step,
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self._learn_steps,
        }, path)
        return path

    def load(self, path: str | Path):
        data = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(data["online_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = data["epsilon"]
        self._learn_steps = data["learn_steps"]
        self.global_step = data.get("global_step", 0)
        return data["episode"]
