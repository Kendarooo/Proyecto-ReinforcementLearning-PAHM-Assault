"""Experience replay buffer for DQN training."""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Uniform experience replay buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, seed: int = 42):
        self._buffer = deque(maxlen=capacity)
        random.seed(seed)

    @staticmethod
    def _pack_observation(obs):
        """Store normalized image observations compactly as uint8."""
        arr = np.asarray(obs)
        if arr.dtype == np.uint8:
            return arr.copy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _unpack_observations(obs_batch):
        return np.asarray(obs_batch, dtype=np.float32) / 255.0

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self._buffer.append((
            self._pack_observation(state),
            action,
            reward,
            self._pack_observation(next_state),
            done,
        ))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            self._unpack_observations(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            self._unpack_observations(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
