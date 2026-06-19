"""Environment wrappers for DemonAttack: resize, RGB frame stack, normalization."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from collections import deque
import cv2

try:
    import ale_py
except ImportError:
    ale_py = None
else:
    from ale_py.registration import register_v5_envs

    register_v5_envs()


class ResizeRGB(gym.ObservationWrapper):
    """Resize RGB frames to target size."""

    def __init__(self, env, width: int, height: int):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)


class FrameStackRGB(gym.Wrapper):
    """Stack n consecutive RGB frames along a new first axis: (n, H, W, 3)."""

    def __init__(self, env, n_frames: int):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames = deque(maxlen=n_frames)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_frames, h, w, c),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self._frames, dtype=np.uint8)


class NormalizePixels(gym.ObservationWrapper):
    """Convert uint8 stacked frames to float32 in [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        low = self.observation_space.low.astype(np.float32) / 255.0
        high = self.observation_space.high.astype(np.float32) / 255.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class ClipReward(gym.Wrapper):
    """Clip rewards for learning while preserving raw rewards in info."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        return obs, np.sign(reward), terminated, truncated, info


def make_env(render_mode: str | None = None, config: dict | None = None) -> gym.Env:
    """Build the DemonAttack environment with all wrappers applied."""
    if ale_py is None:
        raise DependencyNotInstalled(
            "ale-py is required to build ALE/DemonAttack-v5. "
            "Install the Atari extras from requirements.txt."
        )

    cfg = config or {}
    width = cfg.get("frame_width", 84)
    height = cfg.get("frame_height", 84)
    n_frames = cfg.get("n_frames", 4)
    clip_rewards = cfg.get("clip_rewards", True)

    env = gym.make("ALE/DemonAttack-v5", obs_type="rgb", render_mode=render_mode)
    env = ResizeRGB(env, width=width, height=height)
    env = FrameStackRGB(env, n_frames=n_frames)
    env = NormalizePixels(env)
    if clip_rewards:
        env = ClipReward(env)
    return env
