"""Unit tests for the DemonAttack DQN implementation (NFR-6)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
import pytest

from network import DQNNetwork
from replay_buffer import ReplayBuffer
from dqn_agent import DoubleDQNAgent
from wrappers import make_env


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def config():
    with open(os.path.join(os.path.dirname(__file__), "..", "config.json")) as f:
        raw = json.load(f)
    cfg = {**raw["env"], **raw["agent"], **raw["training"]}
    cfg["seed"] = raw["seed"]
    cfg["wandb"] = raw["wandb"]
    return cfg


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def agent(config, device, tmp_path):
    config["checkpoint_dir"] = str(tmp_path)
    return DoubleDQNAgent(config=config, n_actions=6, device=device)


@pytest.fixture
def buffer(config):
    return ReplayBuffer(capacity=config["replay_buffer_capacity"], seed=config["seed"])


# ------------------------------------------------------------------
# Network tests
# ------------------------------------------------------------------

def test_network_output_shape(config, device):
    """Red neuronal produce exactamente n_actions=6 valores Q."""
    net = DQNNetwork(
        n_frames=config["n_frames"],
        height=config["frame_height"],
        width=config["frame_width"],
        n_actions=6,
    ).to(device)
    dummy = torch.zeros(2, config["n_frames"], config["frame_height"], config["frame_width"], 3)
    out = net(dummy)
    assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"


def test_network_output_is_finite(config, device):
    """La red no produce NaN ni Inf en la salida inicial."""
    net = DQNNetwork(config["n_frames"], config["frame_height"], config["frame_width"], 6).to(device)
    dummy = torch.rand(4, config["n_frames"], config["frame_height"], config["frame_width"], 3)
    out = net(dummy)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# Replay buffer tests
# ------------------------------------------------------------------

def test_replay_buffer_stores_and_samples(buffer, config):
    """El buffer almacena transiciones y entrega batches del tamaño correcto."""
    obs_shape = (config["n_frames"], config["frame_height"], config["frame_width"], 3)
    for _ in range(config["batch_size"] + 10):
        s = np.random.rand(*obs_shape).astype(np.float32)
        ns = np.random.rand(*obs_shape).astype(np.float32)
        buffer.push(s, 0, 1.0, ns, False)

    states, actions, rewards, next_states, dones = buffer.sample(config["batch_size"])
    assert states.shape == (config["batch_size"], *obs_shape)
    assert actions.shape == (config["batch_size"],)
    assert rewards.shape == (config["batch_size"],)
    assert next_states.shape == (config["batch_size"], *obs_shape)
    assert dones.shape == (config["batch_size"],)
    assert states.dtype == np.float32
    assert next_states.dtype == np.float32
    assert states.min() >= 0.0 and states.max() <= 1.0
    assert next_states.min() >= 0.0 and next_states.max() <= 1.0


def test_replay_buffer_not_ready_when_small(buffer, config):
    """Buffer no está listo si tiene menos transiciones que el batch_size."""
    assert not buffer.is_ready(config["batch_size"])
    buffer.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
    assert not buffer.is_ready(config["batch_size"])


# ------------------------------------------------------------------
# Agent action tests
# ------------------------------------------------------------------

def test_agent_action_in_valid_range(agent):
    """El agente siempre devuelve una acción dentro de Discrete(6)."""
    obs = np.random.rand(4, 84, 84, 3).astype(np.float32)
    for _ in range(50):
        action = agent.select_action(obs)
        assert 0 <= action < 6, f"Action {action} out of range"


def test_agent_deterministic_when_epsilon_zero(agent):
    """Con epsilon=0 el agente es determinista para el mismo estado."""
    agent.epsilon = 0.0
    obs = np.random.rand(4, 84, 84, 3).astype(np.float32)
    actions = [agent.select_action(obs) for _ in range(10)]
    assert len(set(actions)) == 1, "Agent should be deterministic with epsilon=0"


# ------------------------------------------------------------------
# Epsilon decay test
# ------------------------------------------------------------------

def test_epsilon_decays_and_respects_floor(agent, config):
    """Epsilon decrece y nunca cae por debajo de epsilon_end."""
    initial_eps = agent.epsilon
    for _ in range(1000):
        agent.decay_epsilon()
    assert agent.epsilon < initial_eps
    assert agent.epsilon >= config["epsilon_end"] - 1e-8


def test_linear_epsilon_schedule_reaches_floor(agent, config):
    """El calendario lineal llega al valor mínimo al completar epsilon_decay_steps."""
    agent.update_epsilon(config["epsilon_decay_steps"])
    assert abs(agent.epsilon - config["epsilon_end"]) < 1e-8


# ------------------------------------------------------------------
# Wrapper / observation tests
# ------------------------------------------------------------------

def test_wrappers_observation_shape(config):
    """Los wrappers producen observaciones de shape (n_frames, H, W, 3) en [0,1]."""
    pytest.importorskip("ale_py")
    env = make_env(render_mode=None, config=config)
    obs, _ = env.reset(seed=0)
    expected = (config["n_frames"], config["frame_height"], config["frame_width"], 3)
    assert obs.shape == expected, f"Expected {expected}, got {obs.shape}"
    env.close()


def test_wrappers_pixel_range(config):
    """Los píxeles normalizados están en [0, 1]."""
    pytest.importorskip("ale_py")
    env = make_env(render_mode=None, config=config)
    obs, _ = env.reset(seed=0)
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Pixels out of [0,1] range"
    env.close()


def test_wrappers_step_consistency(config):
    """step() devuelve una observación con el mismo shape que reset()."""
    pytest.importorskip("ale_py")
    env = make_env(render_mode=None, config=config)
    obs, _ = env.reset(seed=0)
    next_obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs.shape == next_obs.shape
    env.close()


def test_clip_reward_preserves_raw_reward(config):
    """El wrapper clipea la recompensa pero conserva el valor real en info."""
    pytest.importorskip("ale_py")
    env = make_env(render_mode=None, config={**config, "clip_rewards": True})
    env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert reward in {-1.0, 0.0, 1.0}
    assert "raw_reward" in info
    env.close()


# ------------------------------------------------------------------
# Checkpoint tests
# ------------------------------------------------------------------

def test_checkpoint_save_and_load(agent, tmp_path):
    """El agente guarda y carga checkpoints correctamente."""
    agent.epsilon = 0.42
    agent._learn_steps = 99
    agent.global_step = 1234
    path = agent.save(episode=50)
    assert path.exists()

    # Mutate agent state then restore
    agent.epsilon = 0.0
    agent._learn_steps = 0
    agent.global_step = 0
    loaded_ep = agent.load(path)

    assert loaded_ep == 50
    assert abs(agent.epsilon - 0.42) < 1e-6
    assert agent._learn_steps == 99
    assert agent.global_step == 1234
