"""
tests/test_environment.py — v1.0
Pruebas unitarias para la Etapa 0 (DQN Boxing) [NFR-6].

Cubre:
    - [NFR-6d] El entorno Gymnasium produce observaciones válidas.
    - [NFR-6e] Los puntos de control se guardan y cargan correctamente.
    - Tests propios de Etapa 0:
        * El entorno tiene el espacio de acción discreto esperado (18 acciones).
        * El entorno tiene el espacio de observación esperado tras frame_stack.
        * El config se carga correctamente y contiene las claves obligatorias.
        * El modelo cargado desde checkpoint produce acciones válidas.
        * La semilla produce trayectorias reproducibles.

Uso:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import json
import os
import tempfile
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# ── Fixtures ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config_dqn.json"


@pytest.fixture(scope="module")
def config() -> dict:
    """Carga el config centralizado una sola vez para todos los tests."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def register_ale():
    """Registra ale_py en Gymnasium una sola vez para el módulo."""
    gym.register_envs(ale_py)


@pytest.fixture(scope="module")
def vec_env(config, register_ale):
    """Crea el entorno vectorizado con frame_stack para tests que lo necesiten."""
    env = make_atari_env(
        config["environment"]["env_id"],
        n_envs=1,
        seed=config["project"]["seed"],
    )
    env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
    yield env
    env.close()


@pytest.fixture(scope="module")
def tiny_model(config, vec_env):
    """Crea un modelo DQN mínimo (sin entrenamiento) para tests de I/O."""
    hp = config["hyperparameters"]
    model = DQN(
        policy=config["model"]["policy"],
        env=vec_env,
        learning_rate=hp["learning_rate"],
        buffer_size=1000,          # Mínimo para que no consuma memoria en tests
        batch_size=hp["batch_size"],
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        verbose=0,
        device="cpu",              # Tests siempre en CPU para reproducibilidad
    )
    return model


# ── Tests de configuración ────────────────────────────────────────────────────

class TestConfig:
    """Verifica que config_dqn.json tiene la estructura esperada [NFR-1]."""

    REQUIRED_TOP_KEYS = {"version", "project", "environment", "model",
                         "hyperparameters", "checkpoints", "paths"}
    REQUIRED_HP_KEYS = {"learning_rate", "buffer_size", "batch_size", "gamma",
                        "exploration_fraction", "exploration_initial_eps",
                        "exploration_final_eps", "learning_starts",
                        "train_freq", "target_update_interval",
                        "max_grad_norm", "total_timesteps"}

    def test_config_file_exists(self):
        assert CONFIG_PATH.exists(), f"No se encontró {CONFIG_PATH}"

    def test_config_top_level_keys(self, config):
        missing = self.REQUIRED_TOP_KEYS - set(config.keys())
        assert not missing, f"Faltan claves en config: {missing}"

    def test_config_hyperparameter_keys(self, config):
        missing = self.REQUIRED_HP_KEYS - set(config["hyperparameters"].keys())
        assert not missing, f"Faltan hiperparámetros en config: {missing}"

    def test_config_seed_is_integer(self, config):
        assert isinstance(config["project"]["seed"], int)

    def test_config_n_envs_positive(self, config):
        assert config["environment"]["n_envs"] >= 1

    def test_config_buffer_size_reasonable(self, config):
        """El buffer no debe exceder 4GB de RAM con frame_stack=4.

        Con optimize_memory_usage=True el buffer almacena uint8 (1 byte/pixel).
        Para buffer_size=100,000: 84x84x4x100000 = ~2.82 GB, dentro del limite.
        """
        buffer_size = config["hyperparameters"]["buffer_size"]
        memory_bytes = 84 * 84 * 4 * buffer_size
        assert memory_bytes < 4 * 1024**3, (
            f"buffer_size={buffer_size} excede 4GB de RAM estimada "
            f"({memory_bytes / 1024**3:.2f} GB)"
        )


# ── Tests del entorno [NFR-6d] ────────────────────────────────────────────────

class TestEnvironment:
    """[NFR-6d] El entorno Gymnasium produce observaciones válidas."""

    def test_env_creates_without_error(self, register_ale, config):
        """El entorno se instancia sin excepciones."""
        env = make_atari_env(
            config["environment"]["env_id"],
            n_envs=1,
            seed=0,
        )
        env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
        obs = env.reset()
        assert obs is not None
        env.close()

    def test_action_space_is_discrete_18(self, vec_env):
        """Boxing tiene exactamente 18 acciones discretas."""
        # VecEnv expone el espacio de acción del env interno
        n_actions = vec_env.action_space.n
        assert n_actions == 18, (
            f"Se esperaban 18 acciones, se encontraron {n_actions}"
        )

    def test_observation_shape_with_frame_stack(self, vec_env, config):
        """La observacion tiene shape (n_envs, H, W, frame_stack) tras VecFrameStack.

        VecFrameStack apila en el ultimo eje (canal), produciendo HWC.
        SB3 transpone internamente a CHW solo al pasar tensores a la red.
        """
        obs = vec_env.reset()
        frame_stack = config["environment"]["frame_stack"]
        assert obs.shape == (1, 84, 84, frame_stack), (
            f"Shape inesperado: {obs.shape}"
        )

    def test_observation_dtype_is_uint8(self, vec_env):
        """Los frames deben ser uint8 (0-255) para eficiencia de memoria."""
        obs = vec_env.reset()
        assert obs.dtype == np.uint8, f"dtype inesperado: {obs.dtype}"

    def test_step_returns_valid_structure(self, vec_env, config):
        """Un paso retorna obs, reward, done, info con tipos correctos."""
        vec_env.reset()
        action = np.array([vec_env.action_space.sample()])
        obs, rewards, dones, infos = vec_env.step(action)

        assert obs.shape[0] == 1                   # n_envs
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert isinstance(infos, (list, tuple))

    def test_observation_values_in_valid_range(self, vec_env):
        """Los píxeles deben estar en [0, 255]."""
        obs = vec_env.reset()
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_reset_produces_different_obs_with_different_seeds(
        self, register_ale, config
    ):
        """Semillas distintas producen estados iniciales distintos."""
        def get_obs(seed):
            env = make_atari_env(
                config["environment"]["env_id"], n_envs=1, seed=seed
            )
            env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
            obs = env.reset()
            env.close()
            return obs

        obs_a = get_obs(seed=0)
        obs_b = get_obs(seed=999)
        assert not np.array_equal(obs_a, obs_b), (
            "Semillas distintas produjeron observaciones idénticas"
        )


# ── Tests de checkpoints [NFR-6e] ─────────────────────────────────────────────

class TestCheckpoints:
    """[NFR-6e] Los puntos de control se guardan y cargan correctamente."""

    def test_model_saves_to_disk(self, tiny_model):
        """El modelo se guarda como archivo .zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            tiny_model.save(str(save_path))
            assert (save_path.with_suffix(".zip")).exists(), (
                "El archivo .zip no fue creado"
            )

    def test_model_loads_from_disk(self, tiny_model, vec_env):
        """El modelo cargado desde disco es funcional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            tiny_model.save(str(save_path))

            loaded = DQN.load(str(save_path), env=vec_env, device="cpu")
            assert loaded is not None

    def test_loaded_model_produces_valid_actions(self, tiny_model, vec_env):
        """El modelo cargado produce acciones dentro del espacio válido."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            tiny_model.save(str(save_path))

            loaded = DQN.load(str(save_path), env=vec_env, device="cpu")
            obs = vec_env.reset()
            action, _ = loaded.predict(obs, deterministic=True)

            assert vec_env.action_space.contains(action[0]), (
                f"Acción inválida: {action[0]}"
            )

    def test_loaded_model_preserves_policy_type(self, tiny_model, vec_env, config):
        """El modelo cargado mantiene el tipo de política original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            tiny_model.save(str(save_path))

            loaded = DQN.load(str(save_path), env=vec_env, device="cpu")
            assert loaded.policy.__class__.__name__ == "CnnPolicy"

    def test_save_and_load_roundtrip_same_action(self, tiny_model, vec_env):
        """El modelo antes y después de guardar/cargar produce la misma acción."""
        obs = vec_env.reset()
        action_before, _ = tiny_model.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            tiny_model.save(str(save_path))
            loaded = DQN.load(str(save_path), env=vec_env, device="cpu")

        action_after, _ = loaded.predict(obs, deterministic=True)
        assert np.array_equal(action_before, action_after), (
            f"La acción cambió tras guardar/cargar: {action_before} → {action_after}"
        )


# ── Tests de reproducibilidad [NFR-4] ─────────────────────────────────────────

class TestReproducibility:
    """[NFR-4] La semilla produce trayectorias reproducibles."""

    def test_same_seed_produces_same_initial_obs(self, register_ale, config):
        """La misma semilla produce la misma observación inicial."""
        def get_obs(seed):
            env = make_atari_env(
                config["environment"]["env_id"], n_envs=1, seed=seed
            )
            env = VecFrameStack(env, n_stack=config["environment"]["frame_stack"])
            obs = env.reset()
            env.close()
            return obs

        seed = config["project"]["seed"]
        obs_1 = get_obs(seed)
        obs_2 = get_obs(seed)
        assert np.array_equal(obs_1, obs_2), (
            "La misma semilla produjo observaciones distintas"
        )