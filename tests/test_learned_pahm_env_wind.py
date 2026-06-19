# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

import copy
from pathlib import Path

import numpy as np

from gym_wrapper.learned_pahm_ode import _load_wrapper_config
from gym_wrapper.learned_pahm_ode import LearnedPAHMODE
from gym_wrapper.wind_source import (
    NoWindSource,
    WindProcessSource,
    build_wind_source_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "pahm_model" / "pahm_fast_v2_best.pth"


def _make_env(**kwargs) -> LearnedPAHMODE:
    return LearnedPAHMODE(
        render_mode=None,
        model_path=str(MODEL_PATH),
        reset_angle_deg=720,
        max_wind_torque=20.0,
        **kwargs,
    )


def _wrapper_config_with_wind(enabled: bool) -> dict:
    config = copy.deepcopy(_load_wrapper_config())
    config["wind"] = {
        "enabled": enabled,
        "source": "wind_process",
        "patterns": ["calm", "gust", "sustained", "turbulent"],
        "default_pattern": "turbulent",
        "max_torque": 20.0,
        "stochastic": True,
    }
    config["domain_randomization"] = {
        "enabled": True,
        "reset_options": {"randomize": True},
    }
    return config


def test_wind_source_can_be_built_from_enabled_config():
    config = _wrapper_config_with_wind(enabled=True)

    source = build_wind_source_from_config(config, seed=123)

    assert isinstance(source, WindProcessSource)
    sample = source.sample(dt=0.02, theta=0.0)
    assert sample.active is True
    assert sample.source == "wind_process"
    assert sample.pattern == "turbulent"
    assert np.isfinite(sample.torque)
    assert 0.0 <= sample.mag <= 1.0


def test_wind_source_can_be_disabled_from_config():
    config = _wrapper_config_with_wind(enabled=False)

    source = build_wind_source_from_config(config, seed=123)
    sample = source.sample(dt=0.02, theta=0.0)

    assert isinstance(source, NoWindSource)
    assert sample.active is False
    assert sample.source == "none"
    assert sample.torque == 0.0


def test_headless_env_with_automatic_wind_returns_valid_info():
    env = _make_env(
        enable_wind=True,
        wind_pattern="turbulent",
        wind_seed=42,
        theta_ref=1.0,
    )
    try:
        obs, reset_info = env.reset(seed=7, options={"randomize": True})
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.2]))

        assert env.observation_space.shape == (3,)
        assert obs.shape == (3,)
        assert next_obs.shape == (3,)
        assert np.isclose(obs[2], 1.0)
        assert np.isclose(next_obs[2], 1.0)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert reset_info["wind_automatic"] is True
        assert reset_info["theta_ref"] == 1.0
        assert info["wind_automatic"] is True
        assert info["wind_active"] is True
        assert info["wind_pattern"] == "turbulent"
        assert info["configured_wind_pattern"] == "turbulent"
        assert info["theta_ref"] == 1.0
        assert np.isfinite(info["tracking_error"])
        assert np.isfinite(info["abs_tracking_error"])
        assert env.configured_wind_pattern == "turbulent"
        assert env.active_wind_pattern == "turbulent"
        assert 0.0 <= info["wind_mag"] <= 1.0
        assert np.isfinite(info["wind_torque"])
    finally:
        env.close()


def test_env_can_enable_automatic_wind_from_external_config():
    config = _wrapper_config_with_wind(enabled=True)
    env = LearnedPAHMODE(
        render_mode=None,
        model_path=str(MODEL_PATH),
        reset_angle_deg=720,
        config=config,
        wind_seed=42,
    )
    try:
        obs, reset_info = env.reset(seed=7, options={"randomize": True})
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.2]))

        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert reset_info["wind_automatic"] is True
        assert info["wind_automatic"] is True
        assert info["wind_pattern"] == "turbulent"
        assert np.isfinite(info["wind_torque"])
    finally:
        env.close()


def test_env_can_disable_automatic_wind_from_external_config():
    config = _wrapper_config_with_wind(enabled=False)
    env = LearnedPAHMODE(
        render_mode=None,
        model_path=str(MODEL_PATH),
        reset_angle_deg=720,
        config=config,
    )
    try:
        _, reset_info = env.reset(seed=7, options={"randomize": True})
        _, _, _, _, info = env.step(np.array([0.2]))

        assert reset_info["wind_automatic"] is False
        assert info["wind_automatic"] is False
        assert info["wind_torque"] == 0.0
    finally:
        env.close()


def test_manual_wind_still_works_when_automatic_wind_is_disabled():
    env = _make_env(enable_wind=False)
    try:
        env.reset(seed=1, options={"initial_state": [0.0, 0.0]})
        env.set_wind(active=True, mag=0.5, angle=0.0)

        _, _, _, _, info = env.step(np.array([0.2]))

        assert info["wind_automatic"] is False
        assert info["configured_wind_pattern"] == "gust"
        assert info["wind_active"] is True
        assert info["wind_mag"] == 0.5
        assert info["wind_angle"] == 0.0
        assert np.isclose(info["wind_torque"], 10.0)
    finally:
        env.close()


def test_set_wind_does_not_override_automatic_wind_process():
    env = _make_env(enable_wind=True, wind_pattern="sustained", wind_seed=123)
    try:
        env.reset(seed=5, options={"initial_state": [0.0, 0.0]})
        env.set_wind(active=True, mag=0.99, angle=1.5)

        _, _, _, _, info = env.step(np.array([0.2]))

        assert info["wind_automatic"] is True
        assert info["wind_pattern"] == "sustained"
        assert info["configured_wind_pattern"] == "sustained"
        assert not np.isclose(info["wind_mag"], 0.99)
        assert not np.isclose(info["wind_angle"], 1.5)
    finally:
        env.close()


def test_wind_process_is_reproducible_with_fixed_wind_seed():
    env_a = _make_env(enable_wind=True, wind_pattern="gust", wind_seed=99)
    env_b = _make_env(enable_wind=True, wind_pattern="gust", wind_seed=99)
    try:
        env_a.reset(seed=11, options={"initial_state": [0.0, 0.0]})
        env_b.reset(seed=11, options={"initial_state": [0.0, 0.0]})

        seq_a = [env_a.step(np.array([0.2]))[-1]["wind_mag"] for _ in range(20)]
        seq_b = [env_b.step(np.array([0.2]))[-1]["wind_mag"] for _ in range(20)]

        np.testing.assert_allclose(seq_a, seq_b)
    finally:
        env_a.close()
        env_b.close()


def test_reset_seed_controls_randomized_wind_pattern_choice():
    env_a = _make_env(
        enable_wind=True,
        wind_pattern="calm",
        wind_seed=1,
        randomize_wind_pattern=True,
    )
    env_b = _make_env(
        enable_wind=True,
        wind_pattern="calm",
        wind_seed=1,
        randomize_wind_pattern=True,
    )
    try:
        _, info_a = env_a.reset(seed=123, options={"initial_state": [0.0, 0.0]})
        _, info_b = env_b.reset(seed=123, options={"initial_state": [0.0, 0.0]})

        assert info_a["wind_pattern"] == info_b["wind_pattern"]
        assert env_a.configured_wind_pattern == "calm"
        assert env_b.configured_wind_pattern == "calm"
        assert info_a["configured_wind_pattern"] == "calm"
        assert info_b["configured_wind_pattern"] == "calm"
        assert env_a.active_wind_pattern == info_a["wind_pattern"]
        assert env_b.active_wind_pattern == info_b["wind_pattern"]
    finally:
        env_a.close()
        env_b.close()


def test_set_theta_ref_updates_observation_and_info():
    env = _make_env(enable_wind=False, theta_ref=0.25)
    try:
        obs, info = env.reset(seed=1, options={"initial_state": [0.0, 0.0]})
        assert np.isclose(obs[2], 0.25)
        assert np.isclose(info["theta_ref"], 0.25)

        env.set_theta_ref(0.75)
        obs, _, _, _, info = env.step(np.array([0.0]))

        assert np.isclose(obs[2], 0.75)
        assert np.isclose(info["theta_ref"], 0.75)
        assert np.isclose(info["tracking_error"], 0.75 - obs[0])
        assert np.isclose(info["abs_tracking_error"], abs(0.75 - obs[0]))
    finally:
        env.close()


def test_changing_theta_ref_changes_tracking_reward():
    env = _make_env(enable_wind=False)
    try:
        env.reset(seed=2, options={"initial_state": [0.0, 0.0]})
        env.set_theta_ref(0.0)
        _, reward_at_zero_ref, _, _, _ = env.step(np.array([0.0]))

        env.reset(seed=2, options={"initial_state": [0.0, 0.0]})
        env.set_theta_ref(1.0)
        _, reward_at_one_ref, _, _, _ = env.step(np.array([0.0]))

        assert reward_at_one_ref < reward_at_zero_ref
    finally:
        env.close()


def test_reset_randomize_with_theta_ref_keeps_valid_observation():
    env = _make_env(enable_wind=False, theta_ref=0.5)
    try:
        obs, info = env.reset(seed=3, options={"randomize": True})

        assert obs.shape == (3,)
        assert np.isfinite(obs).all()
        assert np.isclose(obs[2], 0.5)
        assert np.isclose(info["theta_ref"], 0.5)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_reset_options_can_override_theta_ref_for_episode():
    env = _make_env(enable_wind=True, wind_pattern="gust", wind_seed=12, theta_ref=0.0)
    try:
        obs, info = env.reset(
            seed=4,
            options={"randomize": True, "theta_ref": 1.0},
        )

        assert obs.shape == (3,)
        assert np.isfinite(obs).all()
        assert np.isclose(obs[2], 1.0)
        assert np.isclose(info["theta_ref"], 1.0)
        assert info["wind_automatic"] is True
    finally:
        env.close()
