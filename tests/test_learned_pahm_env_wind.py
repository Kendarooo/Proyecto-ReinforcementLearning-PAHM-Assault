# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

from pathlib import Path

import numpy as np

from gym_wrapper.learned_pahm_ode import LearnedPAHMODE


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


def test_headless_env_with_automatic_wind_returns_valid_info():
    env = _make_env(enable_wind=True, wind_pattern="turbulent", wind_seed=42)
    try:
        obs, reset_info = env.reset(seed=7, options={"randomize": True})
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.2]))

        assert obs.shape == (2,)
        assert next_obs.shape == (2,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert truncated is False
        assert reset_info["wind_automatic"] is True
        assert info["wind_automatic"] is True
        assert info["wind_active"] is True
        assert info["wind_pattern"] == "turbulent"
        assert info["configured_wind_pattern"] == "turbulent"
        assert env.configured_wind_pattern == "turbulent"
        assert env.active_wind_pattern == "turbulent"
        assert 0.0 <= info["wind_mag"] <= 1.0
        assert np.isfinite(info["wind_torque"])
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
