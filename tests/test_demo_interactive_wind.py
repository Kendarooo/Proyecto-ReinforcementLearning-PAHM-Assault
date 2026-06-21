import numpy as np

from gym_wrapper.rl_policy import apply_rl_control_step


class MockPolicy:
    def __init__(self, action):
        self.action = np.array(action, dtype=np.float32)
        self.learn_called = False

    def predict(self, obs, deterministic=True):
        self.received_obs = np.array(obs, dtype=np.float32)
        return self.action, None

    def learn(self, *args, **kwargs):
        self.learn_called = True
        raise AssertionError("demo must not train policies")


class DummyActionSpace:
    low = np.array([0.0], dtype=np.float32)
    high = np.array([1.0], dtype=np.float32)
    shape = (1,)


class DummyWindEnv:
    action_space = DummyActionSpace()

    def __init__(self):
        self.actions = []
        self.step_info = {
            "theta_ref": 0.5,
            "tracking_error": -0.2,
            "wind_torque": 0.75,
            "manual_gust_active": True,
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        obs = np.array([0.1, 0.0, 0.5], dtype=np.float32)
        return obs, 0.25, False, False, dict(self.step_info)


def _config():
    return {
        "demo": {
            "interactive_wind": True,
            "show_particles": False,
            "show_wind_torque": True,
        },
        "wind": {
            "manual_gust_enabled": True,
            "manual_gust_torque": 0.5,
            "manual_gust_duration": 1.0,
        },
    }


def test_demo_config_exposes_interactive_wind_flags():
    from gym_wrapper.demo_wind import demo_wind_settings

    settings = demo_wind_settings(_config())

    assert settings["interactive_wind"] is True
    assert settings["show_particles"] is False
    assert settings["show_wind_torque"] is True


def test_manual_gust_changes_resolved_wind_torque():
    from gym_wrapper.demo_wind import ManualGustController, resolve_demo_wind

    gust = ManualGustController(enabled=True, gust_torque=0.5, duration=1.0)
    gust.activate(t=0.0)

    state = resolve_demo_wind(
        base_active=False,
        base_mag=0.0,
        base_angle=0.0,
        theta=0.0,
        max_wind_torque=2.0,
        gust_controller=gust,
        t=0.2,
    )

    assert state.active is True
    assert state.manual_gust_active is True
    assert np.isclose(state.gust_torque, 0.5)
    assert np.isclose(state.wind_torque, 0.5)
    assert np.isclose(state.mag, 0.25)


def test_manual_gust_deactivation_restores_base_torque():
    from gym_wrapper.demo_wind import ManualGustController, resolve_demo_wind

    gust = ManualGustController(enabled=True, gust_torque=0.5, duration=1.0)
    gust.activate(t=0.0)
    gust.deactivate()

    state = resolve_demo_wind(
        base_active=False,
        base_mag=0.0,
        base_angle=0.0,
        theta=0.0,
        max_wind_torque=2.0,
        gust_controller=gust,
        t=0.2,
    )

    assert state.active is False
    assert state.manual_gust_active is False
    assert state.wind_torque == 0.0


def test_handle_wind_event_uses_mock_event_without_graphical_window():
    from gym_wrapper.demo_wind import ManualGustController, handle_wind_event

    class Keys:
        KEYDOWN = 1
        KEYUP = 2
        K_g = 103

    class Event:
        def __init__(self, event_type):
            self.type = event_type
            self.key = Keys.K_g

    gust = ManualGustController(enabled=True, gust_torque=0.5, duration=1.0)

    assert handle_wind_event(Event(Keys.KEYDOWN), gust, pygame_module=Keys, t=0.0) is True
    assert gust.is_active(0.1) is True
    assert handle_wind_event(Event(Keys.KEYUP), gust, pygame_module=Keys, t=0.2) is True
    assert gust.is_active(0.3) is False


def test_rl_action_still_runs_while_manual_gust_is_active():
    env = DummyWindEnv()
    policy = MockPolicy(action=[0.4])
    obs = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    _, _, _, _, info = apply_rl_control_step(env, policy, obs)

    np.testing.assert_allclose(env.actions[-1], np.array([0.4], dtype=np.float32))
    assert info["control_mode"] == "RL"
    assert info["manual_gust_active"] is True
    assert info["wind_torque"] == 0.75
    assert policy.learn_called is False
