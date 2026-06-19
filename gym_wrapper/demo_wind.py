"""Control de ráfagas interactivas para la demo visual del PAHM."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi

import numpy as np


@dataclass(frozen=True)
class DemoWindState:
    active: bool
    mag: float
    angle: float
    wind_torque: float
    base_torque: float
    gust_torque: float
    manual_gust_active: bool
    particles_enabled: bool = True

    def info(self) -> dict:
        return {
            "wind_torque": float(self.wind_torque),
            "base_wind_torque": float(self.base_torque),
            "manual_gust_torque": float(self.gust_torque),
            "manual_gust_active": bool(self.manual_gust_active),
            "wind_particles_enabled": bool(self.particles_enabled),
        }


@dataclass
class ManualGustController:
    enabled: bool = False
    gust_torque: float = 0.0
    duration: float = 1.0
    _active_until: float | None = None

    @classmethod
    def from_config(cls, config: dict) -> "ManualGustController":
        wind_config = config.get("wind", {})
        return cls(
            enabled=bool(wind_config.get("manual_gust_enabled", False)),
            gust_torque=float(wind_config.get("manual_gust_torque", 0.0)),
            duration=float(wind_config.get("manual_gust_duration", 1.0)),
        )

    def activate(self, t: float = 0.0) -> None:
        if not self.enabled:
            return
        self._active_until = float(t) + max(0.0, float(self.duration))

    def deactivate(self) -> None:
        self._active_until = None

    def is_active(self, t: float = 0.0) -> bool:
        return self._active_until is not None and float(t) <= self._active_until

    def sample(self, t: float = 0.0, theta: float = 0.0) -> float:
        if not self.enabled or not self.is_active(t):
            self.deactivate()
            return 0.0
        return float(self.gust_torque)


def demo_wind_settings(config: dict) -> dict:
    demo_config = config.get("demo", {})
    return {
        "interactive_wind": bool(demo_config.get("interactive_wind", False)),
        "show_particles": bool(demo_config.get("show_particles", True)),
        "show_wind_torque": bool(demo_config.get("show_wind_torque", False)),
    }


def handle_wind_event(
    event,
    wind_controller: ManualGustController,
    *,
    pygame_module=None,
    t: float = 0.0,
) -> bool:
    if pygame_module is None:
        try:
            import pygame as pygame_module
        except ImportError:
            return False

    keydown = getattr(pygame_module, "KEYDOWN", None)
    keyup = getattr(pygame_module, "KEYUP", None)
    gust_key = getattr(pygame_module, "K_g", None)

    if getattr(event, "key", None) != gust_key:
        return False
    if getattr(event, "type", None) == keydown:
        wind_controller.activate(t)
        return True
    if getattr(event, "type", None) == keyup:
        wind_controller.deactivate()
        return True
    return False


def resolve_demo_wind(
    *,
    base_active: bool,
    base_mag: float,
    base_angle: float,
    theta: float,
    max_wind_torque: float,
    gust_controller: ManualGustController | None = None,
    t: float = 0.0,
    particles_enabled: bool = True,
) -> DemoWindState:
    max_torque = float(max_wind_torque)
    theta = float(theta)
    base_torque = 0.0
    if base_active:
        base_torque = float(base_mag) * max_torque * cos(theta - float(base_angle))

    gust_torque = 0.0
    manual_gust_active = False
    if gust_controller is not None:
        gust_torque = gust_controller.sample(t=t, theta=theta)
        manual_gust_active = gust_controller.is_active(t)

    requested_torque = base_torque + gust_torque
    total_torque = _clip_torque(requested_torque, max_torque)
    mag, angle = _torque_to_wind_vector(total_torque, theta, max_torque)
    active = bool(base_active or manual_gust_active or abs(total_torque) > 0.0)

    return DemoWindState(
        active=active,
        mag=mag,
        angle=angle,
        wind_torque=total_torque,
        base_torque=base_torque,
        gust_torque=gust_torque,
        manual_gust_active=manual_gust_active,
        particles_enabled=particles_enabled,
    )


def _torque_to_wind_vector(
    torque: float,
    theta: float,
    max_wind_torque: float,
) -> tuple[float, float]:
    if max_wind_torque <= 0.0 or np.isclose(torque, 0.0):
        return 0.0, float(theta)
    mag = min(1.0, abs(float(torque)) / float(max_wind_torque))
    angle = float(theta) if torque >= 0.0 else float(theta) + pi
    return float(mag), angle


def _clip_torque(torque: float, max_wind_torque: float) -> float:
    if max_wind_torque <= 0.0:
        return 0.0
    return float(np.clip(float(torque), -float(max_wind_torque), float(max_wind_torque)))
