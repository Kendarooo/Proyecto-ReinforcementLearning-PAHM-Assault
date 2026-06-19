# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2
"""Fuentes de viento para el entorno Gymnasium del PAHM.

Este módulo separa la elección de la perturbación de la dinámica del entorno:
`LearnedPAHMODE` solo consume una fuente con `sample(dt, theta)`. Así, un
`WindSampler` aprendido puede reemplazar a `WindProcess` sin tocar el lazo
principal de `env.step()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

try:
    from wind_process import WindProcess
except ImportError:
    from gym_wrapper.wind_process import WindProcess


_PATTERN_ALIASES = {
    "bias": "sustained",
}


@dataclass(frozen=True)
class WindSample:
    """Perturbación aplicada en un paso del entorno."""

    torque: float
    mag: float
    angle: float
    active: bool
    pattern: str
    source: str


class WindSource(Protocol):
    """Contrato mínimo consumido por `LearnedPAHMODE`."""

    source: str
    patterns: tuple[str, ...]

    def set_pattern(self, name: str) -> None:
        ...

    def reset(self) -> None:
        ...

    def sample(
        self,
        dt: float | None = None,
        theta: float = 0.0,
        *,
        t: float | None = None,
    ) -> WindSample:
        ...


def _resolve_dt(dt: float | None, t: float | None) -> float:
    return float(dt if dt is not None else (t if t is not None else 0.0))


class NoWindSource:
    """Fuente nula usada cuando el viento automático está desactivado."""

    source = "none"
    patterns: tuple[str, ...] = ("none",)

    def set_pattern(self, name: str) -> None:
        return None

    def reset(self) -> None:
        return None

    def sample(
        self,
        dt: float | None = None,
        theta: float = 0.0,
        *,
        t: float | None = None,
    ) -> WindSample:
        return WindSample(
            torque=0.0,
            mag=0.0,
            angle=0.0,
            active=False,
            pattern="none",
            source=self.source,
        )


class WindProcessSource:
    """Fuente de viento basada en `WindProcess` y escalada a torque externo."""

    source = "wind_process"

    def __init__(
        self,
        wind_config: dict,
        *,
        pattern: str,
        max_torque: float,
        seed: int | None = None,
        patterns: Sequence[str] | None = None,
    ) -> None:
        self._process = WindProcess(wind_config, seed=seed)
        self.max_torque = float(max_torque)
        self.patterns = self._normalize_patterns(patterns or WindProcess.PATTERNS)
        self.pattern = self._normalize_pattern(pattern)
        self.set_pattern(self.pattern)

    @property
    def process(self) -> WindProcess:
        return self._process

    def set_pattern(self, name: str) -> None:
        pattern = self._normalize_pattern(name)
        if pattern not in self.patterns:
            raise ValueError(f"Patrón de viento no permitido por configuración: {name}")
        self.pattern = pattern
        self._process.set_pattern(pattern)

    def reset(self) -> None:
        self._process.reset()

    def sample(
        self,
        dt: float | None = None,
        theta: float = 0.0,
        *,
        t: float | None = None,
    ) -> WindSample:
        step_dt = _resolve_dt(dt, t)
        mag, angle = self._process.step(step_dt)
        torque = mag * self.max_torque * np.cos(theta - angle)
        return WindSample(
            torque=float(torque),
            mag=float(mag),
            angle=float(angle),
            active=True,
            pattern=self.pattern,
            source=self.source,
        )

    @staticmethod
    def _normalize_pattern(name: str) -> str:
        return _PATTERN_ALIASES.get(str(name), str(name))

    @classmethod
    def _normalize_patterns(cls, patterns: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(cls._normalize_pattern(pattern) for pattern in patterns)
        return tuple(pattern for pattern in normalized if pattern in WindProcess.PATTERNS)


def build_wind_source_from_config(
    config: dict,
    *,
    wind_patterns_config: dict | None = None,
    seed: int | None = None,
    enabled_override: bool | None = None,
    source_override: str | None = None,
    pattern_override: str | None = None,
    max_torque_override: float | None = None,
) -> NoWindSource | WindProcessSource:
    """Construye la fuente de viento configurada para Etapa 3."""

    wind_settings = config.get("wind", {})
    enabled = (
        bool(wind_settings.get("enabled", False))
        if enabled_override is None
        else bool(enabled_override)
    )
    if not enabled:
        return NoWindSource()

    source = source_override or wind_settings.get("source", "wind_process")
    if source != "wind_process":
        raise ValueError(f"Fuente de viento no soportada: {source}")

    pattern_config = wind_patterns_config or config["wind_patterns"]
    max_torque = (
        max_torque_override
        if max_torque_override is not None
        else wind_settings.get("max_torque", pattern_config.get("max_wind_torque", 0.0))
    )
    pattern = pattern_override or wind_settings.get("default_pattern", "gust")
    patterns = wind_settings.get("patterns", WindProcess.PATTERNS)

    return WindProcessSource(
        pattern_config,
        pattern=pattern,
        max_torque=float(max_torque),
        seed=seed,
        patterns=patterns,
    )
