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
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from pahm_stage2.feature_extractor import SUPPORTED_FEATURES

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


def _resolve_config_path(path_value: str | Path, config: dict) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config.get("_config_dir", ".")).resolve() / path


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


class Stage2SamplerWindSource:
    """Fuente de viento que consume features muestreadas por el GMM de Etapa 2."""

    source = "stage2_sampler"
    patterns: tuple[str, ...] = ("stage2_sampler",)

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        max_torque: float,
        seed: int | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        from pahm_stage2.wind_sampler import WindSampler

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint del WindSampler de Etapa 2: "
                f"{self.checkpoint_path}"
            )
        self.max_torque = float(max_torque)
        if self.max_torque <= 0.0:
            raise ValueError("max_torque must be positive for stage2_sampler")
        self.feature_names = tuple(feature_names or SUPPORTED_FEATURES)
        unsupported = sorted(set(self.feature_names) - set(SUPPORTED_FEATURES))
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"Features no soportadas por Stage2SamplerWindSource: {joined}")

        self._sampler = WindSampler.from_checkpoint(str(self.checkpoint_path))
        self._rng = np.random.default_rng(seed)
        self.pattern = self.source
        self._time = 0.0
        self._phase = 0.0
        self._features = np.zeros(len(self.feature_names), dtype=np.float32)
        self.reset()

    def set_pattern(self, name: str) -> None:
        self.pattern = self.source
        self.reset()

    def reset(self) -> None:
        self._time = 0.0
        self._phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        sample = self._sampler.sample(1).detach().cpu().numpy().reshape(-1)
        if sample.size < len(self.feature_names):
            raise ValueError(
                "El WindSampler produjo menos features que las configuradas: "
                f"{sample.size} < {len(self.feature_names)}"
            )
        self._features = sample[: len(self.feature_names)].astype(np.float32)

    def sample(
        self,
        dt: float | None = None,
        theta: float = 0.0,
        *,
        t: float | None = None,
    ) -> WindSample:
        step_dt = _resolve_dt(dt, t)
        values = self._feature_values()
        scale = max(self.max_torque, 1e-6)

        bias = float(np.clip(values.get("mean", 0.0) / scale, -1.0, 1.0))
        std = abs(float(values.get("std", 0.0)))
        max_abs = abs(float(values.get("max_abs", 0.0)))
        energy_amp = float(np.sqrt(max(abs(values.get("energy", 0.0)), 0.0)))
        amplitude = float(np.clip(max(std, max_abs, energy_amp) / scale, 0.0, 1.0))

        smoothness = abs(float(values.get("smoothness", 0.0)))
        skewness = float(values.get("skewness", 0.0))
        frequency_hz = float(np.clip(0.1 + smoothness + 0.05 * abs(skewness), 0.05, 2.0))

        torque_norm = bias + amplitude * np.sin(
            (2.0 * np.pi * frequency_hz * self._time) + self._phase
        )
        torque = float(np.clip(torque_norm * self.max_torque, -self.max_torque, self.max_torque))
        self._time += step_dt

        mag = float(np.clip(abs(torque) / self.max_torque, 0.0, 1.0))
        angle = 0.0 if torque >= 0.0 else float(np.pi)
        return WindSample(
            torque=torque,
            mag=mag,
            angle=angle,
            active=True,
            pattern=self.pattern,
            source=self.source,
        )

    def _feature_values(self) -> dict[str, float]:
        return {
            feature_name: float(self._features[index])
            for index, feature_name in enumerate(self.feature_names)
        }


def build_wind_source_from_config(
    config: dict,
    *,
    wind_patterns_config: dict | None = None,
    seed: int | None = None,
    enabled_override: bool | None = None,
    source_override: str | None = None,
    pattern_override: str | None = None,
    max_torque_override: float | None = None,
) -> NoWindSource | WindProcessSource | Stage2SamplerWindSource:
    """Construye la fuente de viento configurada para Etapa 3."""

    wind_settings = config.get("wind", {})
    enabled = (
        bool(wind_settings.get("enabled", False))
        if enabled_override is None
        else bool(enabled_override)
    )
    if not enabled:
        return NoWindSource()

    pattern_config = wind_patterns_config or config["wind_patterns"]
    max_torque = (
        max_torque_override
        if max_torque_override is not None
        else wind_settings.get("max_torque", pattern_config.get("max_wind_torque", 0.0))
    )
    source = source_override or wind_settings.get("source", "wind_process")
    if source == "stage2_sampler":
        checkpoint = (
            wind_settings.get("stage2_sampler_checkpoint")
            or wind_settings.get("sampler_checkpoint")
            or wind_settings.get("checkpoint_path")
        )
        if checkpoint is None:
            raise ValueError(
                "wind.source='stage2_sampler' requiere "
                "wind.stage2_sampler_checkpoint"
            )
        feature_names = (
            wind_settings.get("stage2_sampler_features")
            or config.get("unsupervised", {}).get("features")
            or SUPPORTED_FEATURES
        )
        return Stage2SamplerWindSource(
            _resolve_config_path(checkpoint, config),
            max_torque=float(max_torque),
            seed=seed,
            feature_names=feature_names,
        )

    if source != "wind_process":
        raise ValueError(f"Fuente de viento no soportada: {source}")

    pattern = pattern_override or wind_settings.get("default_pattern", "gust")
    patterns = wind_settings.get("patterns", WindProcess.PATTERNS)

    return WindProcessSource(
        pattern_config,
        pattern=pattern,
        max_torque=float(max_torque),
        seed=seed,
        patterns=patterns,
    )
