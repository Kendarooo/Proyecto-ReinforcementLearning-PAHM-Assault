# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Versión 0.1.0
# Proyecto 2
"""Fuente exógena de perturbación de viento para el PAHM.

Este módulo es la ÚNICA fuente de verdad (SSOT) de la forma temporal del
viento. Lo consumen dos clientes distintos con el mismo contrato: el entorno
Gymnasium (modo interactivo del GUI) y la herramienta de grabación headless.

Decisiones de diseño acordadas (no evidentes en el código; conservar):

  * "0 m/s NO es lo mismo que ignorar el viento". La activación (active) es
    responsabilidad del CONSUMIDOR, no de esta clase. Si el viento está
    inactivo, NO debe llamarse step(): no se inyecta ningún torque exógeno
    (cero estructural), en lugar de inyectar un viento de magnitud cero.
  * step(dt) produce la fuente exógena (mag, phi) INDEPENDIENTE de theta. El
    acople con el ángulo (tau = mag * max_wind_torque * cos(theta - phi)) lo
    aplica el entorno, no esta clase.
  * Los parámetros de cada patrón viven en config.json -> "wind_patterns".
    Los ángulos se expresan en GRADOS en la configuración (legibilidad y
    consistencia con el GUI), pero esta clase los ENTREGA en RADIANES.
  * Reproducibilidad (NFR-4): el RNG es propio e inyectado por el constructor.
    seed=None => no determinista (convención NumPy/Gym). Misma semilla =>
    misma secuencia; semillas distintas => instancias distintas del mismo
    patrón (p. ej. el instante de disparo de la ráfaga).
"""

import numpy as np


class WindProcess:
    """Genera la señal exógena (mag, phi) de un patrón de viento en el tiempo.

    mag en [0, 1] (magnitud normalizada; el entorno la escala con
    max_wind_torque). phi en radianes.
    """

    PATTERNS = ("calm", "gust", "sustained", "turbulent")

    def __init__(self, wind_config, seed=None):
        """
        Args:
            wind_config: sección "wind_patterns" de config.json ya parseada.
            seed: semilla para el RNG propio. None => no determinista.
        """
        self._cfg = wind_config
        self._rng = np.random.default_rng(seed)
        self._pattern = None
        self._t = 0.0
        self._instance = {}  # parámetros muestreados por episodio

    def set_pattern(self, name):
        """Selecciona el patrón activo y reinicia su ejecución (t=0).

        Muestrea los parámetros aleatorios por episodio usando el RNG propio.
        Habilita el cambio en caliente que requiere el selector del GUI.
        """
        if name not in self.PATTERNS:
            raise ValueError(f"Patrón de viento desconocido: {name}")
        self._pattern = name
        self._t = 0.0
        self._instance = self._sample_instance(name)

    def reset(self):
        """Reinicia el reloj y re-muestrea la instancia del patrón actual."""
        if self._pattern is not None:
            self.set_pattern(self._pattern)

    def step(self, dt):
        """Avanza dt segundos y devuelve (mag, phi_rad) del patrón actual.

        No debe llamarse si el viento está inactivo (ver encabezado del módulo).
        """
        if self._pattern is None:
            raise RuntimeError("step() sin patrón; llame a set_pattern() antes.")
        mag, phi_rad = self._evaluate(self._pattern, self._t)
        self._t += dt
        return float(np.clip(mag, 0.0, 1.0)), float(phi_rad)

    # --- Muestreo de instancia (valores aleatorios por episodio) ---
    def _sample_instance(self, name):
        p = self._cfg[name]
        if name == "gust":
            return {"t_trigger": self._rng.uniform(p["trigger_min_s"],
                                                   p["trigger_max_s"])}
        if name == "sustained":
            return {"t_onset": self._rng.uniform(p["onset_min_s"],
                                                 p["onset_max_s"])}
        return {}  # calm y turbulent no requieren muestreo por instancia

    # --- Evaluación del patrón en el instante t ---
    def _evaluate(self, name, t):
        p = self._cfg[name]
        if name == "calm":
            # Ruido pequeño no negativo: ausencia de viento estructurado.
            return abs(self._rng.normal(0.0, p["noise_std"])), np.deg2rad(p["phi_deg"])
        if name == "gust":
            return self._eval_gust(p, t)
        if name == "sustained":
            return self._eval_sustained(p, t)
        return self._eval_turbulent(p, t)

    def _eval_gust(self, p, t):
        # Ráfaga: ventana de coseno alzado (suave en ambos extremos),
        # disparada en t_trigger y de ancho duration_s.
        elapsed = t - self._instance["t_trigger"]
        if 0.0 <= elapsed <= p["duration_s"]:
            env = 0.5 * (1.0 - np.cos(2.0 * np.pi * elapsed / p["duration_s"]))
            mag = p["peak_mag"] * env
        else:
            mag = 0.0
        return mag, np.deg2rad(p["phi_deg"])

    def _eval_sustained(self, p, t):
        # Sesgo sostenido ("puerta abierta"): escalón en t_onset que se mantiene.
        mag = p["level"] if t >= self._instance["t_onset"] else 0.0
        return mag, np.deg2rad(p["phi_deg"])

    def _eval_turbulent(self, p, t):
        # Turbulento: oscilación + sesgo + ruido (recortado a [0,1] en step()).
        osc = p["amp"] * np.sin(2.0 * np.pi * p["freq_hz"] * t)
        return p["bias"] + osc + self._rng.normal(0.0, p["noise_std"]), np.deg2rad(p["phi_deg"])
