# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2
# Versión 1.3.0

import sys
import os
import torch
import numpy as np
import gymnasium as gym
import json
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from pathlib import Path
from typing import Optional

# Añadir directorio hermano al path para cargar el modelo
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '../pahm_model'))

try:
    from pahm_ode import PAHMHybridODE
    from pahm_fast import PAHMFastModel
except ImportError:
    raise ImportError("No se pudieron importar los modelos. Verifique ../pahm_model/")

try:
    from wind_source import build_wind_source_from_config
except ImportError:
    from gym_wrapper.wind_source import build_wind_source_from_config


def _load_wrapper_config() -> dict:
    """Carga config.json anclado al archivo, no al directorio de ejecucion."""
    config_path = Path(__file__).resolve().with_name("config.json")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _require_pygame():
    """Importa Pygame solo cuando se necesita renderizado."""
    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as exc:
        raise DependencyNotInstalled(
            "pygame required for render_mode='human' or 'rgb_array'"
        ) from exc
    return pygame, gfxdraw



class LearnedPAHMODE(gym.Env):
    """
    Entorno Gymnasium específico para el modelo Neural ODE (PAHM).
    
    A diferencia del modelo GRU (Caja Negra), este entorno simula la física
    integrando paso a paso las ecuaciones diferenciales aprendidas.
    
    Estado interno:
        [theta_rad, theta_dot_rad] (Sistema Internacional)
     
    
    Observación:
        [theta_radians, theta_dot_rad, theta_ref_rad]

        
    Acción:
        [PWM] (0.0 a 1.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 model_path="pahm_ode_best.pth",
                 reset_angle_deg=120,
                 dt=0.02,                 # 50Hz por defecto
                 max_wind_torque: float | None = None,
                 enable_wind: bool | None = None,
                 wind_pattern: str | None = None,
                 wind_config: dict | None = None,
                 wind_seed: int | None = None,
                 randomize_wind_pattern: bool = False,
                 theta_ref: float | None = None,
                 reward_weights: dict | None = None,
                 config: dict | None = None,
                 wind_source: str | None = None):
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.dt = dt
        self.reset_angle_deg = reset_angle_deg
                
        self.device = "cpu" # CPU es preferible para inferencia paso a paso (baja latencia)
        wrapper_config = config or _load_wrapper_config()
        wind_settings = wrapper_config.get("wind", {})
        control_settings = wrapper_config.get("control", {})
        reward_settings = wrapper_config.get("reward", {})
        
        # Cargar Modelo ODE
        print(f"🔄 Cargando modelo ODE desde: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            arch = checkpoint.get('architecture', 'ode')
            
            if arch == 'fast':
                print("⚡ Arquitectura: Fast Physics RNN")
                self.model = PAHMFastModel(device=self.device, dt=self.dt)
            else:
                print("🧠 Arquitectura: Neural ODE")
                self.model = PAHMHybridODE(device=self.device)

            phys_params = self.model.load_model(self.model_path)
            self.model.eval()
            print(f"✅ Modelo cargado. Física aprendida: {phys_params}")
        except Exception as e:
            raise ValueError(f"Error fatal cargando el modelo ODE: {e}")

        # Espacios de Gym
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.limit_rad = float(np.deg2rad(self.reset_angle_deg))
        self.theta_ref_min = float(control_settings.get("theta_ref_min", -self.limit_rad))
        self.theta_ref_max = float(control_settings.get("theta_ref_max", self.limit_rad))
        if self.theta_ref_min > self.theta_ref_max:
            raise ValueError("control.theta_ref_min cannot exceed control.theta_ref_max")

        self.theta_ref = 0.0
        configured_theta_ref = control_settings.get("theta_ref", 0.0)
        self.set_theta_ref(configured_theta_ref if theta_ref is None else theta_ref)
        self.reward_weights = self._build_reward_weights(
            reward_settings=reward_settings,
            overrides=reward_weights,
        )

        # Observación: [ángulo, velocidad angular, ángulo objetivo]
        low_obs = np.array(
            [-self.limit_rad, -10.0, self.theta_ref_min],
            dtype=np.float32,
        )
        high_obs = np.array(
            [self.limit_rad, 10.0, self.theta_ref_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Estado visual
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        
        # Estado interno (Normalizado)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        
        # Perturbación de viento (estado latente de simulación)
        self.wind_active = False
        self.wind_mag = 0.0
        self.wind_angle = 0.0
        self.wind_torque = 0.0
        self.enable_wind = (
            bool(wind_settings.get("enabled", False))
            if enable_wind is None
            else bool(enable_wind)
        )
        self.configured_wind_pattern = (
            wind_pattern
            if wind_pattern is not None
            else wind_settings.get("default_pattern", "gust")
        )
        self.active_wind_pattern = self.configured_wind_pattern
        self.randomize_wind_pattern = randomize_wind_pattern
        self.wind_seed = wind_seed
        self.wind_patterns_config = wind_config or wrapper_config["wind_patterns"]
        self.max_wind_torque = (
            float(max_wind_torque)
            if max_wind_torque is not None
            else float(
                wind_settings.get(
                    "max_torque",
                    self.wind_patterns_config.get("max_wind_torque", 0.0),
                )
            )
        )

        self.wind_source = build_wind_source_from_config(
            wrapper_config,
            wind_patterns_config=self.wind_patterns_config,
            seed=wind_seed,
            enabled_override=self.enable_wind,
            source_override=wind_source,
            pattern_override=self.configured_wind_pattern,
            max_torque_override=self.max_wind_torque,
        )
        self.wind_process = getattr(self.wind_source, "process", None)
        if self.enable_wind:
            self.wind_active = True

        # Cargar configuración visual de partículas
        self.wind_cfg = {"count": 40, "speed": 15.0, "length": 20.0, "color": [100, 240, 255]}
        if "components" in wrapper_config and "wind_particles" in wrapper_config["components"]:
            self.wind_cfg = wrapper_config["components"]["wind_particles"]
        
        self.particles = np.random.rand(self.wind_cfg["count"], 2) * self.screen_dim        
        
    def set_wind(self, active, mag, angle):
        """Actualiza viento manual para demo; no sobreescribe WindProcess automatico."""
        if self.enable_wind:
            return
        self.wind_active = active
        self.wind_mag = mag
        self.wind_angle = angle

    def _build_reward_weights(
        self,
        *,
        reward_settings: dict,
        overrides: dict | None,
    ) -> dict:
        """Normaliza pesos de recompensa desde config y overrides historicos."""
        weights = {
            "error": float(
                reward_settings.get(
                    "tracking_error_weight",
                    reward_settings.get("error", 4.0),
                )
            ),
            "velocity": float(
                reward_settings.get(
                    "velocity_weight",
                    reward_settings.get("velocity", 0.1),
                )
            ),
            "action": float(
                reward_settings.get(
                    "control_weight",
                    reward_settings.get("action", 0.01),
                )
            ),
        }
        for key, value in (overrides or {}).items():
            normalized_key = {
                "tracking_error_weight": "error",
                "velocity_weight": "velocity",
                "control_weight": "action",
            }.get(key, key)
            weights[normalized_key] = float(value)
        return weights

    def set_theta_ref(self, value: float) -> None:
        """Actualiza el ángulo objetivo usado por la observación y recompensa."""
        theta_ref = float(value)
        if not np.isfinite(theta_ref):
            raise ValueError("theta_ref must be finite")
        if theta_ref < self.theta_ref_min or theta_ref > self.theta_ref_max:
            raise ValueError(
                f"theta_ref={theta_ref} outside configured bounds "
                f"[{self.theta_ref_min}, {self.theta_ref_max}]"
            )
        self.theta_ref = theta_ref

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.state[0], self.state[1], self.theta_ref],
            dtype=np.float32,
        )
        
    def _sample_wind_pattern(self) -> str:
        if self.randomize_wind_pattern:
            patterns = list(getattr(self.wind_source, "patterns", ()))
            if not patterns:
                return self.configured_wind_pattern
            return str(self.np_random.choice(patterns))
        return self.configured_wind_pattern

    def _reset_wind_process(self) -> None:
        if not self.enable_wind:
            self.wind_torque = 0.0
            return
        self.active_wind_pattern = self._sample_wind_pattern()
        self.wind_source.set_pattern(self.active_wind_pattern)
        self.wind_active = True

    def _resolve_wind_for_step(self) -> float:
        if self.enable_wind:
            sample = self.wind_source.sample(self.dt, float(self.state[0]))
            self.wind_mag = sample.mag
            self.wind_angle = sample.angle
            self.wind_torque = sample.torque
            self.wind_active = sample.active
            self.active_wind_pattern = sample.pattern
            return self.wind_torque

        tau_val = 0.0
        if self.wind_active:
            tau_val = (
                self.wind_mag
                * self.max_wind_torque
                * np.cos(self.state[0] - self.wind_angle)
            )
        self.wind_torque = float(tau_val)
        return self.wind_torque

    def _wind_info(self) -> dict:
        return {
            "wind_active": bool(self.wind_active),
            "wind_mag": float(self.wind_mag),
            "wind_angle": float(self.wind_angle),
            "wind_torque": float(self.wind_torque),
            "wind_pattern": self.active_wind_pattern if self.enable_wind else "manual",
            "configured_wind_pattern": self.configured_wind_pattern,
            "wind_automatic": bool(self.enable_wind),
        }

    def _tracking_info(self) -> dict:
        error = float(self.state[0] - self.theta_ref)
        return {
            "theta_ref": float(self.theta_ref),
            "tracking_error": error,
            "abs_tracking_error": abs(error),
        }

    def _info(self) -> dict:
        return {**self._wind_info(), **self._tracking_info()}

    def _tracking_reward(self, theta: float, theta_dot: float, action: float) -> float:
        error = float(theta - self.theta_ref)
        return -(
            self.reward_weights["error"] * float(error ** 2)
            + self.reward_weights["velocity"] * float(theta_dot ** 2)
            + self.reward_weights["action"] * float(action ** 2)
        )

    def step(self, action):
        self.last_action = action
        
        # 1. Preparar Tensores
        # Estado actual: [theta, theta_dot] (Normalizados)
        # Optimizacion: from_numpy es más eficiente y evita warnings
        current_state = torch.from_numpy(self.state).float().unsqueeze(0) # (1, 2)
        
        # Tiempo de integración [0, dt]
        t = torch.tensor([0.0, self.dt], dtype=torch.float32)
        
        # Control (Zero Order Hold para este paso)
        u_val = float(np.clip(action[0], 0, 1))
        # Necesitamos 2 puntos para el spline: t=0 y t=dt con el mismo valor
        u_seq = torch.tensor([[[u_val], [u_val]]], dtype=torch.float32) # (1, 2, 1)

        # 1.5. Calcular torque externo de viento.
        # En entrenamiento headless, enable_wind=True hace que WindProcess mande.
        # En demo, enable_wind=False conserva el control manual via set_wind().
        tau_val = self._resolve_wind_for_step()
        tau_seq = torch.tensor([[[tau_val], [tau_val]]], dtype=torch.float32)
        
        # 2. Integración Numérica
        with torch.no_grad():
            # model(...) retorna (Batch, T, State) -> (1, 2, 2)
            # Queremos el estado en t=dt (índice 1)
            next_state_tensor = self.model(t, u_seq, initial_state=current_state, tau_ext=tau_seq)

            new_state = next_state_tensor[0, 1, :].numpy()
            
        # 3. Actualizar estado
        self.state = new_state

        # --- Ángulo SIN envolver (decisión de diseño deliberada; conservar) ---
        # theta se mantiene continuo y acumulativo = ángulo físico real.
        # Envolver aquí a [-pi, pi] ocultaba los sobregiros: el péndulo daba
        # vueltas sin que la terminación lo detectara (comparaba el ángulo
        # envuelto contra el límite) y desincronizaba el estado del error de
        # control. Regla acordada: el envoltorio NUNCA va en el estado
        # observado. La terminación de abajo usa este theta real contra el
        # límite físico de la planta (reset_angle_deg).

        # 4. Calcular observables reales
        # Como el modelo ya trabaja en radianes, el estado es directo
        angle_rad = self.state[0]
        vel_rad_s = self.state[1]

        # 5. Recompensa y Terminación
        terminated = bool(abs(angle_rad) > self.limit_rad)
        
        reward = self._tracking_reward(
            theta=angle_rad,
            theta_dot=vel_rad_s,
            action=u_val,
        )
        
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, self._info()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reinicia el entorno a un estado inicial, soportando configuraciones flexibles
        para el entrenamiento y evaluación de algoritmos de Aprendizaje por Refuerzo (RL).

        Para evitar que las políticas sufran de sobreajuste (overfitting) a la trayectoria 
        monótona desde el reposo y garantizar robustez ante perturbaciones exógenas (como 
        los procesos de viento), este método implementa tres estrategias de inicialización 
        mediante el diccionario de opciones.

        Args:
            seed (Optional[int]): Semilla para el generador de números aleatorios interno 
                (self.np_random), asegurando la reproducibilidad del entorno.
            options (Optional[dict]): Parámetros de control de reinicio. Soporta:
                * "initial_state" (Iterable[float]): Arreglo explícito [theta, theta_dot] 
                  para forzar un punto operativo específico. Útil para pruebas de regresión.
                * "randomize" (bool): Si es True, activa Domain Randomization. Muestrea 
                  un ángulo seguro (no terminal, < 50% del límite físico) y una velocidad 
                  inicial no nula para maximizar la exploración del espacio de estados.
                * "theta_ref" (float): Ángulo objetivo opcional para este episodio.

        Returns:
            tuple: (observación_inicial [theta, theta_dot, theta_ref], info_dict)
        """        
        super().reset(seed=seed)
        
        self.last_action = None
        
        options = options or {}
        if "theta_ref" in options:
            self.set_theta_ref(options["theta_ref"])

        # 1. Forzar un estado inicial específico
        if "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        # 2. Inicialización aleatoria (Domain Randomization)
        elif options.get("randomize", False):
            max_angle = np.deg2rad(self.reset_angle_deg * 0.5)
            rand_angle = self.np_random.uniform(low=-max_angle, high=max_angle)
            rand_vel = self.np_random.uniform(low=-1.0, high=1.0)
            self.state = np.array([rand_angle, rand_vel], dtype=np.float32)
        # 3. Comportamiento por defecto (reposo absoluto)
        else:
            self.state = np.array([0.0, 0.0], dtype=np.float32)
        
        self._reset_wind_process()
        return self._get_obs(), self._info()        

    def render(self):
        if self.render_mode is None:
            return
        pygame, gfxdraw = _require_pygame()

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        # --- Flujo de Viento (Sistema de Partículas en Fondo) ---
        if self.wind_cfg.get("enabled", True) and self.wind_active and self.wind_mag > 0.01:
            dx = np.cos(self.wind_angle) * self.wind_mag * self.wind_cfg["speed"]
            dy = np.sin(self.wind_angle) * self.wind_mag * self.wind_cfg["speed"]
            c_color = tuple(self.wind_cfg["color"])
            
            for i in range(self.wind_cfg["count"]):
                self.particles[i, 0] = (self.particles[i, 0] + dx) % self.screen_dim
                self.particles[i, 1] = (self.particles[i, 1] + dy) % self.screen_dim
                px, py = self.particles[i]

                # La estela sigue la cinemática, pero amplificada visualmente 
                # para que el rastro sea notorio incluso con vientos débiles
                visual_stretch = 2.0
                prev_x = px - (dx * visual_stretch)
                prev_y = py - (dy * visual_stretch)

                pygame.draw.line(self.surf, c_color, (int(prev_x), int(prev_y)), (int(px), int(py)), 2)

        # --- Dibujado (Idéntico a learned_pahm.py original) ---
        bound = 2.2 # Rango visual aprox en radianes
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        rod_length = 1 * scale
        rod_width = 0.2 * scale

        # Obtener ángulo real en radianes para dibujar
        theta = self.state[0]

        left, right, top, bottom = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
        ]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(theta - np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)

        gfxdraw.filled_polygon(self.surf, transformed_coords, (31,47,95,255))
        gfxdraw.aapolygon(self.surf, transformed_coords, (31,47,95,255))

        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width/2), (31,47,95,255))
        
        rod_end = pygame.math.Vector2((rod_length, 0)).rotate_rad(theta - np.pi/2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width/2), (31,47,95,255))

        if self.last_action is not None:
            val = float(self.last_action[0])
            wind_len = rod_length * val
            wind = pygame.math.Vector2((rod_length, -wind_len)).rotate_rad(theta - np.pi/2)
            wind = (int(wind[0] + offset), int(wind[1] + offset))
            if abs(val) > 0.01:
                pygame.draw.line(self.surf, (210,64,64), rod_end, wind, 3)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame, _ = _require_pygame()
            pygame.display.quit()
            pygame.quit()
