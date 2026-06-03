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
from typing import Optional

# Añadir directorio hermano al path para cargar el modelo
sys.path.append(os.path.join(os.path.dirname(__file__), '../pahm_model'))

try:
    from pahm_ode import PAHMHybridODE
    from pahm_fast import PAHMFastModel
except ImportError:
    raise ImportError("No se pudieron importar los modelos. Verifique ../pahm_model/")

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled("pygame required")



class LearnedPAHMODE(gym.Env):
    """
    Entorno Gymnasium específico para el modelo Neural ODE (PAHM).
    
    A diferencia del modelo GRU (Caja Negra), este entorno simula la física
    integrando paso a paso las ecuaciones diferenciales aprendidas.
    
    Estado interno:
        [theta_rad, theta_dot_rad] (Sistema Internacional)
     
    
    Observación:
        [theta_radians, theta_dot_rad]

        
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
                 max_wind_torque=20.0):   # Escala física del torque de viento): # 50Hz por defecto
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.dt = dt
        self.reset_angle_deg = reset_angle_deg
                
        self.device = "cpu" # CPU es preferible para inferencia paso a paso (baja latencia)
        
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

        # Observación: [Ángulo en radianes, Velocidad angular en rad/s]
        high_obs = np.array([np.deg2rad(180.0), 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

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
        # Escala recibida por constructor (SSOT en config.json; la lee el
        # script que instancia el entorno, no el entorno).
        self.max_wind_torque = max_wind_torque

        # Cargar configuración visual de partículas
        self.wind_cfg = {"count": 40, "speed": 15.0, "length": 20.0, "color": [100, 240, 255]}
        try:
            with open('config.json', 'r') as f:
                cfg = json.load(f)
                if "components" in cfg and "wind_particles" in cfg["components"]:
                    self.wind_cfg = cfg["components"]["wind_particles"]
        except Exception: pass
        
        self.particles = np.random.rand(self.wind_cfg["count"], 2) * self.screen_dim        
        
    def set_wind(self, active, mag, angle):
        self.wind_active = active
        self.wind_mag = mag
        self.wind_angle = angle
        

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

        # 1.5. Calcular torque externo de viento si está activo
        tau_val = 0.0
        if self.wind_active:
            tau_val = self.wind_mag * self.max_wind_torque * np.cos(self.state[0] - self.wind_angle)
            
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
        limit_rad = np.deg2rad(self.reset_angle_deg)
        terminated = bool(abs(angle_rad) > limit_rad)
        
        # Recompensa simple: mantenerlo vertical (0) y quieto
        reward = 2.0 * np.exp(-abs(angle_rad)) - 0.1 * abs(vel_rad_s) - 1.0
        
        obs = np.array([angle_rad,vel_rad_s], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

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

        Returns:
            tuple: (observación_inicial [theta, theta_dot], info_dict)
        """        
        super().reset(seed=seed)
        
        self.last_action = None
        
        options = options or {}

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
        
        return np.array([self.state[0], self.state[1]], dtype=np.float32), {}        

    def render(self):
        if self.render_mode is None: return

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
        if self.wind_active and self.wind_mag > 0.01:
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

        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
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
            pygame.display.quit()
            pygame.quit()

