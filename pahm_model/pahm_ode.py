# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

# Módulo para modelo basado en Neural ODE (Universal Differential Equations)
# VERSIÓN OPTIMIZADA PARA GPU

import torch
import torch.nn as nn
from utils import odeint_simple, CubicSplineInterpolation

class PAHMHybridODE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- Parámetros Físicos (espacio logarítmico) ---
        # Inicialización conservadora
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(272.15))) # Ganancia
        self.log_beta  = nn.Parameter(torch.log(torch.tensor(0.52)))  # Fricción
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(18.43))) # Gravedad
        
        # --- Red Residual ---
        self.residual_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Inicialización casi nula para comenzar con física pura
        nn.init.uniform_(self.residual_net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.residual_net[-1].bias)

        self._current_control_spline = None
        self.use_residual = True

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self): return torch.exp(self.log_beta)
    @property
    def gamma(self): return torch.exp(self.log_gamma)

    def get_physical_parameters(self):
        return {
            "alpha (gain)": self.alpha.item(),
            "beta (friction)": self.beta.item(),
            "gamma (gravity)": self.gamma.item()
        }

    def _ode_func(self, t, state):
        """Dinámica del sistema dx/dt = f(x, u, t)"""
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]
        
        # Evaluar control interpolado (operación GPU pura ahora)
        u = self._current_control_spline.evaluate(t)

        accel_tau = self._current_tau_spline.evaluate(t) if self._current_tau_spline is not None else 0.0
        # Física: accel = alpha*u^2 - beta*theta_dot - gamma*sin(theta)
        accel_physics = (self.alpha * (u**2)) - (self.beta * theta_dot) - (self.gamma * torch.sin(theta))
        
        # Residual
        if self.use_residual:
            nn_input = torch.cat([torch.sin(theta), torch.cos(theta), theta_dot, u], dim=1)
            accel_residual = self.residual_net(nn_input)
        else:
            accel_residual = 0.0
            
        theta_ddot = accel_physics + accel_residual + accel_tau
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def forward(self, t, u_sequences, initial_state=None, tau_ext=None):
        batch_size = u_sequences.shape[0]
        
        # Crear spline (esto mueve datos a la estructura interna, debe ser rápido)
        self._current_control_spline = CubicSplineInterpolation(t, u_sequences)
        self._current_tau_spline = CubicSplineInterpolation(t, tau_ext) if tau_ext is not None else None
        
        if initial_state is None:
            y0 = torch.zeros(batch_size, 2, device=self.device)
        else:
            y0 = initial_state
            
        # Usamos nuestro solver nativo (RK4)
        # Es más rápido, estable y no requiere librerías externas
        solution = odeint_simple(self._ode_func, y0, t)
            
        return solution.permute(1, 0, 2)

    @staticmethod
    def estimate_state_from_angles(times, angles):
        """Pre-procesamiento (puede correr en CPU o GPU según donde estén los tensores)"""
        spline = CubicSplineInterpolation(times, angles)

        # Recuperar derivada usando la propiedad compartida
        velocity = spline.derivatives
        
        return torch.cat([angles, velocity], dim=2)

    def save_model(self, path):
        torch.save({'state_dict': self.state_dict(), 'physics': self.get_physical_parameters()}, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint.get('physics', {})
    
