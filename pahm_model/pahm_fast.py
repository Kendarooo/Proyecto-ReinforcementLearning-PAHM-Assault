# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Versión 0.1.1
# Proyecto 2

import torch
import torch.nn as nn
from utils import CubicSplineInterpolation

class PAHMPhysicsCell(nn.Module):
    """
    Célula Recurrente con Física Integrada.
    
    Ejecuta explícitamente un paso de integración Runge-Kutta 4 (RK4) 
    sobre la ecuación diferencial del PAHM.
    Combina la velocidad de una RNN con la robustez física de una ODE.
    """
    def __init__(self, dt=0.02):
        super().__init__()
        self.dt = dt

        # --- Parámetros Físicos Aprendibles ---
        # Inicializados en espacio logarítmico para garantizar positividad

        self.log_alpha = nn.Parameter(torch.log(torch.tensor(20.0)))
        self.log_beta  = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(10.0)))
        
        # --- Red Residual Pequeña ---
        # Aprende la dinámica no modelada (fricción compleja, viento, etc.)
        self.residual_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Inicializar cerca de cero para empezar con física pura
        nn.init.uniform_(self.residual_net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.residual_net[-1].bias)
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

    def _dynamics(self, state, u, tau):
        """Calcula las derivadas [theta_dot, theta_ddot]"""
        theta = state[:, 0:1]
        theta_dot = state[:, 1:2]
        
        # Física base: J*theta'' + b*theta' + mgd*sin(theta) = L*k*u^2
        # Normalizado: theta'' = alpha*u^2 - beta*theta' - gamma*sin(theta)
        accel_physics = (self.alpha * (u**2)) - (self.beta * theta_dot) - (self.gamma * torch.sin(theta))
        
        # Residual
        if self.use_residual:
            nn_input = torch.cat([torch.sin(theta), torch.cos(theta), theta_dot, u], dim=1)
            accel_residual = self.residual_net(nn_input)
        else:
            accel_residual = 0.0
            
        return torch.cat([theta_dot, accel_physics + accel_residual + tau], dim=1)

    def forward(self, state, u, tau=0.0):
        """
        Paso RK4 manual vectorizado.
        state: (Batch, 2)
        u: (Batch, 1)
        """
        k1 = self._dynamics(state, u, tau)
        k2 = self._dynamics(state + 0.5 * self.dt * k1, u, tau)
        k3 = self._dynamics(state + 0.5 * self.dt * k2, u, tau)
        k4 = self._dynamics(state + self.dt * k3, u, tau)
        
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class PAHMFastModel(nn.Module):
    """
    Modelo secuencial que desenrolla la célula física RK4.
    Funciona como una RNN estándar en PyTorch, pero con física dura.
    """
    def __init__(self, device='cpu', dt=0.02):
        super().__init__()
        self.cell = PAHMPhysicsCell(dt=dt)
        self.device = device
        
    def forward(self, t, u_sequences, initial_state=None, tau_ext=None):
        """
        Args:
            t: Ignorado (se asume paso fijo dt implícito por indexación)
            u_sequences: (Batch, Seq_Len, 1)
            initial_state: (Batch, 2)
            tau_ext: (Batch, Seq_Len, 1) opcional
        """
        batch_size, seq_len, _ = u_sequences.shape
        
        if initial_state is None:
            state = torch.zeros(batch_size, 2, device=self.device)
        else:
            state = initial_state
            
        outputs = []
        
        # Incluir estado inicial (t=0) para consistencia de fase con Gym y ODE
        outputs.append(state.unsqueeze(1))
        
        # Simular N-1 pasos para mantener la longitud de salida igual a seq_len
        for i in range(seq_len - 1):
            u = u_sequences[:, i, :] # Control actual
            tau = tau_ext[:, i, :] if tau_ext is not None else 0.0
            state = self.cell(state, u, tau) # Integrar al siguiente paso            
            outputs.append(state.unsqueeze(1))
            
        # Concatenar tiempo: (Batch, Seq_Len, 2)
        return torch.cat(outputs, dim=1)

    # --- Compatibilidad con interfaz ODE ---
    @property
    def residual_net(self): return self.cell.residual_net
    @property
    def use_residual(self): return self.cell.use_residual
    @use_residual.setter
    def use_residual(self, val): self.cell.use_residual = val
    
    def get_physical_parameters(self): return self.cell.get_physical_parameters()
    
    def estimate_state_from_angles(self, times, angles):
        """
        Cálculo de velocidad usando Splines Cúbicos (vía utils).
        Alinea el ground truth con el modelo ODE para comparación justa.
        """

        spline = CubicSplineInterpolation(times, angles)
        velocity = spline.derivatives
        
        return torch.cat([angles, velocity], dim=2)
        
    def save_model(self, path):
        # Guardamos la etiqueta 'fast' para auto-detección
        torch.save({
            'state_dict': self.state_dict(), 
            'physics': self.get_physical_parameters(),
            'architecture': 'fast' 
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint.get('physics', {})
