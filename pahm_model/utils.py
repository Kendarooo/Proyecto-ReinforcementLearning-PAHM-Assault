# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

import torch

# --- SOLVER ODE NATIVO (RK4) ---
def rk4_step(func, t, y, dt):
    """Un paso de integración Runge-Kutta 4."""
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt/2 * k1)
    k3 = func(t + dt/2, y + dt/2 * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def odeint_simple(func, y0, t):
    """Solver ODE minimalista en PyTorch puro."""
    trajectory = [y0]
    curr_y = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        curr_y = rk4_step(func, t[i], curr_y, dt)
        trajectory.append(curr_y)
    return torch.stack(trajectory)

# --- INTERPOLACIÓN CÚBICA (SPLINES) ---
@torch.jit.script
def _eval_spline_jit(t: torch.Tensor, 
                     times_start: float, 
                     times_end: float, 
                     dt_inv: float, 
                     max_idx: int, 
                     values: torch.Tensor, 
                     m: torch.Tensor) -> torch.Tensor:
    """Versión JIT-compilada de la evaluación del spline."""
    t_clamped = torch.clamp(t, times_start, times_end)
    rel_t = (t_clamped - times_start) * dt_inv
    idx = torch.clamp(torch.floor(rel_t).long(), 0, max_idx)
    tau = rel_t - idx.float()
    
    p0 = values[:, idx, :]
    p1 = values[:, idx+1, :]
    m0 = m[:, idx, :]
    m1 = m[:, idx+1, :]
    
    tau2 = tau * tau
    tau3 = tau2 * tau
    
    h00 = 2*tau3 - 3*tau2 + 1
    h10 = tau3 - 2*tau2 + tau
    h01 = -2*tau3 + 3*tau2
    h11 = tau3 - tau2
    
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

class CubicSplineInterpolation:
    """Interpolación cúbica vectorizada."""
    def __init__(self, times, values):
        self.times = times
        self.values = values
        self.device = values.device
        self.t_start = times[0]
        self.dt = times[1] - times[0]
        self.dt_inv = 1.0 / self.dt
        self.t_end = times[-1]
        self.max_idx = values.shape[1] - 2
        self.m = self._compute_gradients(values)

    def _compute_gradients(self, y):
        m = torch.zeros_like(y)
        m[:, 1:-1, :] = (y[:, 2:, :] - y[:, :-2, :]) / 2.0
        m[:, 0, :] = y[:, 1, :] - y[:, 0, :]
        m[:, -1, :] = y[:, -1, :] - y[:, -2, :]
        return m

    def evaluate(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return _eval_spline_jit(t, self.t_start.item(), self.t_end.item(), self.dt_inv.item(), self.max_idx, self.values, self.m)

    @property
    def derivatives(self):
        """Retorna la derivada analítica (velocidad) en los puntos de la grilla"""
        return self.m * self.dt_inv
