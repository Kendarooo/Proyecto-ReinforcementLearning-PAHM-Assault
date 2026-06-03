# Copyright (C) 2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

# Versión: 1.0.0
# Descripción: Implementación de controlador PID con compensación no lineal
# para la PAHM (Péndulo Amortiguado con Hélice a Motor).

import json
import numpy as np

class PIDController:
    """Controlador PID con memoria de estado y compensación de planta."""
    def __init__(self, config_path):
        self.config_path = config_path
        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self._load_config()

    def _load_config(self):
        """Carga las ganancias desde el archivo JSON proporcionado."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            self.kp = data["gains"]["Kp"]
            self.ki = data["gains"]["Ki"]
            self.kd = data["gains"]["Kd"]
            print(f"✅ Configuración PID cargada exitosamente: {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de configuración PID: {self.config_path}")
        except KeyError as e:
            raise KeyError(f"El archivo JSON no tiene el formato esperado. Falta la llave: {e}")

    def reset(self):
        """Limpia la memoria del controlador (acumulador integral y derivada)."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, current_value, dt):
        """
        Calcula la acción de control compensada no linealmente.
        
        Args:
            setpoint (float): Ángulo deseado en radianes.
            current_value (float): Ángulo actual en radianes.
            dt (float): Paso de tiempo en segundos.
            
        Returns:
            float: Acción de control (PWM) saturada entre [0, 1].
        """
        error = setpoint - current_value
        
        # Actualización de estados
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # Esfuerzo de control linealizado (v)
        v = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Compensación de la no linealidad de la planta (u = sqrt(v))
        # y saturación natural
        action = np.sqrt(max(0.0, v))
        
        return min(1.0, action)