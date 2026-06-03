# Copyright (C) 2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

# Versión: 1.0.0
# Descripción: Script de sintonización analítica para los PID.
# Extrae parámetros de la PAHM desde un modelo ODE entrenado y 
# aplica asignación de polos para garantizar cero sobreimpulso.

import argparse
import json
import torch
import os
import sys

def calculate_pid_gains(alpha, beta, gamma, ts=2.0, pole_ratio=10.0):
    """
    Calcula las ganancias de los PID mediante asignación de polos.
    
    Args:
        alpha, beta, gamma: Parámetros físicos del modelo.
        ts: Tiempo de asentamiento deseado al 2% (segundos).
        pole_ratio: Relación de velocidad del tercer polo (p1 = pole_ratio * wn).
    """
    # Para un sistema de segundo orden críticamente amortiguado, 
    # el tiempo de asentamiento al 2% se aproxima por ts = 5.8 / wn
    wn = 5.8 / ts
    p1 = pole_ratio * wn
    
    kd = (p1 + 2 * wn - beta) / alpha
    kp = (2 * p1 * wn + (wn ** 2) - gamma) / alpha
    ki = (p1 * (wn ** 2)) / alpha
    
    return kp, ki, kd

def main():
    parser = argparse.ArgumentParser(description='Sintonización Analítica PID para PAHM')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al modelo ODE (.pth)')
    parser.add_argument('--output', type=str, default='pid_config.json', help='Ruta de salida del JSON')
    parser.add_argument('--ts', type=float, default=2.0, help='Tiempo de asentamiento deseado (segundos)')
    parser.add_argument('--pole_ratio', type=float, default=10.0, help='Ubicación del polo no dominante')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: No se encontró el modelo {args.model_path}")
        sys.exit(1)
        
    print(f"🔄 Cargando parámetros físicos desde {args.model_path}...")
    try:
        # Cargar diccionario directo para no depender de la clase si no hace falta
        checkpoint = torch.load(args.model_path, map_location='cpu')
        
        # El modelo guarda 'physics' como dict: {'alpha (gain)': val, ...}
        physics = checkpoint.get('physics')
        if not physics:
            raise KeyError("El archivo .pth no contiene el diccionario 'physics'")
            
        alpha = physics['alpha (gain)']
        beta = physics['beta (friction)']
        gamma = physics['gamma (gravity)']
        
        print(f"✅ Física identificada -> α: {alpha:.4f}, β: {beta:.4f}, γ: {gamma:.4f}")
        
    except Exception as e:
        print(f"❌ Error al leer el checkpoint: {e}")
        sys.exit(1)

    # Cálculo analítico
    kp, ki, kd = calculate_pid_gains(alpha, beta, gamma, ts=args.ts, pole_ratio=args.pole_ratio)
    
    # Empaquetado JSON
    pid_config = {
        "metadata": {
            "source_model": args.model_path,
            "settling_time_target": args.ts,
            "pole_ratio": args.pole_ratio
        },
        "gains": {
            "Kp": kp,
            "Ki": ki,
            "Kd": kd
        },
        "plant_params": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        }
    }
    
    try:
        with open(args.output, 'w') as f:
            json.dump(pid_config, f, indent=4)
        print(f"✅ Configuración PID guardada exitosamente en {args.output}")
        print(f"   Kp: {kp:.4f} | Ki: {ki:.4f} | Kd: {kd:.4f}")
    except Exception as e:
        print(f"❌ Error al guardar JSON: {e}")

if __name__ == "__main__":
    main()