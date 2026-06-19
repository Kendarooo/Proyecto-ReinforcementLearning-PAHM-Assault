# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_dataloaders
from pahm_ode import PAHMHybridODE
from pahm_fast import PAHMFastModel

__test__ = False

def test_ode():
    parser = argparse.ArgumentParser(description='Test Neural ODE for PAHM')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the .pth model file')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--extension', type=str, default='zero')
    parser.add_argument('--no_fig', action='store_true', help='No plot')
    
    args = parser.parse_args()
    
    device = torch.device('cpu') # Testeo rápido en CPU
    
    # Cargar datos (Test set)
    _, _, test_loader, _ = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=1, # Evaluar una por una para visualizar
        extension=args.extension,
        normalize_angles=False,
        verbose=False
    )
    
    # Cargar Modelo

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        arch = checkpoint.get('architecture', 'ode') # Default a 'ode'
        
        if arch == 'fast':
            print(f"⚡ Detectado modelo FAST (RK4)")
            model = PAHMFastModel(device=device, dt=args.dt)
        else:
            print(f"🧠 Detectado modelo ODE (torchdiffeq)")
            model = PAHMHybridODE(device=device)
        
        phys_params = model.load_model(args.model_path)
        print(f"✅ Modelo cargado. Parámetros físicos aprendidos:")
        for k, v in phys_params.items():
            print(f"   - {k}: {v:.4f}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    model.eval()
    
    mse_list = []
    
    # Listas para graficar (concatenadas)
    all_pred = []
    all_gt = []
    all_u = []
    
    print("Evaluando secuencias...")
    
    with torch.no_grad():
        for i, (pwm, lengths, angle) in enumerate(test_loader):
            # pwm: (1, T, F), angle: (1, T, 1)
            
            u_seq = pwm[:, :, 0:1].to(device)
            
            # Convertir grados (datos) a radianes (física)
            angle_gt = torch.deg2rad(angle).to(device)
            
            max_len = angle_gt.shape[1]
            
            t = torch.linspace(0, (max_len-1)*args.dt, max_len, device=device)
            
            # Estimar estado inicial real
            state_gt = model.estimate_state_from_angles(t, angle_gt)
            initial_state = state_gt[:, 0, :]
            
            # Predicción (Trayectoria completa)
            # Nota: Esto es Open-Loop simulation (integración desde t0 hasta t_end)
            # Es la prueba más dura para un modelo.
            state_pred = model(t, u_seq, initial_state=initial_state)
            
            # Extraer solo ángulos
            theta_pred = state_pred[:, :, 0].numpy().flatten()
            theta_gt = angle_gt[:, :, 0].numpy().flatten()
            u_vals = u_seq[:, :, 0].numpy().flatten()
            
            # Calcular error
            mse = np.mean((theta_pred - theta_gt)**2)
            mse_list.append(mse)
            
            all_pred.append(theta_pred)
            all_gt.append(theta_gt)
            all_u.append(u_vals)
            
            if i >= 5: break # Solo graficar las primeras 5 para no saturar
            
    print(f"MSE Promedio en Test: {np.mean(mse_list):.6f}")
    
    if not args.no_fig:
        # Graficar concatenado
        full_pred = np.concatenate(all_pred)
        full_gt = np.concatenate(all_gt)
        full_u = np.concatenate(all_u)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(full_gt, 'k--', label='Ground Truth', alpha=0.7)
        ax1.plot(full_pred, 'r-', label='Neural ODE Prediction', linewidth=1.5)
        ax1.set_ylabel('Ángulo (Norm)')
        ax1.set_title(f'Trayectoria Predicha (Open Loop) - Modelo: {args.model_path}')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(full_u, 'g-', label='PWM Input')
        ax2.set_ylabel('PWM')
        ax2.set_xlabel('Time steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_ode()
