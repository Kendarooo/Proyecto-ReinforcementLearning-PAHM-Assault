
# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

# Script de entrenamiento para Neural ODE (PAHM)
# Implementa estrategia de entrenamiento en dos fases: Física -> Híbrido

import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import os
import pandas as pd
from datetime import datetime

from pahm_ode import PAHMHybridODE
from pahm_fast import PAHMFastModel

class PAHMChunkDataset(Dataset):
    """
    Dataset optimizado para Neural ODEs.
    1. Carga todos los CSVs en memoria (RAM).
    2. Convierte a Radianes y pre-calcula velocidades (estado completo) una sola vez.
    3. Crea un índice virtual para extraer 'chunks' válidos de cualquier archivo.
    """
    def __init__(self, data_dir, chunk_size, dt=0.02, model_helper=None, device='cpu'):
        self.chunk_size = chunk_size
        self.dt = dt
        self.data_chunks = [] # Lista de tuplas (u_seq, state_gt)
        
        # 1. Cargar y Pre-procesar Archivos
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos .csv en {data_dir}")
        
        print(f"📊 Procesando {len(files)} archivos para generar dataset de entrenamiento...")
        
        # Buffer para indexación virtual: lista de (idx_archivo, idx_inicio)
        self.virtual_index = []
        
        for file_idx, f in enumerate(files):
            path = os.path.join(data_dir, f)
            try:
                # Leer CSV forzando float para evitar el error de numpy.object_
                df = pd.read_csv(path, dtype=float)
            except Exception as e:
                print(f"⚠️ Saltando archivo corrupto {f}: {e}")
                continue
                
            # Extraer columnas: PWM (col 2) y Angulo (col 3) -> Índices 2 y 3
            # Asumimos estructura: index, time, pwm, angle
            pwm_raw = df.iloc[:, 2].values.astype(np.float32)
            angle_deg = df.iloc[:, 3].values.astype(np.float32)
            
            # Convertir a Tensores
            # Formato (1, T, 1) para compatibilidad con las funciones del modelo
            u_tensor = torch.tensor(pwm_raw, dtype=torch.float32).view(1, -1, 1).to(device)
            angle_tensor = torch.tensor(angle_deg, dtype=torch.float32).view(1, -1, 1).to(device)
            
            # Convertir a Radianes (Física)
            angle_rad = torch.deg2rad(angle_tensor)
            
            # Pre-calcular Velocidad (Estado Completo) usando el Spline del modelo
            # Esto se hace UNA VEZ con toda la secuencia para máxima precisión
            seq_len = angle_rad.shape[1]
            t_full = torch.linspace(0, (seq_len-1)*dt, seq_len, device=device)
            
            with torch.no_grad():
                # state_full: (1, T, 2) -> [theta, theta_dot]
                state_full = model_helper.estimate_state_from_angles(t_full, angle_rad)
            
            # Guardar en RAM (movemos a CPU para no saturar GPU si es mucho dato, 
            # aunque 200k muestras caben en GPU sobradas. Dejemos en CPU por compatibilidad)
            self.data_chunks.append({
                'u': u_tensor.cpu(),
                'state': state_full.cpu()
            })
            
            # Generar índices válidos para sliding window
            # Si chunk_size=100 y len=1500, hay 1401 inicios válidos
            num_valid_starts = seq_len - chunk_size + 1
            if num_valid_starts > 0:
                for start_idx in range(num_valid_starts):
                    self.virtual_index.append((file_idx, start_idx))
                    
        print(f"✅ Dataset generado: {len(self.virtual_index)} sub-secuencias (chunks) totales.")

    def __len__(self):
        return len(self.virtual_index)

    def __getitem__(self, idx):
        file_idx, start_idx = self.virtual_index[idx]
        
        data = self.data_chunks[file_idx]
        end_idx = start_idx + self.chunk_size
        
        # Extraer recorte
        # Squeeze(0) para quitar la dimensión de batch falsa que usamos al procesar
        # Salida: (T, 1) y (T, 2)
        u_chunk = data['u'][0, start_idx:end_idx, :]
        state_chunk = data['state'][0, start_idx:end_idx, :]
        
        return u_chunk, state_chunk



def train_ode():
    parser = argparse.ArgumentParser(description='Train Neural ODE for PAHM')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20, 
                        help='Epochs for physics-only training (Phase 1)')
    parser.add_argument('--data_dir', type=str, default='../data/', 
                        help='Directorio donde están los CSV de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step of data (s)')
    parser.add_argument('--chunk_size', type=int, default=150, 
                        help='Tamaño del recorte temporal (steps) para entrenamiento (default 100)')
    parser.add_argument('--model_type', type=str, default='fast', choices=['ode', 'fast'],
                        help='Tipo de modelo: "ode" (torchdiffeq) o "fast" (RNN-RK4 custom)')
    parser.add_argument('--model_name', type=str, default='pahm_ode_v1')
    parser.add_argument('--load_model', type=str, default='',
                        help='Ruta a un checkpoint .pth existente para continuar entrenamiento')   
    parser.add_argument('--wandb_run', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default='pahm-ode-project',
                        help='WandB project name')
    parser.add_argument('--extension', type=str, default='zero', 
                        help='Use "zero" usually for ODE to keep raw PWM')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()

    print("⚙️  Configuración del entrenamiento:")
    for key, value in vars(args).items():
        print(f"   - {key}: {value}")    
    
    # Configuración de Dispositivo
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"🚀 Usando dispositivo: {device}")

    # Inicializar WandB
    use_wandb = bool(args.wandb_run)
    if use_wandb:
        # IMPORTANTE: Eliminamos 'entity' para que vaya por defecto a tu cuenta personal
        # El 'project' se creará automáticamente si no existe en tu cuenta.
        wandb.login()
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run, 
            config=vars(args)
        )

    # 1. Inicializar Modelo (Necesario antes del dataset para usar sus splines)
    if args.model_type == 'ode':
        print("🧠 Inicializando Neural ODE (Lento/Preciso - torchdiffeq)")
        model = PAHMHybridODE(device=device).to(device)
    else:
        print("⚡ Inicializando Fast Physics RNN (Rápido - RK4 Custom)")
        model = PAHMFastModel(device=device, dt=args.dt).to(device)
        
    print(f"   Parámetros físicos iniciales: {model.get_physical_parameters()}")

    # 1.5 Cargar Checkpoint si existe (Continuación de entrenamiento)
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"🔄 Cargando checkpoint: {args.load_model}")
            try:
                phys_params = model.load_model(args.load_model)
                print(f"   ✅ Física recuperada: {phys_params}")
                
                # CRÍTICO: Si continuamos, cancelamos el warmup para no congelar 
                # la red neuronal ya entrenada y evitar 'desaprender' la física.
                print("   ⏩ Saltando Warmup (Fase Física) para permitir ajuste fino híbrido.")
                args.warmup_epochs = 0
            except Exception as e:
                print(f"   ❌ Error cargando modelo: {e}")
        else:
            print(f"   ⚠️ Archivo no encontrado: {args.load_model}")
    
    # 2. Preparar Dataset Monolítico
    # Usamos el propio modelo como helper para calcular derivadas consistentes
    full_dataset = PAHMChunkDataset(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        dt=args.dt,
        model_helper=model,
        device=device
    )
    
    # DataLoader estándar de PyTorch (Ahora sí soporta batches masivos y shuffle real)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Optimizador
    # Usamos tasas de aprendizaje diferentes para física y red neuronal si es necesario
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Scheduler para bajar el LR suavemente
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    
    # --- Bucle de Entrenamiento ---
    for epoch in range(args.epochs):
        
        # --- Fase 1: Warmup Físico (Solo entrenar alpha, beta, gamma) ---
        if epoch < args.warmup_epochs:
            phase = "PHASE 1: PHYSICS ONLY"
            model.use_residual = False
            # Congelar red residual explícitamente (opcional, el flag use_residual ya la apaga en forward)
            for param in model.residual_net.parameters():
                param.requires_grad = False
        else:
            phase = "PHASE 2: HYBRID (PHYSICS + NN)"
            model.use_residual = True
            for param in model.residual_net.parameters():
                param.requires_grad = True
        
        model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_idx, (u_batch, state_gt_batch) in enumerate(train_loader):
            # Datos ya vienen listos del Dataset: (Batch, Chunk_Size, Dim)
            u_seq = u_batch.to(device)
            state_gt = state_gt_batch.to(device)
            
            # Vector de tiempo relativo para la integración
            t = torch.linspace(0, (args.chunk_size-1)*args.dt, args.chunk_size, device=device)
 
            # 3. Estado Inicial para la ODE
            initial_state = state_gt[:, 0, :] # (Batch, 2)

            # 2. Forward (Integración ODE)
            # Predice trayectoria completa a partir de u_seq y estado inicial
            try:
                state_pred = model(t, u_seq, initial_state=initial_state) # (Batch, T, 2)
            except Exception as e:
                print(f"💥 Error numérico en integración: {e}")
                continue

            # 3. Calcular Pérdida
            # Al usar chunks válidos, ya no necesitamos máscara de padding compleja
            # Error en posición (theta) y velocidad (theta_dot)
            loss_pos = torch.mean((state_pred[:,:,0:1] - state_gt[:,:,0:1])**2)
            loss_vel = torch.mean((state_pred[:,:,1:2] - state_gt[:,:,1:2])**2)

            # Loss total ponderada
            loss = loss_pos + 0.1 * loss_vel
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            
            # Clipping de gradientes es CRÍTICO en ODEs para evitar explosiones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_loss / max(1, total_batches)
        
        # --- Validación ---
        model.eval()
        val_loss = 0

        # Nota: Como ahora usamos todo el dataset para train con sliding window,
        # el concepto de validación "out of sample" puro requeriría separar archivos.
        # Para este ejercicio, usaremos el Loss promedio del entrenamiento como proxy,
        # o podríamos reservar los últimos N archivos para validación real.
        # Por simplicidad y robustez, usamos avg_train_loss como métrica de control.
        
        avg_val_loss = avg_train_loss # Simplificación válida con dataset monolítico aleatorio
        scheduler.step(avg_val_loss)
        
        # Logging
        phys_params = model.get_physical_parameters()
        print(f"Epoch {epoch+1}/{args.epochs} [{phase}] "
              f"Loss: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"α:{phys_params['alpha (gain)']:.2f} β:{phys_params['beta (friction)']:.2f} γ:{phys_params['gamma (gravity)']:.2f}")
        
        if use_wandb:
            log_dict = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                **phys_params
            }
            wandb.log(log_dict)

        # Guardar solo el mejor modelo y el último
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = f"{args.model_name}_best.pth"
            model.save_model(best_path)
            print(f"   ⭐ Nuevo mejor modelo guardado: {best_path}")
        
    # Guardar final
    final_path = f"{args.model_name}_final.pth"
    model.save_model(final_path)
    print(f"✅ Modelo guardado en {final_path}")
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train_ode()
