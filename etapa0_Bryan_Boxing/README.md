# Etapa 0 — DQN Boxing Atari | v1.0
**EL5857 Aprendizaje Automático | I Semestre 2026**
Autor: Bryan

---

## Descripción

Agente Double DQN entrenado para jugar **ALE/Boxing-v5** usando Stable-Baselines3.
Etapa individual de inducción al Aprendizaje Reforzado del Proyecto 2 (PAHM).

---

## Estructura del proyecto

```
etapa0_Bryan_Boxing/
├── config_dqn.json       # [NFR-1] Configuración centralizada
├── pyproject.toml        # Dependencias gestionadas con uv
├── setup_env.sh          # Script de instalación del entorno
├── sanity_check.py       # Verificación de dependencias y GPU
├── train_dqn.py          # [FR-1] Sistema de entrenamiento (headless)
├── demo_dqn.py           # [FR-2] Sistema de demostración (render)
├── AI_AUDIT.md           # [CON-4] Auditoría de uso de IA
├── tests/
│   └── test_environment.py  # [NFR-6] Pruebas unitarias
├── checkpoints/          # Puntos de control periódicos
├── logs/                 # Logs de tensorboard y SB3
└── models/               # Modelos finales guardados
```

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd /Documentos/Proyectos_de_AA/Proyecto_2/etapa0_Bryan_Boxing

# 2. Ejecutar el script de setup (requiere uv instalado)
chmod +x setup_env.sh
./setup_env.sh

# 3. Activar el entorno
source .venv/bin/activate
```

---

## Uso

### Entrenamiento (headless, para dejar corriendo de noche)
```bash
source .venv/bin/activate
python train_dqn.py
```

### Demostración (carga modelo entrenado y renderiza)
```bash
source .venv/bin/activate
python demo_dqn.py
```

### Verificar instalación
```bash
python sanity_check.py
```

---

## Requisitos de hardware

- Python 3.11 (gestionado por uv)
- GPU NVIDIA con ≥4 GB VRAM (probado en RTX 3050 6GB)
- CUDA compatible con PyTorch ≥2.2.0

---

## Notas

- El renderizado está **desactivado por defecto** en entrenamiento (`debug_render: false` en config).
- Los checkpoints se guardan cada 50,000 pasos en `checkpoints/`.
- Todas las curvas de recompensa y pérdida se registran en W&B.