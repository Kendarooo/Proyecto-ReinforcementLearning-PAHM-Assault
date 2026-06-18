# Instrucciones de Ejecucion y Verificacion

Este documento resume como correr el estado actual del proyecto PAHM y que artefactos espera cada etapa.

## 1. Entorno

Desde la raiz del repositorio, use la venv incluida:

```bash
source .venv/bin/activate
```

O ejecute los comandos directamente con:

```bash
.venv/bin/python
```

## 2. Datos y artefactos esperados

La carpeta de datos reales debe estar en la raiz del repositorio:

```text
data/
```

El dataloader de Etapa 1 espera archivos CSV en esa carpeta. La estructura actual relevante es:

```text
data/*.csv
pahm_model/pahm_fast_v2_best.pth
checkpoints/estimator_checkpoint_epoch_40.pth
outputs/tau_w/estimador_wind.pth
outputs/tau_w/tau_w_<split>_<idx>.npy
artifacts/stage2/gmm_wind_model.pkl
```

Uso actual de checkpoints:

- `pahm_model/pahm_fast_v2_best.pth`: modelo fisico base usado por el entorno, residuos y FR-7.
- `checkpoints/estimator_checkpoint_epoch_40.pth`: checkpoint del estimador GRU usado por FR-7 y `export_tau_w.py`.
- `outputs/tau_w/estimador_wind.pth`: copia/exportacion del estimador GRU para integrar Etapa 1 -> Etapa 2.
- `artifacts/stage2/gmm_wind_model.pkl`: modelo no supervisado GMM de Etapa 2.

## 3. Verificar pruebas

Ejecutar toda la suite:

```bash
.venv/bin/python -m pytest tests test -q
```

Estado verificado:

```text
33 passed, 1 warning
```

La advertencia proviene de `torch.jit.script` y no bloquea la ejecucion.

## 4. Correr la app visual

Desde `gym_wrapper/`:

```bash
cd gym_wrapper
../.venv/bin/python test_pahm_ode_env.py \
  --model ../pahm_model/pahm_fast_v2_best.pth \
  --pid pid_config.json \
  --reset_angle 720 \
  --max_steps 10000000
```

La app permite probar:

- modos `Off`, `Step`, `Sine`, `Random`, `Manual`;
- controlador `PID`;
- viento manual;
- patrones `Calm`, `Gust`, `Sust`, `Turb`;
- osciloscopio de PWM y angulo.

El modo `RL` todavia no ejecuta una politica entrenada. Actualmente deja `rl_action = 0.0`; esa integracion corresponde a Etapa 3.

## 5. Etapa 1: exportar tau_w

El script de exportacion consume el checkpoint configurado en `gym_wrapper/config.json`:

```bash
.venv/bin/python pahm_model/export_tau_w.py
```

Salida esperada:

```text
outputs/tau_w/tau_w_train_*.npy
outputs/tau_w/tau_w_val_*.npy
outputs/tau_w/tau_w_test_*.npy
outputs/tau_w/estimador_wind.pth
```

## 6. Etapa 2: entrenamiento no supervisado

Ruta historica con autoencoder:

```bash
.venv/bin/python etapa2_unsupervised/train_unsupervised.py \
  --config gym_wrapper/config.json
```

Ruta G2 con GMM, BIC y validacion sintético-real:

```bash
.venv/bin/python -m pahm_stage2.validate_synthetic_real \
  --config configs/stage2_config.json
```

Entrenar GMM desde un manifiesto de `tau_w`:

```bash
.venv/bin/python -m pahm_stage2.train_unsupervised \
  --config configs/stage2_config.json
```

## 7. Estado para Etapa 3

Etapa 1 y Etapa 2 pasan las pruebas actuales. Para avanzar con Etapa 3A y Etapa 3B falta implementar:

- observacion con `theta_ref` configurable;
- recompensa de seguimiento respecto a `theta_ref`;
- `train_rl.py` headless con Stable Baselines3;
- agentes naive y robusto;
- carga de politica entrenada en el modo `RL` de la demo;
- comparacion cuantitativa de error de seguimiento, tiempo de estabilizacion y sobreimpulso.

Antes de entregar, limpiar `requirements.txt`: actualmente contiene marcadores de conflicto Git.
