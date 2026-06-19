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
38 passed, 1 warning
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

## 4.1. Etapa 3: entorno con perturbaciones

El branch `feature/etapa3-entorno-perturbaciones` trabaja sobre el entorno existente en `gym_wrapper/`; no se creo una carpeta nueva `pahm_stage3`.

El entorno `LearnedPAHMODE` soporta ahora perturbaciones automaticas para entrenamiento headless:

```python
from gym_wrapper.learned_pahm_ode import LearnedPAHMODE

env = LearnedPAHMODE(
    render_mode=None,
    model_path="pahm_model/pahm_fast_v2_best.pth",
    reset_angle_deg=720,
    enable_wind=True,
    wind_pattern="gust",
    wind_seed=42,
    theta_ref=1.0,
)

obs, info = env.reset(seed=123, options={"randomize": True})
obs, reward, terminated, truncated, info = env.step([0.2])
```

Tambien puede tomar la fuente automatica desde `gym_wrapper/config.json`:

```json
"control": {
  "theta_ref": 0.0,
  "theta_ref_min": -12.566370614359172,
  "theta_ref_max": 12.566370614359172
},
"reward": {
  "tracking_error_weight": 4.0,
  "velocity_weight": 0.1,
  "control_weight": 0.01
},
"wind": {
  "enabled": false,
  "source": "wind_process",
  "patterns": ["calm", "gust", "sustained", "turbulent"],
  "default_pattern": "gust",
  "max_torque": 20.0,
  "stochastic": true
}
```

La observacion del entorno para Etapa 3 es:

```text
[theta, theta_dot, theta_ref]
```

Campos utiles en `info`:

- `wind_active`
- `wind_mag`
- `wind_angle`
- `wind_torque`
- `wind_pattern`: patron activo del episodio.
- `configured_wind_pattern`: patron configurado en el constructor.
- `wind_automatic`: indica si manda `WindProcess`.
- `theta_ref`: angulo objetivo actual.
- `tracking_error`: `theta - theta_ref`.
- `abs_tracking_error`: valor absoluto del error de seguimiento.

Reglas importantes:

- `render_mode=None` no requiere Pygame durante el import ni durante entrenamiento headless.
- Si `enable_wind` no se pasa al constructor, el entorno usa `wind.enabled` desde `config.json`.
- `enable_wind=True` da prioridad a `WindProcess`; `set_wind()` queda para demo/manual cuando `enable_wind=False`.
- La fuente automatica se construye mediante `WindSource`, lo que permite sustituir `WindProcess` por otra fuente futura sin cambiar el lazo principal del entorno.
- `theta_ref` se puede configurar en `config.json`, por constructor, por `set_theta_ref(value)` o por `reset(options={"theta_ref": value})`.
- Los pesos de recompensa se leen de `reward.tracking_error_weight`, `reward.velocity_weight` y `reward.control_weight`.
- `wind_seed` controla el RNG interno de `WindProcess`.
- `reset(seed=...)` controla el RNG del entorno Gymnasium, incluyendo estado inicial y seleccion de patron cuando `randomize_wind_pattern=True`.

Tests especificos de este branch:

```bash
.venv/bin/python -m pytest tests/test_learned_pahm_env_wind.py -q
```

Estado verificado:

```text
16 passed
```

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

Etapa 1, Etapa 2 y los primeros bloques de Etapa 3 para entorno con perturbaciones, `theta_ref` configurable y recompensa de seguimiento pasan las pruebas actuales. Para continuar con Etapa 3B falta implementar:

- `train_rl.py` headless con Stable Baselines3;
- agentes naive y robusto;
- carga de politica entrenada en el modo `RL` de la demo;
- comparacion cuantitativa de error de seguimiento, tiempo de estabilizacion y sobreimpulso.

Estado verificado de la suite completa:

```text
42 passed, 1 warning
```

Antes de entregar, limpiar `requirements.txt`: actualmente contiene marcadores de conflicto Git.
