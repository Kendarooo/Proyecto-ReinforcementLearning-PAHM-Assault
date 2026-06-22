# Instrucciones de ejecución — Proyecto 2 PAHM

## Requisitos previos

- Python 3.12
- Venv del proyecto en `.venv/` (raíz del repositorio)
- Dependencias instaladas: `pip install -r requirements.txt`

---

## 1. Activar el entorno virtual

Siempre activar el venv del proyecto antes de correr cualquier comando:

```bash
source .venv/bin/activate
```

> **Importante:** si tienes otro venv activo (por ejemplo, `etapa-0-DemonAttack/.venv`),
> desactívalo primero con `deactivate`. Usar el venv incorrecto causa errores de
> importación de `pandas`, `sklearn` y `matplotlib`.

Verificar que el intérprete correcto está activo:

```bash
which python   # debe apuntar a .venv/bin/python
```

---

## 2. Estructura de archivos esperada

```text
data/                                        # CSVs de trayectorias reales
pahm_model/pahm_fast_v2_best.pth             # Modelo físico base (profesor)
pahm_model/pahm_ode_v4_best.pth              # Modelo híbrido ODE + red residual (profesor)
checkpoints/estimator_checkpoint_epoch_40.pth # Estimador GRU entrenado (Etapa 1)
outputs/tau_w/tau_w_<split>_<idx>.npy        # Secuencias τ̂_w exportadas
outputs/tau_w/estimador_wind.pth             # Copia del estimador GRU para Etapa 2
artifacts/stage2/gmm_wind_model.pkl          # Modelo GMM entrenado (Etapa 2)
artifacts/stage3/evaluation/                 # Métricas y tablas de Etapa 3
artifacts/stage1/fr7_comparacion_modelos.md  # Tabla comparativa FR-7
```

---

## 3. Correr las pruebas

### Suite completa

```bash
python -m pytest -v
```


### Solo Etapa 1

```bash
python -m pytest test/test_verification.py -v -s
```

Imprime la tabla FR-7 con 3 modelos comparados (ODE pura, ODE + red residual, ODE + GRU).

### Solo Etapa 2

```bash
python -m pytest tests/test_learned_pahm_env_wind.py tests/test_unsupervised_model.py -v
```

---

## 4. Calidad de código

### Pylint

```bash
pylint pahm_model/train_estimator.py pahm_model/custom_loss.py \
       pahm_model/sequence_estimator.py pahm_model/pahm_fast.py \
       pahm_model/pahm_ode.py
```

Resultado esperado: **≥ 8.0/10** (cumple NFR-5). Configuración en `pyproject.toml`.

### Ruff

```bash
ruff check pahm_model/ gym_wrapper/ test/ tests/
```

---

## 5. Etapa 1 — Estimador GRU de viento

### Entrenar el estimador

```bash
python pahm_model/train_estimator.py
```

Requiere: `data/` con CSVs, `pahm_model/pahm_fast_v2_best.pth`, y `gym_wrapper/config.json`.
Guarda checkpoints en `checkpoints/estimator_checkpoint_epoch_<N>.pth`.
Telemetría en W&B bajo el proyecto `etapa-1`.

### Exportar τ̂_w para Etapa 2

```bash
python pahm_model/export_tau_w.py
```

Salida en `outputs/tau_w/`.

---

## 6. Etapa 2 — Representación no supervisada (GMM)

### Validar con datos sintéticos

```bash
python -m pahm_stage2.validate_synthetic_real --config configs/stage2_config.json
```

Genera figuras en `artifacts/stage2/validation/` y métricas ARI/NMI en `synthetic_metrics.json`.

### Entrenar GMM

```bash
python -m pahm_stage2.train_unsupervised --config configs/stage2_config.json
```

Guarda modelo en `artifacts/stage2/gmm_wind_model.pkl`.

---

## 7. Etapa 3 — Control robusto con RL

### Entrenar agentes (naive y robust)

```bash
# Solo naive
python train_rl.py --config gym_wrapper/config.json

# Ambos (naive + robust)
python train_rl.py --config gym_wrapper/config.json --mode all
```

Modelos guardados en `pahm_stage3/`. Telemetría en W&B bajo el proyecto `etapa-3`.

### Evaluar controladores

```bash
python evaluate_controllers.py --config configs/stage3_config.json
```

Genera en `artifacts/stage3/evaluation/`:
- `controller_metrics.json` y `controller_metrics.csv`
- `informe.md` con tabla naive vs robust y desglose por perturbación

### Demo visual interactiva

```bash
python gym_wrapper/test_pahm_ode_env.py \
  --model pahm_model/pahm_fast_v2_best.pth \
  --reset_angle 720 \
  --config configs/stage3_config.json
```

Durante la demo:
- Tecla `G`: activa ráfaga de viento manual
- Modos disponibles: `Off`, `Step`, `Sine`, `Random`, `Manual`, `RL`
- Controladores: `PID`, política RL naive o robust
- Patrones de viento: `Calm`, `Gust`, `Sustained`, `Turbulent`

---

## 8. Configuración centralizada

Todos los hiperparámetros viven en dos archivos:

| Archivo | Cubre |
|---|---|
| `gym_wrapper/config.json` | Etapa 1: estimador GRU, dt, seq_len, λ₁, λ₂, epochs; Etapa 3: RL training, entorno, recompensa, viento |
| `configs/stage2_config.json` | Etapa 2: GMM, número de componentes, rutas de τ̂_w |
| `configs/stage3_config.json` | Etapa 3: evaluación, demo, rutas de políticas |

No modificar parámetros directamente en el código (NFR-1).

---

## 9. Notas importantes

- **CON-1**: los parámetros físicos `α, β, γ` de `pahm_fast_v2_best.pth` permanecen
  congelados durante el entrenamiento del estimador GRU.
- **CON-2**: el estimador GRU recibe únicamente `(sin θ, cos θ, dθ/dt, u)` como
  entrada. Nunca recibe el residuo `r(t) = θ_obs − θ_ODE`.
- **Reentrenar GRU**: el checkpoint actual (`epoch_40`) fue entrenado con un bug de
  gradientes ya corregido. Para beneficiarse del fix, correr `train_estimator.py`
  de nuevo y reemplazar el checkpoint.
