[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24080422&assignment_repo_type=AssignmentRepo)
# PAHM (Péndulo Amortiguado con Hélice a Motor) — Código Base
## EL5857 Aprendizaje Automático
### Escuela de Ingeniería Electrónica, Tecnológico de Costa Rica (TEC)
### I Semestre, 2026

Este repositorio contiene la línea base de software e infraestructura de simulación para el **Proyecto 2**. El objetivo del proyecto es modelar las perturbaciones de viento latentes no observadas mediante celdas recurrentes (aprendizaje de secuencias) y diseñar una política de control robusta utilizando **Aprendizaje por Refuerzo (RL)** sobre un entorno Gymnasium acoplado a un modelo físico híbrido (Caja Gris).

## Presentación del proyecto

El proyecto integra **modelado físico híbrido, estimación recurrente de perturbaciones, representación no supervisada y control robusto mediante aprendizaje por refuerzo**. Además, incluye una Etapa 0 individual con un agente **DQN entrenado sobre Atari Assault**, utilizada como ejercicio introductorio de RL antes de abordar el sistema PAHM.

La arquitectura conecta una estimación latente del torque de viento con un modelo no supervisado de perturbaciones y, posteriormente, con políticas RL entrenadas para mantener el seguimiento de referencia bajo condiciones de viento no vistas durante la evaluación.

### Resultados destacados

En la evaluación final de Etapa 3, el controlador **robusto** superó al controlador **naive** en las cinco métricas globales registradas:

| Métrica | Naive | Robust | Mejora del controlador robusto |
| --- | ---: | ---: | ---: |
| MAE de seguimiento | 0.0210443 | 0.0120313 | **42.83% menor** |
| MSE de seguimiento | 0.00469722 | 0.00233672 | **50.25% menor** |
| Tiempo de estabilización | 1.344 | 0.688 | **48.81% menor** |
| Sobreimpulso máximo | 0.397407 | 0.190058 | **52.18% menor** |
| Retorno acumulado | -29.8621 | -14.5259 | **51.36% mejor** |

Los resultados completos por episodio y por tipo de perturbación se encuentran en `artifacts/stage3/evaluation/`.

---

## 1. Estructura del Repositorio

El código está organizado siguiendo principios de diseño limpio, separación de responsabilidades (SRP) y el patrón Composite para la interfaz gráfica.

```text
├── docs/                          # Enunciado del proyecto y división de tareas
│   ├── proy2.pdf
│   └── division_tareas_proy2.pdf
│
├── configs/                       # Configuración por etapa (JSON)
│   ├── stage2_config.json
│   └── stage3_config.json
│
├── gym_wrapper/                   # Entorno de Simulación e Interfaz Gráfica
│   ├── config.json                 # Parámetros visuales, geometría, colores y viento (SSOT Layout)
│   ├── pid_config.json             # Ganancias calculadas Kp, Ki, Kd de la planta linealizada
│   ├── learned_pahm_ode.py         # Entorno Gymnasium de la planta (Neural ODE / Fast RNN)
│   ├── pahm_ui.py                  # Módulo de UI en Pygame (VBox, HBox, Controles Polares)
│   ├── wind_process.py             # Generador exógeno estocástico de patrones de viento
│   ├── wind_source.py              # Fuente de viento configurable (WindSource)
│   ├── pid.py                      # Implementación del controlador PID con compensación cuadrática
│   ├── rl_policy.py                # Carga y acoplamiento de políticas RL en la demo
│   ├── demo_wind.py                # Demostración interactiva de inyección de viento
│   ├── test_pahm_ode_env.py        # Lazo principal interactivo y demostración visual
│   ├── test_pahm_env.py            # Pruebas del entorno Gymnasium
│   └── pahm_ode.py                 # Enlace simbólico → pahm_model/pahm_ode.py
│
├── pahm_model/                     # Modelos Dinámicos y Etapa 1 (Estimador de Viento)
│   ├── dataloader.py                # Carga y segmentación de datos reales en RAM
│   ├── pahm_fast.py                 # Physics RNN con paso RK4 (PAHMFastModel)
│   ├── pahm_fast_v2_best.pth        # Checkpoint preentrenado del profesor (CON-1)
│   ├── pahm_ode.py                  # Modelo híbrido Neural ODE
│   ├── pahm_ode_v4_best.pth         # Checkpoint baseline ODE + residual
│   ├── utils.py                     # Solvers numéricos e interpolación por Splines Cúbicos
│   ├── train_ode.py                 # Entrenamiento en dos fases (Física → Híbrido)
│   ├── train_pid.py                 # Sintonización analítica de ganancias por asignación de polos
│   ├── test_ode.py                  # Evaluación comparativa de trayectorias en lazo abierto
│   ├── sequence_estimator.py        # Etapa 1 (FR-4): GRU que infiere τ_w(t) — CON-2
│   ├── rk4_integrator.py            # Etapa 1 (FR-5): inyecta τ_w(t) aditivo en el paso RK4
│   ├── custom_loss.py               # Etapa 1 (FR-6): pérdida triple (Ec. 13)
│   ├── compute_residuals.py         # Etapa 1 (FR-3): calcula r(t) = θ_obs − θ_ODE
│   ├── train_estimator.py           # Etapa 1 (FR-4/5/6): entrena el estimador GRU
│   └── export_tau_w.py              # Etapa 1 (I-1): exporta τ_w(t) + checkpoint para Grupo 2
│
├── pahm_stage2/                    # Etapa 2 — Representación No Supervisada
│   ├── config.py                     # Configuración de la etapa
│   ├── estimator_interface.py        # Interfaz hacia el estimador de Etapa 1
│   ├── feature_extractor.py          # Extracción de características desde τ_w(t)
│   ├── unsupervised_model.py         # Modelo no supervisado (autoencoder/VAE/GMM)
│   ├── train_unsupervised.py         # Entrenamiento no supervisado
│   ├── generate_tau_w.py             # Generación de τ_w(t) para consumo de Etapa 2
│   ├── tau_dataset.py                # Dataset sin etiquetas sobre τ_w(t)
│   ├── validate_synthetic_real.py    # Validación sintético → real (ARI/NMI)
│   ├── validator.py                  # Lógica de validación de complejidad
│   ├── wind_sampler.py               # Muestreador de perturbaciones (I-3 hacia Grupo 1)
│   ├── artifacts.py                  # Persistencia de artefactos de la etapa
│   └── README.md                     # Documentación específica de Etapa 2
│
├── pahm_stage3/                    # Etapa 3 — Telemetría de Control Robusto (RL)
│   └── wandb_logger.py                # Logger centralizado de W&B para entrenamiento/evaluación
│
├── etapa2_unsupervised/            # Legado experimental: autoencoder simple, no ruta oficial
│   ├── autoencoder.py
│   ├── tau_w_dataset.py
│   └── train_unsupervised.py
│
├── etapa0-assault-kendall/         # Etapa 0 individual — DQN sobre Atari Assault
│
├── test/                           # Pruebas unitarias — Etapa 1 (TDD, NFR-6)
│   ├── test_checkpoint_persistence.py
│   ├── test_custom_loss.py
│   ├── test_rk4_integrator.py
│   ├── test_verification.py
│   └── test_stage2_unsupervised.py
│
├── tests/                          # Pruebas unitarias — Etapas 2 y 3
│   ├── test_feature_extractor.py
│   ├── test_tau_dataset.py
│   ├── test_unsupervised_model.py
│   ├── test_validator.py
│   ├── test_validate_synthetic_real.py
│   ├── test_wind_sampler.py
│   ├── test_generate_tau_w.py
│   ├── test_train_unsupervised.py
│   ├── test_learned_pahm_env_wind.py
│   ├── test_demo_interactive_wind.py
│   ├── test_rl_policy_demo.py
│   ├── test_train_rl.py
│   ├── test_evaluate_controllers.py
│   └── test_stage3_wandb_logger.py
│
├── artifacts/                      # Resultados generados por Etapas 2 y 3
│   ├── stage2/
│   └── stage3/
│
├── data/                           # CSVs reales del laboratorio (provistos por la cátedra)
│
├── outputs/
│   └── tau_w/                      # Hito I-1: datos resultantes de la extracción de Etapa 1
│
├── train_rl.py                     # Etapa 3: entrenamiento de agentes RL (naive/robusto)
├── evaluate_controllers.py         # Etapa 3: comparación cuantitativa de controladores
├── AI_AUDIT.md                     # Auditoría de uso de IA
├── instructions.md                 # Notas/instrucciones internas del equipo
├── state_project_etapa1.md         # Bitácora de estado de avance de Etapa 1
├── requirements.txt
└── README.md
```

> **Nota:** La implementación vigente y defendible de Etapa 2 es `pahm_stage2/` porque cubre GMM, selección por BIC, validación sintético-real con ARI/NMI y `WindSampler`. `etapa2_unsupervised/` se conserva solo como prototipo histórico de autoencoder simple para trazabilidad y pruebas heredadas; no es la ruta recomendada para entrega.

> **Nota:** La auditoría de uso de IA está consolidada en `AI_AUDIT.md`, conforme a CON-4.

> **Nota:** La base estructurada para el informe final vive en `informe.md`. Incluye la tabla NFR-7 para registrar tiempos de entrenamiento/inferencia/evaluación y hardware usado en las corridas finales.

---

## 2. Requisitos e Instalación

El entorno requiere Python 3.12+ y las dependencias estándar de aprendizaje profundo y simulación física. En Ubuntu/Python 3.12 evita usar `pip` del sistema directamente, porque puede fallar con `externally-managed-environment` (PEP 668).

Instalación recomendada desde la raíz del repositorio:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

```

Para usar los entornos Atari de la Etapa 0, acepta las ROMs una vez después de instalar:

```bash
AutoROM --accept-license
```

*Nota: si necesitas instalar PyTorch para CPU explícitamente, puedes seguir la variante recomendada por PyTorch para tu sistema y luego ejecutar `uv pip install -r requirements.txt` para el resto de dependencias.*

---


## 3. Guía de Uso

### A. Ejecución del Entorno Interactivo (GUI)

Para levantar el simulador visual, inspeccionar la dinámica del péndulo, graficar en tiempo real mediante el osciloscopio e inyectar ráfagas de viento interactivas:

```bash
cd gym_wrapper
python test_pahm_ode_env.py  --reset_angle 720 --max_steps 10000000 --model ../pahm_model/pahm_fast_v2_best.pth --pid pid_config.json 
```

**Controles de la Interfaz:**

* **Panel de Modos:** Permite alternar entre entradas manuales, señales senoidales, control analítico PID, o el agente RL.
* **Perturbación de Viento:** Al activar el interruptor de viento, es posible seleccionar patrones preconfigurados (`Calm`, `Gust`, `Sust`, `Turb`) o modular vectorialmente la magnitud y dirección arrastrando el mouse sobre el control polar circular.

### B. Etapa 1 — Estimacion de Perturbaciones de Viento (FR-3 a FR-7)

Esta etapa construye un estimador recurrente (GRU) que infiere el torque de viento latente tau_w(t) a partir del historial cinematico reciente `(sin theta, cos theta, dtheta/dt, u)`, sin acceso directo al residuo `r(t) = theta_obs - theta_ODE` (restriccion **CON-2**). El modelo fisico base del profesor (`pahm_fast_v2_best.pth`) permanece congelado durante todo el proceso (**CON-1**).

**Paso 1 - Calculo de residuos (FR-3).** Diagnostico inicial: corre la ODE base en lazo abierto sobre todo el dataset y calcula `r(t)` por trayectoria, generando una tabla de MSE/maximo/desviacion estandar. Los flags son opcionales — por defecto usa `gym_wrapper/config.json`, `data/` y `outputs/residuals/`:

```bash
python pahm_model/compute_residuals.py
```

Si tu estructura de carpetas difiere de la convención del repo, puedes sobrescribir cualquiera de las tres rutas:

```bash
python pahm_model/compute_residuals.py --config gym_wrapper/config.json --data_dir data --output_dir outputs/residuals
```

**Paso 2 - Entrenamiento del estimador (FR-4, FR-5, FR-6).** Entrena la GRU con la funcion de perdida de tres terminos (reconstruccion + parsimonia + suavidad temporal, Ecuacion 13) usando *Truncated BPTT* y *gradient clipping* para estabilidad numerica:

```bash
python pahm_model/train_estimator.py
```

El directorio de checkpoints (`estimator_hyperparameters.checkpoint_dir` en `config.json`, por defecto `checkpoints/`) se crea automáticamente al iniciar el entrenamiento — no requiere preparación manual.

Todos los hiperparametros (arquitectura de la GRU, `lambda1`/`lambda2`, `learning_rate`, `tau_max`, semilla, `dt`) se leen desde `gym_wrapper/config.json` bajo `estimator_hyperparameters` -- no hay valores hardcodeados (**NFR-1**). El progreso se registra en Weights & Biases bajo el proyecto `etapa-1`, y los checkpoints se guardan periodicamente en `checkpoints/` (**NFR-2**, **NFR-3**).

> **Nota de diseno:** la salida de la GRU esta acotada con `tanh(x) * tau_max` para evitar que el estimador colapse a torques no fisicos durante el entrenamiento. `tau_max` es configurable en `estimator_hyperparameters.tau_max` (valor por defecto: `2.0`, calibrado contra el rango de residuos observado en el Paso 1).

**Paso 3 - Comparacion cuantitativa (FR-7).** La suite de pruebas incluye la evaluacion en lazo abierto sobre el conjunto de prueba retenido, comparando ODE pura vs. ODE + estimador GRU:

```bash
pytest test/test_verification.py::test_open_loop_baseline_comparison_fr7 -v -s
```

El test requiere que exista el checkpoint final del estimador (`checkpoints/estimator_checkpoint_epoch_<epochs>.pth`) y la carpeta `data/` con los CSV del laboratorio. Imprime el MSE de ambas configuraciones y la mejora relativa; falla si el estimador no iguala o supera a la ODE pura.

**Paso 4 - Exportacion hacia el Grupo 2 (Hito I-1).** Una vez completado el entrenamiento, genera la senal tau_w(t) para todas las trayectorias y el checkpoint final empaquetado:

```bash
python pahm_model/export_tau_w.py
```

Esto produce `outputs/tau_w/tau_w_<split>_<idx>.npy` por trayectoria y copia el checkpoint final como `outputs/tau_w/estimador_wind.pth` -- los dos artefactos formales del Hito I-1 (ver seccion 4 de `division_tareas_proy2.pdf`).

**Pruebas unitarias de la Etapa 1 (NFR-6):**

```bash
pytest test/ -v
```

Cubre: reproduccion de parametros fisicos conocidos (a), dimensionalidad correcta del estimador (b), salida nula del estimador ante datos sin perturbacion (c), persistencia de checkpoints (e), y recuperacion de un torque sintetico conocido dentro de tolerancia (f).

### C. Entrenamiento de la Dinamica Hibrida (Neural ODE / Fast RNN)

Para entrenar el modelo de caja gris a partir de capturas de laboratorio usando el enfoque de sliding windows sobre memoria RAM:

```bash
cd pahm_model
python train_ode.py --model_type fast --epochs 100 --warmup_epochs 20 --data_dir ../data/

```

*Este script opera en dos fases autónomas: congela la red neuronal residual durante los `warmup_epochs` para forzar la convergencia de los parámetros físicos reales ($\alpha, \beta, \gamma$), y posteriormente libera los pesos neuronales para ajustar la dinámica fina no modelada.*

### D. Sintonización Analítica del PID

Si requieres recalcular las ganancias del lazo de control clásico a partir de la física aprendida por un punto de control `.pth`:

```bash
python train_pid.py --model_path ../pahm_model/pahm_ode_best.pth --output gym_wrapper/pid_config.json --ts 2.0 --pole_ratio 10.0

```

### E. Conexión Etapa 1 → Etapa 2 no supervisada

La Etapa 1 produce su artefacto formal de salida en el Paso 4 de la sección B (`python pahm_model/export_tau_w.py`), que escribe una señal por trayectoria en `outputs/tau_w/tau_w_<split>_<idx>.npy` y copia el checkpoint del estimador como `outputs/tau_w/estimador_wind.pth`.

```bash
.venv/bin/python pahm_model/export_tau_w.py
```

La ruta vigente de Etapa 2 es `pahm_stage2/`. Primero valida la estructura sintética con patrones conocidos y luego entrena el modelo GMM productivo desde el manifiesto de señales reales:

```bash
.venv/bin/python -m pahm_stage2.validate_synthetic_real \
  --config configs/stage2_config.json

.venv/bin/python -m pahm_stage2.train_unsupervised \
  --config configs/stage2_config.json
```

Esta ruta extrae features de `tau_w(t)`, selecciona complejidad mediante BIC, valida recuperación de estructura con ARI/NMI y guarda `artifacts/stage2/gmm_wind_model.pkl`, consumible mediante `pahm_stage2.wind_sampler.WindSampler`.

La carpeta `etapa2_unsupervised/` contiene un autoencoder simple usado como prototipo temprano. Se mantiene para trazabilidad y pruebas heredadas, pero no debe presentarse como implementación principal de FR-9 a FR-11.

Para registrar Etapa 2 en Weights & Biases, primero autentica la sesión:

```bash
wandb login
```

Luego ajusta la sección `wandb` de `configs/stage2_config.json`. Por defecto la ruta vigente usa el proyecto `pahm-stage2`; Etapa 1 usa el proyecto `etapa-1` desde `pahm_model/train_estimator.py`.

---

## 4. Invariantes de Diseño y Contratos Técnicos

Al desarrollar extensiones sobre este código base, se deben respetar las siguientes decisiones arquitectónicas acordadas:

1. **Espacio de Observaciones 4D:** El entorno Gymnasium entrega un vector continuo $[ \theta, \dot{\theta}, \theta_{ref}, e_{int} ]$, donde $e_{int}$ es la integral acotada del error de seguimiento. Esto permite entrenar politicas de seguimiento de referencia con contexto integral usando librerias RL estandar como *Stable Baselines3*.
2. **Sin Envoltura de Ángulo (*No Wrapping*):** El ángulo en el estado interno se mantiene de forma acumulativa y continua. Envolver el ángulo en el intervalo $[-\pi, \pi]$ dentro del estado de la planta oculta los sobregiros y desincroniza los integradores; cualquier transformación visual o matemática debe realizarse externamente de forma aislada.
3. **Robustez mediante Domain Randomization:** Para mitigar el sobreajuste (*overfitting*) a la trayectoria inicial desde el reposo, el método `reset` acepta el argumento `options={"randomize": True}`. Esto inicializa el episodio en un punto cinemático aleatorio pero seguro.
4. **Cinemática del Viento y Estelas:** La actualización visual de las partículas de viento calcula su origen y destino basándose en el vector de desplazamiento real por cuadro ($dx, dy$), aplicando un factor de amplificación visual estático para que el flujo sea perfectamente visible incluso ante brisas de baja magnitud.
5. **Fuente de Viento Configurable:** El entorno construye la perturbación automática desde `gym_wrapper/config.json` (`wind.enabled`, `wind.source`, `wind.default_pattern`, `wind.max_torque`). La lógica queda detrás de `WindSource`; `wind.source="wind_process"` usa los patrones procedurales y `wind.source="stage2_sampler"` carga `artifacts/stage2/gmm_wind_model.pkl` mediante `WindSampler` para alimentar Etapa 3 con la representación no supervisada de Etapa 2.
6. **Referencia y Recompensa Configurables:** `theta_ref` se inicializa desde `control.theta_ref`, puede sobrescribirse por constructor o `reset(options={"theta_ref": ...})`, y la recompensa usa los pesos de `reward.tracking_error_weight`, `reward.integral_weight`, `reward.velocity_weight`, `reward.control_weight` y `reward.action_delta_weight`.
7. **Robustez y Acciones Suaves:** El entorno mantiene `prev_action` por episodio y penaliza cambios consecutivos mediante `reward.action_delta_weight * (u_t-u_{t-1})^2`. `reset()` reinicia `prev_action=0.0` para evitar fuga entre episodios.
8. **Entrenamiento RL desde Estados Iniciales Diversos:** `rl_training.randomize_reset=true` hace que entrenamiento use `reset(options={"randomize": True})`; `evaluation.randomize_reset` se controla de forma independiente para evaluación.

---

## 5. Integración del Agente de Aprendizaje por Refuerzo (RL)

La demo interactiva (`gym_wrapper/test_pahm_ode_env.py`) puede cargar una política RL entrenada y usarla en el modo `RL` del panel:

```bash
python gym_wrapper/test_pahm_ode_env.py \
  --model pahm_model/pahm_fast_v2_best.pth \
  --reset_angle 720 \
  --config configs/stage3_config.json
```

La sección `demo` permite elegir `rl_model_type` (`naive` o `robust`), rutas de modelos entrenados, `deterministic_policy`, `interactive_wind`, `show_particles` y `show_wind_torque`. También se puede pasar una ruta explícita con `--rl_model`. En la demo, mantener presionada la tecla `G` inyecta una ráfaga manual configurada en `wind.manual_gust_*`; esto no entrena modelos y permite observar la respuesta de la política RL ante perturbaciones en tiempo real.

El script de entrenamiento RL vive en `train_rl.py` y opera en modo *headless* (sin renderizado gráfico). Lee `rl_training`, `experiments` y `wandb` desde `gym_wrapper/config.json`, permite modos `naive` y `robust`, y guarda modelos separados en la ruta configurada. El modo `robust` está configurado para usar `wind_source="stage2_sampler"`, conectando entrenamiento RL con el GMM aprendido en Etapa 2:

```bash
python train_rl.py --config gym_wrapper/config.json
python train_rl.py --config gym_wrapper/config.json --mode all
```

La comparación cuantitativa de controladores vive en `evaluate_controllers.py`. Carga modelos ya entrenados, evalúa `naive` y `robust` en modo headless con viento de evaluación, y cicla los episodios sobre `evaluation.unseen_wind_patterns` para probar perturbaciones no vistas:

```bash
python evaluate_controllers.py --config configs/stage3_config.json
```

El resultado queda en `artifacts/stage3/evaluation/controller_metrics.json`, `controller_metrics.csv` e `informe.md`. El informe incluye tabla resumen naive vs robust, desglose por perturbación, interpretación de MAE/MSE/tiempo de estabilización/sobreimpulso/retorno y una conclusión explícita sobre si el controlador robusto supera o no al naive.

La telemetría de Etapa 3 está centralizada en `pahm_stage3/wandb_logger.py`. La sección `wandb` permite activar/desactivar W&B, usar `mode=disabled` para pruebas locales, y controlar `log_models`/`log_evaluation`. Cuando está habilitado, el entrenamiento registra hiperparámetros, configuración de entorno/recompensa/viento, métricas de entrenamiento y artefactos de modelo/config; la evaluación registra métricas comparativas y artefactos JSON/CSV.

El entorno ya acepta `theta_ref` por configuracion, constructor, `set_theta_ref(value)` o `reset(options={"theta_ref": value})`, observa `[theta, theta_dot, theta_ref, error_integral]` y reporta `theta_ref`, `tracking_error`, `abs_tracking_error` y `error_integral` en `info`.
