# AI Audit - Grupo 2

## Alcance

Este documento registra el uso de modelos de lenguaje durante el trabajo del Grupo 2 en la Etapa 2 del proyecto: representacion no supervisada de perturbaciones y validacion sintetico-real. Tambien registra apoyo en la integracion temprana de Etapa 3, especificamente el entorno Gymnasium con perturbaciones automaticas para entrenamiento headless, referencia configurable `theta_ref` y recompensa de seguimiento.

## Modelos de lenguaje utilizados

- Codex, asistente de programacion basado en GPT-5, usado dentro del entorno local de desarrollo.

## Tareas asistidas por IA

- Revision del enunciado del proyecto y de la division de tareas para identificar los requisitos aplicables al Grupo 2.
- Diseno de la estrategia de Etapa 2 usando GMM sobre features de `tau_w(t)` y seleccion de complejidad mediante BIC.
- Implementacion de modulos de soporte en `pahm_stage2/`:
  - `feature_extractor.py`
  - `unsupervised_model.py`
  - `validator.py`
  - `wind_sampler.py`
  - `validate_synthetic_real.py`
  - `train_unsupervised.py`
- Actualizacion de `configs/stage2_config.json` para centralizar parametros de modelo, validacion, rutas y W&B.
- Creacion y ajuste de pruebas unitarias en `tests/` para extraccion de features, GMM, validacion ARI/NMI, sampler, pipeline sintetico-real y entrenamiento desde manifest.
- Redaccion de documentacion tecnica en `pahm_stage2/README.md`.
- Apoyo en interpretacion de resultados de validacion, incluyendo BIC, ARI y NMI.
- Revision exhaustiva de la integracion entre Etapa 1 y Etapa 2, incluyendo:
  - verificacion de la ubicacion esperada de `data/`;
  - identificacion del checkpoint esperado por `gym_wrapper/config.json`;
  - copia del checkpoint exportado `outputs/tau_w/estimador_wind.pth` a `checkpoints/estimator_checkpoint_epoch_40.pth` para cumplir el contrato actual de pruebas y exportacion;
  - creacion de `instructions.md` con comandos de verificacion, ejecucion visual y estado previo a Etapa 3.
- Implementacion inicial de Etapa 3 en el branch `feature/etapa3-entorno-perturbaciones`, sin crear una carpeta nueva `pahm_stage3`; se extendio el entorno existente en `gym_wrapper/`.
- Ajuste de `gym_wrapper/learned_pahm_ode.py` para:
  - permitir import y ejecucion headless sin depender de Pygame;
  - cargar `config.json` usando `__file__` como ancla en vez del directorio actual de ejecucion;
  - integrar `WindProcess` como fuente automatica opcional de perturbaciones;
  - encapsular la perturbacion automatica detras de `WindSource` para poder sustituir `WindProcess` por una fuente aprendida sin reescribir `env.step()`;
  - permitir construir la fuente de viento desde `gym_wrapper/config.json`;
  - separar `configured_wind_pattern` y `active_wind_pattern`;
  - reportar `wind_active`, `wind_mag`, `wind_angle`, `wind_torque`, `wind_pattern`, `configured_wind_pattern` y `wind_automatic` en `info`;
  - mantener `set_wind()` como mecanismo manual para la demo cuando `enable_wind=False`.
- Creacion de pruebas en `tests/test_learned_pahm_env_wind.py` para el contrato de entorno con perturbaciones.
- Implementacion del bloque `feature/etapa3-theta-ref-reward` para FR-13:
  - `theta_ref` configurable desde `gym_wrapper/config.json`, por constructor, `set_theta_ref(value)` y `reset(options={"theta_ref": value})`;
  - observacion expandida a `[theta, theta_dot, theta_ref]`;
  - `observation_space` actualizado a dimension 3;
  - recompensa de seguimiento basada en error cuadratico, velocidad angular y esfuerzo de accion, con pesos configurables;
  - reporte de `theta_ref`, `tracking_error` y `abs_tracking_error` en `info`;
  - sincronizacion de `theta_ref` con el slider de setpoint en la demo visual.
- Ampliacion de `tests/test_learned_pahm_env_wind.py` para cubrir observacion 3D, cambio de recompensa al cambiar `theta_ref`, `reset(options={"randomize": True})` y compatibilidad entre viento automatico y referencia.
- Implementacion de `train_rl.py` como entrada headless independiente para entrenamiento con Stable Baselines3:
  - carga de `rl_training` desde configuracion externa;
  - modos `naive` y `robust` para desactivar/activar perturbaciones;
  - orquestacion de `train_all_modes(config_path)` para entrenar los modos declarados en `experiments.modes`;
  - construccion de entorno con `render_mode=None`;
  - seleccion configurable de algoritmo (`PPO`, `A2C`, `SAC`);
  - integracion W&B desactivable para registrar hiperparametros, modo y ruta del modelo;
  - guardado final de modelo y callbacks de checkpoint configurables;
  - pruebas smoke en `tests/test_train_rl.py`, con skip controlado si `stable-baselines3` no esta instalado.
- Integracion de politicas RL entrenadas en la demo visual:
  - modulo `gym_wrapper/rl_policy.py` para cargar politica, predecir acciones y ejecutar un paso RL sin entrenamiento;
  - seleccion de modelo `naive`/`robust` desde la seccion `demo` de configuracion;
  - acoplamiento del modo `RL` en `gym_wrapper/test_pahm_ode_env.py`;
  - pruebas con politica mock en `tests/test_rl_policy_demo.py`.

## Estrategias de prompting empleadas

- Se proporciono contexto del enunciado del proyecto, la rubrica y la division de tareas entre grupos.
- Se pidio trabajar por ramas e issues concretos, especialmente sobre la validacion sintetico-real de Etapa 2.
- Se especificaron restricciones de diseno:
  - no rehacer la arquitectura ya existente;
  - mantener responsabilidades separadas por modulo;
  - producir evidencia util para la rubrica L3.2;
  - registrar metricas en W&B;
  - permitir fallback cuando todavia no exista la interfaz I-1 de Grupo 1.
- Se solicitaron explicaciones paso a paso para mantener revision humana y evitar cambios opacos.

## Validacion humana realizada

- El equipo reviso la estrategia antes de implementar la validacion sintetico-real.
- Se verifico manualmente que el enfoque GMM + BIC fuera coherente con los patrones esperados de viento: calma, rafaga, sesgo sostenido y turbulencia.
- Se inspeccionaron los resultados generados en `artifacts/stage2/validation/synthetic_metrics.json`.
- Se confirmo que el resultado sintetico recupera cuatro patrones con:
  - `ARI = 1.0`
  - `NMI = 1.0`
  - `n_components_selected = 4`
- Se reviso que el script no falle cuando todavia no existe el manifest real de `tau_w(t)`.
- Se verifico que `train_unsupervised.py` falle de forma explicita cuando no existe el manifest y guarde un checkpoint cuando recibe un manifest valido.
- Se ejecuto la suite de pruebas unitarias con `pytest`.
- Se ejecuto analisis estatico con `ruff`.
- Se agrego la carpeta `data/` en la raiz del repositorio y se confirmo que los tests de integracion ya pueden leer los CSV reales.
- Se verifico que `outputs/tau_w/estimador_wind.pth` contiene las llaves esperadas:
  - `epoch`
  - `loss`
  - `model_state_dict`
  - `optimizer_state_dict`
- Se ejecuto la suite completa `tests/` + `test/` despues de colocar el checkpoint esperado por la configuracion actual.
- Se verifico que el entorno con `render_mode=None` pueda crearse y avanzar pasos con perturbaciones automaticas sin inicializar Pygame.
- Se verifico la prioridad explicita entre viento automatico (`enable_wind=True`) y viento manual (`set_wind()`).
- Se verifico la reproducibilidad separando el RNG de Gymnasium (`reset(seed=...)`) del RNG de `WindProcess` (`wind_seed`).
- Se verifico que el patron configurado en el constructor no se muta cuando `randomize_wind_pattern=True`; el patron sorteado queda en `active_wind_pattern`.
- Se verifico que la observacion del entorno tenga dimension 3 y conserve `theta_ref`.
- Se verifico que cambiar `theta_ref` modifica la recompensa de seguimiento.
- Se verifico que `reset(options={"randomize": True})` sigue produciendo observaciones validas con `theta_ref`.
- Se verifico que el entorno con viento automatico y `theta_ref` configurable avanza correctamente en modo headless.

## Comandos de verificacion

```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check pahm_stage2 tests
.venv/bin/python -m pahm_stage2.validate_synthetic_real --config configs/stage2_config.json
.venv/bin/python -m pytest tests test -q
.venv/bin/python -m pytest tests/test_learned_pahm_env_wind.py -q
```

Resultado registrado de la suite completa:

```text
42 passed, 1 warning
```

Resultado registrado para el contrato especifico del entorno con perturbaciones:

```text
9 passed, 1 warning
```

## Recomendaciones tecnicas

- La configuracion actual usa el checkpoint final del entrenamiento del estimador de viento, derivado de `epochs = 40` y `checkpoint_dir = "checkpoints/"`, es decir `checkpoints/estimator_checkpoint_epoch_40.pth`.
- Para una entrega mas defendible, se recomienda guardar y consumir un checkpoint seleccionado por metrica de validacion, por ejemplo `best_estimator_checkpoint.pth`, usando el menor MSE de validacion o una metrica equivalente.
- Mantener tambien un checkpoint final, por ejemplo `last_estimator_checkpoint.pth`, para reproducibilidad y reanudacion de entrenamiento.
- Exponer en la configuracion una ruta explicita, por ejemplo `estimator_checkpoint_path`, en lugar de reconstruir el nombre del checkpoint solo a partir de `epochs`.
- Documentar en el informe cual checkpoint se uso para exportar `tau_w(t)` y bajo que criterio fue seleccionado.

## Responsabilidad final

El codigo entregado fue revisado por el equipo. La IA se utilizo como apoyo para acelerar implementacion, documentacion y validacion, pero las decisiones tecnicas, la ejecucion de pruebas y la aceptacion final de los cambios corresponden al Grupo 2.
