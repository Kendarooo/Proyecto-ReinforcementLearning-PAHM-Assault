# Auditoría de Uso de Herramientas de Inteligencia Artificial (CON-4)

**Proyecto 2 — PAHM · EL 5857 Aprendizaje Automático · I Semestre 2026**

Este documento consolida el uso de modelos de lenguaje durante el desarrollo del proyecto, dividido por grupo según la división de tareas (`docs/division_tareas_proy2.pdf`). Cada integrante es responsable de comprender y defender todo el código entregado, independientemente del apoyo de IA recibido durante su construcción.

---

## Grupo 1 — Etapa 1: Estimador de Perturbaciones de Viento

**Integrantes:** Alexandra, Bryan, Kendall
**Alcance:** FR-3 a FR-7, NFR-1, NFR-3, NFR-5, NFR-7, CON-1, CON-2
**Modelo de lenguaje utilizado:** Claude (Anthropic), usado como par de revisión de código y depuración iterativa a lo largo de múltiples sesiones de trabajo.

### Tareas asistidas por IA

- Planificación inicial de la división de tareas en una línea de tiempo de 3 días para el Grupo 1, a partir del análisis conjunto del enunciado (`proy2.pdf`) y la división de tareas (`division_tareas_proy2.pdf`).
- Revisión de código generado por el equipo para los módulos `sequence_estimator.py`, `custom_loss.py` y `rk4_integrator.py`, identificando errores y áreas de mejora sin reescribir directamente el código (rol de auditor, no de autor, en las primeras rondas de revisión).
- Identificación de un bug funcional en `custom_loss.py`: el término de parsimonia solo penalizaba el último paso de la ventana temporal (`tau_w_history[:, -1]`) en lugar de toda la ventana, violando la Ecuación 13 del enunciado. Corregido a `torch.mean(tau_w_history ** 2)`.
- Identificación de una discrepancia entre el nombre de la clase `RK4WindIntegrator` y el método numérico realmente implementado (Taylor de segundo orden, no Runge-Kutta de 4to orden). Resuelta renombrando la clase a `TaylorWindIntegrator` y corrigiendo los docstrings.
- Identificación de un test fraudulento en `test_verification.py` (`test_open_loop_baseline_comparison_fr7`): la primera versión usaba valores de MSE hardcodeados; la segunda versión generaba MSE sintéticos con ruido gaussiano en lugar de evaluar modelos reales. Ambas versiones fueron señaladas como inválidas para FR-7 y reemplazadas progresivamente por una evaluación genuina sobre el conjunto de prueba retenido, marcada con `@pytest.mark.skip` hasta contar con un checkpoint entrenado.
- Depuración iterativa de `train_estimator.py`, `compute_residuals.py` y `export_tau_w.py` contra el formato real de `dataloader.py` del profesor, incluyendo:
  - corrección del desempaquetado de `get_dataloaders` (devuelve 4 valores: train, val, test, dataset, no 3);
  - corrección del formato de datos real (el dataloader entrega PWM y ángulo θ como tensores separados `(batch, T, 1)`, no como un tensor combinado de 4 features);
  - corrección de la firma real de `PAHMFastModel.forward(t, u_sequences, initial_state, tau_ext)`, que exige `u_sequences` 3D, y posterior decisión de usar `physics_model.cell` directamente para permitir backpropagation de un solo paso hacia la GRU.
- Diagnóstico de un error de `CUDA out of memory` causado por acumulación del grafo computacional sobre miles de pasos de tiempo por trayectoria; resuelto implementando *Truncated Backpropagation Through Time* (backward cada `truncate_every` pasos).
- Diagnóstico de una divergencia de entrenamiento (`loss_total` subiendo de 1.5 a 28+ en 12 épocas) causada por desbalance de escala entre `loss_reconstruction` (orden de 1e-5, por comparar ángulos normalizados en un solo paso de `dt=0.02s`) y `loss_parsimony` (orden de 1-500). Se intentaron y descartaron correcciones por *gradient clipping* y ajuste de `lambda_1`/`lambda_2` sin éxito completo; la corrección efectiva fue acotar la salida de la GRU con `tanh(x) * tau_max` en `sequence_estimator.py`.
- Verificación de que el prepadding de ceros del dataloader (`np.zeros(1500)` en el código del profesor) se extiende hasta ~2289 pasos tras el `collate_fn` por agrupación de batch; se intentó y se revirtió una estrategia de *skip* de prepadding al confirmar que distorsionaba más el entrenamiento de lo que ayudaba.
- Corrección de un error de compatibilidad: tras cambiar `TaylorWindIntegrator.step` para usar `physics_model.cell` (necesario para el entrenamiento real), los tests unitarios con modelos *dummy* (`DummyModel`, `MockPerfectPhysics`, sin atributo `.cell`) dejaron de pasar; resuelto con un `hasattr(self.physics_model, 'cell')` que mantiene ambos caminos de ejecución sin afectar el comportamiento de entrenamiento real.
- Verificación de que `PAHMFastModel` cargado sin el checkpoint del profesor (`pahm_fast_v2_best.pth`) producía parámetros físicos (α, β, γ) aleatorios, lo que invalidaba la comparación de FR-7; corregido cargando explícitamente el checkpoint en el test antes de instanciar el integrador.

### Estrategias de prompting empleadas

- Revisión de código por rondas: en cada entrega de código se solicitó explícitamente "señala lo que está bien, los errores y las áreas de mejora, sin corregir directamente" — manteniendo al equipo humano como autor final de cada cambio.
- Diagnóstico guiado por evidencia: ante comportamientos anómalos (divergencia de pérdida, OOM, resultados de test sospechosos), se compartieron las salidas reales de consola y gráficas de W&B en lugar de pedir una solución genérica, permitiendo diagnósticos basados en los números observados.
- Verificación cruzada contra el código fuente del profesor: antes de asumir el formato de datos o la firma de un modelo, se solicitó y compartió el contenido real de `dataloader.py` y `pahm_fast.py` para evitar suposiciones incorrectas.
- Confirmación explícita de cambios de configuración antes de relanzar entrenamientos largos, dado el costo de tiempo de cada corrida (~60-90s por época × 40-50 épocas).

### Validación humana realizada

- Cada corrección propuesta fue aplicada manualmente por el equipo sobre su propio entorno de desarrollo (no se entregó código sin pasar por la ejecución real del equipo).
- Se ejecutó la suite completa de pruebas unitarias (`pytest test/ -v`) tras cada ronda de correcciones, confirmando 6/6 tests pasando en la versión final.
- Se verificó manualmente, mediante un script ad-hoc, que el dataloader del profesor entrega 54.3% de pasos con datos reales y 45.7% de prepadding, antes de decidir si intervenir o no esa proporción.
- Se verificó manualmente el formato exacto de los CSV del laboratorio (`pd.read_csv` + inspección de columnas) antes de confiar en la indexación del dataloader.
- Se ejecutó `test_open_loop_baseline_comparison_fr7` de forma iterativa hasta confirmar una mejora genuina y reproducible del estimador GRU sobre la ODE pura (MSE GRU = 0.00000437 vs. MSE ODE pura = 0.00000455; mejora relativa de 3.90%) sobre el conjunto de prueba retenido, sin datos hardcodeados ni sintéticos.
- Se inspeccionaron manualmente los valores de τ_w(t) exportados (`outputs/tau_w/`) antes de considerar el Hito I-1 completo, detectando y corrigiendo un primer intento con torques fuera de rango físico (±194) antes de la entrega final al Grupo 2.

### Comandos de verificación

```bash
python pahm_model/compute_residuals.py
python pahm_model/train_estimator.py
python pahm_model/export_tau_w.py
pytest test/ -v
pytest test/test_verification.py::test_open_loop_baseline_comparison_fr7 -v -s
```

Resultado final registrado de la suite completa de Etapa 1:

```text
6 passed
```

Resultado registrado de FR-7 (comparación cuantitativa en lazo abierto):

```text
MSE ODE pura   : 0.00000455
MSE ODE + GRU  : 0.00000437
Mejora relativa: 3.90%
PASSED
```

---

## Grupo 2 — Etapa 2 (Representación No Supervisada) y Etapa 3 (Control Robusto RL)

**Alcance:** FR-8 a FR-19, NFR-2, NFR-4, NFR-6, CON-3, CON-4 (sección propia)
**Modelo de lenguaje utilizado:** Codex, asistente de programación basado en GPT-5, usado dentro del entorno local de desarrollo.

### Tareas asistidas por IA

- Revisión del enunciado del proyecto y de la división de tareas para identificar los requisitos aplicables al Grupo 2.
- Diseño de la estrategia de Etapa 2 usando GMM sobre features de `tau_w(t)` y selección de complejidad mediante BIC.
- Implementación de módulos de soporte en `pahm_stage2/`: `feature_extractor.py`, `unsupervised_model.py`, `validator.py`, `wind_sampler.py`, `validate_synthetic_real.py`, `train_unsupervised.py`.
- Actualización de `configs/stage2_config.json` para centralizar parámetros de modelo, validación, rutas y W&B.
- Creación y ajuste de pruebas unitarias en `tests/` para extracción de features, GMM, validación ARI/NMI, sampler, pipeline sintético-real y entrenamiento desde manifest.
- Redacción de documentación técnica en `pahm_stage2/README.md`.
- Apoyo en interpretación de resultados de validación, incluyendo BIC, ARI y NMI.
- Revisión exhaustiva de la integración entre Etapa 1 y Etapa 2, incluyendo:
  - verificación de la ubicación esperada de `data/`;
  - identificación del checkpoint esperado por `gym_wrapper/config.json`;
  - copia del checkpoint exportado `outputs/tau_w/estimador_wind.pth` a `checkpoints/estimator_checkpoint_epoch_40.pth` para cumplir el contrato actual de pruebas y exportación;
  - creación de `instructions.md` con comandos de verificación, ejecución visual y estado previo a Etapa 3.
- Implementación inicial de Etapa 3 en el branch `feature/etapa3-entorno-perturbaciones`, sin crear una carpeta nueva `pahm_stage3` en ese momento; se extendió el entorno existente en `gym_wrapper/`.
- Ajuste de `gym_wrapper/learned_pahm_ode.py` para:
  - permitir import y ejecución headless sin depender de Pygame;
  - cargar `config.json` usando `__file__` como ancla en vez del directorio actual de ejecución;
  - integrar `WindProcess` como fuente automática opcional de perturbaciones;
  - encapsular la perturbación automática detrás de `WindSource` para poder sustituir `WindProcess` por una fuente aprendida sin reescribir `env.step()`;
  - permitir construir la fuente de viento desde `gym_wrapper/config.json`;
  - separar `configured_wind_pattern` y `active_wind_pattern`;
  - reportar `wind_active`, `wind_mag`, `wind_angle`, `wind_torque`, `wind_pattern`, `configured_wind_pattern` y `wind_automatic` en `info`;
  - mantener `set_wind()` como mecanismo manual para la demo cuando `enable_wind=False`.
- Creación de pruebas en `tests/test_learned_pahm_env_wind.py` para el contrato de entorno con perturbaciones.
- Implementación del bloque `feature/etapa3-theta-ref-reward` para FR-13:
  - `theta_ref` configurable desde `gym_wrapper/config.json`, por constructor, `set_theta_ref(value)` y `reset(options={"theta_ref": value})`;
  - observación expandida inicialmente a `[theta, theta_dot, theta_ref]` y luego enriquecida a `[theta, theta_dot, theta_ref, error_integral]`;
  - `observation_space` actualizado para incluir la referencia y la integral del error como contexto adicional;
  - recompensa de seguimiento basada en error cuadrático, integral del error, velocidad angular y esfuerzo de acción, con pesos configurables;
  - reporte de `theta_ref`, `tracking_error`, `abs_tracking_error` y `error_integral` en `info`;
  - sincronización de `theta_ref` con el slider de setpoint en la demo visual.
- Ampliación de `tests/test_learned_pahm_env_wind.py` para cubrir observación con referencia/configuración de tracking, cambio de recompensa al cambiar `theta_ref`, `reset(options={"randomize": True})` y compatibilidad entre viento automático y referencia.
- Implementación de `train_rl.py` como entrada headless independiente para entrenamiento con Stable Baselines3:
  - carga de `rl_training` desde configuración externa;
  - modos `naive` y `robust` para desactivar/activar perturbaciones;
  - orquestación de `train_all_modes(config_path)` para entrenar los modos declarados en `experiments.modes`;
  - construcción de entorno con `render_mode=None`;
  - selección configurable de algoritmo (`PPO`, `A2C`, `SAC`);
  - integración W&B desactivable para registrar hiperparámetros, modo y ruta del modelo;
  - guardado final de modelo y callbacks de checkpoint configurables;
  - pruebas smoke en `tests/test_train_rl.py`, con skip controlado si `stable-baselines3` no está instalado.
- Integración de políticas RL entrenadas en la demo visual:
  - módulo `gym_wrapper/rl_policy.py` para cargar política, predecir acciones y ejecutar un paso RL sin entrenamiento;
  - selección de modelo `naive`/`robust` desde la sección `demo` de configuración;
  - acoplamiento del modo `RL` en `gym_wrapper/test_pahm_ode_env.py`;
  - pruebas con política mock en `tests/test_rl_policy_demo.py`.
- Evaluación cuantitativa headless de controladores:
  - script `evaluate_controllers.py` para cargar modelos ya entrenados y comparar `naive` vs `robust`;
  - métricas puras de MAE/MSE de seguimiento, tiempo de estabilización, sobreimpulso y recompensa acumulada;
  - evaluación por episodios ciclando sobre `evaluation.unseen_wind_patterns`;
  - exportación de `controller_metrics.json`, `controller_metrics.csv` e `informe.md`;
  - tabla resumen naive vs robust, desglose por perturbación, interpretación de métricas y conclusión automática sobre superioridad o no del controlador robusto;
  - pruebas con políticas y entornos mock en `tests/test_evaluate_controllers.py`.
- Telemetría W&B de Etapa 3:
  - módulo `pahm_stage3/wandb_logger.py` para inicializar corridas, armar payloads de hiperparámetros/entorno, registrar métricas y artefactos;
  - integración en entrenamiento RL y evaluación cuantitativa con `mode=disabled` usable en pruebas;
  - controles `wandb.enabled`, `wandb.log_models` y `wandb.log_evaluation`;
  - pruebas con mocks en `tests/test_stage3_wandb_logger.py`.
- Integración runtime de Etapa 2 como consumidor real de Etapa 3:
  - nueva fuente `Stage2SamplerWindSource` en `gym_wrapper/wind_source.py`;
  - soporte de `wind.source="stage2_sampler"` y `wind.stage2_sampler_checkpoint`;
  - uso de `WindSampler.from_checkpoint()` para cargar el GMM no supervisado de Etapa 2;
  - conversión explícita de features (`mean`, `std`, `max_abs`, `skewness`, `smoothness`, `energy`) a un torque suave acotado por `wind.max_torque`;
  - configuración de `experiments.robust.wind_source="stage2_sampler"` para que el entrenamiento robusto consuma la representación aprendida;
  - reporte de `wind_source` en `info` para depuración y evidencia experimental.

### Estrategias de prompting empleadas

- Se proporcionó contexto del enunciado del proyecto, la rúbrica y la división de tareas entre grupos.
- Se pidió trabajar por ramas e issues concretos, especialmente sobre la validación sintético-real de Etapa 2.
- Se especificaron restricciones de diseño:
  - no rehacer la arquitectura ya existente;
  - mantener responsabilidades separadas por módulo;
  - producir evidencia útil para la rúbrica L3.2;
  - registrar métricas en W&B;
  - permitir fallback cuando todavía no exista la interfaz I-1 de Grupo 1.
- Se solicitaron explicaciones paso a paso para mantener revisión humana y evitar cambios opacos.

### Validación humana realizada

- El equipo revisó la estrategia antes de implementar la validación sintético-real.
- Se verificó manualmente que el enfoque GMM + BIC fuera coherente con los patrones esperados de viento: calma, ráfaga, sesgo sostenido y turbulencia.
- Se inspeccionaron los resultados generados en `artifacts/stage2/validation/synthetic_metrics.json`.
- Se confirmó que el resultado sintético recupera cuatro patrones con `ARI = 1.0`, `NMI = 1.0`, `n_components_selected = 4`.
- Se revisó que el script no falle cuando todavía no existe el manifest real de `tau_w(t)`.
- Se verificó que `train_unsupervised.py` falle de forma explícita cuando no existe el manifest y guarde un checkpoint cuando recibe un manifest válido.
- Se ejecutó la suite de pruebas unitarias con `pytest`.
- Se ejecutó análisis estático con `ruff`.
- Se agregó la carpeta `data/` en la raíz del repositorio y se confirmó que los tests de integración ya pueden leer los CSV reales.
- Se verificó que `outputs/tau_w/estimador_wind.pth` contiene las llaves esperadas: `epoch`, `loss`, `model_state_dict`, `optimizer_state_dict`.
- Se ejecutó la suite completa `tests/` + `test/` después de colocar el checkpoint esperado por la configuración actual.
- Se verificó que el entorno con `render_mode=None` pueda crearse y avanzar pasos con perturbaciones automáticas sin inicializar Pygame.
- Se verificó la prioridad explícita entre viento automático (`enable_wind=True`) y viento manual (`set_wind()`).
- Se verificó la reproducibilidad separando el RNG de Gymnasium (`reset(seed=...)`) del RNG de `WindProcess` (`wind_seed`).
- Se verificó que el patrón configurado en el constructor no se muta cuando `randomize_wind_pattern=True`; el patrón sorteado queda en `active_wind_pattern`.
- Se verificó que la observación del entorno conserve `theta_ref` y el contexto adicional de integral del error.
- Se verificó que cambiar `theta_ref` modifica la recompensa de seguimiento.
- Se verificó que `reset(options={"randomize": True})` sigue produciendo observaciones válidas con `theta_ref`.
- Se verificó que el entorno con viento automático y `theta_ref` configurable avanza correctamente en modo headless.
- Se verificó que `stage2_sampler` puede construirse desde configuración, cargar un checkpoint GMM y producir perturbaciones finitas.
- Se verificó que `LearnedPAHMODE` puede ejecutar `reset()` y `step()` usando `stage2_sampler` como fuente automática de viento.
- Se verificó que `train_rl.py` propaga `experiments.robust.wind_source` al entorno de entrenamiento.

### Comandos de verificación

```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check pahm_stage2 tests
.venv/bin/python -m pahm_stage2.validate_synthetic_real --config configs/stage2_config.json
.venv/bin/python -m pytest tests test -q
.venv/bin/python -m pytest tests/test_learned_pahm_env_wind.py -q
```

Resultado registrado de la suite completa:

```text
93 passed, 1 warning
```

Resultado registrado para el contrato específico del entorno con perturbaciones:

```text
22 passed, 1 warning
```

### Cierre final registrado de Etapa 3

#### Corrección de calidad estática (`ruff`)

- Se revisaron y corrigieron bloqueos de NFR-5 asociados a estilo y errores estáticos:
  - imports no usados;
  - `if` compactos en una sola línea;
  - `except` desnudos;
  - f-string sin placeholders;
  - detalles menores detectados durante los refactors de evaluación y fuentes de viento.
- Los archivos tocados incluyeron principalmente `gym_wrapper/pahm_ui.py`, `gym_wrapper/test_pahm_env.py`, `pahm_model/pahm_ode.py`, `train_rl.py`, `evaluate_controllers.py`, `gym_wrapper/wind_source.py` y pruebas asociadas.
- Verificación registrada:

```bash
.venv/bin/ruff check train_rl.py evaluate_controllers.py gym_wrapper pahm_stage3 pahm_stage2 tests
```

Resultado:

```text
All checks passed!
```

#### Entrenamiento con Stable Baselines3

- Se implementó y documentó `train_rl.py` como punto de entrada headless para entrenamiento con Stable Baselines3.
- La configuración RL quedó centralizada en `gym_wrapper/config.json` y `configs/stage3_config.json`.
- Se agregó `seed = 42` a la configuración de RL y se propaga a:
  - construcción del agente SB3 (`seed=...`);
  - `env.reset(seed=..., options={"randomize": True})`.
- El modo `naive` desactiva viento y el modo `robust` activa perturbaciones con `wind_source="stage2_sampler"`, conectando Etapa 2 con Etapa 3.
- Comandos preparados para la corrida final:

```bash
.venv/bin/python train_rl.py --config gym_wrapper/config.json --mode naive
.venv/bin/python train_rl.py --config gym_wrapper/config.json --mode robust
.venv/bin/python train_rl.py --config gym_wrapper/config.json --mode all
```

- Artefactos esperados:
  - `artifacts/stage3/models/pahm_ppo_naive.zip`;
  - `artifacts/stage3/models/pahm_ppo_robust.zip`;
  - checkpoints periódicos según `checkpoint_freq`;
  - logs TensorBoard/W&B según configuración.

#### Evaluación final de controladores

- Se fortaleció `evaluate_controllers.py` para producir evidencia defendible de comparación entre `naive` y `robust`.
- La evaluación:
  - corre en modo headless;
  - carga modelos entrenados sin reentrenar;
  - cicla episodios sobre `evaluation.unseen_wind_patterns`;
  - reporta MAE, MSE, tiempo de estabilización, sobreimpulso y retorno acumulado;
  - separa resumen global y resumen por perturbación;
  - genera conclusión explícita sobre si `robust` supera o no a `naive`.
- Comando de evaluación final:

```bash
.venv/bin/python evaluate_controllers.py --config configs/stage3_config.json
```

- Artefactos esperados:
  - `artifacts/stage3/evaluation/controller_metrics.json`;
  - `artifacts/stage3/evaluation/controller_metrics.csv`;
  - `artifacts/stage3/evaluation/informe.md`.

#### Weights & Biases

- Se centralizó la telemetría de Etapa 3 en `pahm_stage3/wandb_logger.py`.
- Se integró W&B en:
  - entrenamiento RL (`train_rl.py`);
  - evaluación de controladores (`evaluate_controllers.py`).
- La configuración permite activar o desactivar W&B sin cambiar código:
  - `wandb.enabled`;
  - `wandb.mode`;
  - `wandb.log_models`;
  - `wandb.log_evaluation`.
- En pruebas se usaron mocks y `mode=disabled` para verificar el contrato sin depender de conectividad externa.
- Para entrega/presentación, el equipo debe confirmar manualmente en W&B que las corridas finales contienen:
  - hiperparámetros de entrenamiento;
  - modo (`naive` o `robust`);
  - fuente de viento;
  - métricas de entrenamiento;
  - artefactos de modelos;
  - artefactos de evaluación JSON/CSV/Markdown.

#### Validación humana final

- El equipo humano debe realizar la aceptación final de los resultados, no solo de la ejecución de pruebas.
- Lista mínima de validación final:
  - confirmar que `ruff` pasa sin errores;
  - confirmar que la suite completa pasa;
  - confirmar que los modelos `naive` y `robust` fueron entrenados con la configuración final;
  - confirmar que los checkpoints/modelos usados en evaluación corresponden a esas corridas finales;
  - revisar `artifacts/stage3/evaluation/informe.md` y verificar que la conclusión automática coincide con los números;
  - revisar que W&B contenga las corridas y artefactos esperados;
  - completar la tabla NFR-7 de `informe.md` con tiempos reales y hardware usado;
  - documentar en el informe del curso si el controlador robusto fue superior, equivalente o inferior al naive, usando la tabla y las métricas exportadas.

#### NFR-7: tiempos y hardware

- Se agregó `informe.md` como base estructurada del informe final.
- La sección NFR-7 incluye campos para registrar:
  - entrenamiento del estimador;
  - inferencia del estimador / exportación de `tau_w(t)`;
  - entrenamiento no supervisado Stage 2;
  - entrenamiento RL `naive`;
  - entrenamiento RL `robust`;
  - evaluación de controladores;
  - hardware usado.
- Los campos de hardware quedan intencionalmente en blanco para que sean completados por la persona que ejecutó las corridas finales.
- No se registran tiempos inventados en esta auditoría; los valores finales deben provenir de ejecuciones reales, idealmente medidas con `/usr/bin/time -v` o logs equivalentes de W&B/TensorBoard.

### Recomendaciones técnicas de Grupo 2 hacia el repositorio

- La configuración actual usa el checkpoint final del entrenamiento del estimador de viento, derivado de `epochs = 40` y `checkpoint_dir = "checkpoints/"`, es decir `checkpoints/estimator_checkpoint_epoch_40.pth`.
- Para una entrega más defendible, se recomienda guardar y consumir un checkpoint seleccionado por métrica de validación, por ejemplo `best_estimator_checkpoint.pth`, usando el menor MSE de validación o una métrica equivalente.
- Mantener también un checkpoint final, por ejemplo `last_estimator_checkpoint.pth`, para reproducibilidad y reanudación de entrenamiento.
- Exponer en la configuración una ruta explícita, por ejemplo `estimator_checkpoint_path`, en lugar de reconstruir el nombre del checkpoint solo a partir de `epochs`.
- Documentar en el informe cuál checkpoint se usó para exportar `tau_w(t)` y bajo qué criterio fue seleccionado.

---

## Correcciones post-evaluación LLM — Grupo 1 y Proyecto

**Fecha:** 2026-06-21
**Evaluación de referencia:** `evaluacion_BKA_LLM_parte_grupal.pdf`
**Modelo de lenguaje utilizado:** Claude Sonnet 4.6 (Anthropic), sesión interactiva de corrección guiada.

Tras recibir la retroalimentación del evaluador LLM, el equipo realizó una revisión estricta del proyecto e implementó las siguientes correcciones:

### C2 — Bug de gradientes en `train_estimator.py` (L2.3)

**Problema identificado:** La asignación en línea `tau_w_history[:, -1] = tau_w_current.squeeze(-1).detach()` desconectaba el gradiente del paso actual de la historia de torques. Esto impedía que los términos de parsimonia (λ₁) y suavidad (λ₂) de la función de pérdida triple influyeran en la optimización de la GRU. El bug no estaba solo en el `.detach()` explícito, sino en la asignación in-place sobre un tensor pre-inicializado con `torch.zeros`, que rompe el grafo computacional de PyTorch aunque no se use `.detach()`.

**Corrección:** Se reemplazó la estrategia de tensor pre-inicializado + asignación in-place por una lista de pasos `history_steps` construida con `torch.no_grad()` para los pasos históricos (TBPTT) y sin `no_grad` para el paso actual, luego ensamblada con `torch.stack(history_steps, dim=1)`. Esto preserva el gradiente del paso actual y permite que λ₁ y λ₂ afecten la optimización.

**Verificación:** Script ad-hoc confirmó `grad is None: False` con `grad norm: 0.1389` tras el fix. Commit aplicado en `feat/etapa3-reentrenamiento`.

### C3 — Test NFR-6c con GRU real (L5.3)

**Problema identificado:** El test `test_wind_estimator_output_near_zero_with_clean_ode_data_nfr6c` usaba un estimador dummy en lugar del modelo real entrenado, lo que no verificaba el cumplimiento de CON-2.

**Corrección:** El test ahora carga `WindSequenceEstimator` desde `checkpoints/estimator_checkpoint_epoch_40.pth` y le pasa 50 pasos de trayectoria ODE pura (τ_w = 0). Se marcó con `@pytest.mark.xfail(strict=False)` porque el modelo entrenado con el bug C2 satura a ~1.95 en datos limpios — limitación honestamente documentada en el motivo del xfail. El test verifica la GRU real tal como exige el espíritu de NFR-6c; pasará cuando se reentrenar con el fix de C2 aplicado.

### C1 — Tercera columna en FR-7 (L2.5)

**Problema identificado:** `test_open_loop_baseline_comparison_fr7` solo comparaba dos configuraciones (ODE pura y ODE + GRU). Faltaba la línea base de ODE + red residual del profesor (`pahm_ode_v4_best.pth`).

**Corrección:** Se añadió `_hybrid_ode_step()`, un integrador RK4 de un solo paso que usa los parámetros físicos (α, β, γ) y la `residual_net` (4→64→32→1, Tanh) del checkpoint `PAHMHybridODE`. El test ahora imprime y compara tres columnas. Se creó `artifacts/stage1/fr7_comparacion_modelos.md` como artefacto persistente de referencia para el PDF.

**Resultados registrados:**

```text
MSE ODE pura          : 0.00000522
MSE ODE + red residual: 0.00000522  (+0.06%)
MSE ODE + GRU         : 0.00000497  (+4.81%)
```

La red residual no mejora la predicción en el test set (diseñada para corregir discrepancias sistemáticas del modelo, no viento estocástico). La GRU gana a ambas líneas base.

### C4 — Configuración de calidad de código (L5.4)

**Problema identificado:** No existía `pyproject.toml` en la raíz del repositorio; pytest no tenía configuración formal de rutas; pylint carecía de configuración para el entorno del proyecto.

**Corrección:** Se creó `pyproject.toml` con:
- `[tool.pytest.ini_options]`: `testpaths = ["test", "tests"]`
- `[tool.ruff]`: reglas E/F/W/I, longitud 88, target py312
- `[tool.pylint."messages control"]`: reglas deshabilitadas justificadas: `E0401` (falso positivo — paquetes en `.venv` no visibles al pylint del sistema), `R0801` (duplicación en código del profesor), `C0413`/`C0411` (patrón obligatorio de `sys.path` previo a imports locales), `R0402` (estilo menor)

**Resultado:** Pylint **8.18/10** sobre los módulos principales. Cumple NFR-5.

### Resultado final de la suite tras correcciones

```bash
python -m pytest -v
```

```text
9 passed, 1 xfailed
```

El `xfailed` corresponde a NFR-6c — documentado y esperado hasta reentrenar el estimador con el fix C2.

---

## Responsabilidad final

El código entregado en ambas etapas fue revisado por los respectivos equipos humanos. La IA se utilizó como apoyo para acelerar el diagnóstico de errores, la implementación y la documentación, pero las decisiones técnicas, la ejecución de pruebas y la aceptación final de los cambios corresponden a cada grupo sobre su propio alcance. Cada integrante es responsable de comprender y defender todo el código entregado, conforme a CON-4.
