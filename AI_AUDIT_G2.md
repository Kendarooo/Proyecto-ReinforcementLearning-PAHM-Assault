# AI Audit - Grupo 2

## Alcance

Este documento registra el uso de modelos de lenguaje durante el trabajo del Grupo 2 en la Etapa 2 del proyecto: representacion no supervisada de perturbaciones y validacion sintetico-real.

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

## Comandos de verificacion

```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check pahm_stage2 tests
.venv/bin/python -m pahm_stage2.validate_synthetic_real --config configs/stage2_config.json
```

## Responsabilidad final

El codigo entregado fue revisado por el equipo. La IA se utilizo como apoyo para acelerar implementacion, documentacion y validacion, pero las decisiones tecnicas, la ejecucion de pruebas y la aceptacion final de los cambios corresponden al Grupo 2.
