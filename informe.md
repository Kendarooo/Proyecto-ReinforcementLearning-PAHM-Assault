# Informe del Proyecto 2 - PAHM

Este documento sirve como base guia para el informe final. Las tablas estan
estructuradas para completar resultados medidos, hardware y evidencia de
ejecucion sin mezclar datos preliminares con resultados finales.

## 1. Resumen Ejecutivo

Completar con una sintesis breve del objetivo del proyecto, las etapas
implementadas y la conclusion principal sobre el controlador robusto.

## 2. Hardware y Entorno de Ejecucion

Completar esta seccion con las caracteristicas del equipo usado para las
corridas finales.

| Campo | Valor |
| --- | --- |
| Responsable de las corridas |  |
| Fecha de ejecucion |  |
| Sistema operativo |  |
| CPU |  |
| RAM |  |
| GPU |  |
| VRAM |  |
| Version de Python |  |
| Version de PyTorch |  |
| Version de Stable Baselines3 |  |
| Version de CUDA/cuDNN, si aplica |  |
| Entorno virtual o gestor de dependencias |  |

## 3. NFR-7 - Tiempos de Ejecucion

Registrar los tiempos finales usando la misma maquina declarada en la seccion
anterior. Si se repite una corrida, reportar media y desviacion estandar o
indicar explicitamente que se hizo una sola medicion.

| Proceso | Comando o script | Configuracion relevante | Tiempo total | Unidad | Artefacto de salida | Observaciones |
| --- | --- | --- | ---: | --- | --- | --- |
| Entrenamiento estimador | `python pahm_model/train_estimator.py` | `epochs=40`, `checkpoint_interval_epochs=5` |  |  | `checkpoints/estimator_checkpoint_epoch_40.pth` |  |
| Inferencia estimador / exportacion tau_w | `python pahm_model/export_tau_w.py` | checkpoint usado:  |  |  | `outputs/tau_w/` |  |
| Entrenamiento Stage 2 | `.venv/bin/python -m pahm_stage2.train_unsupervised --config configs/stage2_config.json` | GMM + BIC, features configuradas |  |  | `artifacts/stage2/gmm_wind_model.pkl` |  |
| Entrenamiento RL naive | `.venv/bin/python train_rl.py --config gym_wrapper/config.json --mode naive` | `wind_enabled=false`, `seed=42` |  |  | `artifacts/stage3/models/pahm_ppo_naive.zip` |  |
| Entrenamiento RL robust | `.venv/bin/python train_rl.py --config gym_wrapper/config.json --mode robust` | `wind_source=stage2_sampler`, `seed=42` |  |  | `artifacts/stage3/models/pahm_ppo_robust.zip` |  |
| Evaluacion controladores | `.venv/bin/python evaluate_controllers.py --config configs/stage3_config.json` | patrones no vistos: `gust`, `turbulent` |  |  | `artifacts/stage3/evaluation/` |  |

### 3.1 Comandos sugeridos para medir

Usar `/usr/bin/time -v` cuando se quiera capturar tiempo de pared y memoria
maxima residente:

```bash
/usr/bin/time -v .venv/bin/python train_rl.py --config gym_wrapper/config.json --mode naive
/usr/bin/time -v .venv/bin/python train_rl.py --config gym_wrapper/config.json --mode robust
/usr/bin/time -v .venv/bin/python evaluate_controllers.py --config configs/stage3_config.json
```

Para scripts ejecutados con `python` del sistema en vez del entorno virtual,
mantener consistencia y registrar cual interprete se uso.

## 4. Etapa 1 - Estimador de Perturbaciones

Completar con:

- arquitectura del estimador;
- checkpoint usado;
- metrica de comparacion en lazo abierto;
- tiempo de entrenamiento;
- tiempo de inferencia/exportacion de `tau_w(t)`.

## 5. Etapa 2 - Representacion No Supervisada

Completar con:

- features extraidas de `tau_w(t)`;
- criterio BIC para seleccion de componentes;
- metricas ARI/NMI;
- checkpoint GMM generado;
- tiempo de entrenamiento.

## 6. Etapa 3 - Control Robusto RL

Completar con:

- configuracion de `theta_ref`;
- observacion usada: `[theta, theta_dot, theta_ref, error_integral]`;
- recompensa de seguimiento;
- diferencia entre entrenamiento `naive` y `robust`;
- uso de `stage2_sampler` como fuente de perturbaciones robustas;
- tiempos de entrenamiento RL.

## 7. Comparacion de Controladores

Usar los resultados generados por:

```bash
.venv/bin/python evaluate_controllers.py --config configs/stage3_config.json
```

Artefactos esperados:

- `artifacts/stage3/evaluation/controller_metrics.json`
- `artifacts/stage3/evaluation/controller_metrics.csv`
- `artifacts/stage3/evaluation/informe.md`

Copiar aqui la tabla resumen naive vs robust y la conclusion generada, revisada
por el equipo humano.

## 8. Weights & Biases

Completar con enlaces o identificadores de corridas finales:

| Corrida | W&B run/link | Artefactos asociados | Observaciones |
| --- | --- | --- | --- |
| Estimador Etapa 1 |  |  |  |
| Stage 2 |  |  |  |
| RL naive |  |  |  |
| RL robust |  |  |  |
| Evaluacion controladores |  |  |  |

## 9. Validacion Final

Checklist antes de entregar:

- [ ] `ruff` pasa sin errores.
- [ ] La suite completa de pruebas pasa.
- [ ] Los tiempos de NFR-7 estan completos.
- [ ] El hardware usado esta documentado.
- [ ] Los modelos evaluados corresponden a las corridas finales.
- [ ] El informe de evaluacion de controladores fue revisado.
- [ ] Los enlaces o artefactos W&B finales estan registrados.
- [ ] La conclusion sobre el controlador robusto esta respaldada por metricas.
