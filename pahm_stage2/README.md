# pahm_stage2

Este paquete contiene la infraestructura de la Etapa 2 del proyecto: generar o consumir señales estimadas de viento `tau_w(t)`, extraer representaciones de tamaño fijo y entrenar modelos no supervisados sobre esas perturbaciones.

## Módulos

### `config.py`

Carga el archivo JSON de configuración, valida que existan las secciones principales y fija semillas para reproducibilidad en Python, NumPy y PyTorch cuando está disponible.

### `tau_dataset.py`

Carga trayectorias `.npy` con columnas `[sin_theta, cos_theta, theta_dot, u]`. Permite alternar entre datos sintéticos y reales usando `data.source` en la configuración.

### `estimator_interface.py`

Define la interfaz común para estimadores de viento. Incluye `DummyWindEstimator`, que produce ceros mientras no exista el checkpoint real de Etapa 1, y deja preparado `PAHMWindEstimator` para integrar el modelo recurrente del Grupo 1.

### `artifacts.py`

Centraliza la escritura de artefactos: guarda cada señal `tau_w(t)` como `.npy` y escribe el manifest JSON con las rutas y metadatos de las señales generadas.

### `generate_tau_w.py`

Pipeline ejecutable por CLI para generar señales `tau_w(t)` por trayectoria. Carga config, dataset y estimador; luego guarda las señales, escribe el manifest y registra métricas básicas en W&B o en un fallback local.

### `feature_extractor.py`

Convierte señales `tau_w(t)` de longitud variable en vectores de tamaño fijo. Extrae features como media, desviación estándar, máximo absoluto, asimetría, suavidad y energía.

### `unsupervised_model.py`

Implementa el modelo no supervisado basado en GMM. Entrena varios modelos, selecciona el número de componentes con BIC, predice clusters y permite guardar/cargar checkpoints.

### `train_unsupervised.py`

Entrena el modelo no supervisado productivo a partir del manifest de señales `tau_w(t)` generado por `generate_tau_w.py`. Extrae features, ajusta el GMM con selección por BIC, guarda el checkpoint configurado y registra métricas en W&B.

### `validator.py`

Contiene utilidades de validación. Para datos sintéticos con etiquetas conocidas calcula ARI y NMI; para datos reales sin etiquetas resume la distribución de clusters de forma exploratoria.

### `validate_synthetic_real.py`

Ejecuta la validación sintético-real de la representación no supervisada. Genera patrones sintéticos conocidos, entrena el GMM, reporta BIC, ARI y NMI, guarda gráficas para el PDF y omite el análisis real si todavía no existe el manifest de `tau_w(t)` entregado por Etapa 1.

### `wind_sampler.py`

Carga un checkpoint del modelo no supervisado y expone `sample()` para producir muestras como tensores de PyTorch. Este módulo sirve como consumidor inicial de la representación aprendida.

## Flujo esperado

```text
trayectorias .npy
  -> tau_dataset.py
  -> estimator_interface.py
  -> generate_tau_w.py
  -> tau_w_<id>.npy + manifest
  -> feature_extractor.py
  -> train_unsupervised.py
  -> validator.py / wind_sampler.py
```

## Entrenamiento no supervisado productivo

```bash
python -m pahm_stage2.train_unsupervised --config configs/stage2_config.json
```

Este script requiere que ya exista el manifest configurado en `outputs.manifest_path`. Si el manifest no existe, primero se debe ejecutar `generate_tau_w.py` o esperar la entrega I-1 de Etapa 1.

## Validación sintético-real

```bash
python -m pahm_stage2.validate_synthetic_real --config configs/stage2_config.json
```

El script escribe artefactos en `artifacts/stage2/validation/`:

- `synthetic_metrics.json`
- `bic_curve.png`
- `synthetic_clusters.png`
- `real_cluster_counts.json`, cuando exista el manifest real
- `real_cluster_distribution.png`, cuando exista el manifest real
