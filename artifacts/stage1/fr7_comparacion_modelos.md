# FR-7 — Comparación de modelos en lazo abierto (test set)

## Configuración

- Conjunto: test set retenido (split del dataloader, no visto en entrenamiento)
- Métrica: MSE de predicción de θ un paso adelante
- dt: 0.02 s | seq\_len: 5 | batch: 1

## Tabla de resultados

| Modelo | MSE | Mejora vs. ODE pura |
| --- | ---: | ---: |
| ODE pura (`pahm_fast_v2_best.pth`) | 0.00000522 | — |
| ODE + red residual (`pahm_ode_v4_best.pth`) | 0.00000522 | +0.06% |
| ODE + estimador GRU (época 40) | 0.00000497 | **+4.81%** |

## Descripción de modelos

- **ODE pura**: `PAHMFastModel` con parámetros físicos (α, β, γ) entrenados.
  Sin estimador de viento; torque externo τ_w = 0.
- **ODE + red residual**: `PAHMHybridODE` del profesor (`pahm_ode_v4_best.pth`).
  Incluye red residual 4→64→32→1 (Tanh) que corrige errores de modelado
  sistemáticos. No estima viento — captura discrepancias estáticas.
- **ODE + estimador GRU**: `PAHMFastModel` + `WindSequenceEstimator` (GRU,
  hidden=64, layers=2). Recibe ventana (sin θ, cos θ, dθ/dt, u) de longitud 5
  y produce τ̂\_w(t) inyectado en el paso RK4. Entrenado con pérdida triple
  (reconstrucción + parsimonia λ₁ + suavidad λ₂).

## Interpretación

La red residual del profesor prácticamente no mejora la predicción en el test
set (+0.06%), lo cual es esperado: fue diseñada para corregir discrepancias
sistemáticas del modelo físico, no perturbaciones estocásticas de viento.

El estimador GRU supera a ambas líneas base con una mejora de **4.81%** en
MSE, demostrando que la señal de viento τ_w(t) es recuperable a partir de la
secuencia de observaciones (sin θ, cos θ, dθ/dt, u) y que su inyección en la
ODE mejora la predicción de trayectorias.

## Reproducción

```bash
pytest test/test_verification.py::test_open_loop_baseline_comparison_fr7 -v -s
```
