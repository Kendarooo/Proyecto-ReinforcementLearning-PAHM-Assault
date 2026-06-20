# Informe de comparacion de controladores RL

## Contexto

- Controladores evaluados: naive, robust.
- Perturbaciones no vistas: gust, turbulent.
- Tolerancia de estabilizacion: 0.05.
- Ventana de estabilizacion: 50 pasos.

## Tabla resumen naive vs robust

| Metrica | Naive | Robust | Delta robust-naive | Cambio relativo | Mejor |
| --- | ---: | ---: | ---: | ---: | --- |
| MAE seguimiento | 0.026862 | 0.0231753 | -0.00368675 | -13.72% | robust |
| MSE seguimiento | 0.00544213 | 0.00456963 | -0.000872506 | -16.03% | robust |
| Tiempo de estabilizacion | 0.908889 | 0.946667 | 0.0377778 | 4.16% | naive |
| Sobreimpulso maximo | 0.374866 | 0.229189 | -0.145677 | -38.86% | robust |
| Retorno acumulado | -53.2384 | -46.9144 | 6.32397 | 11.88% | robust |

## Interpretacion de metricas

- **MAE seguimiento** (menor es mejor): mide el error absoluto promedio respecto a theta_ref; valores menores indican seguimiento mas preciso.
- **MSE seguimiento** (menor es mejor): penaliza errores grandes con mas fuerza que MAE; ayuda a detectar episodios con desviaciones severas.
- **Tiempo de estabilizacion** (menor es mejor): primer instante en que el error permanece dentro de la tolerancia durante la ventana configurada.
- **Sobreimpulso maximo** (menor es mejor): exceso maximo sobre la referencia final; valores altos sugieren respuesta agresiva o poca amortiguacion.
- **Retorno acumulado** (mayor es mejor): suma de recompensas del episodio; valores mayores indican mejor desempeno bajo la funcion objetivo entrenada.

## Resumen por perturbacion

| Controlador | Perturbacion | MAE | MSE | Settling time | Overshoot | Retorno |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive | gust | 0.0142639 | 0.00209549 | 0.508 | 0.281203 | -13.9419 |
| naive | turbulent | 0.0394602 | 0.00878878 | 1.41 | 0.468529 | -92.5349 |
| robust | gust | 0.00659809 | 0.00110764 | 0.428 | 0.142539 | -7.65863 |
| robust | turbulent | 0.0397525 | 0.00803161 | 1.595 | 0.315839 | -86.1703 |

## Conclusion

El controlador robust muestra superioridad global frente a naive en esta evaluacion: gana 4 metricas, pierde 1 y empata 0. Esta conclusion debe defenderse junto con la tabla por perturbacion para confirmar que la mejora no proviene de un unico escenario.
