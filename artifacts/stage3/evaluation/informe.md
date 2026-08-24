# Informe de comparacion de controladores RL

## Contexto

- Controladores evaluados: naive, robust.
- Perturbaciones no vistas: gust, turbulent.
- Tolerancia de estabilizacion: 0.05.
- Ventana de estabilizacion: 50 pasos.

## Tabla resumen naive vs robust

| Metrica | Naive | Robust | Delta robust-naive | Cambio relativo | Mejor |
| --- | ---: | ---: | ---: | ---: | --- |
| MAE seguimiento | 0.0210443 | 0.0120313 | -0.00901302 | -42.83% | robust |
| MSE seguimiento | 0.00469722 | 0.00233672 | -0.0023605 | -50.25% | robust |
| Tiempo de estabilizacion | 1.344 | 0.688 | -0.656 | -48.81% | robust |
| Sobreimpulso maximo | 0.397407 | 0.190058 | -0.207348 | -52.18% | robust |
| Retorno acumulado | -29.8621 | -14.5259 | 15.3361 | 51.36% | robust |

## Interpretacion de metricas

- **MAE seguimiento** (menor es mejor): mide el error absoluto promedio respecto a theta_ref; valores menores indican seguimiento mas preciso.
- **MSE seguimiento** (menor es mejor): penaliza errores grandes con mas fuerza que MAE; ayuda a detectar episodios con desviaciones severas.
- **Tiempo de estabilizacion** (menor es mejor): primer instante en que el error permanece dentro de la tolerancia durante la ventana configurada.
- **Sobreimpulso maximo** (menor es mejor): exceso maximo sobre la referencia final; valores altos sugieren respuesta agresiva o poca amortiguacion.
- **Retorno acumulado** (mayor es mejor): suma de recompensas del episodio; valores mayores indican mejor desempeno bajo la funcion objetivo entrenada.

## Resumen por perturbacion

| Controlador | Perturbacion | MAE | MSE | Settling time | Overshoot | Retorno |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| naive | gust | 0.0170301 | 0.00321298 | 0.936 | 0.349321 | -20.654 |
| naive | turbulent | 0.0250585 | 0.00618147 | 1.752 | 0.445492 | -39.0701 |
| robust | gust | 0.00978158 | 0.00165569 | 0.4 | 0.122792 | -12.3256 |
| robust | turbulent | 0.014281 | 0.00301776 | 0.976 | 0.257325 | -16.7263 |

## Conclusion

El controlador robust muestra superioridad global frente a naive en esta evaluacion: gana 5 metricas, pierde 0 y empata 0. Esta conclusion debe defenderse junto con la tabla por perturbacion para confirmar que la mejora no proviene de un unico escenario.
