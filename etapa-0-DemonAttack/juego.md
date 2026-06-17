# Juego Usado: Demon Attack

## Descripción general

**Demon Attack** es un juego tipo *shoot 'em up* publicado por Imagic para Atari 2600.
En esta etapa del proyecto se usa mediante el entorno de Gymnasium:

```text
ALE/DemonAttack-v5
```

El jugador controla un cañón láser ubicado en la parte inferior de la pantalla. En la
parte superior aparecen oleadas de enemigos, llamados demonios, que se mueven
horizontalmente y atacan con proyectiles. El objetivo es sobrevivir la mayor cantidad de
tiempo posible, destruir enemigos y acumular puntos.

En el contexto de aprendizaje reforzado, el agente observa imágenes del juego y aprende
a seleccionar acciones que maximicen la recompensa acumulada. La recompensa del
entorno corresponde al cambio en el puntaje del juego.

## Objetivo del juego

El objetivo principal es:

1. Mover el cañón hacia la izquierda o derecha.
2. Disparar a los demonios.
3. Evitar los ataques enemigos.
4. Sobrevivir oleadas sucesivas.
5. Obtener el mayor puntaje posible.

El jugador comienza con bunkers de reserva. Si el cañón recibe impactos, esos bunkers
se pierden. Cuando ya no quedan bunkers, otro impacto termina la partida.

## Acciones disponibles en Gymnasium

El entorno `ALE/DemonAttack-v5` expone un espacio de acciones discreto:

```text
Discrete(6)
```

Las acciones son:

| Acción | Significado |
|---:|---|
| 0 | `NOOP`: no hacer nada |
| 1 | `FIRE`: disparar |
| 2 | `RIGHT`: moverse a la derecha |
| 3 | `LEFT`: moverse a la izquierda |
| 4 | `RIGHTFIRE`: moverse a la derecha y disparar |
| 5 | `LEFTFIRE`: moverse a la izquierda y disparar |

Para el agente DQN, cada decisión consiste en escoger una de estas seis acciones a partir
de la observación visual actual.

## Sistema de puntos

El puntaje depende del tipo de enemigo destruido y de la oleada actual. A medida que
avanza el juego, las oleadas se vuelven más difíciles y los enemigos valen más puntos.

Tabla de puntuación según el manual de Atari 2600:

| Oleadas | Demonios normales | Demonios divididos | Demonios en picada |
|---:|---:|---:|---:|
| 1-2 | 10 | - | - |
| 3-4 | 15 | - | - |
| 5-6 | 20 | 40 | 80 |
| 7-8 | 25 | 50 | 100 |
| 9-10 | 30 | 60 | 120 |
| 11-12+ | 35 | 70 | 140 |

En oleadas iniciales, los enemigos son más simples. Desde oleadas posteriores, algunos
demonios pueden dividirse en enemigos más pequeños o atacar en picada, lo que aumenta
la dificultad y el valor potencial de cada destrucción.

## Recompensa en el entrenamiento

Durante el entrenamiento se usa recompensa clipeada:

```text
reward_clipped = sign(reward_raw)
```

Esto significa:

| Puntaje real ganado | Recompensa usada para aprender |
|---:|---:|
| Mayor que 0 | +1 |
| Igual a 0 | 0 |
| Menor que 0 | -1 |

Esta práctica es común en DQN para Atari porque estabiliza el aprendizaje: el agente no
se enfoca en la escala exacta del puntaje, sino en descubrir qué acciones producen eventos
positivos.

Sin embargo, para análisis y demostración también se registra el puntaje real:

```text
train/reward_raw
```

Así, W&B muestra tanto la señal usada para entrenar como el puntaje real del juego.

## Por qué es adecuado para Etapa 0

Demon Attack es apropiado para esta etapa porque:

- Tiene un espacio de acciones discreto.
- Requiere exploración y explotación.
- Permite aprender desde imágenes.
- Produce recompensas claras por destruir enemigos.
- No está en la lista de juegos excluidos del enunciado.
- Es menos complejo que juegos de estrategia o planificación extensa.

Además, el comportamiento aprendido es fácil de observar visualmente en `demo.py`: el
agente debe moverse, disparar y evitar proyectiles.

## Interpretación para el agente DQN

Desde la perspectiva del agente:

```text
Observación: pila de 4 frames RGB redimensionados a 84x84
Acción: una de 6 acciones discretas
Recompensa: cambio en el puntaje del juego, clipeado durante entrenamiento
Episodio: una partida completa hasta perder o finalizar por límite del entorno
Objetivo: maximizar la recompensa acumulada
```

El agente no conoce las reglas explícitamente. Aprende por interacción: prueba acciones,
almacena transiciones en el replay buffer y ajusta la red neuronal para estimar valores Q.

## Fuentes

- Manual HTML de AtariAge para **Demon Attack**: https://www.atariage.com/manual_html_page.php?SoftwareLabelID=134
- Manual remasterizado y resumen de **Demon Attack**: https://atari.365indies.com/manuals/demon-attack/
