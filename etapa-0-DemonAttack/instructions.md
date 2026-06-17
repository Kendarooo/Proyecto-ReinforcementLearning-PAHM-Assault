# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

# Instrucciones — Etapa 0: DQN en Demon Attack

## ¿Qué hace esta etapa?

Entrena un agente de aprendizaje reforzado (Double DQN) para jugar el juego Demon Attack
de Atari. El agente aprende a partir de imágenes RGB del juego, sin conocimiento previo
de las reglas. Al terminar el entrenamiento, se puede cargar el modelo y verlo jugar
en tiempo real a colores.

---

## Requisitos previos

- Python 3.12
- `uv` instalado (`curl -Lsf https://astral.sh/uv/install.sh | sh`)
- Cuenta en [Weights & Biases](https://wandb.ai)
- `tmux` recomendado para entrenamientos largos:
  ```bash
  sudo apt install tmux
  ```

---

## Instalación

```bash
# Desde la carpeta etapa-0-DemonAttack/
uv venv .venv
source .venv/bin/activate

uv pip install -r ../requirements.txt

AutoROM --accept-license
```

Conectar cuenta de W&B (solo una vez):
```bash
wandb login
```

---

## Entrenar el agente

Desde la carpeta de esta etapa:

```bash
cd ~/Desktop/proyecto-2-aprendi/proyecto-2-p2-kbka/etapa-0-DemonAttack
source .venv/bin/activate
```

Verifique primero la sesión de W&B:

```bash
wandb status
```

Si no aparece una cuenta activa, inicie sesión:

```bash
wandb login
```

Entrenamiento normal:

```bash
python train.py --config config.json
```

Esto corre el entrenamiento sin ventana gráfica (headless). Puedes ver las métricas
en tiempo real en tu dashboard de W&B, dentro del proyecto `demon-attack-dqn`.

### Entrenamiento nocturno con tmux

```bash
tmux new -s demonattack-night
cd ~/Desktop/proyecto-2-aprendi/proyecto-2-p2-kbka/etapa-0-DemonAttack
source .venv/bin/activate
python train.py --config config.json
```

Para salir de `tmux` dejando el entrenamiento corriendo:

```text
Ctrl+B
D
```

Para volver a la sesión:

```bash
tmux attach -t demonattack-night
```

Para pausar/detener el entrenamiento guardando checkpoint:

```text
Ctrl+C
```

El entrenamiento registra en W&B:

| Métrica | Qué evidencia |
|---|---|
| `train/reward_raw` | Puntaje real por episodio |
| `train/reward_clipped` | Recompensa usada por DQN para estabilizar aprendizaje |
| `train/loss` | Evolución de la pérdida de TD |
| `train/epsilon` | Calendario de exploración |
| `train/exploration_rate` | Probabilidad de explorar |
| `train/exploitation_rate` | Probabilidad de explotar la política aprendida |
| `eval/mean_reward` | Desempeño de la política con `epsilon=0` |
| `eval/best_reward` | Mejor episodio durante evaluación periódica |

Para reanudar desde un checkpoint:
```bash
python train.py --config config.json --resume checkpoints/checkpoint_ep500.pth
```

Para reanudar desde el último checkpoint disponible:

```bash
ls -lh checkpoints/
python train.py --config config.json --resume checkpoints/checkpoint_epXXXX.pth
```

Ejemplo para reanudar el entrenamiento largo anterior:

```bash
cd ~/Desktop/proyecto-2-aprendi/proyecto-2-p2-kbka/etapa-0-DemonAttack
source .venv/bin/activate
ls -lh checkpoints/
python train.py --config config.json --resume checkpoints/checkpoint_ep1586.pth
```

Para reanudarlo y dejarlo corriendo en `tmux`:

```bash
tmux new -s demonattack-train
cd ~/Desktop/proyecto-2-aprendi/proyecto-2-p2-kbka/etapa-0-DemonAttack
source .venv/bin/activate
python train.py --config config.json --resume checkpoints/checkpoint_ep1586.pth
```

Para salir de la sesión sin detener el entrenamiento:

```text
Ctrl+B
D
```

Para volver a revisar la sesión:

```bash
tmux attach -t demonattack-train
```

Para usar un archivo de configuración distinto:
```bash
python train.py --config mi_config.json
```

### Sweep de hiperparámetros en W&B

Para evidenciar los hiperparámetros explorados:

```bash
wandb sweep sweep.yaml
wandb agent <ENTITY/PROJECT/SWEEP_ID> --count 6
```

El sweep explora:

| Hiperparámetro | Valores |
|---|---|
| `agent.learning_rate` | `0.00003`, `0.00006`, `0.0001`, `0.00018`, `0.00035`, `0.0006` |
| `agent.epsilon_decay_steps` | `100000`, `150000`, `250000`, `300000`, `500000`, `800000` |
| `agent.epsilon_end` | `0.01`, `0.05`, `0.1` |
| `agent.target_update_freq` | `500`, `1000`, `2500`, `5000`, `10000` |
| `agent.gamma` | `0.99`, `0.995`, `0.997` |
| `agent.batch_size` | `32`, `64` |

La métrica objetivo del sweep es `eval/mean_reward`. Cada run del sweep entrena
100 episodios, evalúa cada 20 episodios y usa 3 episodios por evaluación.

### Sweep corto de 5 perfiles

Para una exploración rápida y controlada, use `sweep_50_profiles.yaml`. Este sweep corre
5 perfiles fijos, uno por run, con 50 episodios por run:

| Perfil | Idea principal |
|---|---|
| `base_short` | Configuración base actual, pero corta |
| `fast_learning` | Mayor `learning_rate` y actualización frecuente del target |
| `more_exploration` | Exploración más sostenida y `batch_size=64` |
| `quick_exploitation` | Decaimiento rápido de epsilon y `epsilon_end=0.01` |
| `long_horizon` | `gamma` más alto para ponderar recompensas futuras |

Crear el sweep en el proyecto correcto:

```bash
wandb sweep --project demon-attack-dqn sweep_50_profiles.yaml
```

Ejecutar los 5 perfiles:

```bash
wandb agent aprendi1s26/demon-attack-dqn/SWEEP_ID --count 5
```

Al terminar, ordenar los runs en W&B por:

```text
eval/mean_reward
```

El mejor perfil de 50 episodios no necesariamente reemplaza al modelo final, pero sirve
para escoger configuraciones candidatas para un entrenamiento largo posterior.

### Sweep candidato para entrenamiento largo

Para buscar una configuración candidata con más criterio que el sweep de 5 perfiles, use
`sweep_candidate.yaml`. Este sweep usa búsqueda bayesiana sobre hiperparámetros
discretos razonables y entrena 120 episodios por run:

```bash
wandb sweep --project demon-attack-dqn sweep_candidate.yaml
wandb agent aprendi1s26/demon-attack-dqn/SWEEP_ID --count 10
```

Explora:

| Hiperparámetro | Valores |
|---|---|
| `agent.learning_rate` | `0.00006`, `0.0001`, `0.00018`, `0.00025`, `0.00035` |
| `agent.epsilon_decay_steps` | `100000`, `150000`, `250000`, `300000`, `500000` |
| `agent.epsilon_end` | `0.01`, `0.03`, `0.05`, `0.08`, `0.1` |
| `agent.target_update_freq` | `500`, `1000`, `2500`, `5000` |
| `agent.gamma` | `0.99`, `0.995`, `0.997` |
| `agent.batch_size` | `32`, `64` |

Cada run evalúa cada 20 episodios usando 5 episodios de evaluación. Los checkpoints de
sweeps se guardan en carpetas aisladas para no sobrescribir `checkpoints/best_model.pth`.

### Entrenar el mejor candidato del sweep

El archivo `config_candidate.json` contiene la mejor combinación encontrada en el sweep
candidato:

```text
learning_rate = 0.00025
gamma = 0.997
batch_size = 64
epsilon_decay_steps = 150000
epsilon_end = 0.03
target_update_freq = 5000
```

Para entrenar este candidato largo en W&B:

```bash
tmux new -s demonattack-candidate-train
cd ~/Desktop/proyecto-2-aprendi/proyecto-2-p2-kbka/etapa-0-DemonAttack
source .venv/bin/activate
python train.py --config config_candidate.json
```

Para salir dejando el entrenamiento corriendo:

```text
Ctrl+B
D
```

Para volver:

```bash
tmux attach -t demonattack-candidate-train
```

Los checkpoints del candidato se guardan en:

```text
checkpoints/candidate_lr25e-5_g997_b64_decay150k_eps003_target5000/
```

Esto evita sobrescribir el modelo principal:

```text
checkpoints/best_model.pth
```

Para probar el mejor modelo del candidato:

```bash
python demo.py \
  --checkpoint checkpoints/candidate_lr25e-5_g997_b64_decay150k_eps003_target5000/best_model.pth \
  --episodes 5
```

---

## Ver el agente jugar (demo)

El sistema de demo carga un checkpoint ya entrenado y usa `epsilon=0`. Esto significa
que el agente no explora ni aprende durante la demostración: solo ejecuta la política
aprendida por la red.

Probar el mejor modelo guardado hasta ahora:

```bash
python demo.py --episodes 5
```

Por defecto, `demo.py` carga:

```text
checkpoints/best_model.pth
```

Probar un checkpoint específico:

```bash
python demo.py --checkpoint checkpoints/checkpoint_ep500.pth --episodes 5
```

Abre una ventana con el juego a colores. El agente usa el modelo entrenado y no aprende
nada nuevo durante la demo. Al final de cada episodio imprime:

```text
Score
Time
Steps
```

Opciones:
```bash
python demo.py --checkpoint checkpoints/checkpoint_ep500.pth --episodes 10
```

---

## Correr las pruebas

```bash
pytest tests/test_agent.py -v
```

Debe mostrar todos los tests en verde. Los tests verifican:

| Test | Qué verifica |
|---|---|
| `test_network_output_shape` | La red produce exactamente 6 Q-values |
| `test_network_output_is_finite` | No hay NaN ni Inf en la salida |
| `test_replay_buffer_stores_and_samples` | El buffer entrega batches del tamaño correcto |
| `test_replay_buffer_not_ready_when_small` | El buffer no entrena antes de tener suficientes datos |
| `test_agent_action_in_valid_range` | Las acciones siempre están en [0, 5] |
| `test_agent_deterministic_when_epsilon_zero` | Con epsilon=0 el agente es determinista |
| `test_epsilon_decays_and_respects_floor` | Epsilon decae pero nunca baja de `epsilon_end` |
| `test_linear_epsilon_schedule_reaches_floor` | El calendario lineal llega al mínimo configurado |
| `test_wrappers_observation_shape` | Las observaciones tienen shape `(4, 84, 84, 3)` |
| `test_wrappers_pixel_range` | Los píxeles están en [0, 1] |
| `test_wrappers_step_consistency` | `step()` devuelve el mismo shape que `reset()` |
| `test_clip_reward_preserves_raw_reward` | La recompensa clipeada conserva el score real en `info` |
| `test_checkpoint_save_and_load` | Los checkpoints se guardan y restauran correctamente |

---

## Ajustar hiperparámetros

Todo está en `config.json`. Los más relevantes:

| Parámetro | Valor por defecto | Efecto |
|---|---|---|
| `learning_rate` | 0.0001 | Velocidad de aprendizaje |
| `epsilon_start` | 1.0 | Exploración inicial (100% aleatoria) |
| `epsilon_end` | 0.05 | Exploración mínima (5%) |
| `epsilon_schedule` | `linear` | Tipo de calendario de exploración |
| `epsilon_decay_steps` | 300000 | Pasos para bajar de `epsilon_start` a `epsilon_end` |
| `gamma` | 0.99 | Importancia de recompensas futuras |
| `batch_size` | 32 | Transiciones por paso de aprendizaje |
| `n_episodes` | 3000 | Total de episodios de entrenamiento |
| `checkpoint_every` | 25 | Guardar checkpoint cada N episodios |
| `eval_every` | 25 | Evaluar la política sin exploración cada N episodios |
| `eval_episodes` | 5 | Partidas usadas en cada evaluación |

El replay buffer almacena las observaciones internamente como `uint8` para reducir memoria
y las devuelve como `float32` normalizado al entrenar.

---

## Estructura de archivos

```
etapa-0-DemonAttack/
├── config.json       ← hiperparámetros (editar aquí, no en el código)
├── train.py          ← entrenamiento headless
├── demo.py           ← demostración a colores
├── dqn_agent.py      ← agente Double DQN
├── network.py        ← red neuronal CNN
├── replay_buffer.py  ← memoria de experiencias
├── sweep.yaml        ← exploración de hiperparámetros en W&B
├── wrappers.py       ← preprocesamiento del entorno
├── checkpoints/      ← modelos guardados automáticamente
└── tests/
    └── test_agent.py ← pruebas unitarias
```
