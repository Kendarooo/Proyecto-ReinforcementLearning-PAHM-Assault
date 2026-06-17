# Etapa 0 — DQN en Demon Attack

## Contexto del proyecto

Curso EL 5857 Aprendizaje Automático, I Semestre 2026, ITCR.
Profesor: Dr. Pablo Alvarado Moya. Entrega: 19 de junio, 2026.

Esta carpeta es la **Etapa 0 (individual)** del Proyecto 2. El objetivo es familiarizarse
con aprendizaje reforzado implementando un agente Double DQN que aprenda a jugar
Demon Attack (Atari) con observaciones RGB a colores.

## Estructura de archivos

```
etapa-0-DemonAttack/
├── config.json          # Todos los hiperparámetros (NFR-1, sin hardcoding)
├── train.py             # Entrenamiento headless + W&B
├── demo.py              # Demostración con render a colores (sin entrenamiento adicional)
├── dqn_agent.py         # Lógica Double DQN: selección de acción, aprendizaje, checkpoints
├── network.py           # CNN: 3 conv layers + 2 FC, entrada RGB apilada
├── replay_buffer.py     # Replay buffer uniforme
├── wrappers.py          # Preprocesamiento: resize 84x84, stack 4 frames, normalización [0,1]
├── sweep.yaml           # Sweep W&B para hiperparámetros principales
├── checkpoints/         # Pesos guardados cada N episodios
└── tests/
    └── test_agent.py    # 11 pruebas unitarias con pytest
```

## Entorno

- **Juego:** `ALE/DemonAttack-v5`
- **Observaciones:** RGB a colores `(210, 160, 3)` → resize `(84, 84, 3)` → stack 4 frames → `(4, 84, 84, 3)`
- **Acciones:** `Discrete(6)` — NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
- **Recompensa:** clipeada a {-1, 0, +1} durante entrenamiento; real durante demo

## Algoritmo

**Double DQN** con:
- Red online (aprende) + red target (se actualiza cada `target_update_freq` pasos)
- Epsilon-greedy con calendario lineal por pasos (`epsilon_decay_steps`)
- Replay buffer uniforme
- Loss: Smooth L1 (Huber), gradientes clipados a norma 10

## Entorno virtual

```bash
# Activar
source .venv/bin/activate

# Instalar dependencias (primera vez)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install ale-py "gymnasium[atari]" opencv-python wandb pytest
AutoROM --accept-license
```

## Comandos frecuentes

```bash
# Correr tests
pytest tests/test_agent.py -v

# Entrenar corrida principal (headless)
python train.py --config config.json

# Reanudar entrenamiento desde checkpoint
python train.py --config config.json --resume checkpoints/checkpoint_ep500.pth

# Demostración a colores
python demo.py --checkpoint checkpoints/checkpoint_ep500.pth --episodes 5

# Sweep de hiperparámetros
wandb sweep sweep.yaml
wandb agent <ENTITY/PROJECT/SWEEP_ID> --count 6
```

## W&B

Requiere login previo:
```bash
wandb login   # pegar API key de wandb.ai/authorize
```

El proyecto se llama `demon-attack-dqn` (configurable en `config.json` → `wandb.project`).
Registra por episodio: `train/reward_raw`, `train/reward_clipped`, `train/loss`,
`train/epsilon`, `train/exploration_rate`, `train/exploitation_rate`,
`train/buffer_size` y `train/episode_length`.

Cada `eval_every` episodios corre evaluación sin exploración (`epsilon=0`) y registra:
`eval/mean_reward`, `eval/best_reward`, `eval/std_reward` y `eval/mean_length`.

## Decisiones de diseño importantes

- `ale_py` debe importarse explícitamente en `wrappers.py` para registrar el namespace ALE en gymnasium.
- Los parámetros físicos del ODE (otras etapas) no tienen relación con este módulo.
- `clip_rewards: false` en demo para ver puntajes reales del juego.
- La red recibe los frames como `(batch, n_frames, H, W, 3)` y los reorganiza internamente a `(batch, n_frames*3, H, W)` antes de pasar por las convoluciones.

## Requisitos del PDF que cubre esta etapa

- **FR-1:** Agente DQN/Double DQN entrenado con curvas en W&B ✓
- **FR-2:** Sistema de demostración separado que carga modelo entrenado ✓
- **NFR-1:** Config centralizada en `config.json` ✓
- **NFR-2:** Telemetría W&B ✓
- **NFR-3:** Checkpoints periódicos ✓
- **NFR-4:** Semilla aleatoria configurable ✓
- **NFR-6:** Pruebas unitarias con pytest (11 tests) ✓
