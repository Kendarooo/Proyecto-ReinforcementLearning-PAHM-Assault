# _Autores_: Alexandra Alfaro Elizondo, Kendall Madrigal Campos /Codex

[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24080422&assignment_repo_type=AssignmentRepo)
# PAHM (Péndulo Amortiguado con Hélice a Motor) — Código Base
## EL5857 Aprendizaje Automático
### Escuela de Ingeniería Electrónica, Tecnológico de Costa Rica (TEC)
### I Semestre, 2026

Este repositorio contiene la línea base de software e infraestructura de simulación para el **Proyecto 2**. El objetivo del proyecto es modelar las perturbaciones de viento latentes no observadas mediante celdas recurrentes (aprendizaje de secuencias) y diseñar una política de control robusta utilizando **Aprendizaje por Refuerzo (RL)** sobre un entorno Gymnasium acoplado a un modelo físico híbrido (Caja Gris).

---

## 1. Estructura del Repositorio

El código está organizado siguiendo principios de diseño limpio, separación de responsabilidades (SRP) y el patrón Composite para la interfaz gráfica.


```text
├── gym_wrapper/               # Entorno de Simulación e Interfaz Gráfica
│   ├── config.json            # Parámetros visuales, geometría y colores (SSOT Layout)
│   ├── pid_config.json        # Ganancias calculadas Kp, Ki, Kd de la planta linealizada
│   ├── learned_pahm_ode.py    # Entorno Gymnasium de la planta (Neural ODE / Fast RNN)
│   ├── pahm_ui.py             # Módulo de UI en Pygame (VBox, HBox, Controles Polares)
│   ├── wind_process.py        # Generador exógeno estocástico de patrones de viento
│   ├── pid.py                 # Implementación del controlador PID con compensación cuadrática
│   └── test_pahm_ode_env.py   # Lazo principal interactivo y demostración visual
│
└── pahm_model/                # Modelos Dinámicos y Scripts de Entrenamiento
    ├── dataloader.py          # Utilidad para carga y segmentación de datos reales en RAM
    ├── pahm_fast.py           # Modelo de célula recurrente física (Physics RNN con RK4)
    ├── pahm_ode.py            # Modelo híbrido Neural ODE (vía torchdiffeq)
    ├── utils.py               # Solvers numéricos nativos e interpolación por Splines Cúbicos
    ├── train_ode.py           # Script de entrenamiento en dos fases (Física -> Híbrido)
    └── train_pid.py           # Sintonización analítica de ganancias por asignación de polos

```

---

## 2. Requisitos e Instalación

El entorno requiere Python 3.12+ y las dependencias estándar de aprendizaje profundo y simulación física. En Ubuntu/Python 3.12 evita usar `pip` del sistema directamente, porque puede fallar con `externally-managed-environment` (PEP 668).

Instalación recomendada desde la raíz del repositorio:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

```

Para usar los entornos Atari de la Etapa 0, acepta las ROMs una vez después de instalar:

```bash
AutoROM --accept-license
```

*Nota: si necesitas instalar PyTorch para CPU explícitamente, puedes seguir la variante recomendada por PyTorch para tu sistema y luego ejecutar `uv pip install -r requirements.txt` para el resto de dependencias.*

---


## 3. Guía de Uso

### A. Ejecución del Entorno Interactivo (GUI)

Para levantar el simulador visual, inspeccionar la dinámica del péndulo, graficar en tiempo real mediante el osciloscopio e inyectar ráfagas de viento interactivas:

```bash
cd gym_wrapper
python test_pahm_ode_env.py  --reset_angle 720 --max_steps 10000000 --model ../pahm_model/pahm_fast_v2_best.pth --pid pid_config.json 
```

**Controles de la Interfaz:**

* **Panel de Modos:** Permite alternar entre entradas manuales, señales senoidales, control analítico PID, o el agente RL.
* **Perturbación de Viento:** Al activar el interruptor de viento, es posible seleccionar patrones preconfigurados (`Calm`, `Gust`, `Sust`, `Turb`) o modular vectorialmente la magnitud y dirección arrastrando el mouse sobre el control polar circular.

### B. Entrenamiento de la Dinámica Híbrida (Neural ODE / Fast RNN)

Para entrenar el modelo de caja gris a partir de capturas de laboratorio usando el enfoque de sliding windows sobre memoria RAM:

```bash
cd pahm_model
python train_ode.py --model_type fast --epochs 100 --warmup_epochs 20 --data_dir ../data/

```

*Este script opera en dos fases autónomas: congela la red neuronal residual durante los `warmup_epochs` para forzar la convergencia de los parámetros físicos reales ($\alpha, \beta, \gamma$), y posteriormente libera los pesos neuronales para ajustar la dinámica fina no modelada.*

### C. Sintonización Analítica del PID

Si requieres recalcular las ganancias del lazo de control clásico a partir de la física aprendida por un punto de control `.pth`:

```bash
python train_pid.py --model_path ../pahm_model/pahm_ode_best.pth --output gym_wrapper/pid_config.json --ts 2.0 --pole_ratio 10.0

```

---

## 4. Invariantes de Diseño y Contratos Técnicos

Al desarrollar extensiones sobre este código base, se deben respetar las siguientes decisiones arquitectónicas acordadas:

1. **Espacio de Observaciones 2D:** El entorno Gymnasium está configurado nativamente para entregar un vector continuo de estado $[ \theta, \dot{\theta} ]$ de tamaño 2. Esto garantiza compatibilidad directa con librerías de RL estándar como *Stable Baselines3*.
2. **Sin Envoltura de Ángulo (*No Wrapping*):** El ángulo en el estado interno se mantiene de forma acumulativa y continua. Envolver el ángulo en el intervalo $[-\pi, \pi]$ dentro del estado de la planta oculta los sobregiros y desincroniza los integradores; cualquier transformación visual o matemática debe realizarse externamente de forma aislada.
3. **Robustez mediante Domain Randomization:** Para mitigar el sobreajuste (*overfitting*) a la trayectoria inicial desde el reposo, el método `reset` acepta el argumento `options={"randomize": True}`. Esto inicializa el episodio en un punto cinemático aleatorio pero seguro.
4. **Cinemática del Viento y Estelas:** La actualización visual de las partículas de viento calcula su origen y destino basándose en el vector de desplazamiento real por cuadro ($dx, dy$), aplicando un factor de amplificación visual estático para que el flujo sea perfectamente visible incluso ante brisas de baja magnitud.

---

## 5. Integración del Agente de Aprendizaje por Refuerzo (RL)

Para completar la Etapa 3 del proyecto, se ha dispuesto una sección explícita dentro del lazo de ejecución de `test_pahm_ode_env.py`:

```python
elif controller.current_mode == "RL":
    # -----------------------------------------------------------
    # POR HACER: Inserte aquí el acoplamiento de su modelo RL.
    # Deben pasar la observación (obs) a su agente entrenado
    # y asignar la acción calculada a la variable `rl_action`.
    # -----------------------------------------------------------
    rl_action = agente.predict(obs)
```

El script de entrenamiento para RL (`train_rl.py`) deberá desarrollarse de forma independiente y configurarse para operar en modo *headless* (sin renderizado gráfico) para maximizar la tasa de cuadros por segundo y acelerar la convergencia del entrenamiento por gradiente de política.
