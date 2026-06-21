# AI_AUDIT.md — v1.0
## EL5857 Aprendizaje Automático | Etapa 0 — Boxing DQN
### Autor: Bryan | Fecha de inicio: 2026

---

## 1. Modelos de lenguaje utilizados

| Modelo | Versión | Plataforma |
|--------|---------|------------|
| Claude | Sonnet 4.6 | claude.ai |

---

## 2. Tareas en las que se utilizó IA

| Tarea | Descripción del uso | Validación humana realizada |
|-------|--------------------|-----------------------------|
| Diseño de `config_dqn.json` | Propuesta de estructura y valores iniciales de hiperparámetros | Revisión y ajuste manual de buffer_size y total_timesteps para restricción de VRAM |
| Estructura de directorios | Propuesta de organización modular del proyecto | Confirmada contra requisitos del enunciado (NFR-1, FR-1, FR-2) |
| `setup_env.sh` | Generación del script de instalación con uv | Revisado paso a paso; cada comando fue verificado contra la documentación de uv y AutoROM |
| `sanity_check.py` | Esqueleto del script de verificación | Revisado y entendido en su totalidad antes de ejecutar |

---

## 3. Estrategias de prompting utilizadas

- **Contexto técnico completo al inicio:** Se proporcionó el sistema operativo, GPU, restricciones de VRAM y versiones de herramientas antes de pedir cualquier código.
- **Confirmación iterativa:** Cada bloque de decisiones fue confirmado explícitamente antes de avanzar al siguiente (juego → algoritmo → hiperparámetros → estructura).
- **Solicitud de justificación:** Se pidió razonamiento explícito para cada elección técnica (ej. por qué Double DQN, por qué buffer_size=80000).

---

## 4. Lo que NO se delegó a IA

- La comprensión de los conceptos de política, función de valor y exploración-explotación (objetivo pedagógico de la Etapa 0).
- La decisión final sobre el juego seleccionado (Boxing).
- La interpretación de resultados de entrenamiento en W&B.
- El análisis del comportamiento del agente (clinch problem, convergencia).

---

## 5. Compromiso de comprensión

Cada archivo de código generado con asistencia de IA fue leído, comprendido y podría ser defendido técnicamente en la presentación al profesor. El autor es capaz de explicar:
- Por qué se usa Double DQN en lugar de DQN estándar.
- Qué hace cada parámetro en `config_dqn.json`.
- Cómo funciona el pipeline de instalación de ROMs con AutoROM.

---

*Este archivo se actualizará conforme avance el desarrollo.*