---

### Archivo 2: `AI_AUDIT.md`

```markdown
# Auditoría de Uso de Herramientas de Inteligencia Artificial (CON-4)

**Módulo:** Documentación de Transparencia Académica y Validación de Grafo de Gradientes  
**Versión:** 1.1.0  
**Autor del Repositorio:** Alexandra, Bryan, Katherine, Kendall  
**Socio Tecnológico de IA:** Gemini (Sénior Software Engineer & ML Expert)  

---

## 1. Tareas Delegadas a la IA

En la consolidación del primer hito de software (Etapa 1: Infraestructura Base e Integración de Sistemas), el equipo delegó en la herramienta de IA las siguientes tareas técnicas:
1. **Modelado y Diseño de Patrones Arquitectónicos:** Estructuración de firmas de clases orientadas a objetos y diseño de interfaces con paso matricial para su correcta ejecución en aceleradores de hardware (GPU).
2. **Generación de Pruebas de Grafo en PyTorch:** Escritura automatizada de scripts en `pytest` para aislar el comportamiento de optimización del autograd de PyTorch empleando simulación de clases base (*Mocking*).
3. **Migración de Entorno Virtual y Diagnóstico de Errores:** Diseño de un plan de contingencia portable en `uv` para sustituir dependencias con rutas absolutas locales (`file:///...`) por llamadas nativas genéricas multiplataforma.

---

## 2. Estrategias de Prompting Empleadas

El equipo humano mitigar activamente el riesgo de alucinaciones de código o violaciones de las pautas universitarias mediante técnicas avanzadas de inyección de contexto:

* **Role Prompting y Enfoque Multidisciplinario:** Se obligó a la IA a adoptar la perspectiva concurrente de un *Ingeniero de Software Sénior* (exigiéndole cumplimiento estricto de SOLID y penalizaciones de `pylint`) junto con la de un *Experto en Machine Learning* (exigiéndole flujos vectorizados y operaciones tensoriales eficientes).
* **Restricciones Semánticas por Bloqueo:** Se parametrizaron directrices absolutas extraídas del enunciado oficial, tales como: *"La red no puede recibir el residuo directo $r(t)$"* y *"La física teórica provista debe permanecer congelada"*.
* **TDD Incremental Step-by-Step:** Se prohibió la generación masiva de scripts completos de entrenamiento, forzando a la IA a escribir primero la prueba unitaria que fallara estructuralmente antes de proponer el código definitivo.

---

## 3. Validación Manual de las Soluciones Propuestas

El equipo humano (**Alexandra, Bryan, Katherine y Kendall**) actuó como el filtro crítico y responsable del código integrado en el repositorio central mediante las siguientes auditorías manuales:

1. **Resolución del Error de Inicialización de Object:** Ante el fallo de firma detectado en la suite de pruebas (`TypeError: object.__init__() takes exactly one argument`), el equipo auditó manualmente el archivo `pahm_model/rk4_integrator.py` para corregir la indentación del método `__init__` con respecto a la cabecera de la clase y asegurar la inyección de dependencias.
2. **Corrección de Rutas de Importación de Módulos (sys.path):** El equipo humano identificó que el entorno de ejecución de Python arrojaba un `ModuleNotFoundError` al buscar el paquete `pahm_model`. Se validó e inyectó una solución robusta calculando de forma dinámica la raíz absoluta del proyecto (`project_root`) dentro de `sys.path.insert(0, project_root)` y forzando la invocación mediante el comando nativo de ejecución de módulos de Python (`python -m pytest`).
3. **Auditoría de Congelación de Parámetros (CON-1):** Se constató mediante revisión visual que la instrucción `with torch.set_grad_enabled(False):` envolviera exclusivamente la evaluación de `physics_model`, garantizando manualmente que el gradiente se detenga antes de tocar la física base y se propague correctamente hacia la GRU.