# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2
# Versión: 1.1.0

# Contiene contribuciones de Claude y Gemini

from gymnasium.wrappers import TimeLimit
import numpy as np
import pygame
import sys
import signal
import argparse

try:
    from learned_pahm import LearnedPAHM
except ImportError:
    try:
        from learned_pahm_ode import LearnedPAHMODE as LearnedPAHM
    except ImportError:
        from gym_wrapper.learned_pahm_ode import LearnedPAHMODE as LearnedPAHM

try:
    from pahm_ui import PAHMController, Oscilloscope, CONFIG
except ImportError:
    from gym_wrapper.pahm_ui import PAHMController, Oscilloscope, CONFIG


def _build_env(render_mode: str, model_name: str, reset_angle: float):
    try:
        return LearnedPAHM(
            render_mode=render_mode,
            model_name=model_name,
            reset_angle=reset_angle,
        )
    except TypeError:
        return LearnedPAHM(
            render_mode=render_mode,
            model_path=model_name,
            reset_angle_deg=reset_angle,
            max_wind_torque=CONFIG["wind_patterns"]["max_wind_torque"],
        )

def main():
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Test del entorno PAHM con controles interactivos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Usar modelo por defecto:
    python test_pahm_env.py

  Especificar modelo personalizado:
    python test_pahm_env.py --model mi_modelo_entrenado

  Configurar ángulo de reset:
    python test_pahm_env.py --reset_angle 15.5

  Configurar máximo de pasos:
    python test_pahm_env.py --max_steps 5000

  Combinando parámetros:
    python test_pahm_env.py --model mi_modelo --reset_angle 170 --max_steps 1000
        """
    )
    
    parser.add_argument('--model', type=str, default='p2_base',
                        help='Nombre del modelo PAHM a cargar (default: p2_base)')
    
    parser.add_argument('--reset_angle', type=float, default=120,
                        help='Ángulo inicial en grados para reset del entorno (default: 120)')
    
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Número máximo de pasos antes de truncar episodio (default: 2000)')
    
    args = parser.parse_args()
    
    # Mostrar configuración
    print("🤖 Configuración:")
    print(f"   - Modelo: {args.model}")
    print(f"   - Ángulo de reset: {args.reset_angle}°")
    print(f"   - Máximo pasos por episodio: {args.max_steps}")
    print()
    
    pygame.init()
    
    # Variable para manejo de cierre elegante
    running = True
    
    def signal_handler(sig, frame):
        """Maneja Ctrl-C de forma elegante"""
        nonlocal running
        print("\n🛑 Ctrl-C detectado. Cerrando de forma elegante...")
        running = False
    
    # Registrar el manejador de señales
    signal.signal(signal.SIGINT, signal_handler)
    
    screen_width = CONFIG["window"]["width"]
    screen_height = CONFIG["window"]["height"]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"PAHM Environment - Model: {args.model}")
    clock = pygame.time.Clock()
    
    # Crear entorno con parámetros personalizados
    try:
        base_env = _build_env(
            render_mode="rgb_array",
            model_name=args.model,
            reset_angle=args.reset_angle,
        )
        # Aplicar wrapper TimeLimit para manejar truncamiento automático
        env = TimeLimit(base_env, max_episode_steps=args.max_steps)
        print(f"✅ Modelo cargado exitosamente: {args.model}")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el modelo '{args.model}'")
        print("   Verifica que el archivo existe en el directorio actual.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        sys.exit(1)
    
    controller = PAHMController(screen_width, screen_height)
    oscilloscope = Oscilloscope()
    
    pad = CONFIG["layout"]["padding"]
    panel_w = CONFIG["layout"]["panel_width"]
    oscilloscope.rect.width = screen_width - panel_w - (3 * pad)
    oscilloscope.rect.height = 120
    oscilloscope.set_pos(pad, screen_height - oscilloscope.rect.height - pad)
    
    # Inicializar
    observation, info = env.reset(seed=123)
    
    # Contador de steps real del entorno
    env_step_count = 0
    
    print("🎮 Controles:")
    print("   - Click en modos para cambiar")
    print("   - En Manual: arrastra slider")
    print("   - Esc o Ctrl-C para salir")
    print("   - Osciloscopio: Verde=PWM, Naranja=Ángulo")
    print(f"📊 Debug: Episodio se truncará a los {args.max_steps} pasos")
    print()
    
    try:
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("🛑 Esc presionado. Saliendo...")
                    running = False
                else:
                    controller.handle_event(event)
            
            # Actualizar
            controller.update(mouse_pos)
            
            # Acción y step del entorno
            action_value = controller.get_action()
            action = np.array([action_value])
            observation, reward, terminated, truncated, info = env.step(action)
            env_step_count += 1

            # Agregar datos al osciloscopio
            angle_radians = observation[0] if len(observation) > 0 else 0.0
            oscilloscope.add_sample(action_value, angle_radians)
            
            # Renderizar entorno
            env_surface = env.render()
            if env_surface is not None:
                env_surface = pygame.surfarray.make_surface(env_surface.swapaxes(0, 1))
                # Escalar simétricamente ocupando espacio disponible
                render_size = min(oscilloscope.rect.width, oscilloscope.rect.y - (2 * pad))
                env_surface = pygame.transform.scale(env_surface, (render_size, render_size))

            # Dibujar todo
            screen.fill((255, 255, 255))
            if env_surface is not None:
                screen.blit(env_surface, (pad, pad))
            controller.draw(screen)
            oscilloscope.draw(screen, controller.font_dict)
            
            pygame.display.flip()
            clock.tick(50)
            
            # Reset si termina
            if terminated or truncated:
                if truncated:
                    print(f"🔄 Fin de episodio tras {args.max_steps} pasos (ángulo: {np.rad2deg(angle_radians)}°)")
                else:
                    print(f"🔄 Episodio abortado (ángulo: {np.rad2deg(angle_radians)}°)")
                observation, info = env.reset()
                env_step_count = 0  # Reset contador
    
    except KeyboardInterrupt:
        # Backup en caso de que el signal handler no funcione
        print("\n🛑 KeyboardInterrupt detectado. Cerrando...")

    except Exception:
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup garantizado
        print("🧹 Limpiando recursos...")
        try:
            env.close()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        print("✅ Programa terminado correctamente.")
        sys.exit(0)

if __name__ == "__main__":
    main()
