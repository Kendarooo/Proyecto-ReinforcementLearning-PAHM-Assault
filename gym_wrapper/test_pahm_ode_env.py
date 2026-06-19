# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2
# Versión: 1.6.0

from gymnasium.wrappers import TimeLimit
from learned_pahm_ode import LearnedPAHMODE
import numpy as np
import pygame
import signal
import argparse
from pahm_ui import PAHMController, Oscilloscope, CONFIG
from pid import PIDController
from wind_process import WindProcess

def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE Environment')
    parser.add_argument('--model', type=str, required=True, help='Ruta al archivo .pth del modelo ODE')
    parser.add_argument('--reset_angle', type=float, default=180, help='Ángulo de reset en grados')
    parser.add_argument('--max_steps', type=int, default=2000, help='Máximo número de pasos por episodio')
    parser.add_argument('--pid', type=str, default='', help='Ruta al archivo json de configuración del PID')
    args = parser.parse_args()
    
    pygame.init()
    screen_width = CONFIG["window"]["width"]
    screen_height = CONFIG["window"]["height"]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Neural ODE PAHM: {args.model}")
    clock = pygame.time.Clock()
    
    try:
        base_env = LearnedPAHMODE(render_mode="rgb_array", 
                                  model_path=args.model, 
                                  reset_angle_deg=args.reset_angle,
                                  max_wind_torque=CONFIG["wind_patterns"]["max_wind_torque"])
        env = TimeLimit(base_env, max_episode_steps=args.max_steps)
    except Exception as e:
        print(f"❌ Error iniciando entorno: {e}")
        return

    controller = PAHMController(screen_width, screen_height)

    pid_controller = None
    if args.pid:
        try:
            pid_controller = PIDController(args.pid)
        except Exception as e:
            print(f"❌ Error al inicializar PID: {e}")
            
    if not pid_controller and "PID" in controller.radio_group.options:
        controller.radio_group.options.remove("PID")
        controller.ui_tree.pack()
    
    scope = Oscilloscope()

    # WindProcess vive aquí (el lazo), no en la UI ni en el entorno.
    # Recibe su sub-config parseada por constructor (el entorno no lee config).
    wind_process = WindProcess(CONFIG["wind_patterns"])
    # Mapa mnemónico UI -> nombre de patrón en WindProcess.
    PATTERN_MAP = {"Calm": "calm", "Gust": "gust", "Sust": "sustained", "Turb": "turbulent"}
    last_wind_pattern = None

    
    pad = CONFIG["layout"]["padding"]
    panel_w = CONFIG["layout"]["panel_width"]
    scope.rect.width = screen_width - panel_w - (3 * pad)
    scope.rect.height = 120
    scope.set_pos(pad, screen_height - scope.rect.height - pad)
    
    obs, _ = env.reset()
    running = True
    
    # Manejo de señal para cierre limpio
    def signal_handler(sig, frame):
        nonlocal running
        print("\n🛑 Interrupción recibida.")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🎮 Entorno Interactivo Listo.")
    
    try:
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                controller.handle_event(event)

            controller.update(mouse_pos)

            # --- Resolución del viento: Manual (polar) vs patrón (WindProcess) ---
            active, man_mag, man_angle = controller.wind_state
            base_env.set_theta_ref(controller.setpoint_rad)
            mode = controller.wind_pattern_mode
            if active and mode != "Manu":
                # set_pattern SOLO al cambiar de patrón (si no, resetea t=0 cada frame)
                if mode != last_wind_pattern:
                    wind_process.set_pattern(PATTERN_MAP[mode])
                    last_wind_pattern = mode
                w_mag, w_angle = wind_process.step(base_env.dt)
            else:
                # Manual o viento inactivo: usar el control polar (o no inyectar)
                w_mag, w_angle = man_mag, man_angle
                last_wind_pattern = None
            env.unwrapped.set_wind(active, w_mag, w_angle)
            
            pid_action = 0.0
            rl_action = 0.0
            if pid_controller and controller.current_mode == "PID":
                current_angle = obs[0] if len(obs) > 0 else 0.0
                pid_action = pid_controller.compute(controller.setpoint_rad, current_angle, base_env.dt)
            elif controller.current_mode == "RL":
                # -----------------------------------------------------------
                # POR HACER: Inserte aquí el acoplamiento de su modelo RL.
                # Deben pasar la observación (obs) a su agente entrenado
                # y asignar la acción calculada a la variable `rl_action`.
                # -----------------------------------------------------------
                rl_action = 0.0
                
            if pid_controller and controller.current_mode != "PID":
                pid_controller.reset()
            
            action_value = controller.get_action(rl_agent_action=rl_action, pid_action=pid_action)

            action = np.array([action_value])
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Osciloscopio
            angle_disp = obs[0] if len(obs) > 0 else 0.0
            scope.add_sample(action[0], angle_disp)
            
            # Render
            screen.fill((255,255,255))
            
            env_frame = env.render()
            if env_frame is not None:
                surf = pygame.surfarray.make_surface(env_frame.swapaxes(0,1))

                # Escalar simétricamente ocupando espacio disponible
                render_size = min(scope.rect.width, scope.rect.y - (2 * pad))
                surf = pygame.transform.scale(surf, (render_size, render_size))
                screen.blit(surf, (pad, pad))
                
            controller.draw(screen)
            scope.draw(screen, controller.font_dict)
            
            pygame.display.flip()
            clock.tick(50)
            
            if terminated or truncated:
                env.reset()
                if pid_controller:
                    pid_controller.reset()
                
    except Exception as e:
        print(f"💥 Error en runtime: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        pygame.quit()
        print("✅ Fin del programa.")

if __name__ == "__main__":
    main()
