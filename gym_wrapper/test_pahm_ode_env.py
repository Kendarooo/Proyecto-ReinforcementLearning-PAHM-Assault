# Copyright (C) 2024-2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2
# Versión: 1.6.0

from gymnasium.wrappers import TimeLimit
import numpy as np
import pygame
import signal
import argparse
from pathlib import Path

try:
    from learned_pahm_ode import LearnedPAHMODE
except ImportError:
    from gym_wrapper.learned_pahm_ode import LearnedPAHMODE

try:
    from pahm_ui import PAHMController, Oscilloscope, CONFIG
except ImportError:
    from gym_wrapper.pahm_ui import PAHMController, Oscilloscope, CONFIG

try:
    from pid import PIDController
except ImportError:
    from gym_wrapper.pid import PIDController

try:
    from wind_process import WindProcess
except ImportError:
    from gym_wrapper.wind_process import WindProcess

try:
    from demo_wind import (
        ManualGustController,
        demo_wind_settings,
        handle_wind_event,
        resolve_demo_wind,
    )
except ImportError:
    from gym_wrapper.demo_wind import (
        ManualGustController,
        demo_wind_settings,
        handle_wind_event,
        resolve_demo_wind,
    )

try:
    from rl_policy import (
        apply_rl_control_step,
        get_demo_model_path,
        load_rl_policy,
        predict_rl_action,
    )
except ImportError:
    from gym_wrapper.rl_policy import (
        apply_rl_control_step,
        get_demo_model_path,
        load_rl_policy,
        predict_rl_action,
    )


def _load_stage3_config(config_path: str | None) -> tuple[dict, Path | None]:
    if not config_path:
        return {}, None
    path = Path(config_path).resolve()
    import json

    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file), path.parent


def _init_rl_policy(config: dict, config_dir: Path | None, explicit_model: str | None):
    demo_config = config.get("demo", {})
    if explicit_model:
        model_path = Path(explicit_model)
        if not model_path.is_absolute() and config_dir is not None:
            model_path = config_dir / model_path
    elif demo_config:
        model_path = get_demo_model_path(
            config,
            model_type=demo_config.get("rl_model_type"),
            base_dir=config_dir,
        )
    else:
        return None, None

    policy = load_rl_policy(
        model_path,
        algorithm=config.get("rl_training", {}).get("algorithm", "PPO"),
    )
    return policy, model_path


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE Environment')
    parser.add_argument('--model', type=str, required=True, help='Ruta al archivo .pth del modelo ODE')
    parser.add_argument('--reset_angle', type=float, default=180, help='Ángulo de reset en grados')
    parser.add_argument('--max_steps', type=int, default=2000, help='Máximo número de pasos por episodio')
    parser.add_argument('--pid', type=str, default='', help='Ruta al archivo json de configuración del PID')
    parser.add_argument('--config', type=str, default='', help='Ruta a configuración Stage 3 para demo RL')
    parser.add_argument('--rl_model', type=str, default='', help='Ruta explícita a política RL .zip')
    args = parser.parse_args()

    stage3_config, stage3_config_dir = _load_stage3_config(args.config)
    visual_config = stage3_config or CONFIG
    wind_demo_settings = demo_wind_settings(visual_config)
    
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
        base_env.wind_cfg["enabled"] = wind_demo_settings["show_particles"]
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

    rl_policy = None
    rl_model_path = None
    try:
        rl_policy, rl_model_path = _init_rl_policy(
            stage3_config,
            stage3_config_dir,
            args.rl_model or None,
        )
        if rl_policy is not None:
            print(f"✅ Política RL cargada: {rl_model_path}")
    except Exception as exc:
        print(f"⚠️ No se cargó política RL: {exc}")
        if "RL" in controller.radio_group.options:
            controller.radio_group.options.remove("RL")
            controller.ui_tree.pack()

    deterministic_policy = bool(
        stage3_config.get("demo", {}).get("deterministic_policy", True)
    )
    manual_gust = ManualGustController.from_config(visual_config)
    if not wind_demo_settings["interactive_wind"]:
        manual_gust.enabled = False
    
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
    sim_time = 0.0
    
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
                handle_wind_event(event, manual_gust, pygame_module=pygame, t=sim_time)
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

            demo_wind_state = resolve_demo_wind(
                base_active=active,
                base_mag=w_mag,
                base_angle=w_angle,
                theta=float(base_env.state[0]),
                max_wind_torque=base_env.max_wind_torque,
                gust_controller=manual_gust,
                t=sim_time,
                particles_enabled=wind_demo_settings["show_particles"],
            )
            env.unwrapped.set_wind(
                demo_wind_state.active,
                demo_wind_state.mag,
                demo_wind_state.angle,
            )
            
            pid_action = 0.0
            rl_action = 0.0
            if pid_controller and controller.current_mode == "PID":
                current_angle = obs[0] if len(obs) > 0 else 0.0
                pid_action = pid_controller.compute(controller.setpoint_rad, current_angle, base_env.dt)
            elif controller.current_mode == "RL" and rl_policy is not None:
                obs = base_env._get_obs()
                rl_action = float(
                    predict_rl_action(
                        rl_policy,
                        obs,
                        env.action_space,
                        deterministic=deterministic_policy,
                    )[0]
                )
                
            if pid_controller and controller.current_mode != "PID":
                pid_controller.reset()
            
            action_value = controller.get_action(rl_agent_action=rl_action, pid_action=pid_action)

            action = np.array([action_value], dtype=np.float32)
            if controller.current_mode == "RL" and rl_policy is not None:
                obs, reward, terminated, truncated, info = apply_rl_control_step(
                    env,
                    rl_policy,
                    obs,
                    deterministic=deterministic_policy,
                    model_path=rl_model_path,
                )
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            info.update(demo_wind_state.info())
            info.setdefault("control_mode", controller.current_mode)
            
            # Osciloscopio
            angle_disp = obs[0] if len(obs) > 0 else 0.0
            scope.add_sample(info.get("rl_action", action[0]), angle_disp)
            if wind_demo_settings["show_wind_torque"]:
                pygame.display.set_caption(
                    "Neural ODE PAHM: "
                    f"{args.model} | mode={controller.current_mode} "
                    f"| wind_tau={info['wind_torque']:.3f} "
                    f"| gust={info['manual_gust_active']}"
                )
            
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
            sim_time += base_env.dt
            
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
