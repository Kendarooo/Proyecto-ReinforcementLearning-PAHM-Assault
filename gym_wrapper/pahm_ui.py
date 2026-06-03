# Copyright (C) 2026 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2026
# Proyecto 2

# Versión: 1.4.0
# Descripción: Módulo SSOT para elementos gráficos. Implementa patrón Composite 
# para layouts (VBox) y aísla la responsabilidad de la UI de los scripts de test (SRP).

import pygame
import numpy as np
import json
import math
import os
from collections import deque

# --- Carga de Configuración SSOT ---
def load_config(filename='config.json'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Archivo de configuración {filename} no encontrado.")
    with open(filename, 'r') as f:
        return json.load(f)

CONFIG = load_config()

class UIElement:
    """Clase base abstracta para el patrón Composite."""
    def __init__(self, w=0, h=0):
        self.rect = pygame.Rect(0, 0, w, h)
        self.visible = True

    def set_pos(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def draw(self, screen, font_dict):
        pass

    def handle_event(self, event):
        return False

    def update(self, mouse_pos):
        pass

class VBox(UIElement):
    """Contenedor vertical que posiciona a sus hijos automáticamente."""
    def __init__(self, children=None, spacing=None):
        super().__init__()
        self.children = children if children else []
        self.spacing = spacing if spacing is not None else CONFIG["layout"]["group_spacing"]

        self.pack()

    def pack(self):
        current_y = 0
        max_w = 0
        for child in self.children:
            if not child.visible: continue
            child.set_pos(self.rect.x, self.rect.y + current_y)
            current_y += child.rect.height + self.spacing
            max_w = max(max_w, child.rect.width)
        self.rect.height = max(0, current_y - self.spacing)
        self.rect.width = max_w

    def set_pos(self, x, y):
        super().set_pos(x, y)
        self.pack() # Reposicionar hijos al mover el padre

    def draw(self, screen, font_dict):
        for child in self.children:
            if child.visible: child.draw(screen, font_dict)

    def handle_event(self, event):
        handled = False
        for child in self.children:
            if child.visible and child.handle_event(event):
                handled = True
        return handled

    def update(self, mouse_pos):
        for child in self.children:
            if child.visible: child.update(mouse_pos)

class HBox(UIElement):
    """Contenedor horizontal. Análogo a VBox para colocar elementos en fila
    (usado para poner el selector de patrón a la derecha del círculo manual)."""
    def __init__(self, children=None, spacing=None):
        super().__init__()
        self.children = children if children else []
        self.spacing = spacing if spacing is not None else CONFIG["layout"]["group_spacing"]
        self.pack()

    def pack(self):
        current_x = 0
        max_h = 0
        for child in self.children:
            if not child.visible: continue
            child.set_pos(self.rect.x + current_x, self.rect.y)
            current_x += child.rect.width + self.spacing
            max_h = max(max_h, child.rect.height)
        self.rect.width = max(0, current_x - self.spacing)
        self.rect.height = max_h

    def set_pos(self, x, y):
        super().set_pos(x, y)
        self.pack()

    def draw(self, screen, font_dict):
        for child in self.children:
            if child.visible: child.draw(screen, font_dict)

    def handle_event(self, event):
        handled = False
        for child in self.children:
            if child.visible and child.handle_event(event):
                handled = True
        return handled

    def update(self, mouse_pos):
        for child in self.children:
            if child.visible: child.update(mouse_pos)

class TextLabel(UIElement):
    def __init__(self, text, font_type='default'):
        self.text = text
        self.font_type = font_type
        # Estimación de altura para el layout
        h = CONFIG["fonts"][f"{font_type}_size"]
        super().__init__(200, h)

    def draw(self, screen, font_dict):
        font = font_dict.get(self.font_type, font_dict['default'])
        surface = font.render(self.text, True, CONFIG["colors"]["text"])
        screen.blit(surface, (self.rect.x, self.rect.y))

class RadioGroup(UIElement):
    """Agrupa RadioButtons para asegurar selección única."""
    def __init__(self, options, default_idx=0):
        self.options = options
        self.selected_text = options[default_idx]
        h = len(options) * (CONFIG["components"]["radio"]["radius"] * 2 + CONFIG["layout"]["item_spacing"])
        super().__init__(200, h)
        self.hovered_idx = -1

    def draw(self, screen, font_dict):
        font = font_dict['default']
        r = CONFIG["components"]["radio"]["radius"]
        for i, opt in enumerate(self.options):
            y_c = self.rect.y + i * (2 * r + CONFIG["layout"]["item_spacing"]) + r
            center = (self.rect.x + r, y_c)
            
            color = CONFIG["colors"]["secondary"] if i == self.hovered_idx else CONFIG["colors"]["secondary_hover"]
            pygame.draw.circle(screen, color, center, r, 2)
            
            if opt == self.selected_text:
                pygame.draw.circle(screen, CONFIG["colors"]["primary"], center, r - 3)
            
            surf = font.render(opt, True, CONFIG["colors"]["text"])
            screen.blit(surf, (center[0] + r + 10, center[1] - r))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            r = CONFIG["components"]["radio"]["radius"]
            for i, opt in enumerate(self.options):
                y_c = self.rect.y + i * (2 * r + CONFIG["layout"]["item_spacing"]) + r
                if math.hypot(event.pos[0] - (self.rect.x + r), event.pos[1] - y_c) <= r + 5:
                    self.selected_text = opt
                    return True
        return False

    def update(self, mouse_pos):
        self.hovered_idx = -1
        r = CONFIG["components"]["radio"]["radius"]
        for i in range(len(self.options)):
            y_c = self.rect.y + i * (2 * r + CONFIG["layout"]["item_spacing"]) + r
            if math.hypot(mouse_pos[0] - (self.rect.x + r), mouse_pos[1] - y_c) <= r + 5:
                self.hovered_idx = i

class Slider(UIElement):
    def __init__(self, min_val, max_val, initial_val, fmt="{:.3f}"):
        w = CONFIG["components"]["slider"]["width"]
        h = CONFIG["components"]["slider"]["height"]
        super().__init__(w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.fmt = fmt
        self.dragging = False
        self.hovered = False

    def draw(self, screen, font_dict):
        bg_color = CONFIG["colors"]["secondary_hover"] if self.hovered else CONFIG["colors"]["secondary"]
        pygame.draw.rect(screen, bg_color, self.rect)
        pygame.draw.rect(screen, CONFIG["colors"]["border"], self.rect, 2)
        
        rel_pos = (self.val - self.min_val) / (self.max_val - self.min_val)
        hx = self.rect.x + rel_pos * self.rect.width
        h_rect = pygame.Rect(hx - 5, self.rect.y - 2, 10, self.rect.height + 4)
        
        h_color = CONFIG["colors"]["primary_hover"] if self.dragging else CONFIG["colors"]["primary"]
        pygame.draw.rect(screen, h_color, h_rect)
        
        # Etiqueta de valor
        font = font_dict['small']
        surf = font.render(self.fmt.format(self.val), True, CONFIG["colors"]["text"])
        screen.blit(surf, (self.rect.right + 10, self.rect.y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_val(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_val(event.pos[0])
            return True
        return False

    def _update_val(self, x_pos):
        rel_x = max(0, min(x_pos - self.rect.x, self.rect.width))
        self.val = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

class ToggleButton(UIElement):
    def __init__(self, text="Off/On"):
        super().__init__(80, 30)
        self.active = False
        self.text = text

    def draw(self, screen, font_dict):
        color = CONFIG["colors"]["primary"] if self.active else CONFIG["colors"]["secondary"]
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        surf = font_dict['small'].render(self.text, True, [255,255,255])
        
        txt_rect = surf.get_rect(center=self.rect.center)
        screen.blit(surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
        return False

class WindPolarControl(UIElement):
    """Control polar para magnitud y dirección del viento."""
    def __init__(self):
        self.r = CONFIG["components"]["wind_control"]["radius"]
        super().__init__(self.r * 2, self.r * 2)
        self.mag = 0.0   # 0.0 a 1.0
        self.angle = 0.0 # Radianes
        self.dragging = False

    def draw(self, screen, font_dict):
        cx, cy = self.rect.center
        
        pygame.draw.circle(screen, CONFIG["colors"]["wind_bg"], (cx, cy), self.r)
        pygame.draw.circle(screen, CONFIG["colors"]["border"], (cx, cy), self.r, 2)
        
        # Ejes
        pygame.draw.line(screen, CONFIG["colors"]["secondary"], (cx - self.r, cy), (cx + self.r, cy))
        pygame.draw.line(screen, CONFIG["colors"]["secondary"], (cx, cy - self.r), (cx, cy + self.r))
        
        # Vector de viento
        if self.mag > 0.01:
            vx = cx + math.cos(self.angle) * self.mag * self.r
            vy = cy - math.sin(self.angle) * self.mag * self.r # Pygame Y invertido
            pygame.draw.line(screen, CONFIG["colors"]["wind_arrow"], (cx, cy), (vx, vy), 3)
            pygame.draw.circle(screen, CONFIG["colors"]["wind_arrow"], (int(vx), int(vy)), 5)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if math.hypot(event.pos[0] - self.rect.centerx, event.pos[1] - self.rect.centery) <= self.r:
                self.dragging = True
                self._update_vector(event.pos)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_vector(event.pos)
            return True
        return False

    def _update_vector(self, pos):
        dx = pos[0] - self.rect.centerx
        dy = -(pos[1] - self.rect.centery) # Invertir para plano cartesiano normal
        dist = math.hypot(dx, dy)
        self.mag = min(1.0, dist / self.r)
        self.angle = math.atan2(dy, dx)

class Oscilloscope(UIElement):
    """Osciloscopio robusto. Fusiona filtrado de NaNs y renderizado completo."""
    def __init__(self, max_samples=500):
        # Asume que será dimensionado externamente según ventana
        super().__init__(100, 100)
        self.max_samples = max_samples
        self.input_data = deque(maxlen=max_samples)
        self.output_data = deque(maxlen=max_samples)
        
        self.input_min, self.input_max = 0.0, 1.0
        self.output_min, self.output_max = -2.1, 2.1

    def add_sample(self, in_val, out_val):
        self.input_data.append(in_val)
        self.output_data.append(out_val)
        
        valid_out = [v for v in self.output_data if np.isfinite(v)]
        if valid_out:
            c_min, c_max = min(valid_out), max(valid_out)
            if c_min < self.output_min: self.output_min = c_min - 0.1
            if c_max > self.output_max: self.output_max = c_max + 0.1

    def _draw_signal(self, screen, data, min_v, max_v, color):
        if len(data) < 2: return
        pts = []
        denom = max_v - min_v if max_v != min_v else 1.0
        for i, v in enumerate(data):
            if not np.isfinite(v): continue
            x = self.rect.x + (i * self.rect.width // self.max_samples)
            y = self.rect.bottom - ((v - min_v)/denom * self.rect.height)
            pts.append((float(x), float(y)))
        if len(pts) > 1:
            pygame.draw.lines(screen, color, False, pts, 2)

    def draw(self, screen, font_dict):
        pygame.draw.rect(screen, CONFIG["colors"]["osc_bg"], self.rect)
        pygame.draw.rect(screen, CONFIG["colors"]["border"], self.rect, 1)
        
        # Grilla
        for i in range(1, 4):
            y = self.rect.y + (i * self.rect.height // 4)
            pygame.draw.line(screen, CONFIG["colors"]["osc_grid"], (self.rect.x, y), (self.rect.right, y))
            
        self._draw_signal(screen, self.input_data, self.input_min, self.input_max, CONFIG["colors"]["osc_in"])
        self._draw_signal(screen, self.output_data, self.output_min, self.output_max, CONFIG["colors"]["osc_out"])
        
        # Etiquetas
        f = font_dict['small']
        screen.blit(f.render("PWM Input", True, CONFIG["colors"]["osc_in"]), (self.rect.x + 5, self.rect.y + 5))
        screen.blit(f.render("Angle Output", True, CONFIG["colors"]["osc_out"]), (self.rect.x + 5, self.rect.y + 25))

        # Valores numéricos alineados
        offset = CONFIG["components"]["oscilloscope"]["value_offset"]
        in_str = f"{self.input_data[-1]:.3f}" if (self.input_data and np.isfinite(self.input_data[-1])) else "---"
        out_str = f"{np.rad2deg(self.output_data[-1]):.1f}°" if (self.output_data and np.isfinite(self.output_data[-1])) else "---"

        screen.blit(f.render(in_str, True, CONFIG["colors"]["osc_in"]), (self.rect.x + 5 + offset, self.rect.y + 5))
        screen.blit(f.render(out_str, True, CONFIG["colors"]["osc_out"]), (self.rect.x + 5 + offset, self.rect.y + 25))

        
class PAHMController:
    """Controlador principal de interfaz. Expone estados para el lazo de simulación."""
    def __init__(self, screen_w, screen_h):
        self.font_dict = {
            'default': pygame.font.Font(None, CONFIG["fonts"]["default_size"]),
            'title': pygame.font.Font(None, CONFIG["fonts"]["title_size"]),
            'small': pygame.font.Font(None, CONFIG["fonts"]["small_size"])
        }
        
        self.panel_rect = pygame.Rect(
            screen_w - CONFIG["layout"]["panel_width"], 0, 
            CONFIG["layout"]["panel_width"], screen_h
        )
        
        self.step_counter = 0
        self.last_action = 0.0
        
        # Modos incluyendo PID y RL
        modes = ["Off", "Step", "Sine", "Random", "Manual", "PID", "RL"]
        
        # Construcción del Layout
        self.radio_group = RadioGroup(modes, default_idx=2)
        self.slider_pwm = Slider(0.0, 1.0, 0.0)
        self.slider_sp = Slider(CONFIG["physics"]["min_setpoint_deg"], CONFIG["physics"]["max_setpoint_deg"], 0.0, fmt="{:.1f}°")
        
        self.toggle_wind = ToggleButton("Wind Off")
        self.wind_polar = WindPolarControl()

        # Selector de patrón de viento (mnemónicos de 4 letras).
        # "Manu" = control polar manual; el resto activan un WindProcess.
        # El WindProcess vive en el lazo (test_*), no aquí (SRP: la UI
        # expone intención, el lazo ejecuta el modelo).
        self.wind_pattern = RadioGroup(["Manu", "Calm", "Gust", "Sust", "Turb"])
        
        # Ocultar elementos según el contexto se maneja en el update_logic
        g_space = CONFIG["layout"]["group_spacing"]
        i_space = CONFIG["layout"]["item_spacing"]
        
        self.ui_tree = VBox([
            TextLabel("Control Panel", "title"),
            VBox([TextLabel("Mode:", "default"), self.radio_group], spacing=i_space),
            VBox([TextLabel("PWM Manual:", "default"), self.slider_pwm], spacing=i_space),
            VBox([TextLabel("Setpoint:", "default"), self.slider_sp], spacing=i_space),
            VBox([TextLabel("Wind Perturbation:", "default"), self.toggle_wind,
                  HBox([self.wind_polar, self.wind_pattern], spacing=g_space)], spacing=i_space)
        ], spacing=g_space)
        
        # Posición inicial del layout
        pad = CONFIG["layout"]["padding"]
        self.ui_tree.set_pos(self.panel_rect.x + pad, pad)

    @property
    def current_mode(self):
        return self.radio_group.selected_text

    @property
    def setpoint_rad(self):
        return np.deg2rad(self.slider_sp.val)

    @property
    def wind_state(self):
        """Retorna (activo: bool, magnitud: float, angulo_rad: float)"""
        return (self.toggle_wind.active, self.wind_polar.mag, self.wind_polar.angle)

    @property
    def wind_pattern_mode(self):
        """Mnemónico del patrón de viento seleccionado ("Manu","Calm",...)."""
        return self.wind_pattern.selected_text
    
    def handle_event(self, event):
        self.ui_tree.handle_event(event)

    def update(self, mouse_pos):
        self.ui_tree.update(mouse_pos)
        # Actualizar texto del toggle
        self.toggle_wind.text = "Wind ON" if self.toggle_wind.active else "Wind OFF"

    def get_action(self, rl_agent_action=0.0, pid_action=0.0):
        """
        Calcula la acción según el modo. 
        Acepta inyecciones del lazo principal para modos PID/RL.
        """
        self.step_counter += 1
        mode = self.current_mode
        
        if mode == "Off": act = 0.0
        elif mode == "Step": act = 0.2
        elif mode == "Sine": act = 0.25 * (1 + np.sin(self.step_counter * 2 * np.pi / 150)) / 2
        elif mode == "Random": act = np.random.rand() * 0.25
        elif mode == "Manual": act = self.slider_pwm.val
        elif mode == "PID": act = pid_action
        elif mode == "RL": act = rl_agent_action
        else: act = 0.0
        
        # Saturación de seguridad
        self.last_action = np.clip(act, 0.0, 1.0)
        return self.last_action

    def draw(self, screen):
        # Fondo del panel
        pygame.draw.rect(screen, CONFIG["layout"]["panel_bg"], self.panel_rect)
        pygame.draw.line(screen, CONFIG["colors"]["border"], 
                         (self.panel_rect.x, 0), (self.panel_rect.x, self.panel_rect.height), 2)
        
        self.ui_tree.draw(screen, self.font_dict)
