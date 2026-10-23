"""
Spatial Droplet Floating Copilot Widget — Anthropic Metallic-Glass Edition.
Implements PART C UI Architecture:
  1. Removed '+' icon on bottom-left and aligned left margin cleanly with prompt input.
  2. Split [Chat], [Quick], and [Droplets] into three separate independent metallic-glass
     pill boxes with smooth delayed hover transitions (75ms entry delay + multi-step interpolation).
  3. Upgraded Send button to a multi-layered 3D depth terracotta squircle with top specular bevel
     and smooth delayed transition; upgraded Mic button to a modern studio condenser vector icon
     with hover transition and 1-click / hotkey toggle (Alt+V, F8 — no holding required).
  4. Applied Anthropic typography stack (Styrene A, Copernicus, Charter, Georgia, Segoe UI Variable Display).
  5. Humanized RobotMascotAnimator with natural micro-saccades, double-blinks, dual corneal catchlights,
     curved eyelid expressions, warm cheek glow, and a sleek reflective metallic-glass outer coating.
  6. Upgraded main chat bar from flat dark grey to a brushed gunmetal metallic-glass finish, removed
     the overlapping thunder icon on the model selector, and implemented a smoothly animated
     Model Selector Popover matching the reference design (Model header, checkmark, New/Notice pills,
     right '>' chevrons, and 'Select Model  Ctrl+/' footer).
"""

import os
import sys
import math
import time
import random
import ctypes
from ctypes import wintypes
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any, List
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter

from ..core.models import register_own_hwnd, DiagonalDroplet

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _ensure_assets():
    """Ensures robot mascot chassis asset is available."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    chassis_path = ASSETS_DIR / "robot_chassis.png"
    if chassis_path.exists():
        return

    sources = [
        Path(r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose\robot_clean_base_plate.png"),
        Path(r"C:\Users\NANDHA K\.gemini\antigravity\brain\77172af1-65f6-41cc-bf5d-0a2f27cf94ff\.user_uploaded\media_1792177279344_5c1c7073.png")
    ]
    for src in sources:
        if src.exists():
            try:
                import shutil
                shutil.copy(src, chassis_path)
                return
            except Exception:
                pass


def _resolve_font_family(preferred: List[str], fallback: str = "Segoe UI") -> str:
    """Resolves the first installed font family from the Anthropic preference stack."""
    try:
        installed = {f.lower(): f for f in tkfont.families()}
        for cand in preferred:
            if cand.lower() in installed:
                return installed[cand.lower()]
    except Exception:
        pass
    return fallback


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class RobotMascotAnimator:
    """
    Humanized Interactive Robot Mascot with Sleek Reflective Metallic-Glass Outer Coating.
    Features:
      - Brushed metallic-glass specular outer coating with dynamic fresnel sheen sweep
      - Organic human micro-saccades, natural double-blinks, and typing-reactive gaze
      - Dual corneal catchlight glints for warm, lifelike human eye exposure
      - Expressive curved upper/lower eyelids & subtle cheek warmth
      - Real-time cognitive state coupling (idle, listening, thinking, executing, happy, confused)
    """

    GLOW_COLORS: Dict[str, Tuple[int, int, int]] = {
        "idle": (0, 229, 255),
        "thinking": (192, 132, 252),
        "executing": (56, 189, 248),
        "happy": (16, 185, 129),
        "confused": (244, 63, 94),
        "listening": (0, 240, 255),
        "transcribing": (255, 180, 0),
        "scanning": (0, 229, 255),
        "vimium": (251, 191, 36),
    }

    def __init__(self, base_w: int = 60, base_h: int = 44, base_size: Optional[int] = None, **kwargs):
        if base_size is not None:
            base_w = base_size
            base_h = int(base_size * 44 / 60)
        self.base_w = base_w
        self.base_h = base_h
        self.emotion = "idle"

        # Continuous emotional & human micro-saccade parameters
        self.gaze_x = 0.0
        self.target_gaze_x = 0.0
        self.gaze_y = 0.0
        self.target_gaze_y = 0.0
        self.saccade_x = 0.0
        self.saccade_y = 0.0
        self.saccade_timer = random.uniform(0.9, 2.1)

        self.eye_scale = 1.0
        self.target_eye_scale = 1.0

        # Natural blinking with occasional human double-blink
        self.blink_progress = 0.0
        self.is_blinking = False
        self.pending_double_blink = False
        self.blink_timer = random.uniform(2.6, 4.8)

        # Pulse, breathing, and metallic sheen phase
        self.pulse_phase = 0.0
        self.sheen_phase = 0.0
        self.last_tick_time = time.perf_counter()
        self.is_hovered = False
        self.typing_glance_until = 0.0

        # Load chassis base and apply sleek metallic-glass outer coating
        _ensure_assets()
        chassis_path = ASSETS_DIR / "robot_chassis.png"
        raw_chassis: Optional[Image.Image] = None
        if chassis_path.exists():
            try:
                raw_chassis = Image.open(chassis_path).convert("RGBA")
            except Exception:
                raw_chassis = None

        if raw_chassis is None:
            raw_chassis = Image.new("RGBA", (490, 332), (0, 0, 0, 0))
            d = ImageDraw.Draw(raw_chassis)
            d.ellipse((80, 40, 410, 310), fill=(242, 245, 250, 255), outline=(195, 204, 218, 255), width=6)
            d.rounded_rectangle((125, 75, 365, 275), radius=45, fill=(14, 16, 22, 255))

        self.orig_w, self.orig_h = raw_chassis.size
        self.base_chassis = self._apply_metallic_outer_coating(raw_chassis)

    @property
    def target_glow_rgb(self) -> Tuple[int, int, int]:
        return self.GLOW_COLORS.get(self.emotion, (0, 229, 255))

    @property
    def target_eye_scale_y(self) -> float:
        return self.target_eye_scale * (1.18 if self.is_hovered else 1.0)

    def _apply_metallic_outer_coating(self, img: Image.Image) -> Image.Image:
        """Enhances the outer helmet shell with a sleek anodized metallic-glass rim and specular sheen."""
        arr = np.array(img).astype(np.float32)
        h, w, _ = arr.shape
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Identify the bright outer helmet shell vs dark visor screen
        shell_mask = (alpha > 40) & (lum > 135)
        yy, xx = np.mgrid[0:h, 0:w]
        ny = yy / max(1.0, float(h))
        nx = xx / max(1.0, float(w))

        # Anodized titanium-pearl metallic shading across the outer shell
        metallic_grad = (
            1.06 * np.exp(-((ny - 0.18) ** 2) / 0.045)
            - 0.10 * np.clip((ny - 0.58) / 0.42, 0.0, 1.0)
            + 0.05 * np.cos((nx - 0.32) * math.pi * 1.8)
        )
        for c, tint in enumerate((0.97, 0.99, 1.03)):
            ch = rgb[:, :, c]
            ch_coated = np.clip(ch * (0.92 + 0.14 * metallic_grad) * tint, 0, 255)
            rgb[:, :, c] = np.where(shell_mask, ch_coated, ch)

        arr[:, :, :3] = rgb
        coated = Image.fromarray(arr.astype(np.uint8), "RGBA")

        # Add crisp glass visor specular arc and outer chrome rim highlight
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        # Upper helmet chrome reflection arc
        od.arc((96, 34, 394, 168), start=200, end=340, fill=(255, 255, 255, 115), width=5)
        # Visor glass diagonal specular reflection
        od.arc((134, 84, 356, 200), start=198, end=332, fill=(195, 225, 255, 52), width=4)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.2))
        coated.alpha_composite(overlay)
        return coated

    def notify_typing(self):
        """Causes the mascot to glance curiously toward the user's text input."""
        self.typing_glance_until = time.perf_counter() + 1.1
        self.target_gaze_x = -0.32
        self.target_gaze_y = -0.18
        self.target_eye_scale = 1.08

    def set_hover(self, is_hover: bool):
        self.is_hovered = is_hover
        if is_hover:
            self.target_eye_scale = 1.16
            self.target_gaze_y = -0.14
        else:
            if self.emotion == "idle":
                self.target_eye_scale = 1.0
                self.target_gaze_y = 0.0

    def set_emotion(self, emotion: str):
        valid = ("idle", "thinking", "listening", "executing", "happy", "confused", "vimium", "transcribing", "scanning")
        self.emotion = emotion if emotion in valid else "idle"
        if self.emotion == "listening":
            self.target_eye_scale = 1.22
            self.target_gaze_x = 0.0
            self.target_gaze_y = -0.05
        elif self.emotion == "thinking":
            self.target_eye_scale = 1.06
            self.target_gaze_x = 0.34
            self.target_gaze_y = -0.28
        elif self.emotion == "executing":
            self.target_eye_scale = 1.02
            self.target_gaze_x = 0.0
            self.target_gaze_y = 0.08
        elif self.emotion == "happy":
            self.target_eye_scale = 1.16
            self.target_gaze_x = 0.0
            self.target_gaze_y = -0.10
        elif self.emotion == "confused":
            self.target_eye_scale = 1.05
            self.target_gaze_x = -0.22
            self.target_gaze_y = 0.12
        elif self.emotion == "transcribing":
            self.target_eye_scale = 1.12
            self.target_gaze_x = 0.18
            self.target_gaze_y = -0.16
        elif self.emotion == "scanning":
            self.target_eye_scale = 1.14
            self.target_gaze_x = -0.15
            self.target_gaze_y = 0.0
        elif self.emotion == "vimium":
            self.target_eye_scale = 1.15
            self.target_gaze_x = 0.0
            self.target_gaze_y = 0.0
        else:
            self.target_eye_scale = 1.0
            self.target_gaze_x = 0.0
            self.target_gaze_y = 0.0

    def set_status(self, status_text: str, is_recording: bool):
        low = status_text.lower()
        if is_recording:
            self.set_emotion("listening")
        elif "transcrib" in low:
            self.set_emotion("transcribing")
        elif "scanning" in low:
            self.set_emotion("scanning")
        elif "vimium" in low:
            self.set_emotion("vimium")
        elif any(k in low for k in ("live auto-sync", "ready", "idle", "dots ready", "switched to")):
            self.set_emotion("idle")
        elif any(k in low for k in ("think", "reason", "llm", "ai", "qwen", "groq", "ponder", "process", "analyz", "resolv", "evaluat", "search", "query", "voice")):
            self.set_emotion("thinking")
        elif any(k in low for k in ("execut", "run", "fast-path", "click", "turn", "step", "navigat", "open", "launch", "type", "press")):
            self.set_emotion("executing")
        elif any(k in low for k in ("done", "verified", "success", "✔", "complete")):
            self.set_emotion("happy")
        elif any(k in low for k in ("error", "fail", "aborted", "⚠️", "❌", "🛑", "cancel")):
            self.set_emotion("confused")
        else:
            self.set_emotion("idle")

    def tick(self) -> bool:
        now = time.perf_counter()
        dt = max(0.001, min(0.1, now - self.last_tick_time))
        self.last_tick_time = now

        # Breathing & metallic sheen oscillators
        self.pulse_phase += dt * (4.6 if self.emotion == "listening" else 2.8)
        self.sheen_phase += dt * 1.6

        # Restore gaze after typing glance expires
        if self.typing_glance_until > 0 and now > self.typing_glance_until:
            self.typing_glance_until = 0.0
            if self.emotion == "idle" and not self.is_hovered:
                self.target_gaze_x = 0.0
                self.target_gaze_y = 0.0
                self.target_eye_scale = 1.0

        # Organic human micro-saccades
        self.saccade_timer -= dt
        if self.saccade_timer <= 0:
            self.saccade_timer = random.uniform(0.85, 2.2)
            if self.emotion in ("idle", "listening"):
                self.saccade_x = random.uniform(-0.10, 0.10)
                self.saccade_y = random.uniform(-0.07, 0.07)
            elif self.emotion == "thinking":
                self.saccade_x = random.choice((-0.25, 0.28, 0.36))
                self.saccade_y = random.uniform(-0.32, -0.14)
            else:
                self.saccade_x = 0.0
                self.saccade_y = 0.0

        # Smooth gaze & scale lerp
        lerp_speed = min(1.0, 8.5 * dt)
        goal_gx = self.target_gaze_x + self.saccade_x
        goal_gy = self.target_gaze_y + self.saccade_y
        self.gaze_x += (goal_gx - self.gaze_x) * lerp_speed
        self.gaze_y += (goal_gy - self.gaze_y) * lerp_speed
        self.eye_scale += (self.target_eye_scale - self.eye_scale) * lerp_speed

        # Natural blink cycle with occasional human double-blink
        self.blink_timer -= dt
        if self.blink_timer <= 0:
            self.is_blinking = True
            self.blink_progress = 0.0
            if self.pending_double_blink:
                self.pending_double_blink = False
                self.blink_timer = random.uniform(2.8, 5.2)
            else:
                if random.random() < 0.22:
                    self.pending_double_blink = True
                    self.blink_timer = 0.18
                else:
                    self.blink_timer = random.uniform(2.6, 5.0)

        if self.is_blinking:
            self.blink_progress += dt * 11.0
            if self.blink_progress >= 1.0:
                self.is_blinking = False
                self.blink_progress = 0.0

        return True

    def render_frame(self, target_w: int = 60, target_h: int = 44, size: Optional[int] = None) -> Image.Image:
        if size is not None:
            target_w = size
            target_h = size
        W, H = self.orig_w, self.orig_h
        canvas = self.base_chassis.copy()

        # Subtle breathing vertical bob
        bob_y = int(math.sin(self.pulse_phase) * 2.5)
        cx = 246 + int(self.gaze_x * 15)
        cy = 172 + int(self.gaze_y * 11) + bob_y
        lx = cx - 48
        rx = cx + 48
        ey = cy - 20

        cur_blink = max(0.0, min(1.0, math.sin(self.blink_progress * math.pi))) if self.is_blinking else 0.0

        color_map = {
            "idle": (245, 250, 255),
            "thinking": (225, 242, 255),
            "listening": (165, 240, 255),
            "executing": (150, 225, 255),
            "happy": (185, 255, 225),
            "confused": (255, 210, 175),
            "vimium": (255, 215, 115)
        }
        glow_rgb = color_map.get(self.emotion, (245, 250, 255))

        solid_mask = Image.new("L", (W, H), 0)
        sd = ImageDraw.Draw(solid_mask)

        # 1. Left Eye with Human Eyelid Curvature
        left_h = max(4, int(46 * self.eye_scale * (1.0 - cur_blink * 0.92)))
        left_w = int(46 * self.eye_scale)

        if cur_blink > 0.86:
            sd.arc((lx - left_w // 2, ey - 6, lx + left_w // 2, ey + 8), start=10, end=170, fill=255, width=7)
        elif self.emotion == "happy":
            sd.arc((lx - 24, ey - 22, lx + 24, ey + 18), start=192, end=348, fill=255, width=13)
        elif self.emotion == "confused":
            sd.ellipse((lx - 24, ey - 25, lx + 24, ey + 23), fill=255)
        else:
            sd.rounded_rectangle(
                (lx - left_w // 2, ey - left_h // 2, lx + left_w // 2, ey + left_h // 2),
                radius=16, fill=255
            )

        # 2. Right Eye with Expressive Asymmetry
        right_h = max(4, int(46 * self.eye_scale * (1.0 - cur_blink * 0.92)))
        right_w = int(46 * self.eye_scale)

        if cur_blink > 0.86:
            sd.arc((rx - right_w // 2, ey - 6, rx + right_w // 2, ey + 8), start=10, end=170, fill=255, width=7)
        elif self.emotion == "thinking":
            sd.rounded_rectangle((rx - 24, ey - 3, rx + 24, ey + 9), radius=6, fill=255)
            # Subtle raised eyebrow arc above thinking eye
            sd.arc((rx - 22, ey - 32, rx + 24, ey - 10), start=200, end=345, fill=255, width=6)
        elif self.emotion == "happy":
            sd.arc((rx - 24, ey - 22, rx + 24, ey + 18), start=192, end=348, fill=255, width=13)
        elif self.emotion == "confused":
            sd.rounded_rectangle((rx - 21, ey - 1, rx + 21, ey + 9), radius=5, fill=255)
        else:
            sd.rounded_rectangle(
                (rx - right_w // 2, ey - right_h // 2, rx + right_w // 2, ey + right_h // 2),
                radius=16, fill=255
            )

        # 3. Expressive Humanized Mouth / Voice Equalizer
        my = cy + 33
        if self.emotion == "happy":
            sd.pieslice((cx - 29, my - 13, cx + 29, my + 26), start=0, end=180, fill=255)
        elif self.emotion == "thinking":
            pts = [(cx - 20, my + 2), (cx + 22, my - 1), (cx + 12, my + 12), (cx - 10, my + 12)]
            sd.polygon(pts, fill=255)
        elif self.emotion == "listening":
            # Dynamic 5-bar human voice equalizer mouth
            for idx_b, bx_off in enumerate((-22, -11, 0, 11, 22)):
                bh = int(6 + 9 * abs(math.sin(self.pulse_phase * 1.9 + idx_b * 0.85)))
                sd.rounded_rectangle((cx + bx_off - 3, my + 4 - bh, cx + bx_off + 3, my + 4 + bh), radius=3, fill=255)
        elif self.emotion == "confused":
            sd.rounded_rectangle((cx - 17, my - 2, cx + 17, my + 5), radius=3, fill=255)
        else:
            # Gentle warm curved smile
            sd.chord((cx - 22, my - 10, cx + 22, my + 15), start=8, end=172, fill=255)

        # Convert solid shape into ultra-crisp LED scanlines with fuller fill (3px line, 1px micro-gap)
        solid_arr = np.array(solid_mask)
        scan_arr = np.zeros_like(solid_arr)
        for y in range(H):
            if (y % 4) != 3:
                scan_arr[y, :] = solid_arr[y, :]
            else:
                scan_arr[y, :] = (solid_arr[y, :] * 0.45).astype(np.uint8)

        # Soft bloom halo
        glow_im = Image.fromarray(scan_arr).filter(ImageFilter.GaussianBlur(radius=4.5))
        glow_arr = np.array(glow_im).astype(np.float32) / 255.0

        face_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        core_mask = (scan_arr > 45)
        pulse_boost = 1.0 + 0.12 * math.sin(self.pulse_phase)
        for c in range(3):
            c_val = glow_arr * glow_rgb[c] * 0.78 * pulse_boost + (core_mask * 255.0)
            face_rgba[:, :, c] = np.clip(c_val, 0, 255).astype(np.uint8)
        face_rgba[:, :, 3] = np.clip(glow_arr * 175 + (core_mask * 255.0), 0, 255).astype(np.uint8)

        face_layer = Image.fromarray(face_rgba)
        canvas.alpha_composite(face_layer)

        # 4. Human Corneal Specular Catchlights & Warm Cheek Exposure
        detail_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(detail_layer)
        if cur_blink < 0.55 and self.emotion != "happy":
            # Primary top-left corneal catchlight glint + secondary lower-right reflection
            for eye_cx, is_wink in ((lx, False), (rx, self.emotion in ("thinking", "confused"))):
                if not is_wink:
                    gx_pos = eye_cx - 9 + int(self.gaze_x * 4)
                    gy_pos = ey - 9 + int(self.gaze_y * 3)
                    dd.ellipse((gx_pos - 5, gy_pos - 5, gx_pos + 5, gy_pos + 5), fill=(255, 255, 255, 245))
                    dd.ellipse((eye_cx + 8, ey + 7, eye_cx + 13, ey + 12), fill=(215, 245, 255, 190))

        # Warm human cheek glow when happy, hovered, or listening
        if self.emotion in ("happy", "listening") or self.is_hovered:
            cheek_color = (255, 135, 155, 78) if self.emotion == "happy" or self.is_hovered else (110, 220, 255, 72)
            dd.ellipse((lx - 36, my - 10, lx - 6, my + 10), fill=cheek_color)
            dd.ellipse((rx + 6, my - 10, rx + 36, my + 10), fill=cheek_color)

        # Dynamic metallic outer coating specular reflection sweep
        sheen_x = int(140 + (math.sin(self.sheen_phase) * 0.5 + 0.5) * 210)
        dd.arc((sheen_x - 70, 36, sheen_x + 70, 120), start=210, end=330, fill=(255, 255, 255, 95), width=4)

        if self.emotion == "listening":
            wave_rad = int(86 + math.sin(self.pulse_phase * 2.2) * 12)
            dd.arc(
                (cx - wave_rad, cy - wave_rad, cx + wave_rad, cy + wave_rad),
                start=165, end=375,
                fill=(*glow_rgb, int(85 + math.sin(self.pulse_phase * 3) * 40)),
                width=4
            )

        canvas.alpha_composite(detail_layer)
        return canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)


class HoverChatWidget:
    """
    Anthropic Metallic-Glass Floating Copilot Widget (PART C Overhaul).
    Features:
      - Brushed gunmetal metallic-glass coated card (#25272e -> #18191e -> #111216) with specular bevel
      - Removed '+' button; 3 separate independent pill boxes ([Chat] [Quick] [Droplets]) with smooth delayed hover
      - 3D depth terracotta Send button & modern studio condenser toggle Mic button (Alt+V / F8)
      - Anthropic editorial/UI font stack (Styrene A / Copernicus / Georgia / Segoe UI Variable Display)
      - Smooth animated Model Selector Popover matching reference image (Model header, ✓ checkmark,
        New/Notice pills, '>' chevrons, and 'Select Model  Ctrl+/' footer)
      - Humanized Robot Pet Mascot with reflective metallic outer coating
    """

    MODEL_CATALOG = [
        {
            "id": "kaggle_qwen3_27b",
            "label": "Qwen 3.8 27B (Kaggle)",
            "short_label": "Qwen 3.8 27B",
            "tier": "Medium",
            "badge": "New",
            "use_ngrok": "1",
            "model_env": "qwen3.8:27b",
        },
        {
            "id": "kaggle_qwen25_7b",
            "label": "Qwen 2.5 VL 7B (Kaggle)",
            "short_label": "Qwen 2.5 VL 7B",
            "tier": "Fast",
            "badge": "",
            "use_ngrok": "1",
            "model_env": "qwen2.5vl:7b",
        },
        {
            "id": "groq_qwen3_27b",
            "label": "Qwen 3.8 27B (Groq)",
            "short_label": "Qwen 3.8 Groq",
            "tier": "Medium",
            "badge": "New",
            "use_ngrok": "0",
            "model_env": "qwen/qwen3.8-27b",
        },
        {
            "id": "groq_gpt_oss_120b",
            "label": "GPT-OSS 120B (Groq)",
            "short_label": "GPT-OSS 120B",
            "tier": "High",
            "badge": "Notice",
            "use_ngrok": "0",
            "model_env": "openai/gpt-oss-120b",
        },
    ]

    def __init__(
        self,
        root: tk.Tk,
        on_submit_callback: Callable[[str], None],
        on_toggle_callback: Callable[[], None],
        on_refresh_callback: Callable[[], None],
        on_quick_preview_callback: Optional[Callable[[str], Optional[DiagonalDroplet]]] = None,
        on_quick_dispatch_callback: Optional[Callable[[DiagonalDroplet], None]] = None,
        on_abort_callback: Optional[Callable[[], None]] = None,
        on_toggle_droplets_callback: Optional[Callable[[], bool]] = None,
        on_voice_toggle_callback: Optional[Callable[[], None]] = None,
        on_compact_callback: Optional[Callable[[bool], None]] = None,
        on_model_change_callback: Optional[Callable[[bool, str, str], None]] = None,
    ):
        self.root = root
        self.on_submit = on_submit_callback
        self.on_toggle = on_toggle_callback
        self.on_refresh = on_refresh_callback
        self.on_quick_preview = on_quick_preview_callback
        self.on_quick_dispatch = on_quick_dispatch_callback
        self.on_abort = on_abort_callback
        self.on_toggle_droplets = on_toggle_droplets_callback
        self.on_voice_toggle = on_voice_toggle_callback
        self.on_compact_callback = on_compact_callback
        self.on_model_change = on_model_change_callback

        self.droplets_visible = True
        self.is_quick_mode = False
        self.is_expanded = True
        self.is_voice_recording = False
        self.is_submitting = False
        self.active_candidate = None
        self.current_status_text = "✔ Live Auto-Sync"
        self.current_status_color = "#3fb950"
        self.current_status_details: Optional[str] = None
        self.status_tooltip_win: Optional[tk.Toplevel] = None
        self._status_tip_after_id = None

        # Resolve Anthropic typography stack
        self.font_family_ui = _resolve_font_family(
            ["Styrene A", "Styrene B", "Segoe UI Variable Display", "Inter", "Segoe UI"]
        )
        self.font_family_editorial = _resolve_font_family(
            ["Copernicus", "Charter", "Tiempos Text", "Georgia", "Segoe UI Variable Text", "Segoe UI"]
        )
        self.font_family_mono = _resolve_font_family(
            ["JetBrains Mono", "Cascadia Mono", "Consolas"]
        )

        # Active model selection state
        self.selected_model_idx = self._detect_initial_model_index()
        self.active_model_name = self._format_trigger_text()
        self.model_popover: Optional[tk.Toplevel] = None
        self.is_popover_open = False
        self._popover_anim_job = None

        # Delayed hover animation state trackers
        self._hover_timers: Dict[str, Any] = {}
        self._hover_levels: Dict[str, int] = {
            "pill_chat": 0,
            "pill_quick": 0,
            "pill_drop": 0,
            "model_trigger": 0,
            "mic_btn": 0,
            "send_btn": 0,
        }

        # Dimensions
        self.win_w = 684
        self.win_h = 132
        self.compact_w = 160
        self.compact_h = 76

        # Mascot animator
        self.animator = RobotMascotAnimator(base_w=60, base_h=44)
        self.tk_mascot_frame = None

        # Toplevel floating window with Windows transparency colorkey
        self.win = tk.Toplevel(root)
        self.win.title("✥ SPATIAL DROPLET COPILOT")
        self.win.overrideredirect(True)
        self.win.lift()
        self.win.wm_attributes("-topmost", True)
        self.win.wm_attributes("-transparentcolor", "#010101")
        self.win.config(bg="#010101")

        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        self._hwnd = hwnd
        register_own_hwnd(self.win.winfo_id())
        register_own_hwnd(hwnd)

        # Protect against window closure and maintain persistent topmost heartbeat
        self.win.protocol("WM_DELETE_WINDOW", self.ensure_visible)
        self._topmost_tick_counter = 0

        # Thread-safe coordinates cache & locks (prevents background thread winfo_x race conditions)
        self._coord_lock = threading.Lock()
        self._cached_win_rect = (320, 40, 320 + self.win_w, 40 + self.win_h)
        self._cached_popover_rect: Optional[Tuple[int, int, int, int]] = None
        self._popover_hwnd: Optional[int] = None

        def _on_win_configure(event):
            if event.widget == self.win:
                try:
                    wx = self.win.winfo_x()
                    wy = self.win.winfo_y()
                    ww = self.win.winfo_width()
                    wh = self.win.winfo_height()
                    if ww > 50 and wh > 30 and not (wx == 0 and wy == 0 and ww == 200 and wh == 200):
                        with self._coord_lock:
                            self._cached_win_rect = (wx, wy, wx + ww, wy + wh)
                except Exception:
                    pass

        self.win.bind("<Configure>", _on_win_configure, add="+")

        # Initial topmost assertion
        self.ensure_visible()

        # Drag state
        self._drag_x = 0
        self._drag_y = 0

        # Pre-render metallic-glass cards, separate pill boxes, and 3D depth buttons
        self._generate_glass_backgrounds()
        self._generate_ui_component_sprites()

        # Initial geometry
        self.win.geometry(f"{self.win_w}x{self.win_h}+320+40")

        self.canvas = tk.Canvas(
            self.win, width=self.win_w, height=self.win_h,
            bg="#010101", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Bind Ctrl+/ for Model Selector Popover and F8 for Voice Toggle
        self.win.bind("<Control-slash>", lambda e: self.toggle_model_popover())
        self.win.bind("<F8>", lambda e: self._trigger_voice())

        self._render_expanded_ui()

        # Start 25 FPS animation loop
        self._anim_tick()

    def _detect_initial_model_index(self) -> int:
        if os.environ.get("USE_NGROK", "0") == "1":
            model = os.environ.get("KAGGLE_QWEN_MODEL", "qwen2.5vl:7b").lower()
            if "27b" in model or "3.8" in model:
                return 0
            return 1
        groq_m = os.environ.get("GROQ_PRIMARY_MODEL", "qwen/qwen3.8-27b").lower()
        if "120b" in groq_m or "gpt-oss" in groq_m:
            return 3
        return 2

    def _format_trigger_text(self) -> str:
        item = self.MODEL_CATALOG[self.selected_model_idx]
        return f"{item['short_label']}   {item['tier']}"

    def select_model_by_index(self, idx: int):
        """Applies model selection from popover, updates UI pill without overlap, and notifies live reasoner."""
        if not (0 <= idx < len(self.MODEL_CATALOG)):
            return
        self.selected_model_idx = idx
        item = self.MODEL_CATALOG[idx]
        os.environ["USE_NGROK"] = item["use_ngrok"]
        if item["use_ngrok"] == "1":
            os.environ["KAGGLE_QWEN_MODEL"] = item["model_env"]
        else:
            os.environ["GROQ_PRIMARY_MODEL"] = item["model_env"]

        self.active_model_name = self._format_trigger_text()
        if self.is_expanded and hasattr(self, "model_name_tag"):
            self.canvas.itemconfig(self.model_name_tag, text=item["short_label"])
            self.canvas.itemconfig(self.model_tier_tag, text=item["tier"])
            self._layout_model_capsule_items()
        if self.on_model_change:
            try:
                self.on_model_change(item["use_ngrok"] == "1", item["model_env"], item["label"])
            except Exception as e:
                print(f"[Model Switch Notice]: {e}")
        else:
            self.set_status(f"✔ Model set to {item['label']}", color="#3fb950")
        self._close_model_popover()

    def ensure_visible(self):
        """
        Guarantees the chat widget is visible, deiconified, topmost, and raised
        above all application windows and the full-screen HUD overlay.
        Prevents the chat widget from vanishing during multi-window switches or DWM compositing.
        """
        try:
            if not hasattr(self, "win") or not self.win.winfo_exists():
                return
            
            # 1. Deiconify if minimized or withdrawn
            try:
                if self.win.wm_state() in ("iconic", "withdrawn"):
                    self.win.deiconify()
            except Exception:
                pass

            # 2. Tkinter lift above all other toplevel windows
            self.win.lift()

            # 3. Win32 HWND_TOPMOST (-1) re-assertion with SWP_NOACTIVATE
            if hasattr(self, "_hwnd") and self._hwnd:
                user32 = ctypes.windll.user32
                # SWP_NOSIZE (1) | SWP_NOMOVE (2) | SWP_NOACTIVATE (16) | SWP_SHOWWINDOW (64)
                user32.SetWindowPos(
                    self._hwnd, -1, 0, 0, 0, 0,
                    0x0001 | 0x0002 | 0x0010 | 0x0040
                )
                # Ensure WS_EX_TOPMOST style flag is preserved
                GWL_EXSTYLE = -20
                WS_EX_TOPMOST = 0x00000008
                style = user32.GetWindowLongW(self._hwnd, GWL_EXSTYLE)
                if not (style & WS_EX_TOPMOST):
                    user32.SetWindowLongW(self._hwnd, GWL_EXSTYLE, style | WS_EX_TOPMOST)
        except Exception:
            pass

    def toggle_model_provider(self):
        """Compatibility method: opens/closes the smooth Model Selector Popover."""
        self.toggle_model_popover()

    # =========================================================================
    # METALLIC-GLASS SURFACE & 3D COMPONENT SPRITE GENERATION (PIL 3x SUPERSAMPLED)
    # =========================================================================

    def _generate_glass_backgrounds(self):
        """Pre-renders brushed gunmetal metallic-glass card backgrounds with specular top rim."""
        scale = 3

        def _make_metallic_card(w: int, h: int, r: int):
            W, H = w * scale, h * scale
            # Create brushed gunmetal vertical gradient + diagonal metallic sheen
            yy, xx = np.mgrid[0:H, 0:W]
            ny = yy / max(1.0, float(H))
            nx = xx / max(1.0, float(W))

            # Base gunmetal gradient: top #262830 -> mid #191b20 -> bottom #121317
            sheen = 0.08 * np.exp(-(((nx * 0.65 + ny * 0.35) - 0.36) ** 2) / 0.055)
            r_ch = np.clip((38 * (1 - ny) + 18 * ny) + sheen * 55, 0, 255).astype(np.uint8)
            g_ch = np.clip((40 * (1 - ny) + 19 * ny) + sheen * 58, 0, 255).astype(np.uint8)
            b_ch = np.clip((48 * (1 - ny) + 24 * ny) + sheen * 66, 0, 255).astype(np.uint8)

            # Rounded rectangle alpha mask
            mask = Image.new("L", (W, H), 0)
            md = ImageDraw.Draw(mask)
            pad = int(2 * scale)
            md.rounded_rectangle((pad, pad, W - pad - 1, H - pad - 1), radius=r * scale, fill=253)

            rgba = np.stack([r_ch, g_ch, b_ch, np.array(mask)], axis=-1)
            im = Image.fromarray(rgba, "RGBA")
            d = ImageDraw.Draw(im)

            # Outer anodized metallic rim
            d.rounded_rectangle(
                (pad, pad, W - pad - 1, H - pad - 1),
                radius=r * scale,
                fill=None,
                outline=(68, 72, 84, 255),
                width=int(1.4 * scale)
            )
            # Inner subtle dark shadow rim
            d.rounded_rectangle(
                (pad + scale, pad + scale, W - pad - scale - 1, H - pad - scale - 1),
                radius=max(2, (r - 1) * scale),
                fill=None,
                outline=(14, 15, 18, 190),
                width=scale
            )
            # Top glass specular refraction highlight arc
            d.arc(
                (int(12 * scale), int(2.5 * scale), W - int(12 * scale), int(26 * scale)),
                start=196, end=344,
                fill=(255, 255, 255, 58),
                width=int(1.4 * scale)
            )
            return ImageTk.PhotoImage(im.resize((w, h), Image.Resampling.LANCZOS))

        try:
            self.tk_bg_expanded = _make_metallic_card(self.win_w, self.win_h, 18)
            self.tk_bg_compact = _make_metallic_card(self.compact_w, self.compact_h, 18)
        except Exception as e:
            print(f"[!] Warning: Could not pre-render metallic-glass backgrounds: {e}")

    def _make_pill_sprite(
        self,
        w: int,
        h: int,
        r: int,
        top_rgb: Tuple[int, int, int],
        bot_rgb: Tuple[int, int, int],
        border_rgb: Tuple[int, int, int],
        top_bevel_alpha: int = 42,
    ) -> ImageTk.PhotoImage:
        scale = 3
        W, H = w * scale, h * scale
        yy, _ = np.mgrid[0:H, 0:W]
        ny = yy / max(1.0, float(H))
        r_ch = (top_rgb[0] * (1 - ny) + bot_rgb[0] * ny).astype(np.uint8)
        g_ch = (top_rgb[1] * (1 - ny) + bot_rgb[1] * ny).astype(np.uint8)
        b_ch = (top_rgb[2] * (1 - ny) + bot_rgb[2] * ny).astype(np.uint8)

        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((scale, scale, W - scale - 1, H - scale - 1), radius=r * scale, fill=255)

        im = Image.fromarray(np.stack([r_ch, g_ch, b_ch, np.array(mask)], axis=-1), "RGBA")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle(
            (scale, scale, W - scale - 1, H - scale - 1),
            radius=r * scale,
            fill=None,
            outline=(*border_rgb, 255),
            width=int(1.2 * scale),
        )
        if top_bevel_alpha > 0:
            d.line(
                (int((r + 2) * scale), int(2.2 * scale), W - int((r + 2) * scale), int(2.2 * scale)),
                fill=(255, 255, 255, top_bevel_alpha),
                width=scale,
            )
        return ImageTk.PhotoImage(im.resize((w, h), Image.Resampling.LANCZOS))

    def _make_send_button_sprite(self, hover_t: float = 0.0, busy: bool = False) -> ImageTk.PhotoImage:
        """Renders a modern 30x30 3D depth terracotta Send button with crisp rounded-cap upward arrow glyph."""
        w, h, r, scale = 30, 30, 9, 3
        W, H = w * scale, h * scale
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)

        # 1. Outer ambient drop shadow for 3D elevation
        d.rounded_rectangle(
            (int(1.5 * scale), int(3.0 * scale), W - int(1.5 * scale), H - int(0.5 * scale)),
            radius=r * scale,
            fill=(6, 7, 10, 155),
        )

        # 2. Multi-stop terracotta metallic gradient body
        top_c = _lerp_color((226, 109, 80), (244, 128, 98), hover_t)
        bot_c = _lerp_color((186, 74, 48), (208, 88, 60), hover_t)
        border_c = _lerp_color((242, 136, 110), (255, 165, 140), hover_t)

        yy, _ = np.mgrid[0:H, 0:W]
        ny = yy / max(1.0, float(H))
        r_ch = (top_c[0] * (1 - ny) + bot_c[0] * ny).astype(np.uint8)
        g_ch = (top_c[1] * (1 - ny) + bot_c[1] * ny).astype(np.uint8)
        b_ch = (top_c[2] * (1 - ny) + bot_c[2] * ny).astype(np.uint8)

        bmask = Image.new("L", (W, H), 0)
        bmd = ImageDraw.Draw(bmask)
        pad = int(1.5 * scale)
        bmd.rounded_rectangle((pad, pad, W - pad - 1, H - int(2.2 * scale)), radius=r * scale, fill=255)
        body = Image.fromarray(np.stack([r_ch, g_ch, b_ch, np.array(bmask)], axis=-1), "RGBA")
        im.alpha_composite(body)

        d = ImageDraw.Draw(im)
        # Outer rim + top inner 3D specular bevel
        d.rounded_rectangle(
            (pad, pad, W - pad - 1, H - int(2.2 * scale)),
            radius=r * scale,
            outline=(*border_c, 255),
            width=int(1.2 * scale),
        )
        d.line(
            (int(7 * scale), int(3 * scale), W - int(7 * scale), int(3 * scale)),
            fill=(255, 255, 255, int(90 + 40 * hover_t)),
            width=int(1.2 * scale),
        )

        cx, cy = W // 2, int(H * 0.47)
        if busy:
            # Clean rounded stop/busy square icon while running a task (no overlapping text!)
            sq = int(4.2 * scale)
            d.rounded_rectangle(
                (cx - sq, cy - sq, cx + sq, cy + sq),
                radius=int(1.5 * scale),
                fill=(255, 255, 255, 255),
            )
        else:
            # Crisp vector upward arrow with rounded caps and joints
            stem_top = cy - int(5.4 * scale)
            stem_bot = cy + int(5.8 * scale)
            wing_w = int(4.6 * scale)
            wing_y = cy - int(0.8 * scale)
            stroke = int(2.2 * scale)
            rad = stroke // 2
            d.line((cx, stem_bot, cx, stem_top), fill=(255, 255, 255, 255), width=stroke)
            d.line(
                [(cx - wing_w, wing_y), (cx, stem_top), (cx + wing_w, wing_y)],
                fill=(255, 255, 255, 255),
                width=stroke,
                joint="curve",
            )
            for px, py in ((cx, stem_bot), (cx, stem_top), (cx - wing_w, wing_y), (cx + wing_w, wing_y)):
                d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=(255, 255, 255, 255))

        return ImageTk.PhotoImage(im.resize((w, h), Image.Resampling.LANCZOS))

    def _make_mic_button_sprite(self, hover_t: float = 0.0, recording: bool = False) -> ImageTk.PhotoImage:
        """Renders a sleek 30x30 metallic-glass studio condenser microphone button."""
        w, h, r, scale = 30, 30, 9, 3
        W, H = w * scale, h * scale
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        if recording:
            top_c = (72, 22, 26)
            bot_c = (46, 14, 18)
            border_c = (239, 68, 68)
            icon_c = (255, 115, 115, 255)
        else:
            top_c = _lerp_color((36, 39, 46), (52, 56, 66), hover_t)
            bot_c = _lerp_color((25, 27, 32), (38, 41, 49), hover_t)
            border_c = _lerp_color((58, 62, 74), (92, 98, 116), hover_t)
            ic_v = int(175 + 75 * hover_t)
            icon_c = (ic_v, ic_v + 2, min(255, ic_v + 6), 255)

        yy, _ = np.mgrid[0:H, 0:W]
        ny = yy / max(1.0, float(H))
        r_ch = (top_c[0] * (1 - ny) + bot_c[0] * ny).astype(np.uint8)
        g_ch = (top_c[1] * (1 - ny) + bot_c[1] * ny).astype(np.uint8)
        b_ch = (top_c[2] * (1 - ny) + bot_c[2] * ny).astype(np.uint8)

        bmask = Image.new("L", (W, H), 0)
        bmd = ImageDraw.Draw(bmask)
        pad = int(1.5 * scale)
        bmd.rounded_rectangle((pad, pad, W - pad - 1, H - int(2.2 * scale)), radius=r * scale, fill=255)
        body = Image.fromarray(np.stack([r_ch, g_ch, b_ch, np.array(bmask)], axis=-1), "RGBA")
        im.alpha_composite(body)

        d = ImageDraw.Draw(im)
        d.rounded_rectangle(
            (pad, pad, W - pad - 1, H - int(2.2 * scale)),
            radius=r * scale,
            outline=(*border_c, 255),
            width=int(1.2 * scale),
        )
        d.line(
            (int(7 * scale), int(3 * scale), W - int(7 * scale), int(3 * scale)),
            fill=(255, 255, 255, 45),
            width=scale,
        )

        # Studio condenser capsule + U-cradle + base
        cx, cy = W // 2, int(H * 0.47)
        cap_w, cap_h = int(2.8 * scale), int(5.2 * scale)
        d.rounded_rectangle(
            (cx - cap_w, cy - cap_h, cx + cap_w, cy + int(1.2 * scale)),
            radius=cap_w,
            fill=icon_c,
        )
        cradle_r = int(5.4 * scale)
        d.arc(
            (cx - cradle_r, cy - int(3.2 * scale), cx + cradle_r, cy + int(4.6 * scale)),
            start=0, end=180,
            fill=icon_c,
            width=int(1.7 * scale),
        )
        d.line((cx, cy + int(4.6 * scale), cx, cy + int(7.4 * scale)), fill=icon_c, width=int(1.7 * scale))
        d.line(
            (cx - int(3.4 * scale), cy + int(7.4 * scale), cx + int(3.4 * scale), cy + int(7.4 * scale)),
            fill=icon_c,
            width=int(1.7 * scale),
        )
        return ImageTk.PhotoImage(im.resize((w, h), Image.Resampling.LANCZOS))

    def _generate_ui_component_sprites(self):
        """Pre-renders multi-step hover & active sprites for the 3 separate pills, model trigger, mic, and send buttons."""
        self.pill_sprites: Dict[str, List[ImageTk.PhotoImage]] = {}
        specs = {
            "pill_chat": (58, 8),
            "pill_quick": (62, 8),
            "pill_drop": (74, 8),
            "model_trigger": (208, 13),
        }
        for key, (width, rad) in specs.items():
            frames = []
            for step in range(4):
                t = step / 3.0
                top_c = _lerp_color((34, 36, 43), (52, 56, 66), t)
                bot_c = _lerp_color((24, 26, 31), (38, 41, 49), t)
                bdr_c = _lerp_color((52, 56, 66), (88, 94, 110), t)
                frames.append(self._make_pill_sprite(width, 26, rad, top_c, bot_c, bdr_c, int(32 + 30 * t)))
            self.pill_sprites[key] = frames

        # Dedicated active state sprites for Chat, Quick, and Droplets separate boxes
        self.pill_active_chat = self._make_pill_sprite(58, 26, 8, (54, 58, 69), (38, 41, 50), (105, 112, 130), 68)
        self.pill_active_quick = self._make_pill_sprite(62, 26, 8, (68, 52, 26), (46, 35, 18), (215, 145, 42), 68)
        self.pill_active_drop = self._make_pill_sprite(74, 26, 8, (26, 52, 70), (18, 36, 50), (56, 165, 225), 68)

        # Send & Mic button hover ladders
        self.send_sprites = [self._make_send_button_sprite(step / 3.0, busy=False) for step in range(4)]
        self.send_busy_sprite = self._make_send_button_sprite(0.5, busy=True)
        self.mic_sprites = [self._make_mic_button_sprite(step / 3.0, recording=False) for step in range(4)]
        self.mic_recording_sprite = self._make_mic_button_sprite(1.0, recording=True)

    # =========================================================================
    # SMOOTH DELAYED HOVER ENGINE
    # =========================================================================

    def _schedule_smooth_hover(self, key: str, entering: bool, update_fn: Callable[[int], None]):
        """Applies a 70ms intentional hover delay and smoothly interpolates across 4 frames."""
        if key in self._hover_timers and self._hover_timers[key]:
            try:
                self.win.after_cancel(self._hover_timers[key])
            except Exception:
                pass
            self._hover_timers[key] = None

        target_level = 3 if entering else 0
        delay_ms = 68 if entering else 25

        def _step():
            cur = self._hover_levels.get(key, 0)
            if cur == target_level:
                self._hover_timers[key] = None
                return
            nxt = cur + 1 if cur < target_level else cur - 1
            self._hover_levels[key] = nxt
            try:
                update_fn(nxt)
            except Exception:
                pass
            if nxt != target_level and hasattr(self, "win") and self.win.winfo_exists():
                self._hover_timers[key] = self.win.after(18, _step)
            else:
                self._hover_timers[key] = None

        if hasattr(self, "win") and self.win.winfo_exists():
            self._hover_timers[key] = self.win.after(delay_ms, _step)

    # =========================================================================
    # ANIMATION LOOP (25 FPS)
    # =========================================================================

    def _anim_tick(self):
        try:
            if hasattr(self, "animator") and hasattr(self, "canvas"):
                self.animator.tick()
                mw = 60 if self.is_expanded else 56
                mh = 44 if self.is_expanded else 42
                frame_img = self.animator.render_frame(target_w=mw, target_h=mh)
                self.tk_mascot_frame = ImageTk.PhotoImage(frame_img)

                if hasattr(self, "mascot_tag") and self.mascot_tag:
                    self.canvas.itemconfig(self.mascot_tag, image=self.tk_mascot_frame)

                if hasattr(self, "compact_mic_border") and self.compact_mic_border and self.is_voice_recording:
                    alpha_pulse = int(180 + math.sin(self.animator.pulse_phase * 2) * 75)
                    self.canvas.itemconfig(self.compact_mic_border, outline=f"#{alpha_pulse:02x}2020")

            # Periodic Topmost Heartbeat Sentinel:
            # Re-assert topmost every 25 ticks (~1 sec) or immediately if state dropped from normal
            self._topmost_tick_counter = getattr(self, "_topmost_tick_counter", 0) + 1
            if self._topmost_tick_counter >= 25:
                self._topmost_tick_counter = 0
                self.ensure_visible()
            elif hasattr(self, "win"):
                try:
                    if self.win.winfo_exists() and self.win.wm_state() != "normal":
                        self.ensure_visible()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                if hasattr(self, "win") and self.win.winfo_exists():
                    self.win.after(40, self._anim_tick)
            except Exception:
                pass

    # =========================================================================
    # EXPANDED METALLIC-GLASS CHAT BAR
    # =========================================================================

    def _render_expanded_ui(self):
        prev_text = ""
        if hasattr(self, "entry") and self.entry:
            try:
                prev_text = self.entry.get()
            except Exception:
                prev_text = ""

        if hasattr(self, "entry_frame") and self.entry_frame:
            try:
                self.entry_frame.destroy()
            except Exception:
                pass
            self.entry_frame = None
            self.entry = None

        self.canvas.delete("all")

        # 1. Metallic-Glass Card Background
        if hasattr(self, "tk_bg_expanded") and self.tk_bg_expanded:
            self.bg_tag = self.canvas.create_image(self.win_w // 2, self.win_h // 2, image=self.tk_bg_expanded)
        else:
            self.bg_tag = self.canvas.create_rectangle(
                0, 0, self.win_w, self.win_h, fill="#191b20", outline="#444854", width=2
            )

        self.canvas.tag_bind(self.bg_tag, "<ButtonPress-1>", self._start_drag)
        self.canvas.tag_bind(self.bg_tag, "<B1-Motion>", self._do_drag)

        # 2. Top Tier: Prompt Input Entry Area (Anthropic Typography)
        self.entry_frame = tk.Frame(self.canvas, bg="#1e2026", padx=0, pady=0)
        self.placeholder_text = "Ask Copilot or type Dot ID (e.g. 5, click 12)..."
        self.entry = tk.Entry(
            self.entry_frame,
            bg="#1e2026",
            fg="#f4f3ef",
            insertbackground="#e06c4f",
            selectbackground="#3a3f4d",
            relief="flat",
            font=(self.font_family_editorial, 11),
            highlightthickness=0,
            bd=0,
        )
        self.entry.pack(fill="both", expand=True, ipady=3)
        self.entry.bind("<Return>", self._handle_submit)
        self.entry.bind("<KeyRelease>", self._handle_key_release)
        self.entry.bind("<Escape>", self._handle_escape)
        self.entry.bind("<Control-slash>", lambda e: self.toggle_model_popover())
        self.entry.bind("<F8>", lambda e: self._trigger_voice())

        if prev_text:
            self.entry.insert(0, prev_text)
        self._is_placeholder = False

        self.canvas.create_window(
            20, 15, window=self.entry_frame, anchor="nw", width=640, height=30
        )
        self.entry.focus_set()

        # 3. Middle Tier: Dedicated Real-Time Reasoning & Status Banner
        self.status_bg_tag = self.canvas.create_rectangle(
            20, 52, 660, 78, fill="#13151a", outline="#2b2f3a", width=1
        )
        self.canvas.tag_bind(self.status_bg_tag, "<ButtonPress-1>", self._start_drag)
        self.canvas.tag_bind(self.status_bg_tag, "<B1-Motion>", self._do_drag)
        self.canvas.tag_bind(self.status_bg_tag, "<Enter>", self._on_status_enter)
        self.canvas.tag_bind(self.status_bg_tag, "<Leave>", self._on_status_leave)

        cur_col = getattr(self, "current_status_color", "#3fb950")
        cur_txt = getattr(self, "current_status_text", "✔ Live Auto-Sync")
        self.status_dot_tag = self.canvas.create_oval(32, 61, 40, 69, fill=cur_col, outline="")
        self.canvas.tag_bind(self.status_dot_tag, "<Enter>", self._on_status_enter)
        self.canvas.tag_bind(self.status_dot_tag, "<Leave>", self._on_status_leave)

        self.status_text_tag = self.canvas.create_text(
            48, 65,
            text=cur_txt,
            fill=cur_col,
            font=(self.font_family_ui, 9),
            anchor="w"
        )
        self.canvas.tag_bind(self.status_text_tag, "<ButtonPress-1>", self._start_drag)
        self.canvas.tag_bind(self.status_text_tag, "<B1-Motion>", self._do_drag)
        self.canvas.tag_bind(self.status_text_tag, "<Enter>", self._on_status_enter)
        self.canvas.tag_bind(self.status_text_tag, "<Leave>", self._on_status_leave)

        # 4. Bottom Tier Left: Three Separate Pill Boxes ([Chat]  [Quick]  [Droplets]) — '+' Removed!
        self._render_separate_mode_pills(x_start=18, y=90)

        # 5. Bottom Tier Right: Clean Non-Overlapping Model Selector Capsule Trigger
        self._render_model_selector_capsule(x=300, y=90)

        # 6. Modern Studio Condenser Toggle Mic Button (x=518, y=88)
        self._render_mic_button(x=518, y=88)

        # 7. Modern 3D Depth Terracotta Send Button (x=556, y=88)
        self._render_terracotta_submit_button(x=556, y=88)

        # 8. Humanized Robot Mascot with Reflective Metallic Outer Coating (docked far right)
        if not self.tk_mascot_frame:
            frame_img = self.animator.render_frame(target_w=60, target_h=44)
            self.tk_mascot_frame = ImageTk.PhotoImage(frame_img)

        self.mascot_tag = self.canvas.create_image(634, 101, image=self.tk_mascot_frame)
        self.canvas.tag_bind(self.mascot_tag, "<Button-1>", lambda e: self.collapse_to_orb())
        self.canvas.tag_bind(self.mascot_tag, "<Enter>", lambda e: self._on_mascot_hover(True))
        self.canvas.tag_bind(self.mascot_tag, "<Leave>", lambda e: self._on_mascot_hover(False))

    def _render_separate_mode_pills(self, x_start: int, y: int):
        """Renders [Chat], [Quick], and [Droplets] as 3 separate metallic-glass pill boxes with smooth delayed hover."""
        cy = y + 13
        # --- Pill 1: [Chat] (x=18, w=58) ---
        x_chat, w_chat = x_start, 58
        chat_active = not self.is_quick_mode
        chat_img = self.pill_active_chat if chat_active else self.pill_sprites["pill_chat"][0]
        self.pill_chat_bg = self.canvas.create_image(x_chat + w_chat // 2, cy, image=chat_img)
        self.pill_chat_txt = self.canvas.create_text(
            x_chat + w_chat // 2, cy,
            text="Chat",
            fill="#f4f3ef" if chat_active else "#9ca0ab",
            font=(self.font_family_ui, 9, "bold" if chat_active else "normal"),
        )

        def _upd_chat(lvl: int):
            if not self.is_quick_mode:
                self.canvas.itemconfig(self.pill_chat_bg, image=self.pill_active_chat)
                self.canvas.itemconfig(self.pill_chat_txt, fill="#ffffff")
            else:
                self.canvas.itemconfig(self.pill_chat_bg, image=self.pill_sprites["pill_chat"][lvl])
                self.canvas.itemconfig(self.pill_chat_txt, fill="#e5e7eb" if lvl > 0 else "#9ca0ab")

        for tag in (self.pill_chat_bg, self.pill_chat_txt):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self.deactivate_quick_select())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("pill_chat", True, _upd_chat)])
            self.canvas.tag_bind(tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("pill_chat", False, _upd_chat)])

        # --- Pill 2: [Quick] (x=84, w=62) ---
        x_quick, w_quick = x_chat + w_chat + 8, 62
        quick_active = self.is_quick_mode
        quick_img = self.pill_active_quick if quick_active else self.pill_sprites["pill_quick"][0]
        self.pill_quick_bg = self.canvas.create_image(x_quick + w_quick // 2, cy, image=quick_img)
        self.pill_quick_txt = self.canvas.create_text(
            x_quick + w_quick // 2, cy,
            text="Quick",
            fill="#fbbf24" if quick_active else "#9ca0ab",
            font=(self.font_family_ui, 9, "bold" if quick_active else "normal"),
        )

        def _upd_quick(lvl: int):
            if self.is_quick_mode:
                self.canvas.itemconfig(self.pill_quick_bg, image=self.pill_active_quick)
                self.canvas.itemconfig(self.pill_quick_txt, fill="#fcd34d")
            else:
                self.canvas.itemconfig(self.pill_quick_bg, image=self.pill_sprites["pill_quick"][lvl])
                self.canvas.itemconfig(self.pill_quick_txt, fill="#f3f4f6" if lvl > 0 else "#9ca0ab")

        for tag in (self.pill_quick_bg, self.pill_quick_txt):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self.activate_quick_select())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("pill_quick", True, _upd_quick)])
            self.canvas.tag_bind(tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("pill_quick", False, _upd_quick)])

        # --- Pill 3: [Droplets] (x=154, w=74) ---
        x_drop, w_drop = x_quick + w_quick + 8, 74
        drop_active = self.droplets_visible
        drop_img = self.pill_active_drop if drop_active else self.pill_sprites["pill_drop"][0]
        self.pill_drop_bg = self.canvas.create_image(x_drop + w_drop // 2, cy, image=drop_img)
        self.pill_drop_txt = self.canvas.create_text(
            x_drop + w_drop // 2, cy,
            text="Droplets",
            fill="#7dd3fc" if drop_active else "#9ca0ab",
            font=(self.font_family_ui, 9, "bold" if drop_active else "normal"),
        )

        def _upd_drop(lvl: int):
            if self.droplets_visible and lvl == 0:
                self.canvas.itemconfig(self.pill_drop_bg, image=self.pill_active_drop)
                self.canvas.itemconfig(self.pill_drop_txt, fill="#7dd3fc")
            elif self.droplets_visible:
                self.canvas.itemconfig(self.pill_drop_bg, image=self.pill_active_drop)
                self.canvas.itemconfig(self.pill_drop_txt, fill="#bae6fd")
            else:
                self.canvas.itemconfig(self.pill_drop_bg, image=self.pill_sprites["pill_drop"][lvl])
                self.canvas.itemconfig(self.pill_drop_txt, fill="#e5e7eb" if lvl > 0 else "#9ca0ab")

        for tag in (self.pill_drop_bg, self.pill_drop_txt):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self.toggle_droplets())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("pill_drop", True, _upd_drop)])
            self.canvas.tag_bind(tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("pill_drop", False, _upd_drop)])

    def _layout_model_capsule_items(self):
        """Positions Model Name, Tier ('Medium'/'Fast'/'High'), and Vector Chevron side-by-side without overlapping."""
        if not hasattr(self, "model_name_tag") or not hasattr(self, "model_tier_tag"):
            return
        x = getattr(self, "_model_capsule_x", 300)
        y = getattr(self, "_model_capsule_y", 55)
        w = 208
        cy = y + 13

        # Measure rendered bounding box of model_name_tag so tier sits 7px to its right (matching reference image!)
        name_box = self.canvas.bbox(self.model_name_tag)
        if name_box:
            tier_x = min(name_box[2] + 7, x + w - 64)
        else:
            tier_x = x + 104
        self.canvas.coords(self.model_tier_tag, tier_x, cy)

        # Update vector chevron (up '^' when popover open, down 'v' when closed — never a missing glyph box!)
        chev_cx = x + w - 14
        if hasattr(self, "model_chevron_tag") and self.model_chevron_tag:
            if self.is_popover_open:
                self.canvas.coords(self.model_chevron_tag, chev_cx - 4, cy + 2, chev_cx, cy - 2, chev_cx + 4, cy + 2)
            else:
                self.canvas.coords(self.model_chevron_tag, chev_cx - 4, cy - 2, chev_cx, cy + 2, chev_cx + 4, cy - 2)

    def _render_model_selector_capsule(self, x: int, y: int):
        """Renders the reference-style Model Selector trigger capsule ('Model Name  Medium  ^') with vector chevron."""
        self._model_capsule_x = x
        self._model_capsule_y = y
        w = 208
        cy = y + 13
        item = self.MODEL_CATALOG[self.selected_model_idx]

        self.model_capsule_bg = self.canvas.create_image(
            x + w // 2, cy, image=self.pill_sprites["model_trigger"][0]
        )
        self.model_name_tag = self.canvas.create_text(
            x + 14, cy,
            text=item["short_label"],
            anchor="w",
            fill="#d8dae0",
            font=(self.font_family_ui, 9),
        )
        self.model_tier_tag = self.canvas.create_text(
            x + 110, cy,
            text=item["tier"],
            anchor="w",
            fill="#82858e",
            font=(self.font_family_ui, 9),
        )
        chev_cx = x + w - 14
        chev_pts = (
            (chev_cx - 4, cy + 2, chev_cx, cy - 2, chev_cx + 4, cy + 2)
            if self.is_popover_open
            else (chev_cx - 4, cy - 2, chev_cx, cy + 2, chev_cx + 4, cy - 2)
        )
        self.model_chevron_tag = self.canvas.create_line(
            *chev_pts,
            fill="#9ca0ab",
            width=1.7,
            capstyle="round",
            joinstyle="round",
        )
        self._layout_model_capsule_items()
        # Backward compatibility alias
        self.model_text_tag = self.model_name_tag

        def _upd_model(lvl: int):
            self.canvas.itemconfig(self.model_capsule_bg, image=self.pill_sprites["model_trigger"][lvl])
            self.canvas.itemconfig(self.model_name_tag, fill="#ffffff" if lvl > 0 else "#d8dae0")
            self.canvas.itemconfig(self.model_tier_tag, fill="#a6a9b2" if lvl > 0 else "#82858e")
            self.canvas.itemconfig(self.model_chevron_tag, fill="#ffffff" if lvl > 0 else "#9ca0ab")

        for tag in (self.model_capsule_bg, self.model_name_tag, self.model_tier_tag, self.model_chevron_tag):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self.toggle_model_popover())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("model_trigger", True, _upd_model)])
            self.canvas.tag_bind(tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("model_trigger", False, _upd_model)])

    def _render_mic_button(self, x: int, y: int):
        """Renders the 30x30 studio condenser toggle Mic button (Alt+V / F8, 1-click toggle)."""
        cx, cy = x + 15, y + 15
        sprite = self.mic_recording_sprite if self.is_voice_recording else self.mic_sprites[0]
        self.btn_mic_tag = self.canvas.create_image(cx, cy, image=sprite)

        def _upd_mic(lvl: int):
            if self.is_voice_recording:
                self.canvas.itemconfig(self.btn_mic_tag, image=self.mic_recording_sprite)
            else:
                self.canvas.itemconfig(self.btn_mic_tag, image=self.mic_sprites[lvl])

        self.canvas.tag_bind(self.btn_mic_tag, "<Button-1>", lambda e: self._trigger_voice())
        self.canvas.tag_bind(self.btn_mic_tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("mic_btn", True, _upd_mic)])
        self.canvas.tag_bind(self.btn_mic_tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("mic_btn", False, _upd_mic)])

    def _render_terracotta_submit_button(self, x: int, y: int):
        """Renders the 30x30 3D depth terracotta Send button with smooth delayed transition."""
        cx, cy = x + 15, y + 15
        init_sprite = self.send_busy_sprite if self.is_submitting else self.send_sprites[0]
        self.btn_send_bg = self.canvas.create_image(cx, cy, image=init_sprite)
        self.btn_send_glyph = self.canvas.create_text(
            cx, cy, text="", fill="#ffffff", font=(self.font_family_ui, 10, "bold")
        )

        def _upd_send(lvl: int):
            if self.is_submitting:
                self.canvas.itemconfig(self.btn_send_bg, image=self.send_busy_sprite)
            else:
                self.canvas.itemconfig(self.btn_send_bg, image=self.send_sprites[lvl])

        for tag in (self.btn_send_bg, self.btn_send_glyph):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self._handle_submit())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: [self.canvas.config(cursor="hand2"), self._schedule_smooth_hover("send_btn", True, _upd_send)])
            self.canvas.tag_bind(tag, "<Leave>", lambda e: [self.canvas.config(cursor=""), self._schedule_smooth_hover("send_btn", False, _upd_send)])

    def _on_mascot_hover(self, is_hover: bool):
        self.animator.set_hover(is_hover)
        self.canvas.config(cursor="hand2" if is_hover else "")

    # =========================================================================
    # SMOOTH ANIMATED MODEL SELECTOR POPOVER (MATCHING REFERENCE IMAGE)
    # =========================================================================

    def toggle_model_popover(self):
        if self.is_popover_open:
            self._close_model_popover()
        else:
            self._open_model_popover()

    def _open_model_popover(self):
        try:
            if self.is_popover_open and self.model_popover and self.model_popover.winfo_exists():
                return
        except Exception:
            pass
        self.is_popover_open = True
        self._layout_model_capsule_items()

        pop_w = 286
        row_h = 34
        pop_h = 40 + len(self.MODEL_CATALOG) * row_h + 44

        # Safe coordinates acquisition from cache or HWND
        wx, wy, _, _ = self.get_rect(include_popover=False)
        target_x = wx + 286
        # Position cleanly below if chat bar is near top of screen, otherwise above
        open_below = (wy < pop_h + 18)
        target_y = (wy + self.win_h + 6) if open_below else (wy - pop_h - 6)
        start_y = target_y - 10 if open_below else target_y + 10

        with self._coord_lock:
            self._cached_popover_rect = (target_x, start_y, target_x + pop_w, start_y + pop_h)

        pop = tk.Toplevel(self.win)
        self.model_popover = pop
        pop.overrideredirect(True)
        pop.wm_attributes("-topmost", True)
        pop.wm_attributes("-transparentcolor", "#010101")
        pop.wm_attributes("-alpha", 0.0)
        pop.config(bg="#010101")
        pop.geometry(f"{pop_w}x{pop_h}+{target_x}+{start_y}")

        try:
            user32 = ctypes.windll.user32
            phwnd = user32.GetParent(pop.winfo_id()) or pop.winfo_id()
            register_own_hwnd(pop.winfo_id())
            register_own_hwnd(phwnd)
            self._popover_hwnd = phwnd
        except Exception:
            self._popover_hwnd = None

        # Render dark metallic-glass popover surface
        scale = 3
        W, H = pop_w * scale, pop_h * scale
        im = Image.new("RGBA", (W, H), (1, 1, 1, 0))
        d = ImageDraw.Draw(im)
        pad = int(2 * scale)
        d.rounded_rectangle(
            (pad, pad, W - pad - 1, H - pad - 1),
            radius=14 * scale,
            fill=(26, 27, 32, 254),
            outline=(58, 62, 74, 255),
            width=int(1.4 * scale),
        )
        d.arc(
            (int(12 * scale), int(2.5 * scale), W - int(12 * scale), int(22 * scale)),
            start=198, end=342,
            fill=(255, 255, 255, 42),
            width=scale,
        )
        self._tk_popover_bg = ImageTk.PhotoImage(im.resize((pop_w, pop_h), Image.Resampling.LANCZOS))
        self._tk_row_hover = self._make_pill_sprite(pop_w - 16, row_h - 2, 7, (46, 49, 58), (40, 43, 52), (68, 72, 85), 20)
        self._tk_badge_pill = self._make_pill_sprite(44, 18, 9, (48, 52, 62), (38, 41, 50), (74, 79, 94), 15)
        self._tk_footer_pill = self._make_pill_sprite(146, 26, 7, (32, 34, 40), (26, 28, 33), (58, 62, 72), 12)

        pcan = tk.Canvas(pop, width=pop_w, height=pop_h, bg="#010101", highlightthickness=0)
        pcan.pack(fill="both", expand=True)
        pcan.create_image(pop_w // 2, pop_h // 2, image=self._tk_popover_bg)

        # Header: "Model"
        pcan.create_text(
            18, 22, text="Model", anchor="w", fill="#8b8d93", font=(self.font_family_ui, 9, "bold")
        )

        # Model Rows (matching reference image: ModelName + Tier + New/Notice pill + right-side vector ✓ or >)
        for idx, item in enumerate(self.MODEL_CATALOG):
            ry = 38 + idx * row_h
            rcy = ry + row_h // 2
            hover_img_tag = pcan.create_image(pop_w // 2, rcy, image=self._tk_row_hover, state="hidden")

            is_sel = (idx == self.selected_model_idx)

            lbl_tag = pcan.create_text(
                18, rcy,
                text=item["label"],
                anchor="w",
                fill="#ffffff" if is_sel else "#d4d6dc",
                font=(self.font_family_ui, 9, "bold" if is_sel else "normal"),
            )
            lbl_box = pcan.bbox(lbl_tag)
            tier_x = (lbl_box[2] + 7) if lbl_box else 150

            tier_tag = pcan.create_text(
                tier_x, rcy,
                text=item["tier"],
                anchor="w",
                fill="#80838b",
                font=(self.font_family_ui, 9),
            )
            tier_box = pcan.bbox(tier_tag)
            badge_x = (tier_box[2] + 28) if tier_box else (pop_w - 58)

            badge_tags = []
            if item.get("badge"):
                bx = min(badge_x, pop_w - 54)
                b_bg = pcan.create_image(bx, rcy, image=self._tk_badge_pill)
                b_txt = pcan.create_text(bx, rcy, text=item["badge"], fill="#d4d6dc", font=(self.font_family_ui, 8, "bold"))
                badge_tags = [b_bg, b_txt]

            # Crisp vector checkmark ✓ (when selected) or vector right chevron > (when unselected) on far right
            rx = pop_w - 20
            if is_sel:
                ind_tag = pcan.create_line(
                    rx - 5, rcy, rx - 1, rcy + 4, rx + 5, rcy - 4,
                    fill="#f4f3ef",
                    width=1.8,
                    capstyle="round",
                    joinstyle="round",
                )
            else:
                ind_tag = pcan.create_line(
                    rx - 2, rcy - 4, rx + 2, rcy, rx - 2, rcy + 4,
                    fill="#8b8d93",
                    width=1.6,
                    capstyle="round",
                    joinstyle="round",
                )

            hit_rect = pcan.create_rectangle(8, ry, pop_w - 8, ry + row_h, fill="", outline="")

            def _bind_row(row_idx=idx, h_tag=hover_img_tag, l_tag=lbl_tag, t_tag=tier_tag, c_tag=ind_tag, sel=is_sel):
                def _on_ent(e):
                    pcan.config(cursor="hand2")
                    pcan.itemconfig(h_tag, state="normal")
                    pcan.itemconfig(l_tag, fill="#ffffff")
                    pcan.itemconfig(t_tag, fill="#a4a7b0")
                    pcan.itemconfig(c_tag, fill="#ffffff")

                def _on_lv(e):
                    pcan.config(cursor="")
                    pcan.itemconfig(h_tag, state="hidden")
                    pcan.itemconfig(l_tag, fill="#ffffff" if sel else "#d4d6dc")
                    pcan.itemconfig(t_tag, fill="#80838b")
                    pcan.itemconfig(c_tag, fill="#f4f3ef" if sel else "#8b8d93")

                for t in [hit_rect, h_tag, l_tag, t_tag, c_tag] + badge_tags:
                    pcan.tag_bind(t, "<Enter>", _on_ent)
                    pcan.tag_bind(t, "<Leave>", _on_lv)
                    pcan.tag_bind(t, "<Button-1>", lambda e, i=row_idx: self.select_model_by_index(i))

            _bind_row()

        # Footer divider & keycap pill "[ Select Model  Ctrl+/ ]" + right vector chevron '>'
        div_y = 38 + len(self.MODEL_CATALOG) * row_h + 4
        pcan.create_line(14, div_y, pop_w - 14, div_y, fill="#2e313a", width=1)
        fy = div_y + 20
        pcan.create_image(89, fy, image=self._tk_footer_pill)
        pcan.create_text(26, fy, text="Select Model", anchor="w", fill="#b4b7c0", font=(self.font_family_ui, 9))
        pcan.create_text(112, fy, text="Ctrl+/", anchor="w", fill="#7d8088", font=(self.font_family_mono, 8))
        rx_f = pop_w - 20
        pcan.create_line(
            rx_f - 2, fy - 4, rx_f + 2, fy, rx_f - 2, fy + 4,
            fill="#8b8d93", width=1.6, capstyle="round", joinstyle="round"
        )

        # Smooth 6-step ease-out cubic animation (slide + alpha fade)
        steps = 6
        def _animate_in(step: int = 1):
            if not self.is_popover_open or not self.model_popover:
                return
            try:
                if not self.model_popover.winfo_exists():
                    return
            except Exception:
                return
            t = step / float(steps)
            ease = 1.0 - pow(1.0 - t, 3)
            cur_y = int(start_y + (target_y - start_y) * ease)
            try:
                self.model_popover.geometry(f"{pop_w}x{pop_h}+{target_x}+{cur_y}")
                self.model_popover.wm_attributes("-alpha", round(0.98 * ease, 2))
                with self._coord_lock:
                    self._cached_popover_rect = (target_x, cur_y, target_x + pop_w, cur_y + pop_h)
            except Exception:
                return
            if step < steps:
                try:
                    self._popover_anim_job = self.win.after(18, lambda: _animate_in(step + 1))
                except Exception:
                    self._popover_anim_job = None

        _animate_in(1)

    def _close_model_popover(self):
        self.is_popover_open = False
        if hasattr(self, "_popover_anim_job") and self._popover_anim_job:
            try:
                self.win.after_cancel(self._popover_anim_job)
            except Exception:
                pass
            self._popover_anim_job = None
        with self._coord_lock:
            self._cached_popover_rect = None
            self._popover_hwnd = None
        self._layout_model_capsule_items()
        if self.model_popover:
            try:
                if self.model_popover.winfo_exists():
                    self.model_popover.destroy()
            except Exception:
                pass
        self.model_popover = None

    # =========================================================================
    # COMPACT COMPANION POD MODE (160x76)
    # =========================================================================

    def _render_compact_orb_ui(self):
        self._close_model_popover()
        if hasattr(self, "entry_frame") and self.entry_frame:
            try:
                self.entry_frame.destroy()
            except Exception:
                pass
            self.entry_frame = None
            self.entry = None

        self.canvas.delete("all")

        if hasattr(self, "tk_bg_compact") and self.tk_bg_compact:
            self.bg_tag = self.canvas.create_image(self.compact_w // 2, self.compact_h // 2, image=self.tk_bg_compact)
        else:
            self.bg_tag = self.canvas.create_rectangle(
                0, 0, self.compact_w, self.compact_h, fill="#191b20", outline="#444854", width=2
            )

        if not self.tk_mascot_frame:
            frame_img = self.animator.render_frame(target_w=56, target_h=42)
            self.tk_mascot_frame = ImageTk.PhotoImage(frame_img)

        self.mascot_tag = self.canvas.create_image(42, 38, image=self.tk_mascot_frame)
        self.canvas.tag_bind(self.mascot_tag, "<Button-1>", lambda e: self.expand_to_capsule())
        self.canvas.tag_bind(self.mascot_tag, "<Enter>", lambda e: self._on_mascot_hover(True))
        self.canvas.tag_bind(self.mascot_tag, "<Leave>", lambda e: self._on_mascot_hover(False))

        self.canvas.create_line(80, 16, 80, 60, fill="#2e313a", width=1)

        mic_bg = "#38181c" if self.is_voice_recording else "#23262e"
        mic_border = "#ef4444" if self.is_voice_recording else "#454956"
        self.compact_mic_border = self.canvas.create_oval(
            98, 16, 142, 60, fill=mic_bg, outline=mic_border, width=2
        )
        self.compact_mic_bg = self.compact_mic_border
        glyph = "🔴" if self.is_voice_recording else "🎙"
        self.compact_mic_glyph = self.canvas.create_text(
            120, 38, text=glyph, fill="#ffffff", font=("Segoe UI Emoji", 13)
        )
        for tag in (self.compact_mic_border, self.compact_mic_glyph):
            self.canvas.tag_bind(tag, "<Button-1>", lambda e: self._trigger_voice())
            self.canvas.tag_bind(tag, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(tag, "<Leave>", lambda e: self.canvas.config(cursor=""))

        self.canvas.tag_bind(self.bg_tag, "<ButtonPress-1>", self._start_drag)
        self.canvas.tag_bind(self.bg_tag, "<B1-Motion>", self._do_drag)

    # =========================================================================
    # INTERACTIONS & ACTIONS
    # =========================================================================

    def _trigger_voice(self):
        if self.on_voice_toggle:
            self.on_voice_toggle()

    def set_voice_recording(self, recording: bool):
        """Updates voice recording state in-place without destroying the Entry widget."""
        self.is_voice_recording = recording
        self.animator.set_status("listening" if recording else "ready", recording)
        if not self.is_expanded:
            self._render_compact_orb_ui()
        else:
            if hasattr(self, "btn_mic_tag") and self.btn_mic_tag:
                try:
                    sprite = self.mic_recording_sprite if recording else self.mic_sprites[0]
                    self.canvas.itemconfig(self.btn_mic_tag, image=sprite)
                except Exception:
                    pass

    def set_voice_text(self, text: str, auto_submit: bool = False):
        """Inserts transcribed voice command into the entry box and dispatches cleanly."""
        if not self.is_expanded:
            self.expand_to_capsule()
        if hasattr(self, "entry") and self.entry:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, text)
            self.win.update_idletasks()
            if auto_submit and text.strip():
                self._handle_submit()

    def collapse_to_orb(self, animate: bool = True):
        if not self.is_expanded:
            return
        self._hide_status_tooltip()
        self._close_model_popover()
        self.is_expanded = False
        with self._coord_lock:
            cur_x, cur_y = self._cached_win_rect[0], self._cached_win_rect[1]
            try:
                if self.win.winfo_exists() and threading.current_thread() == threading.main_thread():
                    cur_x, cur_y = self.win.winfo_x(), self.win.winfo_y()
            except Exception:
                pass
            self._cached_win_rect = (cur_x, cur_y, cur_x + self.compact_w, cur_y + self.compact_h)
        self.win.geometry(f"{self.compact_w}x{self.compact_h}+{cur_x}+{cur_y}")
        self.canvas.config(width=self.compact_w, height=self.compact_h)
        self._render_compact_orb_ui()
        if self.on_compact_callback:
            self.on_compact_callback(False)

    def expand_to_capsule(self, animate: bool = True):
        if self.is_expanded:
            return
        self.is_expanded = True
        with self._coord_lock:
            cur_x, cur_y = self._cached_win_rect[0], self._cached_win_rect[1]
            try:
                if self.win.winfo_exists() and threading.current_thread() == threading.main_thread():
                    cur_x, cur_y = self.win.winfo_x(), self.win.winfo_y()
            except Exception:
                pass
            self._cached_win_rect = (cur_x, cur_y, cur_x + self.win_w, cur_y + self.win_h)
        self.win.geometry(f"{self.win_w}x{self.win_h}+{cur_x}+{cur_y}")
        self.canvas.config(width=self.win_w, height=self.win_h)
        self._render_expanded_ui()
        if self.on_compact_callback:
            self.on_compact_callback(True)

    def toggle_expanded(self):
        if self.is_expanded:
            self.collapse_to_orb()
        else:
            self.expand_to_capsule()

    def toggle_droplets(self):
        if self.on_toggle_droplets:
            self.droplets_visible = self.on_toggle_droplets()
            self._update_pill_states_in_place()

    def set_droplets_visible(self, visible: bool):
        self.droplets_visible = visible
        self._update_pill_states_in_place()

    def _update_pill_states_in_place(self):
        """Updates the 3 separate pill boxes in-place without tearing down the Entry widget."""
        if not self.is_expanded:
            return
        try:
            chat_active = not self.is_quick_mode
            self.canvas.itemconfig(
                self.pill_chat_bg,
                image=self.pill_active_chat if chat_active else self.pill_sprites["pill_chat"][0],
            )
            self.canvas.itemconfig(
                self.pill_chat_txt,
                fill="#f4f3ef" if chat_active else "#9ca0ab",
                font=(self.font_family_ui, 9, "bold" if chat_active else "normal"),
            )

            quick_active = self.is_quick_mode
            self.canvas.itemconfig(
                self.pill_quick_bg,
                image=self.pill_active_quick if quick_active else self.pill_sprites["pill_quick"][0],
            )
            self.canvas.itemconfig(
                self.pill_quick_txt,
                fill="#fbbf24" if quick_active else "#9ca0ab",
                font=(self.font_family_ui, 9, "bold" if quick_active else "normal"),
            )

            drop_active = self.droplets_visible
            self.canvas.itemconfig(
                self.pill_drop_bg,
                image=self.pill_active_drop if drop_active else self.pill_sprites["pill_drop"][0],
            )
            self.canvas.itemconfig(
                self.pill_drop_txt,
                fill="#7dd3fc" if drop_active else "#9ca0ab",
                font=(self.font_family_ui, 9, "bold" if drop_active else "normal"),
            )
        except Exception:
            pass

    def activate_quick_select(self):
        if not self.is_expanded:
            self.expand_to_capsule()
        self.is_quick_mode = True
        self.active_candidate = None
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        self.win.lift()
        self.win.focus_force()

        if hasattr(self, "entry") and self.entry:
            self.entry.delete(0, tk.END)
            self.entry.focus_set()

        self.set_status("⚡ Quick Mode: Type Dot ID & Enter", color="#f59e0b")
        self._update_pill_states_in_place()

    def deactivate_quick_select(self):
        self.is_quick_mode = False
        self.active_candidate = None
        self.set_status("✔ Live Auto-Sync", color="#3fb950")
        if self.on_quick_preview:
            self.on_quick_preview("")
        self._update_pill_states_in_place()

    def toggle_quick_select(self):
        if self.is_quick_mode:
            self.deactivate_quick_select()
        else:
            self.activate_quick_select()

    def _handle_key_release(self, event=None):
        self.animator.notify_typing()
        if not self.is_quick_mode:
            return
        if event and event.keysym in ("Alt_L", "Alt_R", "Control_L", "Control_R", "Shift_L", "Shift_R", "Escape"):
            return
        query = self.entry.get().strip() if hasattr(self, "entry") and self.entry else ""
        if self.on_quick_preview:
            self.active_candidate = self.on_quick_preview(query)

    def _handle_escape(self, event=None):
        if self.is_popover_open:
            self._close_model_popover()
        elif self.is_quick_mode:
            self.deactivate_quick_select()
        elif hasattr(self, "entry") and self.entry and self.entry.get():
            self.entry.delete(0, tk.END)
        elif self.on_abort:
            self.on_abort()
        self.ensure_visible()

    def _handle_submit(self, event=None):
        self._close_model_popover()
        if not hasattr(self, "entry") or not self.entry:
            return

        if self.is_quick_mode:
            cand = self.active_candidate
            query = self.entry.get().strip()
            self.deactivate_quick_select()
            if cand and self.on_quick_dispatch:
                self.on_quick_dispatch(cand)
                return
            elif query:
                self.on_submit(query)
                return
            return

        cmd = self.entry.get().strip()
        if not cmd:
            return

        self.entry.delete(0, tk.END)
        try:
            self.root.focus_set()
        except Exception:
            pass

        self.is_submitting = True
        self.set_status(f"⚡ Processing: \"{cmd}\"...", color="#ffb300")
        if hasattr(self, "btn_send_bg") and self.btn_send_bg:
            try:
                self.canvas.itemconfig(self.btn_send_bg, image=self.send_busy_sprite)
            except Exception:
                pass
        if hasattr(self, "btn_send_glyph") and self.btn_send_glyph:
            try:
                self.canvas.itemconfig(self.btn_send_glyph, text="")
            except Exception:
                pass

        self.on_submit(cmd)

    def set_status(self, text: str, color: str = "#00e5ff", details: Optional[str] = None):
        self.current_status_text = text
        self.current_status_color = color
        if details is not None:
            self.current_status_details = details
        else:
            self.current_status_details = text
        self.animator.set_status(text, self.is_voice_recording)
        if hasattr(self, "status_text_tag") and self.status_text_tag and hasattr(self, "canvas") and self.canvas:
            def _update_ui():
                try:
                    display_text = text if len(text) <= 90 else (text[:87] + "...")
                    self.canvas.itemconfig(self.status_text_tag, text=display_text, fill=color)
                    if hasattr(self, "status_dot_tag") and self.status_dot_tag:
                        self.canvas.itemconfig(self.status_dot_tag, fill=color)
                    self.ensure_visible()
                except Exception:
                    pass
            try:
                self.win.after(0, _update_ui)
            except Exception:
                pass

    def set_emotion(self, emotion: str):
        self.animator.set_emotion(emotion)

    def reset_button(self):
        self.is_submitting = False
        if hasattr(self, "btn_send_bg") and self.btn_send_bg:
            try:
                self.canvas.itemconfig(self.btn_send_bg, image=self.send_sprites[0])
            except Exception:
                pass
        if hasattr(self, "btn_send_glyph") and self.btn_send_glyph:
            try:
                self.canvas.itemconfig(self.btn_send_glyph, text="")
            except Exception:
                pass

    def get_rect(self, include_popover: bool = True) -> Tuple[int, int, int, int]:
        """
        Thread-safe retrieval of copilot window coordinates.
        Never calls winfo_x() across background threads, completely preventing
        Tcl 'bad window path name' errors during background screen scanning.
        """
        win_rect = None
        if threading.current_thread() == threading.main_thread():
            try:
                if self.win.winfo_exists():
                    ww = self.win.winfo_width()
                    wh = self.win.winfo_height()
                    wx = self.win.winfo_x()
                    wy = self.win.winfo_y()
                    if ww > 50 and wh > 30 and not (wx == 0 and wy == 0 and ww == 200 and wh == 200):
                        win_rect = (wx, wy, wx + ww, wy + wh)
            except Exception:
                pass

        if not win_rect and hasattr(self, "_hwnd") and self._hwnd:
            try:
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(self._hwnd, ctypes.byref(rect)):
                    rw = rect.right - rect.left
                    rh = rect.bottom - rect.top
                    if rw > 50 and rh > 30 and not (rect.left == 0 and rect.top == 0 and rw == 200 and rh == 200):
                        win_rect = (rect.left, rect.top, rect.right, rect.bottom)
            except Exception:
                win_rect = None

        with self._coord_lock:
            if win_rect:
                self._cached_win_rect = win_rect
            else:
                win_rect = self._cached_win_rect

            pop_rect = self._cached_popover_rect if self.is_popover_open else None

        if include_popover and pop_rect:
            return (
                min(win_rect[0], pop_rect[0]),
                min(win_rect[1], pop_rect[1]),
                max(win_rect[2], pop_rect[2]),
                max(win_rect[3], pop_rect[3])
            )
        return win_rect

    def get_popover_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Thread-safe retrieval of model popover coordinates."""
        with self._coord_lock:
            if not self.is_popover_open:
                return None
            return self._cached_popover_rect

    def get_status_tooltip_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if getattr(self, "status_tooltip_win", None):
            try:
                if self.status_tooltip_win.winfo_exists():
                    x = self.status_tooltip_win.winfo_rootx()
                    y = self.status_tooltip_win.winfo_rooty()
                    w = self.status_tooltip_win.winfo_width()
                    h = self.status_tooltip_win.winfo_height()
                    if w > 1 and h > 1:
                        return (x, y, x + w, y + h)
            except Exception:
                pass
        return None

    def get_all_rects(self) -> List[Tuple[int, int, int, int]]:
        """Thread-safe list of active HUD/copilot exclusion boxes."""
        base_rect = self.get_rect(include_popover=False)
        rects = [base_rect]
        pop_rect = self.get_popover_rect()
        if pop_rect:
            rects.append(pop_rect)
        tip_rect = self.get_status_tooltip_rect()
        if tip_rect:
            rects.append(tip_rect)
        return rects

    def _start_drag(self, event):
        self._hide_status_tooltip()
        self._close_model_popover()
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_status_enter(self, event=None):
        if hasattr(self, "_status_tip_after_id") and self._status_tip_after_id:
            try:
                self.win.after_cancel(self._status_tip_after_id)
            except Exception:
                pass
            self._status_tip_after_id = None
        try:
            self._status_tip_after_id = self.win.after(200, self._show_status_tooltip)
        except Exception:
            pass

    def _on_status_leave(self, event=None):
        if hasattr(self, "_status_tip_after_id") and self._status_tip_after_id:
            try:
                self.win.after_cancel(self._status_tip_after_id)
            except Exception:
                pass
            self._status_tip_after_id = None
        self._hide_status_tooltip()

    def _show_status_tooltip(self):
        if not self.is_expanded:
            return
        tip_text = getattr(self, "current_status_details", None) or getattr(self, "current_status_text", "")
        if not tip_text:
            return
        self._hide_status_tooltip()

        try:
            tw = tk.Toplevel(self.win)
            tw.wm_overrideredirect(True)
            tw.wm_attributes("-topmost", True)
            tw.config(bg="#111317")

            container = tk.Frame(tw, bg="#181a20", highlightbackground="#2b2f3a", highlightthickness=1, padx=12, pady=8)
            container.pack(fill="both", expand=True)

            hdr = tk.Label(
                container,
                text="🧠 Cognitive Reasoner Status & Master Plan",
                font=(self.font_family_ui, 8, "bold"),
                fg="#94a3b8",
                bg="#181a20",
                anchor="w",
            )
            hdr.pack(fill="x", pady=(0, 4))

            body = tk.Label(
                container,
                text=tip_text,
                font=(self.font_family_ui, 9),
                fg="#f1f5f9",
                bg="#181a20",
                justify="left",
                wraplength=520,
                anchor="w",
            )
            body.pack(fill="both", expand=True)

            tw.update_idletasks()
            tip_w = tw.winfo_reqwidth()
            tip_h = tw.winfo_reqheight()

            with self._coord_lock:
                win_x, win_y = self._cached_win_rect[0], self._cached_win_rect[1]
            try:
                if self.win.winfo_exists():
                    win_x = self.win.winfo_rootx()
                    win_y = self.win.winfo_rooty()
            except Exception:
                pass

            target_x = win_x + 20
            target_y = win_y + 52 - tip_h - 6
            if target_y < 10:
                target_y = win_y + 82

            tw.geometry(f"{tip_w}x{tip_h}+{target_x}+{target_y}")
            tw.bind("<Leave>", lambda e: self._hide_status_tooltip())
            tw.bind("<ButtonPress-1>", lambda e: self._hide_status_tooltip())
            self.status_tooltip_win = tw
        except Exception:
            self.status_tooltip_win = None

    def _hide_status_tooltip(self):
        if hasattr(self, "status_tooltip_win") and self.status_tooltip_win:
            try:
                if self.status_tooltip_win.winfo_exists():
                    self.status_tooltip_win.destroy()
            except Exception:
                pass
            self.status_tooltip_win = None

    def _do_drag(self, event):
        try:
            deltax = event.x - self._drag_x
            deltay = event.y - self._drag_y
            with self._coord_lock:
                cur_x, cur_y, cur_r, cur_b = self._cached_win_rect
                w = cur_r - cur_x
                h = cur_b - cur_y
            sw = self.win.winfo_screenwidth()
            sh = self.win.winfo_screenheight()
            new_x = max(10, min(sw - w - 10, cur_x + deltax))
            new_y = max(10, min(sh - h - 10, cur_y + deltay))
            self.win.geometry(f"+{new_x}+{new_y}")
            with self._coord_lock:
                self._cached_win_rect = (new_x, new_y, new_x + w, new_y + h)
            self.ensure_visible()
        except Exception:
            pass


# Backward compatibility aliases
SphereMascotAnimator = RobotMascotAnimator
FloatingChatBar = HoverChatWidget
