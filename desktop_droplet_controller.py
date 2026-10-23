"""
SPATIAL DROPLET CONTROLLER — PRODUCTION MODULAR FACADE (PHASE 8).
Full-Bleed 3,726 OpenCV Star Lattice, Triple-Sensor Consensus (UIA 1.0, OCR 0.95, CLAHE 0.70),
Per-Monitor DPI & Input Desktop Attachment, and Closed-Loop Multi-Turn Trajectory Engine.

This module serves as the 100% backwards-compatible entry point and CLI facade,
delegating all subsystems to the modular `spatial_droplet` enterprise package.
"""

import os
import sys
import time
import math
import re
import json
import ctypes
from ctypes import wintypes
import threading
import queue
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import tkinter as tk

# Conditional 3rd-party dependencies (with availability flags)
try:
    import mss
    HAS_MSS = True
except ImportError:
    mss = None
    HAS_MSS = False

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    HAS_PYAUTOGUI = True
except ImportError:
    pyautogui = None
    HAS_PYAUTOGUI = False

try:
    import winocr
    HAS_WINOCR = True
except ImportError:
    winocr = None
    HAS_WINOCR = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

try:
    import uiautomation as auto
    HAS_UIA = True
except ImportError:
    auto = None
    HAS_UIA = False

# Low-level Desktop Session Keeper (Session 0 isolation & GDI thread desktop attachment)
_user32 = ctypes.windll.user32
try:
    _hdesk = _user32.OpenDesktopW("default", 0, False, 0x01FF)
    if _hdesk:
        _user32.SetThreadDesktop(_hdesk)
except Exception:
    _hdesk = None

# Re-export all core models & context definitions
from spatial_droplet.core.models import (
    WindowContext,
    ModalDialogContext,
    ScreenAwarenessContext,
    DiagonalDroplet,
    TrajectoryStep,
    register_own_hwnd,
    is_own_window,
    get_own_hwnds,
    _OWN_HWN_SET as OWN_HWNDS
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory

# Re-export platform, DPI, desktop & actuator drivers
from spatial_droplet.platform.dpi import DpiAwarenessManager
from spatial_droplet.platform.desktop import DesktopThreadManager
from spatial_droplet.platform.actuator import (
    Win32HardwareActuator,
    PrecisionActuator,
    get_cursor_shape,
    MOUSEINPUT,
    KEYBDINPUT,
    HARDWAREINPUT,
    INPUT
)
from spatial_droplet.platform.window import (
    get_window_details,
    get_active_window_context,
    get_desktop_context,
    get_process_name_for_hwnd,
    is_window_cloaked,
    DWMWA_CLOAKED
)

# Re-export vision, CLAHE & icon detectors
from spatial_droplet.vision.grabber import FastScreenGrabber
from spatial_droplet.vision.clahe import AdaptiveContrastNormalizer
from spatial_droplet.vision.icon_detector import EnhancedIconDetector, IconShapeDetector
from spatial_droplet.vision.uia_walker import UiaControlWalker
from spatial_droplet.vision.screen_parser import ScreenStateParser
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer

# Re-export reasoning & API key pool
from spatial_droplet.reasoning.groq_rotator import GroqRotator, GROQ_API_KEYS, GROQ_MODELS
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner

# Re-export verification engines
from spatial_droplet.verification.action_verifier import ActionVerifier, PreActionSnapshot
from spatial_droplet.verification.semantic_verifier import SemanticStateVerifier, SemanticSnapshot

# Re-export execution engines
from spatial_droplet.engine.fast_path import FastPathRouter
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine

# Re-export UI components
from spatial_droplet.ui.hud import FullBleedHUD
from spatial_droplet.ui.chat_widget import HoverChatWidget
from spatial_droplet.ui.hotkeys import GlobalHotkeyListener

# Re-export application runner
from spatial_droplet.app import grab_frame, main

__version__ = "8.0.0"

if __name__ == "__main__":
    main()
