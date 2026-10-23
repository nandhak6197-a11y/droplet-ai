"""
Spatial Droplet — Production Spatial Screen Navigation & Autonomous Agent Framework.
Enterprise modular package providing Triple-Sensor Consensus (UIA 1.0, WinRT OCR 0.95, CLAHE CV 0.70),
Per-Monitor DPI & Desktop Session preservation, and Closed-Loop Multi-Turn Trajectories.
"""

# Core
from .core.models import (
    WindowContext,
    ModalDialogContext,
    ScreenAwarenessContext,
    DiagonalDroplet,
    TrajectoryStep,
    register_own_hwnd,
    is_own_window,
    get_own_hwnds
)
from .core.memory import EpisodicTrajectoryMemory

# Platform
from .platform.dpi import DpiAwarenessManager
from .platform.desktop import DesktopThreadManager
from .platform.actuator import Win32HardwareActuator, PrecisionActuator, get_cursor_shape
from .platform.window import (
    get_window_details,
    get_active_window_context,
    get_desktop_context,
    get_process_name_for_hwnd,
    is_window_cloaked
)

# Vision
from .vision.grabber import FastScreenGrabber
from .vision.clahe import AdaptiveContrastNormalizer
from .vision.icon_detector import EnhancedIconDetector, IconShapeDetector
from .vision.uia_walker import UiaControlWalker
from .vision.screen_parser import ScreenStateParser
from .vision.sprayer import FullBleedLatticeSprayer

# Reasoning
from .reasoning.groq_rotator import GroqRotator, GROQ_API_KEYS, GROQ_MODELS
from .reasoning.reasoner import DiagonalDropletReasoner

# Verification
from .verification.action_verifier import ActionVerifier, PreActionSnapshot
from .verification.semantic_verifier import SemanticStateVerifier, SemanticSnapshot

# Engine
from .engine.fast_path import FastPathRouter
from .engine.trajectory_engine import AutonomousTrajectoryEngine

# UI
from .ui.hud import FullBleedHUD
from .ui.chat_widget import HoverChatWidget
from .ui.hotkeys import GlobalHotkeyListener

__version__ = "8.0.0"

__all__ = [
    # Core
    "WindowContext",
    "ModalDialogContext",
    "ScreenAwarenessContext",
    "DiagonalDroplet",
    "TrajectoryStep",
    "register_own_hwnd",
    "is_own_window",
    "get_own_hwnds",
    "EpisodicTrajectoryMemory",
    # Platform
    "DpiAwarenessManager",
    "DesktopThreadManager",
    "Win32HardwareActuator",
    "PrecisionActuator",
    "get_cursor_shape",
    "get_window_details",
    "get_active_window_context",
    "get_desktop_context",
    "get_process_name_for_hwnd",
    "is_window_cloaked",
    # Vision
    "FastScreenGrabber",
    "AdaptiveContrastNormalizer",
    "EnhancedIconDetector",
    "IconShapeDetector",
    "UiaControlWalker",
    "ScreenStateParser",
    "FullBleedLatticeSprayer",
    # Reasoning
    "GroqRotator",
    "GROQ_API_KEYS",
    "GROQ_MODELS",
    "DiagonalDropletReasoner",
    # Verification
    "ActionVerifier",
    "PreActionSnapshot",
    "SemanticStateVerifier",
    "SemanticSnapshot",
    # Engine
    "FastPathRouter",
    "AutonomousTrajectoryEngine",
    # UI
    "FullBleedHUD",
    "HoverChatWidget",
    "GlobalHotkeyListener",
]
