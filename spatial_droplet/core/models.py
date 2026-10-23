"""
Core data models and context definitions for Spatial Droplet Engine.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Process-global set of internal HWNDs to ignore during window queries
_OWN_HWN_SET = set()

def register_own_hwnd(hwnd: int):
    if hwnd:
        _OWN_HWN_SET.add(hwnd)
        try:
            import ctypes
            # WDA_EXCLUDEFROMCAPTURE (0x11) excludes the window from BitBlt, MSS, and screen captures
            # while keeping it 100% visible to human eyes, completely curing rescan noise feedback!
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            parent = ctypes.windll.user32.GetParent(hwnd)
            if parent:
                _OWN_HWN_SET.add(parent)
                ctypes.windll.user32.SetWindowDisplayAffinity(parent, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

def is_own_window(hwnd: int) -> bool:
    return hwnd in _OWN_HWN_SET

def get_own_hwnds() -> List[int]:
    return list(_OWN_HWN_SET)

def is_our_ui(title: str, class_name: str) -> bool:
    if not title:
        return False
    t_lower = title.lower()
    return any(k in t_lower for k in (
        "spatial droplet", "full-bleed", "droplet hud", "tk", "hoverchat",
        "chat copilot", "diagonal droplet"
    ))

@dataclass
class WindowContext:
    hwnd: int = 0
    title: str = "Desktop"
    rect: Tuple[int, int, int, int] = (0, 0, 1920, 1080)  # (left, top, right, bottom)
    width: int = 1920
    height: int = 1080
    is_valid: bool = True
    class_name: str = ""
    is_elevated: bool = False

@dataclass
class ModalDialogContext:
    hwnd: int
    title: str
    class_name: str
    rect: Tuple[int, int, int, int]
    width: int = 0
    height: int = 0
    is_modal: bool = True
    owner_hwnd: int = 0
    dialog_type: str = "GENERIC_DIALOG"

    def __post_init__(self):
        if (self.width == 0 or self.height == 0) and self.rect:
            self.width = max(0, self.rect[2] - self.rect[0])
            self.height = max(0, self.rect[3] - self.rect[1])
        if not self.dialog_type and self.class_name:
            self.dialog_type = self.class_name

@dataclass
class ScreenAwarenessContext:
    app_title: str = "Desktop"
    app_class: str = ""
    scene_type: str = "DESKTOP"  # 'GAME_LOBBY', 'GAME_CANVAS', 'CODE_EDITOR', 'WEB_APP', 'FORM_INPUT', 'SETTINGS_VIEW', 'MODAL_DIALOG', 'DESKTOP'
    active_modal: Optional[ModalDialogContext] = None
    layout_zones: Dict[str, List[str]] = field(default_factory=dict)
    raw_text_lines: List[str] = field(default_factory=list)
    summary: str = ""
    timestamp: float = field(default_factory=time.time)
    # Dual compatibility fields
    app_name: str = ""
    window_title: str = ""
    modal_context: Optional[ModalDialogContext] = None
    total_text_tokens: int = 0
    active_window_rect: Optional[Tuple[int, int, int, int]] = None
    window_rect: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self):
        if not self.app_name:
            self.app_name = self.app_title
        if not self.app_title:
            self.app_title = self.app_name
        if not self.window_title:
            self.window_title = self.app_title
        if self.modal_context and not self.active_modal:
            self.active_modal = self.modal_context
        if self.active_modal and not self.modal_context:
            self.modal_context = self.active_modal
        if self.active_window_rect and not self.window_rect:
            self.window_rect = self.active_window_rect
        if self.window_rect and not self.active_window_rect:
            self.active_window_rect = self.window_rect

ARCHETYPE_COLORS = {
    "button": "#00E676",    # Emerald Green: Clickable actions, toggles, chevrons, icons
    "text": "#00E5FF",      # Electric Cyan: Text fields, inputs, hyperlinks, editable cells
    "logo": "#D500F9",      # Vibrant Magenta: App logos, brand icons, taskbar buttons, system tray
    "slider": "#FFD600",    # Gold Yellow: Continuous sliders, steppers, ruler markers, scrollbars
    "tab": "#FF6D00",       # Safety Orange: Navigation tabs, ribbon tabs, breadcrumbs
    "label": "#78909C",     # Muted Slate: Static section headers, non-interactive text
    "probe": "#64B5F6",     # Sky Blue: Surface probe lattice stars
}

# OpenCV BGR color mapping for Set-of-Marks image overlay
ARCHETYPE_COLORS_BGR = {
    "button": (118, 230, 0),
    "text": (255, 229, 0),
    "logo": (249, 0, 213),
    "slider": (0, 214, 255),
    "tab": (0, 109, 255),
    "label": (156, 144, 120),
    "probe": (246, 181, 100),
}

@dataclass
class UIContainer:
    id: str                                         # "taskbar", "left_sidebar", "main_canvas", "right_panel", "top_ribbon"
    rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    title: str = ""                                 # e.g. "Slide Transition Panel", "Slides Pane", "Taskbar"
    droplet_ids: List[int] = field(default_factory=list)

@dataclass
class DiagonalDroplet:
    id: int
    center_x: int
    center_y: int
    category: str           # "interactive_app", "interactive_widget", "surface_probe"
    label: str
    diag_wave: int          # Sort key: x + y (45° Diagonal Wavefront)
    display_subtext: str = ""
    is_foreground: bool = True
    confidence: float = 1.0  # 1.0 = UIA, 0.95 = OCR, 0.70 = CV
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    attached_label: str = ""
    semantic_role: str = "widget"  # "control", "header", "input", "cell", "slider", "widget", "probe"
    archetype: str = "button"      # "button", "text", "logo", "slider", "tab", "label", "probe"
    semantic_color: str = "#00E676"
    container_id: str = "main"     # ID of parent UIContainer
    container_title: str = ""      # Title of parent container
    is_minimized: bool = False     # True if control activates a minimized/hidden window
    hwnd: int = 0                  # Associated HWND if applicable

    def __post_init__(self):
        # Auto-infer archetype and color if not explicitly provided
        if not self.archetype or self.archetype == "button":
            lbl_lower = self.label.lower()
            role = self.semantic_role.lower()
            cat = self.category.lower()
            if cat == "surface_probe":
                self.archetype = "probe"
            elif any(k in lbl_lower for k in ("tab", "breadcrumb", "page 1", "sheet1", "sheet2", "ribbon")):
                self.archetype = "tab"
            elif any(k in lbl_lower for k in ("slider", "trackbar", "ruler", "margin", "scrollbar", "spinbox", "increment", "decrement", "stepper")):
                self.archetype = "slider"
            elif any(k in lbl_lower for k in ("logo", "app", "taskbar", "chrome", "spotify", "vscode", "terminal", "explorer", "start menu")):
                self.archetype = "logo"
            elif role in ("input", "cell") or any(k in lbl_lower for k in ("text box", "input", "search field", "url", "link", "http")):
                self.archetype = "text"
            elif role == "header" or (cat == "interactive_app" and not lbl_lower.startswith("[icon]") and not lbl_lower.startswith("[btn]")):
                self.archetype = "label"
            else:
                self.archetype = "button"
        
        self.semantic_color = ARCHETYPE_COLORS.get(self.archetype, "#00E676")

@dataclass
class TrajectoryStep:
    step_num: int
    goal: str
    subgoal: str
    action: str
    target_id: Optional[int]
    target_label: str
    input_text: Optional[str]
    verified: bool
    verification_reason: str
    active_window: str
    semantic_intention: str = ""
    expected_transition: str = ""
    verdict: str = "CONFIRMED"
    timestamp: float = field(default_factory=time.time)
    observation_analysis: str = ""
    critique: str = ""
    strategy_plan: str = ""
    master_plan: List[str] = field(default_factory=list)
    intent_analysis: str = ""
    mistake_critique: str = ""
    # Canonical 5-Stage Deep CoT Deliberation Trace
    user_intent_and_success_criteria: str = ""
    screen_state_and_visual_hierarchy: str = ""
    mistake_critique_and_loop_diagnosis: str = ""
    end_to_end_master_plan: List[str] = field(default_factory=list)
    tactical_action_justification: str = ""

    def __post_init__(self):
        # Bidirectional synchronization between 5-stage CoT and legacy fields
        # Stage 1: User Intent
        if self.user_intent_and_success_criteria and not self.intent_analysis:
            self.intent_analysis = self.user_intent_and_success_criteria
        elif self.intent_analysis and not self.user_intent_and_success_criteria:
            self.user_intent_and_success_criteria = self.intent_analysis

        # Stage 2: Screen State & Hierarchy
        if self.screen_state_and_visual_hierarchy and not self.observation_analysis:
            self.observation_analysis = self.screen_state_and_visual_hierarchy
        elif self.observation_analysis and not self.screen_state_and_visual_hierarchy:
            self.screen_state_and_visual_hierarchy = self.observation_analysis

        # Stage 3: Mistake Critique & Loop Diagnosis
        if self.mistake_critique_and_loop_diagnosis:
            if not self.critique:
                self.critique = self.mistake_critique_and_loop_diagnosis
            if not self.mistake_critique:
                self.mistake_critique = self.mistake_critique_and_loop_diagnosis
        elif self.mistake_critique or self.critique:
            self.mistake_critique_and_loop_diagnosis = self.mistake_critique or self.critique
            if not self.critique:
                self.critique = self.mistake_critique_and_loop_diagnosis
            if not self.mistake_critique:
                self.mistake_critique = self.mistake_critique_and_loop_diagnosis

        # Stage 4: End-to-End Master Plan
        if self.end_to_end_master_plan and not self.master_plan:
            self.master_plan = list(self.end_to_end_master_plan)
        elif self.master_plan and not self.end_to_end_master_plan:
            self.end_to_end_master_plan = list(self.master_plan)

        # Stage 5: Tactical Action Justification
        if self.tactical_action_justification and not self.strategy_plan:
            self.strategy_plan = self.tactical_action_justification
        elif self.strategy_plan and not self.tactical_action_justification:
            self.tactical_action_justification = self.strategy_plan
