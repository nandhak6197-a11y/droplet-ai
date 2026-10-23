"""
Cognitive Spatial Droplet Reasoner.
Evaluates goal, episodic history, live screen awareness, and droplet manifest
to generate predictive intention and next-step actions via Groq 25-key pool.
"""
import os
import re
import json
import ast
import math
import time
import threading
import base64
import cv2
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

from ..core.models import DiagonalDroplet, WindowContext, ScreenAwarenessContext, ARCHETYPE_COLORS_BGR
from ..core.memory import EpisodicTrajectoryMemory
from .groq_rotator import GroqRotator, GROQ_MODELS, HAS_OPENAI

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class _UseConfiguredProvider(Exception):
    """Internal control flow after preparing the shared ranked candidate slate."""


class DecisionResult(dict):
    """Encapsulates a cognitive decision with both dict and attribute access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        return self.get(name, "") if name == "reasoning" else self.get(name)

    @property
    def target_x(self) -> Optional[float]:
        val = self.get("target_x")
        return float(val) if val is not None else None

    @property
    def target_y(self) -> Optional[float]:
        val = self.get("target_y")
        return float(val) if val is not None else None

    @property
    def target_droplet_id(self) -> Optional[int]:
        val = self.get("target_droplet_id")
        if val is None:
            val = self.get("target_dot_id")
        return int(val) if val is not None else None

    @property
    def target_dot_id(self) -> Optional[int]:
        return self.target_droplet_id

    @property
    def action_type(self) -> str:
        return str(self.get("action_type") or self.get("action") or "click")

    @property
    def action(self) -> str:
        return self.action_type

    @property
    def status(self) -> str:
        return str(self.get("status", "in_progress"))

    @property
    def next_subgoal(self) -> str:
        return str(self.get("next_subgoal", ""))

    @property
    def confidence(self) -> float:
        try:
            return float(self.get("confidence", 0.95))
        except (ValueError, TypeError):
            return 0.95

    @property
    def deliberation(self) -> Dict[str, Any]:
        d = self.get("deliberation")
        return d if isinstance(d, dict) else {}

    @property
    def user_intent_and_success_criteria(self) -> str:
        d = self.deliberation
        return str(d.get("user_intent_and_success_criteria") or d.get("intent_analysis") or self.get("user_intent_and_success_criteria") or "")

    @property
    def screen_state_and_visual_hierarchy(self) -> str:
        d = self.deliberation
        return str(d.get("screen_state_and_visual_hierarchy") or d.get("observation_analysis") or d.get("spatial_anchoring") or self.get("screen_state_and_visual_hierarchy") or "")

    @property
    def mistake_critique_and_loop_diagnosis(self) -> str:
        d = self.deliberation
        return str(d.get("mistake_critique_and_loop_diagnosis") or d.get("critique_of_previous_step") or self.get("mistake_critique_and_loop_diagnosis") or "")

    @property
    def end_to_end_master_plan(self) -> Any:
        d = self.deliberation
        val = d.get("end_to_end_master_plan")
        if val is None:
            val = self.get("end_to_end_master_plan")
        if val is None:
            val = d.get("strategy_plan") or d.get("action_plan") or self.get("strategy_plan")
        return val if val is not None else []

    @property
    def tactical_action_justification(self) -> str:
        d = self.deliberation
        return str(d.get("tactical_action_justification") or d.get("candidate_evaluation") or d.get("action_plan") or self.get("tactical_action_justification") or "")

    @property
    def intent_analysis(self) -> str:
        d = self.deliberation
        return str(d.get("intent_analysis") or d.get("user_intent_and_success_criteria") or self.get("intent_analysis") or "")

    @property
    def spatial_anchoring(self) -> str:
        d = self.deliberation
        return str(d.get("spatial_anchoring") or d.get("screen_state_and_visual_hierarchy") or self.get("spatial_anchoring") or "")

    @property
    def candidate_evaluation(self) -> str:
        d = self.deliberation
        return str(d.get("candidate_evaluation") or d.get("tactical_action_justification") or self.get("candidate_evaluation") or "")

    @property
    def action_plan(self) -> str:
        d = self.deliberation
        return str(d.get("action_plan") or d.get("tactical_action_justification") or self.get("action_plan") or "")

    @property
    def observation_analysis(self) -> str:
        d = self.deliberation
        if "observation_analysis" in d:
            return str(d["observation_analysis"])
        if "screen_state_and_visual_hierarchy" in d:
            return str(d["screen_state_and_visual_hierarchy"])
        return str(self.get("observation_analysis") or self.get("screen_state_and_visual_hierarchy") or "")

    @property
    def critique_of_previous_step(self) -> str:
        d = self.deliberation
        if "critique_of_previous_step" in d:
            return str(d["critique_of_previous_step"])
        if "mistake_critique_and_loop_diagnosis" in d:
            return str(d["mistake_critique_and_loop_diagnosis"])
        return str(self.get("critique_of_previous_step") or self.get("mistake_critique_and_loop_diagnosis") or "")

    @property
    def strategy_plan(self) -> str:
        d = self.deliberation
        if "strategy_plan" in d:
            return str(d["strategy_plan"])
        if "end_to_end_master_plan" in d:
            val = d["end_to_end_master_plan"]
            return " | ".join(str(v) for v in val) if isinstance(val, list) else str(val)
        val = self.get("end_to_end_master_plan")
        if val:
            return " | ".join(str(v) for v in val) if isinstance(val, list) else str(val)
        return str(self.get("strategy_plan", ""))

    @property
    def reasoning(self) -> str:
        d = self.deliberation
        if d:
            ordered_keys = [
                ("user_intent_and_success_criteria", "User Intent"),
                ("intent_analysis", "Intent"),
                ("screen_state_and_visual_hierarchy", "Screen State"),
                ("observation_analysis", "Observation"),
                ("spatial_anchoring", "Spatial Anchoring"),
                ("mistake_critique_and_loop_diagnosis", "Mistake Critique"),
                ("critique_of_previous_step", "Critique"),
                ("end_to_end_master_plan", "Master Plan"),
                ("strategy_plan", "Strategy"),
                ("tactical_action_justification", "Tactical Justification"),
                ("candidate_evaluation", "Candidate Evaluation"),
                ("action_plan", "Action Plan"),
            ]
            parts = []
            seen = set()
            for k, label in ordered_keys:
                if k in d and k not in seen:
                    val = d[k]
                    seen.add(k)
                    if isinstance(val, list):
                        val_str = " -> ".join(str(v) for v in val)
                    else:
                        val_str = str(val).strip()
                    if val_str:
                        parts.append(f"[{label}]: {val_str}")
            for k, v in d.items():
                if k not in seen and v:
                    val_str = " -> ".join(str(item) for item in v) if isinstance(v, list) else str(v).strip()
                    if val_str:
                        parts.append(f"[{k.replace('_', ' ').title()}]: {val_str}")
            if parts:
                return " | ".join(parts)
        return str(self.get("reasoning") or self.get("strategy_plan") or "")

    @property
    def semantic_intention(self) -> str:
        return str(self.get("semantic_intention") or self.get("intention") or self.next_subgoal)

    @property
    def expected_transition(self) -> str:
        return str(self.get("expected_transition", "Screen updates accordingly"))

    @property
    def error(self) -> Optional[str]:
        return self.get("error")

def get_system_grounding_context() -> Dict[str, Any]:
    """
    Dynamically discover the real Windows user environment, home directories,
    and active drive letters. Fully generalized across any Windows installation.
    """
    import getpass
    import os
    import pathlib
    import string

    try:
        user_name = getpass.getuser()
    except Exception:
        user_name = os.environ.get("USERNAME", "User")

    try:
        home_path = pathlib.Path.home()
    except Exception:
        home_path = pathlib.Path(os.environ.get("USERPROFILE", f"C:\\Users\\{user_name}"))

    user_profile = os.environ.get("USERPROFILE", str(home_path))
    docs_dir = str(home_path / "Documents")
    downloads_dir = str(home_path / "Downloads")
    desktop_dir = str(home_path / "Desktop")

    active_drives = []
    for letter in string.ascii_uppercase:
        d_path = f"{letter}:\\"
        try:
            if os.path.exists(d_path):
                active_drives.append(d_path)
        except Exception:
            pass
    drives_str = ", ".join(active_drives) if active_drives else "C:\\"

    text = f"""WINDOWS SYSTEM ENVIRONMENT & PATHS:
â€¢ Active Logged-in User: "{user_name}" (Do NOT hallucinate or guess other usernames like "Administrator"!)
â€¢ User Profile Directory: "{user_profile}"
â€¢ Documents: "{docs_dir}"
â€¢ Downloads: "{downloads_dir}"
â€¢ Desktop: "{desktop_dir}"
â€¢ Available Drives: [{drives_str}]"""

    return {
        "text": text,
        "user_name": user_name,
        "user_profile": user_profile,
        "docs_dir": docs_dir,
        "downloads_dir": downloads_dir,
        "desktop_dir": desktop_dir,
        "active_drives": active_drives
    }

class DiagonalDropletReasoner:
    """
    Cognitive Reasoning Brain.
    Rotates across 25 Groq API keys with auto-failover across open-source foundation models.
    Formulates explicit semantic_intention and expected_transition for every turn.
    """
    def __init__(self, rotator: Optional[GroqRotator] = None):
        # Primary vision reasoning path: Kaggle Dual-T4 Qwen via ngrok.
        # Auto-probed if available, with automatic fallback to Groq 25-key pool.
        env_ngrok = os.environ.get("USE_NGROK")
        self.ngrok_url = (
            os.environ.get("KAGGLE_QWEN_URL")
            or os.environ.get("NGROK_URL")
            or "https://greedy-porcupine-precut.ngrok-free.dev/v1"
        )
        self.ngrok_model = (
            os.environ.get("KAGGLE_QWEN_MODEL")
            or os.environ.get("NGROK_MODEL")
            or os.environ.get("OLLAMA_MODEL")
        )
        self.use_ngrok = False

        if env_ngrok != "0":
            try:
                import urllib.request
                import json
                tags_url = self.ngrok_url.replace("/v1", "").rstrip("/") + "/api/tags"
                req = urllib.request.Request(
                    tags_url,
                    headers={"ngrok-skip-browser-warning": "true", "User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    if models:
                        self.use_ngrok = True
                        if not self.ngrok_model:
                            for pref in ["qwen2.5vl", "qwen3-vl", "ui-tars", "qwen"]:
                                for m in models:
                                    if pref in m.lower():
                                        self.ngrok_model = m
                                        break
                                if self.ngrok_model:
                                    break
                            if not self.ngrok_model:
                                self.ngrok_model = models[0]
            except Exception:
                self.use_ngrok = (env_ngrok == "1")
        else:
            self.use_ngrok = False

        if not self.ngrok_model:
            self.ngrok_model = "qwen2.5vl:7b"

        if self.use_ngrok:
            os.environ["USE_NGROK"] = "1"
            os.environ["KAGGLE_QWEN_MODEL"] = self.ngrok_model

        self.rotator = rotator or GroqRotator()
        self.key_pool = self.rotator.key_pool
        self.current_idx = 0
        self.lock = threading.Lock()
        self._key_cooldowns: Dict[Tuple[str, str], float] = {}
        self.preferred_groq_model = os.environ.get("GROQ_PRIMARY_MODEL", "qwen/qwen3.8-27b")

        if not self.use_ngrok:
            os.environ["USE_NGROK"] = "0"

        env_vision = os.environ.get("NGROK_USE_VISION")
        if env_vision is not None:
            self.ngrok_use_vision = env_vision.lower() in ("1", "true", "yes")
        else:
            # Auto-enable multimodal vision if endpoint model is a VLM (e.g. qwen2.5vl, qwen3-vl)
            self.ngrok_use_vision = bool(self.ngrok_model and any(k in self.ngrok_model.lower() for k in ("vl", "vision", "tars", "qwen")))
        self.ngrok_client = None
        if HAS_OPENAI and self.use_ngrok:
            try:
                self.ngrok_client = OpenAI(
                    base_url=self.ngrok_url,
                    api_key="ollama",
                    default_headers={"ngrok-skip-browser-warning": "true"},
                    timeout=90.0
                )
                print(f"[*] Primary Reasoner: ⚡ Kaggle Dual-T4 Qwen ({self.ngrok_model} @ {self.ngrok_url})")
            except Exception as e:
                print(f"[!] Failed to initialize ngrok client: {e}")
                self.ngrok_client = None
                self.use_ngrok = False
                os.environ["USE_NGROK"] = "0"

    def set_active_model(self, use_ngrok: bool, model_name: str):
        """Dynamically switches the live reasoning provider and model selected from the UI."""
        with self.lock:
            if use_ngrok:
                self.use_ngrok = True
                self.ngrok_model = model_name
                self.ngrok_use_vision = bool(model_name and any(k in model_name.lower() for k in ("vl", "vision", "tars", "qwen")))
                os.environ["USE_NGROK"] = "1"
                os.environ["KAGGLE_QWEN_MODEL"] = model_name
                if HAS_OPENAI and self.ngrok_client is None:
                    try:
                        self.ngrok_client = OpenAI(
                            base_url=self.ngrok_url,
                            api_key="ollama",
                            default_headers={"ngrok-skip-browser-warning": "true"},
                            timeout=90.0
                        )
                    except Exception as e:
                        print(f"[!] Failed to initialize ngrok client during model switch: {e}")
                        self.ngrok_client = None
                        self.use_ngrok = False
                        os.environ["USE_NGROK"] = "0"
                print(f"[Model Switch] Active Reasoner switched to Kaggle Endpoint: {self.ngrok_model}")
            else:
                self.use_ngrok = False
                self.preferred_groq_model = model_name
                os.environ["USE_NGROK"] = "0"
                os.environ["GROQ_PRIMARY_MODEL"] = model_name
                print(f"[Model Switch] Active Reasoner switched to Groq 25-Key Pool: {self.preferred_groq_model}")


    @classmethod
    def classify_screen_zone(
        cls,
        x: float,
        y: float,
        sw: int,
        sh: int,
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Classifies coordinates (x, y) into one of 5 functional screen zones:
        [MODAL/POPUP], [HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER].
        Supports window-relative functional zones for non-maximized floating windows
        while maintaining global OS taskbar as [FOOTER] and modal as [MODAL/POPUP].
        """
        sw = max(1, sw)
        sh = max(1, sh)

        # 1. Elevated Modal / Popup takes highest precedence
        if modal_rect and len(modal_rect) == 4:
            mx1, my1, mx2, my2 = modal_rect
            if (mx1 - 10 <= x <= mx2 + 10) and (my1 - 10 <= y <= my2 + 10):
                return "[MODAL/POPUP]"

        # 2. Global OS Taskbar (always footer regardless of window bounds)
        if y >= 0.88 * sh:
            return "[FOOTER]"

        # 3. Active Window-Relative Functional Zones (when inside non-maximized floating window)
        if win_rect and len(win_rect) == 4:
            wx1, wy1, wx2, wy2 = win_rect
            ww = max(1, wx2 - wx1)
            wh = max(1, wy2 - wy1)
            if (wx1 - 5 <= x <= wx2 + 5) and (wy1 - 5 <= y <= wy2 + 5):
                rx = x - wx1
                ry = y - wy1
                if ry < 0.14 * wh:
                    return "[HEADER/NAV]"
                if ry >= 0.88 * wh:
                    return "[FOOTER]"
                if rx < 0.22 * ww:
                    return "[SIDEBAR]"
                return "[WORKSPACE]"

        # 4. Global Screen-Relative Zones (fallback or full-screen)
        if y < 0.12 * sh:
            return "[HEADER/NAV]"
        if x < 0.22 * sw:
            return "[SIDEBAR]"
        return "[WORKSPACE]"

    @classmethod
    def _build_spatial_layout_graph(
        cls,
        nodes: List[DiagonalDroplet],
        sw: int,
        sh: int,
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Builds a 2D Visual Spatial Layout & Connectivity Graph across 5 functional screen zones:
        - [MODAL / POPUP DIALOG]: Elevated confirmation sheets or dialog boxes.
        - [HEADER / NAV BAR] (y < 0.12 * sh or window top 14%): Address bar, tabs, navigation arrows, search inputs.
        - [SIDEBAR / NAV DRAWER] (x < 0.22 * sw or window left 22%): Folder tree, categories, menu items.
        - [WORKSPACE / VIEWPORT]: Central content area.
        - [FOOTER / BOTTOM BAR] (y >= 0.88 * sh or window bottom 12%): Taskbar, zoom sliders, status readouts.
        Supplies spatial coordinates, 2D relative cardinal hints (Right/Left/Above/Below), and archetype tags.
        """
        sw = max(1, sw)
        sh = max(1, sh)
        if not nodes:
            return "=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH ===\n(No droplets present on screen)"

        zone_buckets: Dict[str, List[DiagonalDroplet]] = {
            "[MODAL/POPUP]": [],
            "[HEADER/NAV]": [],
            "[SIDEBAR]": [],
            "[WORKSPACE]": [],
            "[FOOTER]": []
        }

        for d in nodes:
            z = cls.classify_screen_zone(d.center_x, d.center_y, sw, sh, modal_rect, win_rect=win_rect)
            zone_buckets[z].append(d)

        relative_hints: Dict[int, List[str]] = {d.id: [] for d in nodes}
        sorted_by_y = sorted(nodes, key=lambda n: (round(n.center_y / 15.0), n.center_x))
        for i, d1 in enumerate(sorted_by_y):
            # Horizontal Left neighbors (d1 is Right of closest_left)
            left_neighbors = [
                d2 for d2 in sorted_by_y[:i]
                if abs(d2.center_y - d1.center_y) <= 18 and 0 < (d1.center_x - d2.center_x) <= 220
            ]
            if left_neighbors:
                closest_left = max(left_neighbors, key=lambda n: n.center_x)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_left.label).strip()[:24]
                relative_hints[d1.id].append(f"Right of Dot [{closest_left.id}] '{clean_lbl}'")

            # Horizontal Right neighbors (d1 is Left of closest_right)
            right_neighbors = [
                d2 for d2 in sorted_by_y[i+1:]
                if abs(d2.center_y - d1.center_y) <= 18 and 0 < (d2.center_x - d1.center_x) <= 220
            ]
            if right_neighbors:
                closest_right = min(right_neighbors, key=lambda n: n.center_x)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_right.label).strip()[:24]
                relative_hints[d1.id].append(f"Left of Dot [{closest_right.id}] '{clean_lbl}'")

            # Vertical Above neighbors (d2 is physically above d1, so d1 is Below closest_above)
            above_neighbors = [
                d2 for d2 in nodes
                if abs(d2.center_x - d1.center_x) <= 35 and 0 < (d1.center_y - d2.center_y) <= 180
            ]
            if above_neighbors:
                closest_above = max(above_neighbors, key=lambda n: n.center_y)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_above.label).strip()[:24]
                relative_hints[d1.id].append(f"Below Dot [{closest_above.id}] '{clean_lbl}'")

            # Vertical Below neighbors (d2 is physically below d1, so d1 is Above closest_below)
            below_neighbors = [
                d2 for d2 in nodes
                if abs(d2.center_x - d1.center_x) <= 35 and 0 < (d2.center_y - d1.center_y) <= 180
            ]
            if below_neighbors:
                closest_below = min(below_neighbors, key=lambda n: n.center_y)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_below.label).strip()[:24]
                relative_hints[d1.id].append(f"Above Dot [{closest_below.id}] '{clean_lbl}'")

        zone_titles = [
            ("[MODAL/POPUP]", "[MODAL / POPUP DIALOG] (Elevated Foreground Dialog/Sheet)"),
            ("[HEADER/NAV]", f"[HEADER / NAV BAR] (Top Navigation Zone | y < {int(0.12 * sh)})"),
            ("[SIDEBAR]", f"[SIDEBAR / NAV DRAWER] (Left Navigation Zone | x < {int(0.22 * sw)})"),
            ("[WORKSPACE]", f"[WORKSPACE / VIEWPORT] (Central Content Area | 0.12 <= y < 0.88)"),
            ("[FOOTER]", f"[FOOTER / BOTTOM BAR] (Bottom Bar & Status Area | y >= {int(0.88 * sh)})"),
        ]

        win_str = f" | Window Bounds: ({win_rect[0]},{win_rect[1]})-({win_rect[2]},{win_rect[3]})" if win_rect else ""
        lines = [
            f"=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH ({sw}x{sh}{win_str}) ===",
            "Topology Zones: [HEADER/NAV] (top 14%), [SIDEBAR] (left 22%), [WORKSPACE] (center), [FOOTER] (bottom 12%), [MODAL/POPUP] (elevated)",
            ""
        ]

        for z_key, z_header in zone_titles:
            cands = zone_buckets[z_key]
            if not cands:
                continue
            lines.append(f"--- {z_header} ---")
            for d in sorted(cands, key=lambda n: (n.center_y, n.center_x)):
                arch_tag = f"[{getattr(d, 'archetype', 'button').upper()}]"
                clean_lbl = d.label.replace('"', "'")
                rel_parts = list(relative_hints.get(d.id, []))
                att_val = str(getattr(d, 'attached_label', '')).strip()
                if att_val and att_val.lower() not in d.label.lower():
                    rel_parts.append(f"near='{att_val}'")
                c_title = getattr(d, 'container_title', '')
                if c_title:
                    rel_parts.append(f"in '{c_title}'")
                rel_str = f" ({', '.join(rel_parts)})" if rel_parts else ""
                lines.append(f"• [{d.id}] ({d.center_x},{d.center_y}) {z_key} {arch_tag} \"{clean_lbl}\"{rel_str}")
            lines.append("")

        return "\n".join(lines).strip()

    # =========================================================================
    # CANONICAL ACTION CONSTANTS & SYNONYM DICTIONARIES
    # =========================================================================
    CANONICAL_ACTIONS = {
        "click",
        "double_click",
        "type_and_enter",
        "clear_and_type",
        "launch_app",
        "open_path",
        "switch_window",
        "minimize",
        "maximize",
        "hotkey",
        "stop",
    }

    ACTION_SYNONYMS = {
        # click
        "single_click": "click",
        "left_click": "click",
        "press_button": "click",
        "tap": "click",
        "click_element": "click",
        "select": "click",
        "activate": "click",
        # double_click
        "double-click": "double_click",
        "doubleclick": "double_click",
        "dblclick": "double_click",
        "double_tap": "double_click",
        "open_icon": "double_click",
        # type_and_enter
        "type_enter": "type_and_enter",
        "type_and_submit": "type_and_enter",
        "submit_text": "type_and_enter",
        "input_and_enter": "type_and_enter",
        "write_and_enter": "type_and_enter",
        # clear_and_type
        "replace_text": "clear_and_type",
        "clear_type": "clear_and_type",
        "clear_and_write": "clear_and_type",
        "overwrite_text": "clear_and_type",
        "set_text": "clear_and_type",
        "edit_text": "clear_and_type",
        # launch_app
        "open_app": "launch_app",
        "run_app": "launch_app",
        "start_app": "launch_app",
        "launch": "launch_app",
        "run": "launch_app",
        "start_program": "launch_app",
        "execute_app": "launch_app",
        # open_path
        "open_folder": "open_path",
        "open_file": "open_path",
        "browse_path": "open_path",
        "system_open": "open_path",
        "system_open_path": "open_path",
        "navigate_to": "open_path",
        "open_dir": "open_path",
        # switch_window
        "switch_win": "switch_window",
        "activate_window": "switch_window",
        "focus_window": "switch_window",
        "switch_to_window": "switch_window",
        "bring_to_front": "switch_window",
        # minimize
        "minimize_window": "minimize",
        "min_window": "minimize",
        "minimize_app": "minimize",
        "iconify": "minimize",
        # maximize
        "maximize_window": "maximize",
        "max_window": "maximize",
        "maximize_app": "maximize",
        "restore_window": "maximize",
        # hotkey
        "shortcut": "hotkey",
        "chord": "hotkey",
        "key_combo": "hotkey",
        "press_hotkey": "hotkey",
        "send_hotkey": "hotkey",
        "keyboard_shortcut": "hotkey",
        # press
        "press_enter": "press",
        "press-enter": "press",
        # stop
        "finish": "stop",
        "complete": "stop",
        "completed": "stop",
        "done": "stop",
        "terminate": "stop",
        "exit": "stop",
        "end": "stop",
        "success": "stop",
    }

    PASSIVE_ACTIONS = {
        "", "none", "null", "think", "wait", "passive", "deliberate",
        "commentary", "thought", "reasoning", "ponder", "observe", "analyzing", "inspection",
        "thinking", "observing", "waiting", "examining", "idle", "watching", "reviewing", "evaluating",
        "analysis", "pondering", "deliberating", "inspecting"
    }

    @staticmethod
    def _repair_partial_json(raw: str) -> Optional[str]:
        """Repairs truncated / partial JSON by closing unclosed quotes, brackets, and braces."""
        if not raw or not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s.startswith('{'):
            start = s.find('{')
            if start == -1:
                return None
            s = s[start:]
        
        # Strip trailing commas
        s = re.sub(r',\s*$', '', s)
        
        in_string = False
        escape = False
        stack = []
        
        for char in s:
            if escape:
                escape = False
                continue
            if char == '\\':
                if in_string:
                    escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in ('{', '['):
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    
        if in_string:
            s += '"'
        
        s = re.sub(r',\s*$', '', s)
        
        while stack:
            opener = stack.pop()
            if opener == '{':
                s += '}'
            elif opener == '[':
                s += ']'
                
        return s

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Removes trailing commas and normalizes Python booleans/nulls in JSON-like text."""
        t = re.sub(r',\s*([}\]])', r'\1', text)
        t = re.sub(r'\bTrue\b', 'true', t)
        t = re.sub(r'\bFalse\b', 'false', t)
        t = re.sub(r'\bNone\b', 'null', t)
        return t

    @classmethod
    def _parse_python_dict(cls, text: str) -> Optional[Dict[str, Any]]:
        """Safely parses Python dict literals (e.g. single-quoted dictionaries)."""
        if not text or not isinstance(text, str):
            return None
        s = text.strip()
        s_py = re.sub(r'\bnull\b', 'None', s, flags=re.IGNORECASE)
        s_py = re.sub(r'\btrue\b', 'True', s_py, flags=re.IGNORECASE)
        s_py = re.sub(r'\bfalse\b', 'False', s_py, flags=re.IGNORECASE)
        try:
            d = ast.literal_eval(s_py)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        return None

    @classmethod
    def _clean_decision(cls, decision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not decision or not isinstance(decision, dict):
            return None

        # 1. Normalize target_dot_id / target_droplet_id
        for key in ("target_dot_id", "target_droplet_id"):
            val = decision.get(key)
            if val is not None:
                if isinstance(val, (int, float)):
                    decision[key] = int(val)
                elif isinstance(val, str):
                    val_str = val.strip()
                    if val_str.lower() in ("none", "null", ""):
                        decision[key] = None
                    else:
                        m = re.search(r'\d+', val_str)
                        decision[key] = int(m.group(0)) if m else None
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    try:
                        decision[key] = int(val[0])
                    except (ValueError, TypeError):
                        decision[key] = None
                else:
                    decision[key] = None

        if decision.get("target_dot_id") is not None and decision.get("target_droplet_id") is None:
            decision["target_droplet_id"] = decision["target_dot_id"]
        elif decision.get("target_droplet_id") is not None and decision.get("target_dot_id") is None:
            decision["target_dot_id"] = decision["target_droplet_id"]

        # 2. Synchronize deliberation schema (5-stage deep CoT <-> legacy)
        delib = decision.get("deliberation")
        if delib is None or not isinstance(delib, dict):
            delib = {}
            decision["deliberation"] = delib

        # Move top-level CoT keys into deliberation if misplaced
        for k in (
            "user_intent_and_success_criteria", "screen_state_and_visual_hierarchy",
            "mistake_critique_and_loop_diagnosis", "end_to_end_master_plan",
            "tactical_action_justification", "intent_analysis", "observation_analysis",
            "critique_of_previous_step", "strategy_plan", "candidate_evaluation", "action_plan"
        ):
            if k in decision and k not in delib:
                delib[k] = decision[k]

        # Master plan normalization
        raw_plan = delib.get("end_to_end_master_plan") or decision.get("end_to_end_master_plan")
        if raw_plan is not None:
            if isinstance(raw_plan, list):
                delib["end_to_end_master_plan"] = [str(x).strip() for x in raw_plan if str(x).strip()]
            elif isinstance(raw_plan, str):
                lines_p = [l.strip() for l in raw_plan.splitlines() if l.strip()]
                delib["end_to_end_master_plan"] = lines_p if lines_p else [raw_plan.strip()]
        elif "strategy_plan" in delib or "strategy_plan" in decision:
            strat = str(delib.get("strategy_plan") or decision.get("strategy_plan") or "").strip()
            if strat:
                delib["end_to_end_master_plan"] = [strat]

        # Bidirectional sync between 5-stage CoT and legacy keys
        if "user_intent_and_success_criteria" in delib and "intent_analysis" not in delib:
            delib["intent_analysis"] = delib["user_intent_and_success_criteria"]
        elif "intent_analysis" in delib and "user_intent_and_success_criteria" not in delib:
            delib["user_intent_and_success_criteria"] = delib["intent_analysis"]

        if "screen_state_and_visual_hierarchy" in delib:
            if "observation_analysis" not in delib:
                delib["observation_analysis"] = delib["screen_state_and_visual_hierarchy"]
            if "spatial_anchoring" not in delib:
                delib["spatial_anchoring"] = delib["screen_state_and_visual_hierarchy"]
        elif "observation_analysis" in delib and "screen_state_and_visual_hierarchy" not in delib:
            delib["screen_state_and_visual_hierarchy"] = delib["observation_analysis"]
        elif "spatial_anchoring" in delib and "screen_state_and_visual_hierarchy" not in delib:
            delib["screen_state_and_visual_hierarchy"] = delib["spatial_anchoring"]

        if "mistake_critique_and_loop_diagnosis" in delib and "critique_of_previous_step" not in delib:
            delib["critique_of_previous_step"] = delib["mistake_critique_and_loop_diagnosis"]
        elif "critique_of_previous_step" in delib and "mistake_critique_and_loop_diagnosis" not in delib:
            delib["mistake_critique_and_loop_diagnosis"] = delib["critique_of_previous_step"]

        if "tactical_action_justification" in delib:
            if "candidate_evaluation" not in delib:
                delib["candidate_evaluation"] = delib["tactical_action_justification"]
            if "action_plan" not in delib:
                delib["action_plan"] = delib["tactical_action_justification"]
        elif "candidate_evaluation" in delib and "tactical_action_justification" not in delib:
            delib["tactical_action_justification"] = delib["candidate_evaluation"]
        elif "action_plan" in delib and "tactical_action_justification" not in delib:
            delib["tactical_action_justification"] = delib["action_plan"]

        if "end_to_end_master_plan" in delib:
            plan_val = delib["end_to_end_master_plan"]
            plan_str = " | ".join(plan_val) if isinstance(plan_val, list) else str(plan_val)
            if "strategy_plan" not in delib:
                delib["strategy_plan"] = plan_str
            if "action_plan" not in delib:
                delib["action_plan"] = plan_str
        elif "strategy_plan" in delib and "end_to_end_master_plan" not in delib:
            delib["end_to_end_master_plan"] = [delib["strategy_plan"]]

        # Mirror deliberation fields to top-level decision dict for easy access
        for f in (
            "user_intent_and_success_criteria", "screen_state_and_visual_hierarchy",
            "mistake_critique_and_loop_diagnosis", "end_to_end_master_plan",
            "tactical_action_justification", "intent_analysis", "observation_analysis",
            "spatial_anchoring", "critique_of_previous_step", "strategy_plan", "action_plan"
        ):
            if f in delib and f not in decision:
                decision[f] = delib[f]

        # 3. Dot ID recovery from deliberation text if omitted from top-level
        if decision.get("target_dot_id") is None and decision.get("target_droplet_id") is None:
            scan_texts = [
                str(delib.get("tactical_action_justification", "")),
                str(delib.get("candidate_evaluation", "")),
                str(delib.get("screen_state_and_visual_hierarchy", "")),
                str(delib.get("action_plan", "")),
                str(decision.get("reasoning", "")),
                str(decision.get("semantic_intention", "")),
                str(decision.get("next_subgoal", ""))
            ]
            plan_list = delib.get("end_to_end_master_plan") or []
            if isinstance(plan_list, list):
                scan_texts.extend(str(p) for p in plan_list)
            combined_scan = " ".join(scan_texts)
            m_dot = re.search(r'\b(?:dot|droplet|node|target)[\s\[#:]*(\d+)\b', combined_scan, re.IGNORECASE)
            if m_dot:
                recovered_id = int(m_dot.group(1))
                decision["target_dot_id"] = recovered_id
                decision["target_droplet_id"] = recovered_id

        # 4. Canonical action normalization & ZERO PASSIVE TEXT ENFORCEMENT
        raw_act = decision.get("action") or decision.get("action_type") or ""
        clean_act = str(raw_act).strip().lower().replace("-", "_").replace(" ", "_")

        # Direct synonym mapping
        if clean_act in cls.ACTION_SYNONYMS:
            clean_act = cls.ACTION_SYNONYMS[clean_act]

        # Passive action detection and deterministic conversion
        if clean_act in cls.PASSIVE_ACTIONS or (clean_act not in cls.CANONICAL_ACTIONS and clean_act != "press"):
            scan_texts = [
                str(delib.get("tactical_action_justification", "")),
                str(delib.get("candidate_evaluation", "")),
                str(delib.get("action_plan", "")),
                str(decision.get("reasoning", "")),
                str(decision.get("semantic_intention", "")),
                str(decision.get("next_subgoal", ""))
            ]
            plan_list = delib.get("end_to_end_master_plan") or []
            if isinstance(plan_list, list):
                scan_texts.extend(str(p) for p in plan_list)
            text_to_scan = " ".join(scan_texts).lower()

            if decision.get("status") == "completed" or any(k in text_to_scan for k in ("goal complete", "task finished", "completed", "task goal satisfied")):
                clean_act = "stop"
            elif any(k in text_to_scan for k in ("double click", "double_click", "doubleclick")):
                clean_act = "double_click"
            elif any(k in text_to_scan for k in ("clear and type", "clear_and_type", "replace text", "overwrite text")):
                clean_act = "clear_and_type"
            elif any(k in text_to_scan for k in ("type and enter", "type_and_enter", "press enter")):
                clean_act = "type_and_enter" if decision.get("input_text") else "press"
            elif any(k in text_to_scan for k in ("launch", "open app", "run app", "start app")):
                clean_act = "launch_app"
            elif any(k in text_to_scan for k in ("open path", "open folder", "navigate to", "open file")):
                clean_act = "open_path"
            elif any(k in text_to_scan for k in ("minimize", "minimize_window")):
                clean_act = "minimize"
            elif any(k in text_to_scan for k in ("maximize", "maximize_window")):
                clean_act = "maximize"
            elif any(k in text_to_scan for k in ("hotkey", "shortcut", "alt+", "ctrl+", "win+")):
                clean_act = "hotkey"
            elif any(k in text_to_scan for k in ("switch window", "switch_window")):
                clean_act = "switch_window"
            elif decision.get("target_dot_id") is not None:
                clean_act = "click"
            elif decision.get("input_text"):
                clean_act = "clear_and_type"
            elif decision.get("key_combo"):
                clean_act = "hotkey"
            elif decision.get("key"):
                clean_act = "press"
            else:
                clean_act = "stop" if decision.get("status") == "completed" else "click"

        decision["action"] = clean_act
        decision["action_type"] = clean_act

        # 5. Hotkey auto-recovery & normalization
        act = decision["action"]
        if act in ("hotkey", "chord"):
            combo = decision.get("key_combo")
            if isinstance(combo, str):
                decision["key_combo"] = [k.strip().lower() for k in combo.split('+') if k.strip()]
            elif isinstance(combo, list):
                decision["key_combo"] = [str(k).strip().lower() for k in combo if k and str(k).strip().lower() not in ("none", "null", "")]
            else:
                decision["key_combo"] = []

            if not decision["key_combo"]:
                for field in ("key", "hotkey", "keys", "shortcut", "input_text"):
                    val = decision.get(field)
                    if val:
                        if isinstance(val, list):
                            decision["key_combo"] = [str(k).strip().lower() for k in val if k]
                            break
                        elif isinstance(val, str) and val.strip().lower() not in ("none", "null", ""):
                            decision["key_combo"] = [k.strip().lower() for k in val.split('+') if k.strip()]
                            break

            if not decision["key_combo"]:
                scan_chunks = [
                    str(decision.get("strategy_plan", "")),
                    str(delib.get("strategy_plan", "")),
                    str(decision.get("candidate_evaluation", "")),
                    str(delib.get("candidate_evaluation", "")),
                    str(delib.get("user_intent_and_success_criteria", "")),
                    str(delib.get("screen_state_and_visual_hierarchy", "")),
                    str(delib.get("mistake_critique_and_loop_diagnosis", "")),
                    str(delib.get("tactical_action_justification", "")),
                    str(decision.get("semantic_intention", "")),
                    str(decision.get("next_subgoal", ""))
                ]
                plan_list = delib.get("end_to_end_master_plan") or []
                if isinstance(plan_list, list):
                    scan_chunks.extend(str(p) for p in plan_list)
                text_to_scan = " ".join(scan_chunks).lower()
                m_hk = re.search(r'\b(alt|ctrl|shift|win)\s*[\+\-]\s*([a-z0-9]|tab|enter|esc|f\d+)\b', text_to_scan)
                if m_hk:
                    decision["key_combo"] = [m_hk.group(1).lower(), m_hk.group(2).lower()]
                elif "alt+d" in text_to_scan or "alt + d" in text_to_scan:
                    decision["key_combo"] = ["alt", "d"]
                elif "ctrl+t" in text_to_scan or "ctrl + t" in text_to_scan:
                    decision["key_combo"] = ["ctrl", "t"]
                elif "ctrl+w" in text_to_scan or "ctrl + w" in text_to_scan:
                    decision["key_combo"] = ["ctrl", "w"]
                elif "ctrl+r" in text_to_scan or "ctrl + r" in text_to_scan or "f5" in text_to_scan:
                    decision["key_combo"] = ["ctrl", "r"]
                elif "alt+tab" in text_to_scan:
                    decision["key_combo"] = ["alt", "tab"]

        # 6. Press key auto-recovery & normalization
        elif act == "press":
            k = decision.get("key") or decision.get("input_text") or decision.get("key_name")
            if not k or str(k).strip().lower() in ("none", "null", ""):
                scan_chunks = [
                    str(decision.get("strategy_plan", "")),
                    str(delib.get("strategy_plan", "")),
                    str(decision.get("candidate_evaluation", "")),
                    str(delib.get("candidate_evaluation", "")),
                    str(delib.get("tactical_action_justification", "")),
                    str(delib.get("user_intent_and_success_criteria", ""))
                ]
                text_to_scan = " ".join(scan_chunks).lower()
                for key_name in ("enter", "esc", "tab", "backspace", "delete", "space", "up", "down", "left", "right"):
                    if key_name in text_to_scan:
                        k = key_name
                        break
            decision["key"] = str(k or "enter").strip().lower()

        return decision

    @classmethod
    def _extract_json(cls, raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw or not isinstance(raw, str):
            return None

        # 1. Strip <think>...</think> blocks from reasoning models (including unclosed <think>)
        cleaned_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        if '<think>' in cleaned_raw:
            cleaned_raw = re.sub(r'<think>.*$', '', cleaned_raw, flags=re.DOTALL)
        cleaned_raw = cleaned_raw.strip()
        if not cleaned_raw:
            return None

        # 2. Try markdown fenced JSON blocks
        matches = re.findall(r'```(?:json|python)?\s*(\{.*?\})\s*```', cleaned_raw, re.DOTALL | re.IGNORECASE)
        for block in matches:
            try:
                d = json.loads(cls._clean_json_text(block))
                cleaned = cls._clean_decision(d)
                if cleaned:
                    return cleaned
            except Exception:
                pass
            d_py = cls._parse_python_dict(block)
            if d_py:
                cleaned = cls._clean_decision(d_py)
                if cleaned:
                    return cleaned
            repaired = cls._repair_partial_json(cls._clean_json_text(block))
            if repaired:
                try:
                    d = json.loads(repaired)
                    cleaned = cls._clean_decision(d)
                    if cleaned:
                        return cleaned
                except Exception:
                    pass

        # 3. Stack-based balanced brace extraction
        start_indices = [i for i, c in enumerate(cleaned_raw) if c == '{']
        blocks = []
        for start in start_indices:
            stack = 0
            in_quote = None
            escape = False
            for i in range(start, len(cleaned_raw)):
                c = cleaned_raw[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    if in_quote:
                        escape = True
                    continue
                if in_quote:
                    if c == in_quote:
                        in_quote = None
                    continue
                if c in ('"', "'"):
                    in_quote = c
                    continue
                if c == '{':
                    stack += 1
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        blocks.append(cleaned_raw[start:i+1])
                        break

        # Sort blocks by length descending to prioritize full decision objects
        blocks.sort(key=len, reverse=True)
        for block in blocks:
            try:
                d = json.loads(cls._clean_json_text(block))
                cleaned = cls._clean_decision(d)
                if cleaned:
                    return cleaned
            except Exception:
                pass
            d_py = cls._parse_python_dict(block)
            if d_py:
                cleaned = cls._clean_decision(d_py)
                if cleaned:
                    return cleaned
            repaired = cls._repair_partial_json(cls._clean_json_text(block))
            if repaired:
                try:
                    d = json.loads(repaired)
                    cleaned = cls._clean_decision(d)
                    if cleaned:
                        return cleaned
                except Exception:
                    pass

        # 4. Partial / truncated JSON repair for unclosed outer object
        if start_indices:
            repaired = cls._repair_partial_json(cls._clean_json_text(cleaned_raw[start_indices[0]:]))
            if repaired:
                try:
                    d = json.loads(repaired)
                    cleaned = cls._clean_decision(d)
                    if cleaned:
                        return cleaned
                except Exception:
                    pass

        # 5. Regex key-value extraction fallback (for malformed / unquoted JSON)
        m_act = re.search(r'["\']?(?:action|action_type)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)["\']?', cleaned_raw, re.IGNORECASE)
        if m_act:
            parsed: Dict[str, Any] = {"action": m_act.group(1).strip().lower()}
            m_tgt = re.search(r'["\']?(?:target_dot_id|target_droplet_id)["\']?\s*[:=]\s*["\']?(?:dot[\s_\[#:]*)?(\d+|null|none)[\]\s]*["\']?', cleaned_raw, re.IGNORECASE)
            if m_tgt:
                v = m_tgt.group(1).lower()
                parsed["target_dot_id"] = int(v) if v not in ("null", "none") else None

            m_inp = re.search(r'["\']?input_text["\']?\s*[:=]\s*(?:"([^"]*)"|\'([^\']*)\'|null)', cleaned_raw, re.IGNORECASE)
            if m_inp:
                parsed["input_text"] = m_inp.group(1) or m_inp.group(2) or None

            m_key = re.search(r'["\']?key["\']?\s*[:=]\s*(?:"([^"]*)"|\'([^\']*)\'|null)', cleaned_raw, re.IGNORECASE)
            if m_key:
                parsed["key"] = m_key.group(1) or m_key.group(2) or None

            m_combo = re.search(r'["\']?key_combo["\']?\s*[:=]\s*\[(.*?)\]', cleaned_raw, re.IGNORECASE)
            if m_combo:
                keys = [re.sub(r'["\']', '', k).strip().lower() for k in m_combo.group(1).split(',') if k.strip()]
                parsed["key_combo"] = keys

            m_stat = re.search(r'["\']?status["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)["\']?', cleaned_raw, re.IGNORECASE)
            if m_stat:
                parsed["status"] = m_stat.group(1).strip().lower()

            m_sub = re.search(r'["\']?next_subgoal["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
            if m_sub:
                parsed["next_subgoal"] = m_sub.group(1).strip()

            m_sem = re.search(r'["\']?semantic_intention["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
            if m_sem:
                parsed["semantic_intention"] = m_sem.group(1).strip()

            m_exp = re.search(r'["\']?expected_transition["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
            if m_exp:
                parsed["expected_transition"] = m_exp.group(1).strip()

            m_rsn = re.search(r'["\']?reasoning["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
            if m_rsn:
                parsed["reasoning"] = m_rsn.group(1).strip()

            delib: Dict[str, Any] = {}
            for d_field in (
                "observation_analysis", "critique_of_previous_step", "candidate_evaluation", "strategy_plan",
                "user_intent_and_success_criteria", "screen_state_and_visual_hierarchy",
                "mistake_critique_and_loop_diagnosis", "tactical_action_justification"
            ):
                m_df = re.search(rf'["\']?{d_field}["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
                if m_df:
                    delib[d_field] = m_df.group(1).strip()
                    parsed[d_field] = m_df.group(1).strip()

            m_plan = re.search(r'["\']?end_to_end_master_plan["\']?\s*[:=]\s*\[(.*?)\]', cleaned_raw, re.DOTALL | re.IGNORECASE)
            if m_plan:
                phases = [re.sub(r'["\']', '', p).strip() for p in m_plan.group(1).split(',') if p.strip()]
                delib["end_to_end_master_plan"] = phases
                parsed["end_to_end_master_plan"] = phases
            else:
                m_plan_str = re.search(r'["\']?end_to_end_master_plan["\']?\s*[:=]\s*["\']([^"\']+)["\']', cleaned_raw, re.IGNORECASE)
                if m_plan_str:
                    delib["end_to_end_master_plan"] = [m_plan_str.group(1).strip()]
                    parsed["end_to_end_master_plan"] = [m_plan_str.group(1).strip()]

            if delib:
                parsed["deliberation"] = delib
            return cls._clean_decision(parsed)

        # 6. Conversational / prose extraction:
        # Requires either an explicit action declaration OR an actionable command targeting a dot
        m_explicit_act = re.search(r'(?:^|\n|[\.\?!;])\s*(?:action|next action|step|command)\s*[:=\-]\s*([a-zA-Z_-]+)', cleaned_raw, re.IGNORECASE)
        m_dot = re.search(r'\b(?:dot|droplet|node)[\s\[#:]*(\d+)\b', cleaned_raw, re.IGNORECASE)
        m_cmd = re.search(r'\b(click|double[-_ ]click|clear[-_ ]and[-_ ]type|type[-_ ]and[-_ ]enter)\s+(?:on\s+)?(?:dot|droplet|node)[\s\[#:]*(\d+)\b', cleaned_raw, re.IGNORECASE)

        if m_explicit_act or m_cmd or (m_dot and re.search(r'\b(click|double[-_ ]click|clear[-_ ]and[-_ ]type|type[-_ ]and[-_ ]enter|launch|minimize|maximize|stop|finish)\b', cleaned_raw, re.IGNORECASE)):
            chosen_act = "click"
            if m_explicit_act:
                chosen_act = m_explicit_act.group(1).lower().replace("-", "_")
            elif m_cmd:
                chosen_act = m_cmd.group(1).lower().replace("-", "_").replace(" ", "_")
            else:
                m_verb = re.search(r'\b(click|double[-_ ]click|clear[-_ ]and[-_ ]type|type[-_ ]and[-_ ]enter|launch|minimize|maximize|stop|finish)\b', cleaned_raw, re.IGNORECASE)
                if m_verb:
                    chosen_act = m_verb.group(1).lower().replace("-", "_").replace(" ", "_")

            target_id = None
            if m_cmd:
                target_id = int(m_cmd.group(2))
            elif m_dot:
                target_id = int(m_dot.group(1))

            prose_decision: Dict[str, Any] = {
                "action": chosen_act,
                "target_dot_id": target_id,
                "status": "completed" if chosen_act in ("stop", "finish") else "in_progress",
                "reasoning": cleaned_raw[:240],
                "deliberation": {
                    "user_intent_and_success_criteria": cleaned_raw[:120],
                    "screen_state_and_visual_hierarchy": "Recovered from model conversational prose",
                    "mistake_critique_and_loop_diagnosis": "Model returned conversational text; recovered canonical action.",
                    "end_to_end_master_plan": ["[Phase 1] Recovered from prose deliberation (ACTIVE)"],
                    "tactical_action_justification": cleaned_raw[:180]
                }
            }
            return cls._clean_decision(prose_decision)

        # Pure conversational commentary without action directive or dot -> reject as ungrounded
        return None

    @staticmethod
    def _parse_groq_retry_seconds(err_str: str) -> float:
        """Parses Groq rate limit / TPD retry duration (e.g. '13m20.928s', '1m21s', '45s')."""
        m_h = re.search(r'(\d+)\s*h', err_str)
        m_m = re.search(r'(\d+)\s*m(?!s)', err_str)
        m_s = re.search(r'(\d+(?:\.\d+)?)\s*s', err_str)
        secs = 0.0
        if m_h:
            secs += float(m_h.group(1)) * 3600.0
        if m_m:
            secs += float(m_m.group(1)) * 60.0
        if m_s:
            secs += float(m_s.group(1))
        return secs if secs > 0.0 else 60.0

    @staticmethod
    def _render_unoccluded_som(screenshot: Any, candidates: List[Any], max_image_side: int = 1280, jpeg_quality: int = 82):
        """
        Renders a Set-of-Marks (SoM) visual overlay where candidate icons and controls are 100% UNOCCLUDED:
        1. 1px Archetype Bounding Box Boundary: Gives the model the exact boundary of the control.
        2. Hollow Micro-Anchor: At (cx, cy), draws a tiny 2px hollow ring (radius 2, thickness 1),
           never an opaque filled circle blob, ensuring internal glyph strokes (stars, funnels, arrows) are fully visible.
        3. Border-Pinned Translucent Badges: Badges are docked OUTSIDE the top-left boundary of small controls
           (or flipped below if against the top screen edge), preventing badge text from obscuring the icon.
        Supports high-capacity VLMs up to 1920px with explicit memory buffer deallocation.
        """
        if (
            screenshot is None
            or not hasattr(screenshot, 'shape')
            or len(screenshot.shape) < 2
            or screenshot.shape[0] <= 0
            or screenshot.shape[1] <= 0
        ):
            return None, (0, 0), 1.0

        overlay = None
        marked = None
        buf = None
        try:
            sh_orig, sw_orig = screenshot.shape[:2]
            scale = max_image_side / float(max(sw_orig, sh_orig)) if max(sw_orig, sh_orig) > max_image_side else 1.0
            marked = cv2.resize(screenshot, (int(sw_orig * scale), int(sh_orig * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else screenshot.copy()
            img_h, img_w = marked.shape[:2]

            overlay = marked.copy()
            font_scale = 0.38 if max_image_side >= 1920 else (0.34 if max_image_side >= 1024 else 0.32)

            # Pass 1: Draw bounding box frames and translucent badge pill backgrounds
            badge_positions = []
            for d in candidates:
                cx, cy = int(d.center_x * scale), int(d.center_y * scale)
                arch = getattr(d, 'archetype', 'button')
                b_col = ARCHETYPE_COLORS_BGR.get(arch, (118, 230, 0))

                raw_box = getattr(d, 'bbox', None)
                if raw_box and len(raw_box) == 4 and raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
                    x1 = max(0, int(raw_box[0] * scale))
                    y1 = max(0, int(raw_box[1] * scale))
                    x2 = min(img_w - 1, int(raw_box[2] * scale))
                    y2 = min(img_h - 1, int(raw_box[3] * scale))
                else:
                    x1 = max(0, cx - int(12 * scale))
                    y1 = max(0, cy - int(8 * scale))
                    x2 = min(img_w - 1, cx + int(12 * scale))
                    y2 = min(img_h - 1, cy + int(8 * scale))

                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)

                # Draw 1px element outline
                cv2.rectangle(marked, (x1, y1), (x2, y2), b_col, 1)

                txt = str(d.id)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                pw = tw + 5
                ph = th + 4

                # Border-pinned badge placement:
                # For compact controls (icons, buttons, arrows), dock OUTSIDE above the top edge
                if bh <= 45 or bw <= 55:
                    by1 = y1 - ph - 2
                    by2 = y1 - 2
                    bx1 = max(0, min(img_w - pw, x1))
                    bx2 = bx1 + pw
                    if by1 < 2:  # Flipped below if pinned against top screen border
                        by1 = min(img_h - ph - 1, y2 + 2)
                        by2 = by1 + ph
                else:
                    # For large containers or multi-word text inputs, dock in top-left inside corner
                    bx1 = x1 + 2
                    by1 = y1 + 2
                    bx2 = bx1 + pw
                    by2 = by1 + ph

                # Draw badge pill on overlay
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), b_col, -1)
                cv2.rectangle(marked, (bx1, by1), (bx2, by2), (0, 0, 0), 1)
                badge_positions.append((bx1, by1, txt, arch, cx, cy, th))

            # Alpha blend overlay with marked image (80% opacity)
            cv2.addWeighted(overlay, 0.80, marked, 0.20, 0, marked)
            del overlay
            overlay = None

            # Pass 2: Draw crisp text and non-occluding micro hollow ring anchor
            for bx1, by1, txt, arch, cx, cy, th in badge_positions:
                text_color = (255, 255, 255) if arch in ("logo", "label") else (0, 0, 0)
                cv2.putText(marked, txt, (bx1 + 2, by1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
                # Hollow red micro-ring at exact center (radius 2, thickness 1 - leaves glyph interior completely clear)
                cv2.circle(marked, (cx, cy), 2, (0, 0, 255), 1)

            quality = max(1, min(100, int(jpeg_quality)))
            _, buf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, quality])
            b64_str = base64.b64encode(buf.tobytes() if hasattr(buf, "tobytes") else buf).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64_str}"
            return image_url, (img_w, img_h), scale
        finally:
            if overlay is not None:
                del overlay
            if marked is not None:
                del marked
            if buf is not None:
                del buf

    async def decide(
        self,
        prompt: Optional[str] = None,
        droplets: Optional[List[DiagonalDroplet]] = None,
        win_ctx: Optional[WindowContext] = None,
        user_intent: Optional[str] = None,
        window_ctx: Optional[WindowContext] = None,
        memory: Optional[EpisodicTrajectoryMemory] = None,
        screenshot: Optional[Any] = None,
        **kwargs
    ) -> DecisionResult:
        requested_provider = str(kwargs.get("provider") or os.environ.get("SPATIAL_PROVIDER") or "").strip().lower()
        if not self.key_pool and not kwargs.get("candidate_trace_only") and requested_provider not in (
            "gemini", "gemini-pro", "google", "gemini_pro", "gemini-flash", "gemini-1.5-pro", "gemini-2.0-flash",
            "openai", "azure", "local", "ollama", "vllm", "sglang"
        ):
            return DecisionResult({"error": "No OpenAI/Groq clients available in pool", "target_x": None, "target_y": None, "target_droplet_id": None, "action_type": "error", "confidence": 0.0, "reasoning": "No API keys available"})

        instruction = kwargs.get("instruction") or user_intent or prompt or (memory.goal if memory else "")
        active_droplets = droplets if droplets is not None else []
        active_win = window_ctx or win_ctx

        win_info = "Desktop / Full Screen"
        has_win = bool(active_win and active_win.is_valid and active_win.title != "Desktop")
        if has_win:
            win_info = f"\"{active_win.title}\" [Bounds: {active_win.rect}]"

        # Screen dimensions come from the observed image/window bounds; only
        # fall back to detector extents when neither source is available.
        if screenshot is not None and hasattr(screenshot, "shape") and len(screenshot.shape) >= 2:
            sh_screen, sw_screen = screenshot.shape[:2]
        elif has_win and active_win and active_win.rect:
            wx1, wy1, wx2, wy2 = active_win.rect
            sw_screen, sh_screen = max(1, wx2 - wx1), max(1, wy2 - wy1)
        else:
            sw_screen = max((getattr(d, "bbox", (0, 0, d.center_x, d.center_y))[2] for d in active_droplets), default=1)
            sh_screen = max((getattr(d, "bbox", (0, 0, d.center_x, d.center_y))[3] for d in active_droplets), default=1)

        # Fast-Path 1: Tabular / Spreadsheet 2D Cartesian Grid Intersection
        grid_best_droplet = None
        try:
            from ..vision.grid_resolver import TabularGridResolver
            grid_hit = TabularGridResolver.resolve_cell_intent(instruction, active_droplets)
            if grid_hit:
                gx, gy, gdesc = grid_hit
                best_d = min(active_droplets, key=lambda d: math.hypot(d.center_x - gx, d.center_y - gy))
                grid_best_droplet = best_d
                if not kwargs.get("candidate_trace_only"):
                    return DecisionResult({
                        "target_dot_id": best_d.id,
                        "target_x": float(gx),
                        "target_y": float(gy),
                        "target_droplet_id": best_d.id,
                        "action_type": "click",
                        "confidence": 0.98,
                        "reasoning": f"Tabular Grid Resolver computed exact 2D intersection for {gdesc}",
                        "model_used": "tabular_grid_resolver",
                        "key_used": "Deterministic Geometry"
                    })
        except Exception:
            pass

        # Fast-Path 2: Direct Continuous Slider & Sub-Character Punctuation/Letter Geometry
        try:
            from ..core.spatial_geometry import SpatialGeometryEngine
            geo_hit = SpatialGeometryEngine.resolve_direct_geometry(
                instruction, active_droplets, sw_screen=int(sw_screen), sh_screen=int(sh_screen)
            )
            if geo_hit:
                gx, gy, g_droplet, g_desc = geo_hit
                if grid_best_droplet is None:
                    grid_best_droplet = g_droplet
                if not kwargs.get("candidate_trace_only"):
                    return DecisionResult({
                        "target_dot_id": g_droplet.id,
                        "target_x": float(gx),
                        "target_y": float(gy),
                        "target_droplet_id": g_droplet.id,
                        "action_type": "click",
                        "confidence": 0.98,
                        "reasoning": f"SpatialGeometryEngine resolved exact sub-element coordinate: {g_desc}",
                        "model_used": "spatial_geometry_engine",
                        "key_used": "Deterministic Geometry"
                    })
        except Exception:
            pass

        key_nodes = [
            d for d in active_droplets 
            if d.category != 'surface_probe' or getattr(d, 'semantic_role', '') == 'control' or (getattr(d, 'label', '') and not (d.label.startswith('Lattice Star') or d.label.startswith('Lattice Probe')))
        ]
        max_ui_nodes = 320

        UI_STOPWORDS = {
            "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "by", "from", "with",
            "click", "open", "select", "want", "current", "this", "that", "it", "is", "be",
            "screen", "area"
        }
        prompt_lower = instruction.lower()
        prompt_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', prompt_lower))
        quoted_words = set(w.lower() for w in re.findall(r'["\']([a-zA-Z0-9]+)["\']', prompt_lower) if len(w) >= 2)
        rare_prompt_words = set(w for w in prompt_words if len(w) >= 3 and w not in UI_STOPWORDS) | quoted_words

        ACTION_CONTROL_MAP = [
            # 0. Affirmative / Confirm / Replace / Overwrite Dialog Actions
            ({"confirm", "accept", "replace", "overwrite", "proceed", "agree", "yes", "ok"},
             {"yes", "ok", "confirm", "replace", "apply", "save", "continue"}, 950),

            # Checkbox / Toggle Option
            ({"checkbox", "check", "uncheck", "toggle", "option"}, 
             {"checkbox", "toggle", "square option", "check", "color swatch", "color cell"}, 650),

            # 1. Close / Dismiss / Cancel / Exit
            ({"close", "dismiss", "cancel", "exit", "quit"}, 
             {"close", "cross", "dismiss", "x", "delete", "cancel", "toggle", "action icon"}, 450),
           
            # 2. Expand / Unfold / Show / More / Dropdown
            ({"expand", "unfold", "dropdown", "drop-down", "more", "options", "downward"}, 
             {"expand", "dropdown", "down arrow", "chevron", "plus", "unfold", "toggle", "toolbar tool", "visual button"}, 450),
           
            # 3. Collapse / Fold / Hide / Less / Panel dividers
            ({"collapse", "fold", "hide", "less"}, 
             {"collapse", "up arrow", "chevron", "fold", "toggle", "handle", "divider", "v", "^", ">", "<"}, 350),
            ({"hide", "close", "collapse"}, 
             {"panel divider", "splitter handle", "divider"}, 850),
           
            # 4. Star / Favorite / Bookmark
            ({"star", "favorite", "bookmark"}, 
             {"star", "favorite", "bookmark", "address"}, 800),
            
            # 5. Underline / Strikethrough / Bold / Italic / Formatting
            ({"underline", "underlining"}, 
             {"underline", "u", "format", "toolbar tool", "visual button"}, 750),
            ({"strikethrough", "strike", "bold", "italic", "underline", "font", "bullets", "bullet"}, 
             {"strikethrough", "format", "strike", "font", "bold", "italic", "underline", "bullet", "toolbar tool"}, 400),
            
            # 6. Question / Help / Info / Prompt / Settings
            ({"question", "help", "info", "settings"}, 
             {"question", "help", "prompt", "info", "settings", "toolbar tool", "visual button"}, 750),

            # 7. Undo / Redo / History / Search
            ({"undo", "redo", "history", "search"}, 
             {"undo", "redo", "refresh", "curved arrow", "history", "search", "find"}, 550),
            
            # 8. Navigation: Back / Forward / Previous / Next / Rewind
            ({"back", "previous", "rewind", "beginning"}, 
             {"back", "rewind", "previous", "left arrow", "start", "beginning", "trackbar left", "slider left"}, 500),
            ({"forward", "next", "fast"}, 
             {"forward", "fast", "next", "right arrow", "play"}, 400),
            
            # 9. Numeric adjustments & Percentage / Zoom Sliders
            ({"increase", "raise", "increment", "up", "maximum"}, 
             {"increment", "up arrow", "up", "raise", "plus", "expand", "close"}, 450),
            ({"lower", "decrease", "decrement", "down", "reduce"}, 
             {"decrement", "down arrow", "down", "lower", "toolbar tool", "visual button"}, 450),
            ({"position", "slider", "zoom", "quarter", "percent", "middle", "100", "80", "30", "25"}, 
             {"slider", "trackbar", "25%", "75%", "80%", "handle", "plus", "expand"}, 850),
            
            # 10. Scrolling / Navigation through panels
            ({"bottom", "see", "scroll"}, 
             {"scrollbar", "scroll rail", "slider bottom", "slider", "lower segment", "handle", "down arrow"}, 350),
            
            # 11. Selection / Tools
            ({"brush", "paintbrush", "pencil", "pen", "transform", "wand", "fuzzy", "paths", "ellipse", "symbol", "shape"}, 
             {"paintbrush", "brush", "pencil", "draw", "tool", "toolbar tool", "visual button", "star", "expand", "bullet"}, 650),
            ({"eraser", "erase"}, 
             {"eraser", "tool"}, 450),
            ({"bucket", "fill"}, 
             {"bucket", "fill", "gradient", "tool", "star", "toolbar tool", "visual button"}, 650),
            ({"margin", "ruler", "indent"}, 
             {"ruler", "margin", "indent", "slider", "handle", "marker", "thumb", "arrow", "trackbar"}, 750),
            ({"color", "palette", "swatch", "blue", "column"}, 
             {"color swatch", "swatch", "palette box", "color cell", "blue", "column"}, 650),
            ({"fullscreen", "full"}, 
             {"fullscreen", "maximize", "video fullscreen"}, 650),
            ({"filter"}, 
             {"filter", "funnel"}, 600),
            ({"debug", "panel", "run"}, 
             {"run and debug", "debug panel", "play", "start"}, 650)
        ]

        def _node_bounds(node):
            box = getattr(node, 'bbox', None)
            if box is None:
                return (node.center_x, node.center_y, node.center_x, node.center_y)
            if hasattr(box, 'x1'):
                return (box.x1, box.y1, box.x2, box.y2)
            return tuple(box)

        def _boxes_near(a, b, x_scale=0.65, y_scale=0.35):
            ax1, ay1, ax2, ay2 = _node_bounds(a)
            bx1, by1, bx2, by2 = _node_bounds(b)
            aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            dx = max(0, max(ax1 - bx2, bx1 - ax2))
            dy = max(0, max(ay1 - by2, by1 - ay2))
            return dx <= x_scale * max(aw, bw) and dy <= y_scale * max(ah, bh)

        # Parse directional relations only when the instruction names a
        # repeated layout. This avoids interpreting "scroll to the bottom"
        # or "lower the value" as an ordinal selection.
        layout_relation = None
        layout_group = None
        relation_specs = (
            ({"bottom", "lowest", "last"}, "bottom", {"column", "grid", "list", "sequence"}),
            ({"top", "highest", "first"}, "top", {"column", "grid", "list", "sequence"}),
            ({"leftmost", "left"}, "left", {"row", "grid", "list", "sequence"}),
            ({"rightmost", "right"}, "right", {"row", "grid", "list", "sequence"}),
        )
        for relation_words, relation_name, group_words in relation_specs:
            if prompt_words.intersection(relation_words) and prompt_words.intersection(group_words):
                layout_relation, layout_group = relation_name, next(iter(prompt_words.intersection(group_words)))
                break

        _layout_position_cache = {}

        def _layout_position(node):
            if layout_relation is None:
                return None
            if node.id in _layout_position_cache:
                return _layout_position_cache[node.id]
            node_role = str(getattr(node, "semantic_role", ""))
            node_type = str(getattr(node, "archetype", ""))
            node_container = str(getattr(node, "container_title", ""))
            nx1, ny1, nx2, ny2 = _node_bounds(node)
            nw, nh = max(1.0, nx2 - nx1), max(1.0, ny2 - ny1)
            ncx, ncy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
            siblings = []
            for candidate in key_nodes:
                if candidate.id == node.id:
                    siblings.append(candidate)
                    continue
                if (str(getattr(candidate, "semantic_role", "")) != node_role
                        or str(getattr(candidate, "archetype", "")) != node_type
                        or str(getattr(candidate, "container_title", "")) != node_container):
                    continue
                cx1, cy1, cx2, cy2 = _node_bounds(candidate)
                cw, ch = max(1.0, cx2 - cx1), max(1.0, cy2 - cy1)
                ccx, ccy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
                # Use the smaller control dimension so adjacent grid columns
                # are not merged just because one detector box is oversized.
                x_tolerance = max(2.0, 0.60 * min(nw, cw))
                y_tolerance = max(2.0, 0.60 * min(nh, ch))
                same_track = abs(ccx - ncx) <= x_tolerance if layout_relation in {"top", "bottom"} else abs(ccy - ncy) <= y_tolerance
                if same_track:
                    siblings.append(candidate)
            if len(siblings) < 3:
                _layout_position_cache[node.id] = None
                return None
            siblings.sort(key=(lambda item: (-item.center_y, item.center_x)) if layout_relation == "bottom"
                          else (lambda item: (item.center_y, item.center_x)) if layout_relation == "top"
                          else (lambda item: (-item.center_x, item.center_y)) if layout_relation == "right"
                          else (lambda item: (item.center_x, item.center_y)))
            result = (1 + next(i for i, item in enumerate(siblings) if item.id == node.id), len(siblings))
            _layout_position_cache[node.id] = result
            return result

        # Pre-compute keyword-matching anchor nodes for Same-Row Peer Control Inheritance
        action_exclude_words = {
            "turn", "make", "set", "change", "switch", "toggle", "check", "uncheck",
            "enable", "disable", "choose", "select", "click", "press", "open", "close",
            "adjust", "modify", "use", "find", "show", "hide", "expand", "collapse"
        }
        entity_rare_words = rare_prompt_words - action_exclude_words
        if not entity_rare_words:
            entity_rare_words = rare_prompt_words

        keyword_matched_anchors = []
        for node in key_nodes:
            n_text = f"{getattr(node, 'label', '')} {getattr(node, 'attached_label', '')}".lower()
            n_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', n_text))
            matched_ew = entity_rare_words.intersection(n_words)
            if not matched_ew:
                for pw in entity_rare_words:
                    for cw in n_words:
                        if len(pw) >= 4 and len(cw) >= 4 and (pw.startswith(cw[:4]) or cw.startswith(pw[:4])):
                            matched_ew.add(pw)
                            break
            if matched_ew:
                keyword_matched_anchors.append((node, len(matched_ew)))

        def relevance_score(d: DiagonalDroplet):
            label_lower = d.label.lower()
            att_lower = getattr(d, 'attached_label', '').lower()
            combined_text = label_lower + " " + att_lower
            
            # Extract "(Near '...')" or "(for '...')" or "for '...'"
            near_match = re.search(r"(?:\((?:near|for)\s+'([^']+)'\)|for\s+'([^']+)'|\(near\s+'([^']+)'\))", label_lower)
            near_text = ""
            if near_match:
                near_text = (near_match.group(1) or near_match.group(2) or near_match.group(3) or "").lower()
            if near_text:
                combined_text += " " + near_text

            combined_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', combined_text))
            
            # 1. Exact rare word matches (+300 pts per content word)
            rare_matches = rare_prompt_words.intersection(combined_words)
            score = len(rare_matches) * 300

            if grid_best_droplet is not None and d.id == grid_best_droplet.id:
                score += 3500

            # General relation-aware retrieval for ordered UI collections.
            # Boost only the requested extreme among like controls in the
            # same detected container and geometric track.
            layout_position = _layout_position(d)
            if layout_position and layout_position[0] == 1:
                score += 1200
            
            # Stemming / prefix fuzzy matching (e.g. "donations" <-> "donate", "highlight" <-> "highlighting", "slides" <-> "slide")
            for pw in rare_prompt_words:
                if pw not in rare_matches:
                    for cw in combined_words:
                        if len(pw) >= 4 and len(cw) >= 4:
                            if pw.startswith(cw[:4]) or cw.startswith(pw[:4]):
                                score += 250
                                break

            # Give detectable controls a modest baseline; do not infer their
            # meaning or target from global screen coordinates.
            is_visual_widget = (d.label.startswith("[Icon]") or 
                                d.label.startswith("[Btn]") or
                                getattr(d, 'archetype', '') in ('button', 'icon', 'control', 'slider', 'tab', 'stepper') or
                                getattr(d, 'semantic_role', '') in ('control', 'slider', 'action'))
            if is_visual_widget:
                score += 350

            # Same-Row & Panel-Relative Peer Control Inheritance:
            # In desktop settings/dialogs/forms, the interactive control (toggle, checkbox, dropdown, button, resize corner)
            # sits on the same horizontal row (|dy| <= 24px, |dx| <= 520px) or lower-right corner of its header/label.
            if is_visual_widget and keyword_matched_anchors:
                best_peer_hits = 0
                best_peer_dx = 9999.0
                wants_corner = any(k in prompt_words for k in ("corner", "resize", "adjust", "size", "lower", "bottom"))
                for anchor_node, hit_count in keyword_matched_anchors:
                    if anchor_node.id == d.id:
                        continue
                    dy = abs(d.center_y - anchor_node.center_y)
                    dx = abs(d.center_x - anchor_node.center_x)
                    if dy <= 24 and 4 <= dx <= 520:
                        if hit_count > best_peer_hits or (hit_count == best_peer_hits and dx < best_peer_dx):
                            best_peer_hits = hit_count
                            best_peer_dx = dx
                    elif wants_corner and (0 <= d.center_y - anchor_node.center_y <= 320) and (0 <= d.center_x - anchor_node.center_x <= 380):
                        score += 950
                if best_peer_hits > 0:
                    score += 650 + (best_peer_hits * 350)
                    if best_peer_dx <= 160:
                        score += 250

            # Status-bar Zoom Slider Rescue
            if ("zoom" in prompt_words or "slider" in prompt_words) and d.center_y >= sh_screen * 0.88 and d.center_x >= sw_screen * 0.55:
                if is_visual_widget or "%" in combined_text or any(k in combined_text for k in ("slider", "zoom", "plus", "minus", "trackbar")):
                    score += 1100

            # Stopwords (+15 pts only if already grounded, heavy penalty for standalone OCR stopwords)
            if d.label.lower().strip() in UI_STOPWORDS and not rare_matches and not is_visual_widget:
                score -= 800
            else:
                stop_matches = (prompt_words.intersection(UI_STOPWORDS)).intersection(combined_words)
                score += len(stop_matches) * 15

            # Checkbox / Toggle explicit matching
            if any(k in prompt_words for k in ("checkbox", "check", "toggle", "uncheck", "deselect", "turn", "enable", "disable", "switch")):
                if any(k in label_lower for k in ("checkbox", "toggle", "square option", "radio", "switch")):
                    score += 900
                    # If this checkbox is associated with rare prompt words, give major boost
                    if any(pw in near_text for pw in rare_prompt_words) or any(pw in combined_text for pw in rare_prompt_words):
                        score += 950

            # Prefer explicit dialog ownership metadata when the request names
            # a dialog; never guess a dialog from screen-center coordinates.
            is_dialog_intent = any(k in prompt_words for k in ("dialog", "modal", "popup", "alert", "prompt", "notification", "confirm", "replace"))
            if is_dialog_intent:
                dialog_id = str(getattr(d, 'container_id', '')).lower()
                dialog_title = str(getattr(d, 'container_title', '')).lower()
                if any(term in dialog_id or term in dialog_title for term in ("dialog", "modal", "popup")):
                    score += 700

            # Question / Help / Info matching
            if any(k in prompt_words for k in ("question", "help", "info", "faq")):
                if any(k in label_lower for k in ("help", "question", "info")):
                    score += 1000
                elif d.label.startswith("[Icon]") or getattr(d, 'archetype', '') == 'button':
                    dialog_id = str(getattr(d, 'container_id', '')).lower()
                    dialog_title = str(getattr(d, 'container_title', '')).lower()
                    if is_dialog_intent and any(term in dialog_id or term in dialog_title for term in ("dialog", "modal", "popup")):
                        score += 750

            # Exact multi-word phrase matching (+450 pts)
            words_list = instruction.lower().split()
            for phrase_len in (3, 2):
                for i in range(len(words_list) - phrase_len + 1):
                    phrase = " ".join(words_list[i:i+phrase_len])
                    if len(phrase) >= 5 and phrase in combined_text:
                        score += 450
                        break

            # Attached label / contextual proximity match (+300 pts)
            att_context_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', att_lower + " " + near_text))
            att_rare_hits = rare_prompt_words.intersection(att_context_words)
            if att_rare_hits:
                score += 300
                if d.label.startswith("[Icon]") or getattr(d, 'semantic_role', '') in ('control', 'action'):
                    # An interactive control directly attached to the target entity (e.g. checkbox/toggle for 'Highlight')
                    # is an actionable control for that entity
                    score += len(att_rare_hits) * 500
            else:
                for pw in rare_prompt_words:
                    for cw in att_context_words:
                        if len(pw) >= 4 and len(cw) >= 4 and (pw.startswith(cw[:4]) or cw.startswith(pw[:4])):
                            score += 250
                            if d.label.startswith("[Icon]") or getattr(d, 'semantic_role', '') in ('control', 'action'):
                                score += 350
                            break

            # Role and category base scores
            if getattr(d, 'semantic_role', '') == 'control':
                score += 80
            if d.is_foreground:
                score += 50
            if has_win and active_win and active_win.rect:
                wx1, wy1, wx2, wy2 = active_win.rect
                if wx1 <= d.center_x <= wx2 and wy1 <= d.center_y <= wy2:
                    score += 50

            # Direct Primary Entity Match (+1400 pts):
            # If the droplet's direct label is the target entity itself (e.g. "Chrome", "Brave", "This PC"), award major bonus!
            clean_label = re.sub(r'\[item \d+ of \d+.*?\]', '', label_lower).strip()
            clean_label = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean_label).strip()
            is_auxiliary = any(k in label_lower for k in ["[action control", "right of", "left of", "near '", "for '"])

            direct_label_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', clean_label))
            direct_rare = rare_prompt_words.intersection(direct_label_words)
            if direct_rare and not is_auxiliary:
                score += len(direct_rare) * 600
                if any(pw == clean_label or clean_label.startswith(pw) for pw in rare_prompt_words):
                    score += 800

            # If user asks to "open", "launch", "start", "run", or "click" an app/program:
            if d.category == 'interactive_app':
                score += 100
                if direct_rare:
                    score += 1100

            # Auxiliary penalty: if user directly named an entity without asking for "near" or "control", penalize indirect contours (exclude genuine icons)
            if is_auxiliary and not d.label.startswith("[Icon]") and not any(k in prompt_words for k in ["control", "right", "left", "near", "arrow", "undo", "redo", "icon", "tool", "button"]):
                score -= 500

            if str(d.id) in prompt_words:
                score += 250

            # Generic uninformative icon penalty (only penalize noise when no tool/action intent is present)
            if any(g in label_lower for g in ["small graphic", "graphic element"]) or label_lower.strip() in ("[icon] icon", "icon"):
                score -= 80

            # Hierarchical Container Scoping
            c_title = getattr(d, 'container_title', '').lower()
            c_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', c_title))
            c_hits = rare_prompt_words.intersection(c_words)
            if c_hits:
                score += len(c_hits) * 550
            else:
                for pw in rare_prompt_words:
                    for cw in c_words:
                        if len(pw) >= 4 and len(cw) >= 4 and (pw.startswith(cw[:4]) or cw.startswith(pw[:4])):
                            score += 550
                            break

            # 2. Semantic Action-to-Control mapping
            for act_keywords, ctrl_keywords, boost in ACTION_CONTROL_MAP:
                if bool(prompt_words.intersection(act_keywords)):
                    if any(ck in combined_text for ck in ctrl_keywords):
                        # Explicit affordance compatibility should outweigh a
                        # distractor that shares only the object/context noun.
                        score += boost + 450

            # 3. Direct syntactic association on same row ("for 'X'")
            if "for '" in label_lower and any(pw in near_text for pw in rare_prompt_words):
                score += 350

            # 4. Action + Target Context Synergy
            if any(k in prompt_words for k in ["close", "dismiss", "cancel"]):
                if any(k in label_lower for k in ["close", "cross", "dismiss", "cancel", "clear"]):
                    score += 800
                # Action verb disambiguation: "cancel the [feature]" -> prefer checkbox/toggle for [feature]
                if "cancel" in prompt_words:
                    feature_words = rare_prompt_words - {"cancel"}
                    if feature_words:
                        if (any(k in label_lower for k in ["checkbox", "toggle", "check", "button", "option"]) or d.label.startswith("[Icon]")) and any(fw in combined_text for fw in feature_words):
                            score += 1200  # Prefer checkbox/toggle matching the feature over dialog Cancel button

            if any(k in prompt_words for k in ["expand", "unfold"]):
                if d.label.startswith("[Icon]") and any(k in label_lower for k in ["expand", "toggle", "down arrow", "chevron"]):
                    score += 950
                elif any(k in label_lower for k in ["expand", "unfold", "toggle"]):
                    score += 600
                if any(pw in combined_text for pw in rare_prompt_words):
                    score += 450

            if any(k in prompt_words for k in ["collapse", "fold"]):
                if any(k in label_lower for k in ["collapse", "fold", "toggle", "up arrow", "chevron"]):
                    score += 800
            # 5. Side panel divider handles (Left vs Right)
            if ("panel" in prompt_words or "sidebar" in prompt_words) and "divider" in label_lower:
                score += 500
                if "left" in prompt_words and "left" in label_lower:
                    score += 450
                elif "right" in prompt_words and "right" in label_lower:
                    score += 450
            # 6. Ordinal Reasoning (first, 1st, second, 2nd, fifth, 5th, last, etc.)
            ORDINAL_MAP = {
                "first": 1, "1st": 1,
                "second": 2, "2nd": 2,
                "third": 3, "3rd": 3,
                "fourth": 4, "4th": 4,
                "fifth": 5, "5th": 5,
                "sixth": 6, "6th": 6,
                "seventh": 7, "7th": 7,
                "eighth": 8, "8th": 8,
                "ninth": 9, "9th": 9,
                "tenth": 10, "10th": 10,
            }
            for ord_w, ord_n in ORDINAL_MAP.items():
                if ord_w in prompt_words:
                    if f"[item {ord_n} of " in label_lower:
                        score += 750
                    elif ord_n == 1 and "(first / " in label_lower:
                        score += 750

            if any(k in prompt_words for k in ["last", "bottommost", "rightmost"]):
                if "(last / " in label_lower:
                    score += 650

            # Geometric ordinal fallback: if label doesn't have [item N of M],
            # resolve ordinals by spatial position among semantically similar candidates
            d_norm_base = re.sub(r'\s*(?:for|near)\s*[\'"].*?[\'"]', '', re.sub(r'\[item \d+ of \d+.*?\]', '', d.label)).strip()
            for ord_w, ord_n in ORDINAL_MAP.items():
                if ord_w in prompt_words:
                    # Find all candidates with similar semantic class within the same container scope
                    d_container = getattr(d, 'container_title', '')
                    geo_siblings = sorted(
                        [s for s in key_nodes
                         if re.sub(r'\s*(?:for|near)\s*[\'"].*?[\'"]', '', re.sub(r'\[item \d+ of \d+.*?\]', '', s.label)).strip() == d_norm_base
                         and (getattr(s, 'container_title', '') == d_container or (not d_container and not getattr(s, 'container_title', '')))],
                        key=lambda s: (s.center_y, s.center_x)
                    )
                    if len(geo_siblings) >= ord_n and geo_siblings[ord_n - 1].id == d.id:
                        score += 800
                    break

            if any(k in prompt_words for k in ["last", "bottommost", "rightmost"]):
                d_container = getattr(d, 'container_title', '')
                geo_siblings = sorted(
                    [s for s in key_nodes
                     if re.sub(r'\s*(?:for|near)\s*[\'"].*?[\'"]', '', re.sub(r'\[item \d+ of \d+.*?\]', '', s.label)).strip() == d_norm_base
                     and (getattr(s, 'container_title', '') == d_container or (not d_container and not getattr(s, 'container_title', '')))],
                    key=lambda s: (s.center_y, s.center_x)
                )
                if geo_siblings and geo_siblings[-1].id == d.id:
                    score += 700

            # 7. Numeric spinbox adjustments
            spin_kw = {"increase", "lower", "increment", "decrement", "spin", "raise", "reduce", "value"}
            if bool(prompt_words.intersection(spin_kw)):
                att_val = getattr(d, 'attached_label', '')
                is_num = bool(re.search(r'^-?\d+(\.\d+)?$', att_val.strip()))
                is_arrow = any(k in d.label.lower() for k in ["increment", "decrement", "arrow", "up", "down", "spinbox", "action control", "forward"])
                if is_num or is_arrow:
                    score += 400
                if "increase" in prompt_words or "raise" in prompt_words or "up" in prompt_words:
                    if any(k in d.label.lower() for k in ["up arrow", "increment", "spin"]):
                        score += 650
                elif "lower" in prompt_words or "decrease" in prompt_words or "down" in prompt_words:
                    if any(k in d.label.lower() for k in ["down arrow", "decrement", "spin"]):
                        score += 650
                if is_num:
                    score += 250

            # 8. Fullscreen corner preference
            if "fullscreen" in prompt_words or ("full" in prompt_words and "screen" in prompt_words):
                if "fullscreen" in label_lower or "maximize" in label_lower:
                    score += 750

            # 9. Redo / Undo action preference
            if "redo" in prompt_words and "redo" in d.label.lower():
                score += 550
            elif "undo" in prompt_words and "undo" in d.label.lower():
                score += 550

            # 10. Media player playback controls (Rewind / Beginning / Fast forward / Timeline)
            media_kw = {"video", "playback", "player", "beginning", "rewind", "forward", "fast"}
            if bool(prompt_words.intersection(media_kw)):
                if "media" in label_lower or re.search(r'\d+:\d+', combined_text):
                    score += 400
                if re.search(r'\b\d+:\d+\b', combined_text):
                    # Dedicated media timeline timecode association bonus
                    score += 350
                if re.search(r'\b\d+:\d+\b', combined_text):
                    score += 650
                if "beginning" in prompt_words:
                    if any(k in label_lower for k in ["beginning", "slider left", "trackbar left", "start"]):
                        score += 1200
                    elif any(k in label_lower for k in ["rewind", "previous"]):
                        score += 400
                if "fast" in prompt_words and any(k in label_lower for k in ["fast", "forward", "right arrow"]):
                    score += 600

            # 11. Scrollbar and panel navigation (bottom / scroll)
            if not any(k in prompt_words for k in ["close", "dismiss", "cancel", "remove", "exit"]):
                if any(k in label_lower for k in ["close", "cross", "dismiss"]):
                    score -= 1200

            if any(k in prompt_words for k in ["scrollbar", "scroll", "bottom"]):
                if "scrollbar" in label_lower or "slider" in label_lower or getattr(d, 'archetype', '') == 'slider':
                    score += 650
                    if any(k in prompt_words for k in ["bottom", "down", "lower"]):
                        if "slider bottom" in label_lower or "slider_bottom" in label_lower:
                            score += 850

            # 12. Ruler and document margins
            if any(k in prompt_words for k in ("margin", "ruler", "indent")):
                if any(k in label_lower for k in ["ruler", "margin", "indent", "slider", "trackbar", "handle"]):
                    score += 750
                    if (("right" in prompt_words and "right" in label_lower)
                            or ("left" in prompt_words and "left" in label_lower)):
                        score += 450

            # 13. Color Swatch and palette cell selection
            if "color" in prompt_words and not any(k in prompt_words for k in ["increase", "lower", "increment", "decrement", "raise", "reduce"]):
                if "color swatch" in label_lower or "palette" in label_lower or "color cell" in label_lower:
                    score += 650
                    if any(k in prompt_words for k in ["bottom", "last", "lower"]):
                        if any(k in label_lower for k in ["bottommost", "last", "lowest", "bottom row"]):
                            score += 550
            elif any(k in prompt_words for k in ["increase", "lower", "increment", "decrement", "raise", "reduce", "value"]):
                # User wants to adjust a numeric value or spinbox stepper
                if (
                    any(k in label_lower for k in ["spinbox", "arrow", "increment", "decrement", "stepper", "slider"])
                    or getattr(d, 'archetype', '') in ('slider', 'stepper', 'control')
                    or getattr(d, 'semantic_role', '') in ('control', 'slider', 'action')
                    or (label_lower.startswith('[icon]') and bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", combined_text.lower())))
                ):
                    if getattr(d, 'archetype', '') in ('stepper', 'slider') or bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", combined_text.lower())):
                        score += 850
                    def _has_numeric_context(node):
                        text = f"{getattr(node, 'label', '')} {getattr(node, 'attached_label', '')}"
                        return bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", text.lower()))

                    def _is_value_control(node):
                        label = str(getattr(node, 'label', '')).lower()
                        return (
                            any(k in label for k in ["spinbox", "arrow", "increment", "decrement", "stepper"])
                            or getattr(node, 'archetype', '') in ('slider', 'button', 'control')
                            or (label.startswith('[icon]') and _has_numeric_context(node))
                        )

                    column_nodes = sorted(
                        [s for s in key_nodes if _is_value_control(s) and _boxes_near(s, d, x_scale=0.65, y_scale=0.0)],
                        key=lambda s: (s.center_y, s.center_x)
                    )
                    stepper_siblings = []
                    for sibling in column_nodes:
                        if stepper_siblings and _boxes_near(sibling, stepper_siblings[-1], x_scale=0.0, y_scale=0.35):
                            prev = stepper_siblings[-1]
                            prev_explicit = any(k in prev.label.lower() for k in ("spinbox", "arrow", "stepper"))
                            cur_explicit = any(k in sibling.label.lower() for k in ("spinbox", "arrow", "stepper"))

                            wants_up = any(k in prompt_words for k in ("increase", "raise", "up", "increment", "higher", "more", "add"))
                            wants_dn = any(k in prompt_words for k in ("decrease", "lower", "down", "decrement", "less", "reduce"))
                            prev_dir_match = (wants_up and "up" in prev.label.lower()) or (wants_dn and "down" in prev.label.lower())
                            cur_dir_match = (wants_up and "up" in sibling.label.lower()) or (wants_dn and "down" in sibling.label.lower())

                            if cur_dir_match and not prev_dir_match:
                                stepper_siblings[-1] = sibling
                            elif not prev_dir_match and (abs(sibling.center_x - d.center_x), not cur_explicit) < (abs(prev.center_x - d.center_x), not prev_explicit):
                                stepper_siblings[-1] = sibling
                        else:
                            stepper_siblings.append(sibling)
                    if len(stepper_siblings) >= 2:
                        requested_ordinal = next((n for token, n in ORDINAL_MAP.items() if token in prompt_words), None)
                        # Color channel ordinal mapping: R/G/B or H/S/V/L channels
                        if requested_ordinal is None and any(c in prompt_words for c in ("color", "channel", "rgb", "hsv", "hsl")):
                            for ch_token, ch_ord in (("r", 1), ("red", 1), ("g", 2), ("green", 2), ("b", 3), ("blue", 3)):
                                if ch_token in prompt_words:
                                    requested_ordinal = ch_ord
                                    break
                        if requested_ordinal is not None and requested_ordinal <= len(stepper_siblings):
                            if stepper_siblings[requested_ordinal - 1].id == d.id:
                                score += 850

            # 14. Raw text box selection priority
            if prompt_words & {"text", "word", "phrase", "sentence"}:
                visible_text_role = getattr(d, "semantic_role", "") in ("input", "text", "cell")
                visible_text_type = getattr(d, "archetype", "") in ("text", "input", "cell")
                if (visible_text_role or visible_text_type) and not d.label.startswith("[Icon]"):
                    score += 350

            # 15. Explicit feature / action match bonus
            # When prompt explicitly requests a specific tool/action, prioritize controls whose label contains that exact feature name
            for tool_tok in ("strikethrough", "underline", "star", "redo", "undo", "filter", "funnel", "ruler", "brush", "eraser", "fullscreen", "maximize", "question"):
                if tool_tok in prompt_words and tool_tok in label_lower:
                    score += 900
                    # Text formatting controls (strikethrough, bold, etc.) belong in formatting clusters/panels
                    if tool_tok in ("strikethrough", "bold", "italic", "underline"):
                        container_name = getattr(d, 'container_title', '').lower()
                        has_format_ctx = any(k in container_name for k in ("character", "properties", "format", "font", "text", "paragraph", "toolbar"))
                        has_sibling_tools = any(
                            s.id != d.id and math.hypot(s.center_x - d.center_x, s.center_y - d.center_y) <= 150
                            and any(k in s.label.lower() for k in ("bold", "italic", "underline", "format", "font", "strikethrough", "character", "pt"))
                            for s in key_nodes
                        )
                        if has_format_ctx or has_sibling_tools:
                            score += 850

            # Accordion / Section Fold / Collapse / Expand
            if any(k in prompt_words for k in ("expand", "collapse", "fold", "toggle", "accordion", "section")):
                if "expand" in label_lower or "collapse" in label_lower or "fold" in label_lower or "toggle" in label_lower:
                    score += 750
                    att_label = str(getattr(d, 'attached_label', '')).lower()
                    if att_label and any(w in prompt_words for w in re.findall(r'[a-zA-Z]+|[0-9]+', att_label) if len(w) >= 3):
                        score += 950

            # List tool options / dropdown expander
            if any(k in prompt_words for k in ("bullet", "numbered", "list")) and any(k in prompt_words for k in ("option", "options", "dropdown", "expand", "arrow")):
                if "dropdown / expand options" in label_lower:
                    score += 900

            # Direct primary tool selection vs dropdown/expander
            if any(k in prompt_words for k in ("brush", "pencil", "eraser", "highlighter", "pen", "tool")):
                if any(k in label_lower for k in ("paintbrush", "pencil", "brush", "eraser")):
                    score += 850

            # Checkbox / Toggle / Radio selection
            if any(k in prompt_words for k in ("check", "uncheck", "box", "checkbox", "radio", "toggle")):
                if any(k in label_lower for k in ("checkbox", "radio", "check / checkmark", "toggle / option")):
                    score += 800

            # Clear / Reset / Dismiss button
            if any(k in prompt_words for k in ("clear", "dismiss", "reset", "close", "cancel")):
                if "clear / close for" in label_lower:
                    score += 850

            return score

        score_cache = {}
        def get_score(d):
            if d.id not in score_cache:
                score_cache[d.id] = relevance_score(d)
            return score_cache[d.id]

        sorted_nodes = sorted(key_nodes, key=get_score, reverse=True)
        top_ui = sorted_nodes[:max_ui_nodes]

        if kwargs.get('mode') == 'pure_droplet':
            if not top_ui:
                return DecisionResult({"error": "No UI elements", "target_x": 0, "target_y": 0, "target_droplet_id": 0, "action_type": "click", "confidence": 0.0, "reasoning": ""})
            best = top_ui[0]
            best_score = get_score(best)

            # Strict validation: Pure Droplet is only confident if there is a direct keyword or dot ID match
            clean_best = re.sub(r'\[item \d+ of \d+.*?\]', '', best.label.lower()).strip()
            clean_best = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean_best).strip()
            best_words = set(re.findall(r'[a-zA-Z]+|[0-9]+', clean_best))
            has_word_match = bool(rare_prompt_words.intersection(best_words))
            is_id_match = str(best.id) in prompt_words

            if not (has_word_match or is_id_match):
                return DecisionResult({
                    "error": "No direct entity match on screen for pure droplet",
                    "target_x": 0, "target_y": 0, "target_droplet_id": 0,
                    "action_type": "click", "confidence": 0.0,
                    "reasoning": f"No direct entity match on screen for '{instruction}' (best score: {best_score})"
                })

            conf = 1.0 if (has_word_match and best_score >= 1000) else (0.85 if best_score >= 500 else 0.5)
            return DecisionResult({
                "error": "", "target_x": best.center_x, "target_y": best.center_y,
                "target_droplet_id": best.id, "action_type": "click",
                "confidence": conf,
                "reasoning": f"Pure droplet mode selected rank 1: {best.label} (score: {best_score})"
            })

        # Include surface probe lattice stars near relevant controls
        matching_ui = [d for d in top_ui if bool(prompt_words.intersection(set(re.findall(r'[a-zA-Z]+|[0-9]+', d.label.lower()))))]
        surface_probes = [d for d in active_droplets if d.category == 'surface_probe']
        relevant_probes = []
        if matching_ui:
            for sp in surface_probes:
                if any(_boxes_near(sp, m, x_scale=3.0, y_scale=0.75) for m in matching_ui):
                    relevant_probes.append(sp)
                    if len(relevant_probes) >= 6:
                        break

        # Spatial-Diverse Stratified Candidate Selection (Constraint 1: 144-Candidate 24-Zone Grid)
        cand_budget = 144

        # Pool A: Top relevance score matches (direct keyword matches, peer-inherited controls, active window targets)
        priority_count = 48
        pool_a = []
        chosen_ids = set()
        if grid_best_droplet is not None:
            pool_a.append(grid_best_droplet)
            chosen_ids.add(grid_best_droplet.id)
        for d in sorted_nodes:
            if d.id not in chosen_ids:
                pool_a.append(d)
                chosen_ids.add(d.id)
                if len(pool_a) >= priority_count:
                    break

        # Intent keyword boosting: protect interactive widgets matching intent keywords
        intent_kw = set(re.findall(r'[a-zA-Z]+|[0-9]+', (user_intent or "").lower()))
        intent_boosted = []
        for d in sorted_nodes:
            if d.id in chosen_ids:
                continue
            lbl = f"{getattr(d, 'label', '')} {getattr(d, 'attached_label', '')}".lower()
            is_interactive = (
                d.label.startswith("[Icon]") or 
                d.label.startswith("[Btn]") or
                getattr(d, 'archetype', '') in ('button', 'icon', 'slider', 'tab', 'control', 'spin', 'stepper') or
                getattr(d, 'semantic_role', '') in ('control', 'slider', 'action')
            )
            if is_interactive and any(k in lbl for k in ("star", "bookmark", "favorite", "spinbox", "stepper", "arrow", "filter", "yes", "ok", "confirm", "replace", "slide") if k in intent_kw):
                intent_boosted.append(d)
                chosen_ids.add(d.id)
                if len(chosen_ids) >= cand_budget:
                    break

        # Pool B: Non-text visual icons, tool buttons, checkboxes, and action controls pool
        # Spatially distributed across 6 horizontal screen columns to prevent left-side toolbar clustering
        visual_candidates = []
        col_w = max(1, sw_screen // 6)
        vis_by_col = {i: [] for i in range(6)}
        for d in sorted_nodes:
            if d.id in chosen_ids:
                continue
            is_visual_tool = (d.label.startswith("[Icon]") or 
                              d.label.startswith("[Btn]") or
                              getattr(d, 'archetype', '') in ('button', 'icon', 'slider', 'tab', 'control', 'stepper') or
                              getattr(d, 'semantic_role', '') in ('control', 'slider', 'action'))
            if is_visual_tool:
                col_idx = min(5, max(0, int(d.center_x) // col_w))
                if len(vis_by_col[col_idx]) < 7:
                    vis_by_col[col_idx].append(d)
                    chosen_ids.add(d.id)
                    visual_candidates.append(d)
                if len(visual_candidates) >= 42:
                    break

        # Pool C: 24-Zone Stratified Spatial Grid (6 columns x 4 rows)
        # Guarantees representation across all 24 screen zones (top ribbon, left/right sidebars, status bar, center canvas)
        zone_w = max(1.0, sw_screen / 6.0)
        zone_h = max(1.0, sh_screen / 4.0)
        quadrant_candidates = []
        for r_idx in range(4):
            for c_idx in range(6):
                zx1, zy1 = c_idx * zone_w, r_idx * zone_h
                zx2, zy2 = (c_idx + 1) * zone_w, (r_idx + 1) * zone_h
                z_nodes = [d for d in sorted_nodes if zx1 <= d.center_x < zx2 and zy1 <= d.center_y < zy2]
                z_existing = sum(1 for d in (pool_a + intent_boosted + visual_candidates) if zx1 <= d.center_x < zx2 and zy1 <= d.center_y < zy2)
                needed = max(0, 2 - z_existing)
                for d in z_nodes:
                    if needed <= 0 or len(chosen_ids) >= cand_budget:
                        break
                    if d.id not in chosen_ids:
                        quadrant_candidates.append(d)
                        chosen_ids.add(d.id)
                        needed -= 1

        all_candidates = pool_a + intent_boosted + visual_candidates + quadrant_candidates + relevant_probes[:4]
        if len(all_candidates) < cand_budget:
            for d in sorted_nodes:
                if d.id not in chosen_ids:
                    all_candidates.append(d)
                    chosen_ids.add(d.id)
                    if len(all_candidates) >= cand_budget:
                        break
        all_candidates = all_candidates[:cand_budget]
        all_candidates.sort(key=lambda d: get_score(d), reverse=True)

        # Compute spatial zone for each candidate
        def _spatial_zone(cx, cy):
            x_z = "LEFT" if cx < sw_screen * 0.2 else ("CENTER" if cx < sw_screen * 0.8 else "RIGHT")
            y_z = "TOP" if cy < sh_screen * 0.15 else ("MID" if cy < sh_screen * 0.85 else "BOTTOM")
            return f"{y_z}-{x_z}"

        # Compute vertical column row ordinals for aligned controls (e.g. form fields, steppers, spinboxes, swatches)
        row_ordinal_tags = {}
        col_groups = []
        for d in sorted(all_candidates, key=lambda x: x.center_x):
            is_col_candidate = (
                any(k in d.label.lower() for k in ("spinbox", "arrow", "stepper", "increment", "decrement", "slider", "swatch", "cell", "color", "palette", "box", "toggle"))
                or getattr(d, 'archetype', '') in ('slider', 'stepper', 'control', 'input')
                or getattr(d, 'semantic_role', '') in ('control', 'slider', 'action', 'input')
            )
            if not is_col_candidate:
                continue
            placed = False
            for cg in col_groups:
                if abs(d.center_x - cg[0].center_x) <= 20:
                    cg.append(d)
                    placed = True
                    break
            if not placed:
                col_groups.append([d])

        for cg in col_groups:
            if len(cg) < 3:
                continue
            cg_sorted = sorted(cg, key=lambda x: x.center_y)
            rows = []
            for item in cg_sorted:
                if not rows or (item.center_y - rows[-1][0].center_y) > 14:
                    rows.append([item])
                else:
                    rows[-1].append(item)
            if len(rows) >= 3:
                col_colors = []
                for it in cg:
                    m_col = re.search(r'\((Blue|Red|Green|Yellow|Orange|Purple|Cyan|Black|White|Gray|Slate|Teal|Indigo|Violet|Magenta|Pink|Brown)\)', it.label, re.IGNORECASE)
                    if m_col:
                        col_colors.append(m_col.group(1).capitalize())
                col_color_prefix = ""
                if col_colors:
                    col_color_prefix = f"{Counter(col_colors).most_common(1)[0][0]} Column, "
                for r_idx, row_items in enumerate(rows, start=1):
                    for item in row_items:
                        row_ordinal_tags[item.id] = f" ({col_color_prefix}Row {r_idx} of {len(rows)})"

        # Identify Vertically Stacked Steppers / Spinboxes (Columns of Up/Down arrow pairs)
        stepper_up = [d for d in all_candidates if "Up Arrow" in d.label]
        stepper_dn = [d for d in all_candidates if "Down Arrow" in d.label]
        spinbox_pairs = []
        used_dn = set()
        for u in stepper_up:
            best_d = None
            min_dy = 999
            for d in stepper_dn:
                if d.id in used_dn:
                    continue
                dx = abs(u.center_x - d.center_x)
                dy = d.center_y - u.center_y
                if dx <= 6 and 8 <= dy <= 24 and dy < min_dy:
                    min_dy = dy
                    best_d = d
            if best_d:
                used_dn.add(best_d.id)
                spinbox_pairs.append((u, best_d, (u.center_x + best_d.center_x) / 2.0, (u.center_y + best_d.center_y) / 2.0))

        pairs_sorted = sorted(spinbox_pairs, key=lambda p: p[2])
        spinbox_cols = []
        for p in pairs_sorted:
            placed = False
            for c in spinbox_cols:
                if abs(p[2] - c[0][2]) <= 8:
                    c.append(p)
                    placed = True
                    break
            if not placed:
                spinbox_cols.append([p])

        for c in spinbox_cols:
            if len(c) < 2:
                continue
            c_sorted = sorted(c, key=lambda p: p[3])
            for r_idx, (u, d, _, _) in enumerate(c_sorted, start=1):
                row_ordinal_tags[u.id] = f" (Spinbox Row {r_idx} of {len(c)})"
                row_ordinal_tags[d.id] = f" (Spinbox Row {r_idx} of {len(c)})"

        manifest_lines = []
        for d in all_candidates:
            c_tag = f" [{getattr(d, 'container_title', '')}]" if getattr(d, 'container_title', '') else ""
            att_val = str(getattr(d, 'attached_label', '')).strip()
            att_tag = f" near='{att_val}'" if att_val and att_val.lower() not in d.label.lower() else ""
            ord_tag = row_ordinal_tags.get(d.id, "")
            zone_tag = f" Zone={_spatial_zone(d.center_x, d.center_y)}"
            manifest_lines.append(
                f"• [{d.id}] ({d.center_x},{d.center_y}){zone_tag}{ord_tag}{c_tag}{att_tag} {d.label}"
            )
        manifest = "\n".join(manifest_lines)
        spatial_graph = self._build_spatial_layout_graph(all_candidates, int(sw_screen), int(sh_screen), None)

        som_intro = ""
        # Prepare Multimodal Vision Payload with Set-of-Marks Badges
        image_url = None
        clean_image_url = None
        scale = 1.0
        if screenshot is not None and hasattr(screenshot, 'shape') and len(screenshot.shape) >= 2:
            try:
                image_url, (img_w, img_h), scale = self._render_unoccluded_som(
                    screenshot, all_candidates, max_image_side=1280, jpeg_quality=82
                )
                som_intro = "\nSet-of-Marks: Numbered badges [ID] mark candidates on screen. Green=Buttons/Tools, Cyan=Inputs, Magenta=Dock/App, Yellow=Sliders, Orange=Tabs."
            except Exception:
                image_url = None

        system_instruction = f"""<system_capability>
Role: Autonomous Cognitive Spatial Desktop Automation Brain controlling a Windows PC.
Architecture: Middle Router between Foundation LLM/VLM, Agentic Framework, and OS Input Dispatch.
Grounding: Spatial Droplet Set-of-Marks visual overlay, 2D Visual Spatial Layout & Connectivity Graph, and deterministic candidate manifest.
Active Foreground Window: {win_info}{som_intro}
</system_capability>

<constitution>
Operational Invariants (NON-NEGOTIABLE):
1. 5-STAGE DEEP DELIBERATION: Every turn requires deep autoregressive deliberation across:
   - user_intent_and_success_criteria: Target application, goal requirements, implicit constraints, visual completion proof.
   - screen_state_and_visual_hierarchy: Foreground window focus, blocking modals, 2D functional screen zones ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP]).
   - mistake_critique_and_loop_diagnosis: Prior action critique, root-cause failure diagnosis, avoidance of repeated traps.
   - end_to_end_master_plan: Milestone plan tracking across turns ([Done], [Active], [Pending]).
   - tactical_action_justification: Step-by-step justification for chosen action & Dot ID, with explicit distractor elimination.
2. EVIDENTIARY GROUNDING: Choose ONLY from verified Dot IDs present in the live manifest. You MUST select an integer Dot ID from the manifest for "target_dot_id". Never invent raw coordinates or return null when candidates exist.
3. CONTROL OVER LABEL: When the goal requires toggling, checking, or adjusting a setting, click the interactive control (checkbox, toggle, spinbox, chevron), NEVER the static category header or text label.
4. MODAL PRECEDENCE: When an active modal dialog is present, background canvas/ribbon elements are strictly OUT-OF-BOUNDS. Resolve modal prompts first.
5. CONTEXTUAL GROUNDING: Use detected labels, control roles, owning containers, and relative geometry in the current screenshot. Do not assume an application-specific layout or fixed screen location.
6. DISTRACTOR ELIMINATION: Compare target, requested operation, modifiers, and container evidence; reject candidates that match only one word.
7. EXPLICIT SEMANTIC MATCH OVER ASSUMED POSITIONS: When the user requests a feature (e.g. 'fullscreen', 'strikethrough', 'redo', 'undo', 'filter', 'brush', 'star', 'debug'), ALWAYS select the candidate whose detected icon label explicitly matches that feature. NEVER pick unrelated static text or invent raw coordinates.
8. PRIMARY TOOL BUTTON VS SPLIT EXPANDER: When expanding or accessing a tool function, select the primary tool button itself. Conversely, when expanding a list with a dedicated chevron, select the dropdown chevron explicitly attached to that list.
9. TIMELINE NAVIGATION: When seeking video playback position, select the timeline trackbar start handle, not transport rewind buttons.
10. ORDINAL ROWS IN LISTS, COLOR PALETTES & SPINBOXES: Use explicit row tags in the manifest: '(Row X of N)' or '(Spinbox Row X of N)'.
11. ACTIVE BROWSER/APPLICATION CONTEXT: Select the control located inside the requested active application window, not background windows.
12. CLEAR / CANCEL FORMATTING VS CANVAS DIALOGS: Select the toolbar control dedicated to that feature, never generic dismiss or canvas buttons.
</constitution>

<screen_context>
Foreground Window: {win_info}
Set-of-Marks Legend: Green=Buttons/Tools, Cyan=Inputs, Magenta=Dock/App, Yellow=Sliders, Orange=Tabs, Red Dot=Exact Pixel Center.
</screen_context>

<action_dispatch>
You MUST respond with valid JSON where the "deliberation" object is generated FIRST:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "<query decomposition: identify target application, destination, explicit/implicit constraints, completion proof>",
    "screen_state_and_visual_hierarchy": "<active window, focus state, blocking modals/spinners, visual layout zone topography ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP])>",
    "mistake_critique_and_loop_diagnosis": "<critique of previous turn outcome; diagnose failures, formulate concrete pivot, ban repeated mistakes>",
    "end_to_end_master_plan": [
      "[Phase 1] <milestone description> (ACTIVE)",
      "[Phase 2] <milestone description> (PENDING)"
    ],
    "tactical_action_justification": "<why selected action & Dot ID optimal, with explicit distractor elimination>"
  }},
  "reasoning": "<concise 1-sentence decision rationale>",
  "target_dot_id": <int or null (REQUIRED if interacting with a specific UI element)>,
  "action": "click" | "double_click" | "triple_click" | "right_click" | "middle_click" | "hover" | "drag" | "scroll" | "type" | "press" | "hotkey",
  "x": <int or null (Absolute X coordinate for start/target if dot ID not used)>,
  "y": <int or null (Absolute Y coordinate for start/target if dot ID not used)>,
  "drag_destination_dot_id": <int or null (Required if action is 'drag' and target is an element)>,
  "drag_dest_x": <int or null (Absolute X coordinate for drag destination if dot ID not used)>,
  "drag_dest_y": <int or null (Absolute Y coordinate for drag destination if dot ID not used)>,
  "input_text": <string or null (Required if action is 'type')>,
  "key": <string or null>,
  "key_combo": <list of strings or null>,
  "status": "in_progress" | "completed" | "failed",
  "next_subgoal": "<brief description of what this single turn achieves>",
  "semantic_intention": "<1-sentence clear statement of intent and purpose>",
  "expected_transition": "<exact predicted visual/text transition on screen>"
}}
</action_dispatch>"""

        user_content = f"""<user_request>
{instruction}
</user_request>

<spatial_layout_graph>
{spatial_graph}
</spatial_layout_graph>

<screen_droplet_manifest>
{manifest}
</screen_droplet_manifest>

<instruction>
Autoregressively deliberate across the 5 cognitive pillars in <deliberation>, eliminate distractors, and dispatch the single best Dot ID in the JSON action schema.
</instruction>"""

        # Build the 144-candidate stratified spatial manifest for all providers (ngrok & groq)
        if True:
            try:
                def _norm_word(word):
                    word = word.lower()
                    return word[:-1] if len(word) > 4 and word.endswith("s") else word
                prompt_terms = {_norm_word(w) for w in rare_prompt_words if len(w) >= 3}
                instruction_terms = {_norm_word(w) for w in re.findall(r"[a-zA-Z]+|[0-9]+", instruction.lower()) if len(w) >= 2}
                # Constraint 1: 144-Candidate Spatial Lattice & 24-Zone Stratification
                max_vlm_candidates = 144
                vlm_cands = []
                vlm_ids = set()
                if grid_best_droplet is not None:
                    vlm_cands.append(grid_best_droplet)
                    vlm_ids.add(grid_best_droplet.id)
                for d in top_ui[:48]:
                    if d.id not in vlm_ids:
                        vlm_cands.append(d)
                        vlm_ids.add(d.id)

                # Intent Keyword Boosting: Protect interactive widgets matching intent keywords from pruning
                vlm_intent_kw = set(re.findall(r'[a-zA-Z]+|[0-9]+', instruction.lower()))
                for d in sorted_nodes:
                    if d.id in vlm_ids:
                        continue
                    lbl = f"{getattr(d, 'label', '')} {getattr(d, 'attached_label', '')}".lower()
                    is_interactive = (
                        d.label.startswith("[Icon]") or 
                        d.label.startswith("[Btn]") or
                        getattr(d, 'archetype', '') in ('button', 'icon', 'slider', 'tab', 'control', 'spin', 'stepper') or
                        getattr(d, 'semantic_role', '') in ('control', 'slider', 'action')
                    )
                    if is_interactive and any(k in lbl for k in ("star", "bookmark", "favorite", "spinbox", "stepper", "arrow", "filter", "yes", "ok", "confirm", "replace", "slide") if k in vlm_intent_kw):
                        vlm_cands.append(d)
                        vlm_ids.add(d.id)
                        if len(vlm_cands) >= max_vlm_candidates:
                            break

                semantic_matches = []
                for d in sorted_nodes:
                    if d.id in vlm_ids:
                        continue
                    retrieval_text = " ".join((
                        str(getattr(d, "label", "")),
                        str(getattr(d, "attached_label", "")),
                        str(getattr(d, "container_title", "")),
                        str(getattr(d, "semantic_role", "")),
                        str(getattr(d, "archetype", "")),
                    )).lower()
                    retrieval_terms = {_norm_word(w) for w in re.findall(r"[a-zA-Z]+|[0-9]+", retrieval_text)}
                    if prompt_terms & retrieval_terms:
                        semantic_matches.append(d)
                numeric_row_request = bool(re.search(r"\b(?:increase|lower|decrease|raise|increment|decrement|value|reduce)\b", instruction.lower()))
                semantic_limit = 28
                for d in semantic_matches[:semantic_limit]:
                    if len(vlm_cands) >= max_vlm_candidates:
                        break
                    vlm_cands.append(d)
                    vlm_ids.add(d.id)

                # Recover controls with strong task-relative context even if
                # their visual classifier label is noisy.
                context_matches = []
                row_aligned_ids = set()
                action_intent_present = any(
                    bool(prompt_words.intersection(action_words))
                    for action_words, _, _ in ACTION_CONTROL_MAP
                )
                action_intent_present = action_intent_present or bool(re.search(
                    r"\b(?:click|tap|press|select|activate|open|close|expand|collapse|toggle|redo|undo|star|bookmark|play|pause|forward|back|submit|apply|cancel|turn|make|set|change|switch|use|check|uncheck)\b",
                    instruction.lower(),
                ))
                numeric_adjustment = bool(instruction_terms & {"increase", "lower", "decrease", "raise", "increment", "decrement", "value", "reduce"})
                quoted_field = re.search(r"[\"']([a-z])['\"]", instruction.lower()) if numeric_adjustment else None
                def _node_size(node):
                    box = getattr(node, "bbox", None)
                    if box and len(box) == 4:
                        return max(1.0, float(box[2]) - float(box[0])), max(1.0, float(box[3]) - float(box[1]))
                    return 12.0, 12.0

                if quoted_field:
                    field_name = quoted_field.group(1).lower()
                    field_labels = [
                        d for d in sorted_nodes
                        if field_name in set(re.findall(r"[a-zA-Z]+|[0-9]+", f"{getattr(d, 'label', '')} {getattr(d, 'attached_label', '')}".lower()))
                    ]
                    for field in field_labels:
                        fw, fh = _node_size(field)
                        for d in sorted_nodes:
                            label = str(getattr(d, 'label', '')).lower()
                            attached = str(getattr(d, 'attached_label', '')).strip().lower()
                            is_visual = label.startswith("[icon]") or getattr(d, 'semantic_role', '') in ('control', 'slider', 'action')
                            has_value_context = bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", label + " " + attached))
                            dw, dh = _node_size(d)
                            scale_f = max(fw, fh, dw, dh)
                            dx, dy = d.center_x - field.center_x, abs(d.center_y - field.center_y)
                            if is_visual and has_value_context and -2 * scale_f <= dx <= 12 * scale_f and dy <= 1.5 * scale_f:
                                row_aligned_ids.add(d.id)
                                if d.id not in vlm_ids:
                                    context_matches.append(d)

                    color_component_orders = {
                        "rgb": ("r", "g", "b"),
                        "rgba": ("r", "g", "b", "a"),
                        "hsv": ("h", "s", "v"),
                        "hsl": ("h", "s", "l"),
                    }
                    color_component_row = None
                    if not field_labels and prompt_words & {"color", "colour", "rgb", "rgba", "hsv", "hsl"}:
                        for family, components in color_component_orders.items():
                            if field_name in components:
                                color_component_row = components.index(field_name)
                                break
                    if color_component_row is not None:
                        value_controls = [
                            d for d in sorted_nodes
                            if (str(getattr(d, "label", "")).lower().startswith("[icon]")
                                or getattr(d, "semantic_role", "") in ("control", "slider", "action"))
                            and bool(re.search(
                                r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])",
                                f"{getattr(d, 'label', '')} {getattr(d, 'attached_label', '')}".lower(),
                            ))
                        ]
                        seen_x_tracks = set()
                        for anchor in value_controls:
                            anchor_value = str(getattr(anchor, "attached_label", "")).strip().lower()
                            if not anchor_value:
                                continue
                            aw, _ = _node_size(anchor)
                            column = [
                                node for node in value_controls
                                if str(getattr(node, "container_title", "")) == str(getattr(anchor, "container_title", ""))
                                and str(getattr(node, "attached_label", "")).strip().lower() == anchor_value
                                and abs(node.center_x - anchor.center_x) <= max(2.0, 0.60 * min(aw, _node_size(node)[0]))
                            ]
                            column.sort(key=lambda node: (node.center_y, node.center_x))
                            rows = []
                            for node in column:
                                if rows and abs(node.center_y - rows[-1][-1].center_y) <= max(3.0, 0.60 * min(_node_size(node)[1], _node_size(rows[-1][-1])[1])):
                                    rows[-1].append(node)
                                else:
                                    rows.append([node])
                            track_key = (str(getattr(anchor, "container_title", "")), round(anchor.center_x, 1), anchor_value)
                            if track_key in seen_x_tracks:
                                continue
                            seen_x_tracks.add(track_key)
                            if len(rows) < len(color_component_orders.get("rgb", ())):
                                continue
                            if color_component_row < len(rows):
                                for node in rows[color_component_row]:
                                    row_aligned_ids.add(node.id)
                                    if node.id not in vlm_ids:
                                        context_matches.append(node)

                row_index = None
                ordinal_words = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9}
                for word, index in ordinal_words.items():
                    if word in instruction_terms:
                        row_index = index
                        break
                if numeric_adjustment and row_index is not None:
                    numeric_controls = [
                        d for d in sorted_nodes
                        if (str(getattr(d, 'label', '')).lower().startswith('[icon]') or getattr(d, 'semantic_role', '') in ('control', 'slider', 'action'))
                        and bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", f"{getattr(d, 'label', '')} {getattr(d, 'attached_label', '')}".lower()))
                    ]
                    for anchor in numeric_controls:
                        aw, ah = _node_size(anchor)
                        column = sorted(
                            [n for n in numeric_controls if abs(n.center_x - anchor.center_x) <= 1.5 * max(aw, _node_size(n)[0])],
                            key=lambda n: (n.center_y, -get_score(n)),
                        )
                        rows = []
                        for node in column:
                            if rows and abs(node.center_y - rows[-1][-1].center_y) <= 1.5 * max(ah, _node_size(node)[1]):
                                rows[-1].append(node)
                            else:
                                rows.append([node])
                        if row_index < len(rows):
                            for node in rows[row_index]:
                                row_aligned_ids.add(node.id)
                                if node.id not in vlm_ids:
                                    context_matches.append(node)
                for d in sorted_nodes:
                    if d.id in vlm_ids:
                        continue
                    label = str(getattr(d, 'label', '')).lower()
                    attached = str(getattr(d, 'attached_label', '')).strip()
                    is_visual = label.startswith("[icon]") or label.startswith("[btn]") or getattr(d, 'semantic_role', '') in ('control', 'slider', 'action')
                    generic_visual = any(token in label for token in (
                        "toolbar tool", "visual button", "generic icon", "small graphic", "graphic element", "checkbox", "toggle", "dropdown", "expand"
                    ))
                    if action_intent_present and is_visual and generic_visual:
                        context_matches.append(d)
                    if numeric_adjustment and is_visual and (
                        any(k in label for k in ("increment", "decrement", "up arrow", "down arrow", "spinbox", "stepper"))
                        or bool(re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", label + " " + attached.lower()))
                    ):
                        context_matches.append(d)
                def _context_priority(d):
                    label = str(getattr(d, 'label', '')).lower()
                    priority = get_score(d)
                    if d.id in row_aligned_ids:
                        priority += 8000
                    if numeric_adjustment:
                        if any(term in label for term in ("increment", "decrement", "up arrow", "down arrow", "spinbox", "stepper")):
                            priority += 4000
                        elif re.search(r"(?<![a-z])-?\d+(?:\.\d+)?(?![a-z])", label):
                            priority += 2000
                    return priority
                context_matches = list({d.id: d for d in context_matches}.values())
                context_matches.sort(key=lambda d: (d.id in row_aligned_ids, _context_priority(d)), reverse=True)
                context_limit = 34
                context_selected = []
                context_selected_ids = set()
                context_container_counts = {}
                for per_container_cap in (1, 2, 3, 5):
                    for d in context_matches:
                        if d.id in context_selected_ids:
                            continue
                        container_key = str(getattr(d, "container_title", "")).strip().lower()
                        if container_key and context_container_counts.get(container_key, 0) >= per_container_cap:
                            continue
                        context_selected.append(d)
                        context_selected_ids.add(d.id)
                        if container_key:
                            context_container_counts[container_key] = context_container_counts.get(container_key, 0) + 1
                        if len(context_selected) >= context_limit:
                            break
                    if len(context_selected) >= context_limit:
                        break
                if len(context_selected) < context_limit:
                    for d in context_matches:
                        if d.id in context_selected_ids:
                            continue
                        context_selected.append(d)
                        if len(context_selected) >= context_limit:
                            break
                for d in context_selected:
                    if len(vlm_cands) >= max_vlm_candidates:
                        break
                    vlm_cands.append(d)
                    vlm_ids.add(d.id)

                # 24-Zone Stratified Spatial Grid (6 columns x 4 rows)
                zone_w = max(1.0, sw_screen / 6.0)
                zone_h = max(1.0, sh_screen / 4.0)
                for r_idx in range(4):
                    for c_idx in range(6):
                        zx1, zy1 = c_idx * zone_w, r_idx * zone_h
                        zx2, zy2 = (c_idx + 1) * zone_w, (r_idx + 1) * zone_h
                        z_nodes = [d for d in sorted_nodes if zx1 <= d.center_x < zx2 and zy1 <= d.center_y < zy2]
                        z_existing = sum(1 for d in vlm_cands if zx1 <= d.center_x < zx2 and zy1 <= d.center_y < zy2)
                        needed = max(0, 2 - z_existing)
                        for d in z_nodes:
                            if needed <= 0 or len(vlm_cands) >= max_vlm_candidates:
                                break
                            if d.id not in vlm_ids:
                                vlm_cands.append(d)
                                vlm_ids.add(d.id)
                                needed -= 1

                # Fill any remaining slots up to max_vlm_candidates from the relevance ranking
                for d in sorted_nodes:
                    if len(vlm_cands) >= max_vlm_candidates:
                        break
                    if d.id not in vlm_ids:
                        vlm_cands.append(d)
                        vlm_ids.add(d.id)
                vlm_manifest_lines = []
                for candidate_rank, d in enumerate(vlm_cands, start=1):
                    label = str(d.label).replace('"', "'")[:58]
                    attached = str(getattr(d, "attached_label", "")).replace('"', "'")[:36]
                    box = getattr(d, "bbox", None)
                    box_text = ""
                    if box is not None and len(box) == 4:
                        box_text = f" [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]"
                    layout_pos = _layout_position(d)
                    relation_text = ""
                    if layout_pos is not None:
                        relation_text = f" rel={layout_relation}-{layout_pos[0]}/{layout_pos[1]}"
                    if numeric_adjustment and d.id in row_aligned_ids:
                        row_name = f"row {row_index + 1}" if row_index is not None else "field control"
                        if quoted_field:
                            row_name += f" ({quoted_field.group(1).upper()})"
                        label = f"{row_name}; {label}"
                    ctx_part = f' ctx="{attached}"' if attached and attached.lower() not in label.lower() else ""
                    vlm_manifest_lines.append(
                        f"[{d.id}] ({d.center_x},{d.center_y}){box_text}{relation_text} \"{label}\"{ctx_part}"
                    )
                vlm_manifest = "\n".join(vlm_manifest_lines)

                if kwargs.get("candidate_trace_only"):
                    return DecisionResult({
                        "error": "",
                        "target_x": None,
                        "target_y": None,
                        "target_droplet_id": None,
                        "target_dot_id": None,
                        "action_type": "diagnostic",
                        "confidence": 0.0,
                        "reasoning": "Candidate retrieval trace only; no model call made.",
                        "manifest_candidate_ids": [d.id for d in vlm_cands],
                        "ranked_candidate_ids": [d.id for d in sorted_nodes],
                        "active_candidate_ids": [d.id for d in active_droplets if d.category != "surface_probe"],
                        "ranked_candidate_details": [
                            {
                                "id": d.id,
                                "score": get_score(d),
                                "label": str(getattr(d, "label", ""))[:120],
                                "attached_label": str(getattr(d, "attached_label", ""))[:100],
                                "container_title": str(getattr(d, "container_title", ""))[:100],
                                "semantic_role": str(getattr(d, "semantic_role", "")),
                                "archetype": str(getattr(d, "archetype", "")),
                                "bbox": list(_node_bounds(d)),
                                "layout_position": _layout_position(d),
                            }
                            for d in sorted_nodes
                        ],
                    })

                # Constraint 2: 1280px High-Resolution Unoccluded Hollow-Frame Set-of-Marks
                vlm_image_url = None
                scale_vlm = 1.0
                vlm_image_size = (int(sw_screen), int(sh_screen))
                if screenshot is not None and hasattr(screenshot, 'shape') and len(screenshot.shape) >= 2:
                    vlm_image_url, vlm_image_size, scale_vlm = self._render_unoccluded_som(
                        screenshot, vlm_cands, max_image_side=1280, jpeg_quality=82
                    )

                vlm_prompt = f"""You are a GUI grounding agent. Choose the exact visible control or object that satisfies the request.
USER REQUEST: "{instruction}"

GROUNDING DIRECTIVES:
1. TOOL VS CONTAINER: When the user requests a specific tool, filter, toggle, or action located inside or beside an input field or toolbar (e.g. 'filter function', 'search filter', 'clear input'), choose the specific dedicated interactive icon/button control ([Icon] / [Btn]), NOT the surrounding text input field ([Input]) or window canvas. When asked to expand, unfold, collapse, or toggle an accordion or section (e.g. 'expand advanced options', 'unfold settings'), select the dedicated expander triangle/toggle control ([Icon] Expand / Toggle for '...'), NEVER the static text heading beside it!
2. CONTENT ORDINALS VS APP CHROME: When selecting an ordinal content item (e.g. 'first slide', 'bottom color', 'third row'), target the control belonging to that specific content card/thumbnail/cell in the content/sidebar area (e.g. Slide thumbnails under the 'Slides' sidebar panel), NEVER an icon in the top application toolbar or ribbon header (y < 160).
3. SIDEBAR & PANEL TOGGLES: Panel dividers and splitter handles along a sidebar or panel edge (e.g. 'Left Panel Divider / Splitter Handle', 'Right Panel Divider / Splitter Handle') ARE the primary interactive toggle controls used to collapse or hide that panel. When asked to hide, close, or collapse a side panel, ALWAYS select the panel divider/splitter handle along that panel's border. NEVER select buttons inside the panel itself (such as buttons in the panel header or top corner) or static panel text headers!
4. VISUAL GLYPH PRECEDENCE: Numbered candidate badges [ID] are docked directly on or above the bounding box with a 1px frame and hollow center. Inspect the visible icon glyph in the screenshot directly to confirm the tool's true function.
5. ACTION & TOOL SEMANTIC MATCH PRECEDENCE: When opening, launching, running, debugging, filtering, or modifying settings (e.g. 'debug panel', 'run and debug', 'search filter', 'star'), ALWAYS select the dedicated interactive icon control ([Icon] Play / Run and Debug / Start, [Icon] Filter / Funnel / Search Filter, [Icon] Star / Favorite / Bookmark), NEVER top application menu bar text labels ('Run', 'File', 'Edit', 'View') when an explicit icon control exists! In web browsers, the star icon to bookmark the current page is located at the right end of the URL address bar. NEVER select a generic unlabelled button ('[Icon] Toolbar Tool / Visual Button') or external system dock/launcher icon when an explicit semantic match exists in the application workspace!
6. TARGET APPLICATION SCOPING & DESKTOP DOCK DISTRACTORS: When the user requests an action in an application (e.g. VS Code, Chrome, Word, LibreOffice), choose the control located inside THAT application's window/workspace (e.g. x > 100, inside the editor/toolbar). NEVER select generic launcher icons on the outer host system taskbar/dock (x <= 50 or y >= screen_height - 45) when the requested action is within an open application!
7. LIST DROPDOWNS & LIST TOOLS: When expanding or configuring bullet or numbered list options (e.g. 'Expand the non-numeric bullet point options'), select the dropdown chevron explicitly attached to that list type ('[Icon] Dropdown / Expand Options for Bullet List (Non-numeric / Unordered Dots)'), NEVER a formatting dropdown in another panel!
8. CLEAR / CANCEL FORMATTING VS CANVAS DIALOGS: When asked to cancel, clear, or modify a formatting tool (e.g. 'Cancel the highlight of the picture', 'clear formatting'), ALWAYS select the toolbar control/toggle dedicated to that feature (e.g. control for 'Highlight'), NEVER a generic 'None / Clear / Cancel / No Fill' color button on the canvas or an unrelated dialog dismiss button!
9. MULTI-WINDOW TARGET SCOPING: When the goal targets a specific named application (e.g. 'Chrome'), and multiple applications are visible, select the control within THAT application's window/address bar (e.g. control with 'Chrome' attached context), NEVER an icon in an image editor (like GIMP) or background window above it!

The screenshot is {vlm_image_size[0]} by {vlm_image_size[1]} pixels and has numbered candidate badges; each badge maps to exactly one manifest ID.
Candidates:
{vlm_manifest}

Respond with ONLY JSON:
{{"reasoning": "<short rationale>", "target_dot_id": <int|null>, "target_point": [<x>,<y>]|null}}"""

                user_msg_content = [{"type": "text", "text": vlm_prompt}]
                if vlm_image_url and getattr(self, "ngrok_use_vision", False):
                    user_msg_content.append({"type": "image_url", "image_url": {"url": vlm_image_url}})

                manifest = vlm_manifest
                image_url = vlm_image_url
                scale = scale_vlm
                user_content = vlm_prompt
                all_candidates = vlm_cands
                kwargs["_prepared_manifest_ids"] = [d.id for d in vlm_cands]

                if (
                    requested_provider == "groq"
                    or requested_provider not in {"", "ngrok"}
                    or not (getattr(self, "use_ngrok", False) and self.ngrok_client is not None)
                ):
                    raise _UseConfiguredProvider()

                high_reasoning_system = (
                    "You are an expert GUI visual grounding agent with deep visual perception and rapid quantitative reasoning. "
                    "MANDATORY THINKING DIRECTIVE: You MUST bound your internal <think> reasoning phase strictly to 350-400 tokens of dense quantitative analysis. "
                    "Do NOT engage in verbose commentary or philosophical rambling. Direct coordinate/glyph audit only: "
                    "1. Identify target control function from user instruction. "
                    "2. Locate candidate Dot IDs matching icon glyph morphology, rendered text, and spatial container alignment. "
                    "3. Systematically eliminate distractors based on spatial quadrant and role. "
                    "4. Conclude </think> immediately and output strictly valid JSON on line 1: "
                    "{\"reasoning\": \"<dense visual rationale under 30 words>\", \"target_dot_id\": <int|null>, \"target_point\": [<x>,<y>]|null}."
                )

                try:
                    resp = self.ngrok_client.chat.completions.create(
                        model=self.ngrok_model,
                        messages=[
                            {"role": "system", "content": high_reasoning_system},
                            {"role": "user", "content": user_msg_content}
                        ],
                        temperature=0.01,
                        max_tokens=700,
                        timeout=25.0
                    )
                except Exception as call_err:
                    if "image input is not supported" in str(call_err).lower() or "mmproj" in str(call_err).lower():
                        resp = self.ngrok_client.chat.completions.create(
                            model=self.ngrok_model,
                            messages=[
                                {"role": "system", "content": high_reasoning_system},
                                {"role": "user", "content": [{"type": "text", "text": vlm_prompt}]}
                            ],
                            temperature=0.01,
                            max_tokens=700,
                            timeout=25.0
                        )
                    else:
                        raise
                msg = resp.choices[0].message
                raw = (msg.content or "").strip()
                thought = (getattr(msg, "reasoning", "") or "").strip()
                if not thought and "<think>" in raw:
                    m_th = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
                    if m_th:
                        thought = m_th.group(1).strip()
                if thought:
                    safe_thought = thought[:800].encode("ascii", "backslashreplace").decode("ascii")
                    print(f"\n[*] [Qwen 3-VL Deep Reasoning Trace]:\n{safe_thought}...\n")

                if not raw and thought:
                    raw = thought

                decision = self._extract_json(raw)
                if decision is None:
                    id_match = re.search(
                        r'"(?:target_dot_id|target_id|dot_id|id)"\s*:\s*(\d+|null)',
                        raw,
                        flags=re.IGNORECASE,
                    )
                    if not id_match:
                        id_match = re.search(r'(?:dot|droplet|candidate|target)\s*(?:id)?\s*\[?#?(\d+)\]?', raw, flags=re.IGNORECASE)
                    if id_match:
                        raw_id = id_match.group(1)
                        decision = {
                            "target_dot_id": None if raw_id.lower() == "null" else int(raw_id),
                            "reasoning": re.sub(r"\s+", " ", raw)[:300],
                        }
                if decision is None:
                    print(f"[WARNING] ngrok/{self.ngrok_model} output could not be parsed as JSON: {raw[:150]}")
                else:
                    decision['model_used'] = f"ngrok/{self.ngrok_model}"
                    decision["provider"] = "ngrok"
                    decision['key_used'] = "Kaggle Qwen GPU"
                    decision['manifest_candidate_ids'] = [d.id for d in vlm_cands]

                    target_id = decision.get("target_dot_id") or decision.get("target_id") or decision.get("dot_id") or decision.get("id")
                    if isinstance(target_id, list) and len(target_id) > 0:
                        target_id = target_id[0]
                    elif isinstance(target_id, str):
                        m = re.search(r'\d+', target_id)
                        target_id = int(m.group()) if m else None

                    raw_px = decision.get("x")
                    raw_py = decision.get("y")
                    raw_point = decision.get("target_point") or decision.get("point") or decision.get("coordinate")
                    if raw_px is None and isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
                        raw_px, raw_py = raw_point[0], raw_point[1]

                    target_droplet = (
                        next((d for d in vlm_cands if d.id == target_id), None)
                        or next((d for d in key_nodes if d.id == target_id), None)
                    ) if target_id is not None else None

                    point_snap = None
                    if raw_px is not None and raw_py is not None:
                        try:
                            f_px, f_py = float(raw_px), float(raw_py)
                            # Note: in vlm_prompt, xy=(center_x, center_y) are native screen coordinates,
                            # whereas visual pixels in vlm_image_url are scaled by scale_vlm (or [0..1000] normalized).
                            candidate_points = [(f_px, f_py)]
                            if abs(scale_vlm - 1.0) > 1e-3:
                                candidate_points.append((f_px / max(scale_vlm, 1e-9), f_py / max(scale_vlm, 1e-9)))
                            if 0 <= f_px <= 1000 and 0 <= f_py <= 1000:
                                candidate_points.append((f_px * sw_screen / 1000.0, f_py * sh_screen / 1000.0))

                            def _point_to_box_gap(point_x, point_y, box):
                                if box is None or len(box) != 4:
                                    return math.hypot(point_x, point_y), False
                                x1, y1, x2, y2 = map(float, box)
                                if x2 <= x1 or y2 <= y1:
                                    return math.inf, False
                                inside = x1 <= point_x <= x2 and y1 <= point_y <= y2
                                dx = max(x1 - point_x, 0.0, point_x - x2)
                                dy = max(y1 - point_y, 0.0, point_y - y2)
                                return math.hypot(dx, dy), inside

                            snapped_candidates = []
                            for px, py in candidate_points:
                                for d in key_nodes:
                                    box = getattr(d, "bbox", None)
                                    gap, inside = _point_to_box_gap(px, py, box)
                                    box_w = box_h = 1.0
                                    if box and len(box) == 4:
                                        box_w = max(1.0, float(box[2]) - float(box[0]))
                                        box_h = max(1.0, float(box[3]) - float(box[1]))
                                        snap_tolerance = max(12.0, min(32.0, 0.45 * max(box_w, box_h)))
                                    else:
                                        snap_tolerance = max(12.0, 0.012 * min(sw_screen, sh_screen))
                                    if inside or gap <= snap_tolerance:
                                        snapped_candidates.append((not inside, gap, box_w * box_h if box else math.inf, d))
                            if snapped_candidates:
                                _, _, _, point_snap = min(
                                    snapped_candidates, key=lambda item: (item[0], item[1], item[2])
                                )
                        except (ValueError, TypeError):
                            pass

                    if point_snap is not None and target_droplet is None:
                        target_droplet = point_snap
                        decision["selection_source"] = "full_droplet_point_snap"
                    elif target_droplet is not None:
                        decision["selection_source"] = "manifest_id"

                    action_groups = [
                        (actions, controls)
                        for actions, controls, _ in ACTION_CONTROL_MAP
                        if prompt_words.intersection(actions)
                    ]
                    action_tokens = set().union(*(actions for actions, _ in action_groups)) if action_groups else set()
                    spatial_tokens = {
                        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                        "top", "bottom", "left", "right", "upper", "lower", "middle", "near", "above", "below", "under",
                    }
                    target_terms = rare_prompt_words - action_tokens - spatial_tokens
                    chosen_text = ""
                    if target_droplet is not None:
                        chosen_text = " ".join((
                            str(getattr(target_droplet, "label", "")),
                            str(getattr(target_droplet, "attached_label", "")),
                            str(getattr(target_droplet, "container_title", "")),
                            str(getattr(target_droplet, "semantic_role", "")),
                            str(getattr(target_droplet, "archetype", "")),
                        )).lower()
                    chosen_terms = {_norm_word(w) for w in re.findall(r"[a-zA-Z]+|[0-9]+", chosen_text)}
                    has_target_evidence = bool(target_terms & chosen_terms)
                    has_action_evidence = any(
                        any(control in chosen_text for control in controls)
                        for _, controls in action_groups
                    )
                    needs_visual_refinement = (
                        target_droplet is None
                        or (bool(target_terms) and not has_target_evidence)
                        or (bool(action_groups) and not has_action_evidence)
                    )
                    if needs_visual_refinement and screenshot is not None and self.ngrok_client is not None and getattr(self, "ngrok_use_vision", False):
                        try:
                            src_h, src_w = screenshot.shape[:2]
                            refine_scale = min(1.0, 1280.0 / max(src_w, src_h))
                            refine_img = cv2.resize(
                                screenshot,
                                (max(1, int(src_w * refine_scale)), max(1, int(src_h * refine_scale))),
                                interpolation=cv2.INTER_AREA,
                            )
                            _, refine_buf = cv2.imencode(".jpg", refine_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                            refine_url = f"data:image/jpeg;base64,{base64.b64encode(refine_buf).decode('utf-8')}"
                            refine_prompt = (
                                f"Locate the exact visible UI target for this request: {instruction!r}. "
                                f"Image dimensions are {refine_img.shape[1]}x{refine_img.shape[0]} pixels. "
                                "Return JSON only: {\"target_point\":[x,y]} with the center of the actual clickable target. "
                                "Use the screenshot itself, not assumed application layouts. If it is not visually identifiable, return {\"target_point\":null}."
                            )
                            refine_resp = self.ngrok_client.chat.completions.create(
                                model=self.ngrok_model,
                                messages=[
                                    {"role": "system", "content": "Return only the requested JSON object."},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": refine_prompt},
                                        {"type": "image_url", "image_url": {"url": refine_url}},
                                    ]},
                                ],
                                temperature=0,
                                max_tokens=80,
                                timeout=35.0,
                            )
                            refine_raw = re.sub(r"<think>.*?</think>", "", refine_resp.choices[0].message.content or "", flags=re.DOTALL)
                            refine_point_match = re.search(
                                r'"target_point"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
                                refine_raw,
                            )
                            if refine_point_match:
                                rx = float(refine_point_match.group(1)) / max(refine_scale, 1e-9)
                                ry = float(refine_point_match.group(2)) / max(refine_scale, 1e-9)
                                containing = []
                                for candidate in key_nodes:
                                    candidate_box = getattr(candidate, "bbox", None)
                                    if candidate_box and len(candidate_box) == 4:
                                        x1, y1, x2, y2 = map(float, candidate_box)
                                        inside = x1 <= rx <= x2 and y1 <= ry <= y2
                                        if inside:
                                            area = max(1.0, (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1]))
                                            containing.append((area, 0.0, candidate))
                                if containing:
                                    refined_target = min(containing, key=lambda item: (item[0], item[1]))[2]
                                    if target_droplet is None:
                                        target_droplet = refined_target
                                        decision["target_dot_id"] = int(refined_target.id)
                                        decision["target_droplet_id"] = int(refined_target.id)
                                        decision["selection_source"] = "visual_refinement_point_snap"
                                        decision["reasoning"] = f"{decision.get('reasoning', '')} [clean-image point snapped to detected droplet]"
                        except Exception as refine_error:
                            print(f"[WARNING] visual point refinement skipped ({type(refine_error).__name__})")

                    if target_droplet is not None:
                        decision["target_dot_id"] = int(target_droplet.id)
                        decision["target_x"] = float(target_droplet.center_x)
                        decision["target_y"] = float(target_droplet.center_y)
                        decision["target_droplet_id"] = int(target_droplet.id)

                    if target_droplet is not None:
                        # Continuous and sub-character geometry interpolation
                        try:
                            from ..core.spatial_geometry import SpatialGeometryEngine
                            sub_coords = SpatialGeometryEngine.parse_and_resolve(instruction, target_droplet)
                            if sub_coords:
                                decision["target_x"] = float(sub_coords[0])
                                decision["target_y"] = float(sub_coords[1])
                                decision["reasoning"] = f"{decision.get('reasoning', '')} [SubGeometry: ({sub_coords[0]:.1f}, {sub_coords[1]:.1f})]"
                        except Exception:
                            pass
                        return DecisionResult(decision)
                    else:
                        print(f"[WARNING] ngrok/{self.ngrok_model} returned no valid manifest target; trying configured fallback pool")
            except _UseConfiguredProvider:
                pass
            except Exception as e:
                if kwargs.get("candidate_trace_only"):
                    return DecisionResult({
                        "error": f"Candidate trace failed: {type(e).__name__}: {str(e)[:180]}",
                        "target_x": None,
                        "target_y": None,
                        "target_droplet_id": None,
                        "target_dot_id": None,
                        "action_type": "diagnostic_error",
                        "confidence": 0.0,
                        "reasoning": f"Candidate trace failed: {type(e).__name__}: {str(e)[:180]}",
                    })
                print(f"[WARNING] ngrok/{self.ngrok_model} failed ({type(e).__name__}): {str(e)[:180]}; using configured fallback pool")

        if requested_provider in (
            "local", "ollama", "vllm", "sglang", "openai", "azure",
            "gemini", "gemini-pro", "google", "gemini_pro", "gemini-flash", "gemini-1.5-pro", "gemini-2.0-flash"
        ):
            try:
                from .providers.registry import get_grounding_provider
                provider_inst = get_grounding_provider(requested_provider)
                res = await provider_inst.ground(
                    instruction=instruction,
                    manifest_text=manifest,
                    all_candidates=all_candidates,
                    som_image_url=image_url,
                    win_info=win_info,
                    **kwargs
                )
                if not getattr(res, "error", ""):
                    return DecisionResult(res.to_dict())
                print(f"[WARNING] Provider {requested_provider} error: {res.error}; falling back to configured pool.")
                requested_provider = ""
            except Exception as e:
                print(f"[WARNING] Provider {requested_provider} failed: {e}; falling back to configured pool.")
                requested_provider = ""

        provider_entries = [
            (index, entry) for index, entry in enumerate(self.key_pool)
            if not requested_provider
            or requested_provider == "ngrok"
            or str(entry.get("provider", "groq")).strip().lower() == requested_provider
        ]
        if not provider_entries:
            return DecisionResult({
                "error": f"No configured clients for provider={requested_provider}",
                "target_x": None,
                "target_y": None,
                "target_droplet_id": None,
                "target_dot_id": None,
                "action_type": "error",
                "confidence": 0.0,
                "reasoning": "Requested provider is unavailable; refusing provider fallback.",
            })
        num_keys = len(provider_entries)
        last_err = ""
        with self.lock:
            start_idx = self.current_idx % num_keys
            self.current_idx = (self.current_idx + 1) % num_keys

        target_model = kwargs.get("target_model") or getattr(self, "preferred_groq_model", None) or os.environ.get("GROQ_PRIMARY_MODEL")

        # Proactively guard token budget against Groq 7,000 ITPM and provider limits
        from .token_guard import TokenBudgetGuard
        system_instruction, user_content, all_candidates = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=all_candidates,
            max_tokens=4800,
            goal=instruction,
            sw=int(sw_screen),
            sh=int(sh_screen),
            win_info=win_info,
        )

        if requested_provider == "pollinations":
            model_tiers = [target_model] if target_model else []
        elif target_model:
            default_tiers = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
            model_tiers = [target_model] + [m for m in default_tiers if m != target_model]
        else:
            model_tiers = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

        now = time.time()
        for model_name in model_tiers:
            for offset in range(num_keys):
                active_idx, entry = provider_entries[(start_idx + offset) % num_keys]
                client = entry["client"]
                masked_key = entry["masked"]
                if entry.get("provider") == "pollinations":
                    if model_name not in entry.get("models", []):
                        continue

                cooldown_until = self._key_cooldowns.get((masked_key, model_name), 0.0)
                if now < cooldown_until:
                    continue

                try:
                    if "qwen" in model_name and entry.get("provider", "groq") == "groq" and "Candidates:\n" in user_content:
                        head, tail = user_content.split("Candidates:\n", 1)
                        if "\n\nRespond with ONLY JSON:" in tail:
                            cand_block, footer = tail.split("\n\nRespond with ONLY JSON:", 1)
                            c_lines = [ln for ln in cand_block.splitlines() if ln.strip()]
                            if len(c_lines) > 76:
                                user_content = head + "Candidates:\n" + "\n".join(c_lines[:76]) + "\n\nRespond with ONLY JSON:" + footer
                    if "qwen" in model_name and image_url:
                        user_msg_content = [
                            {"type": "text", "text": user_content},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    else:
                        user_msg_content = user_content

                    sys_content = (
                        (
                            "You are an expert GUI visual grounding agent with deep visual perception and rapid quantitative reasoning. "
                            "MANDATORY THINKING DIRECTIVE: You MUST bound your internal <think> reasoning phase strictly to 350-400 tokens of dense quantitative analysis. "
                            "Do NOT engage in verbose commentary or philosophical rambling. Direct coordinate/glyph audit only: "
                            "1. Identify target control function from user instruction. "
                            "2. Locate candidate Dot IDs matching icon glyph morphology, rendered text, and spatial container alignment. "
                            "3. Systematically eliminate distractors based on spatial quadrant and role. "
                            "4. Conclude </think> immediately and output strictly valid JSON on line 1: "
                            "{\"reasoning\": \"<dense visual rationale under 30 words>\", \"target_dot_id\": <int|null>, \"target_point\": [<x>,<y>]|null}."
                        )
                        if "qwen" in model_name
                        else system_instruction
                    )
                    print(f"[*] Calling Groq Rotator Key #{active_idx + 1} ({masked_key}) with model '{model_name}'...")
                    call_kwargs = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": sys_content},
                            {"role": "user", "content": user_msg_content}
                        ],
                        "temperature": 0.05 if "qwen" in model_name else 0.1,
                        "max_tokens": 800,
                        "timeout": 25.0 if "qwen" in model_name else 45.0
                    }
                    if "gpt-oss" in model_name:
                        call_kwargs["reasoning_effort"] = "low"
                    response = client.chat.completions.create(**call_kwargs)
                    raw = response.choices[0].message.content or ''
                    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
                    if think_match:
                        think_text = think_match.group(1).strip()
                        if think_text:
                            print(f"\n🧠 [Groq Qwen Thinking Trace]:\n{think_text}\n")
                    print(f"[DEBUG RAW] {model_name} output: {raw[:200]}...")
                    decision = self._extract_json(raw)

                    if decision is None:
                        m_id = re.search(r'["\']?(?:target_dot_id|target_id|dot_id|id)["\']?\s*:\s*["\']?(?:dot[\s_\[#:]*)?(\d+)', raw, re.IGNORECASE)
                        m_act = re.search(r'["\']?action["\']?\s*:\s*["\']([a-z_]+)["\']', raw)
                        m_x = re.search(r'["\']?x["\']?\s*:\s*(\d+(?:\.\d+)?)', raw)
                        m_y = re.search(r'["\']?y["\']?\s*:\s*(\d+(?:\.\d+)?)', raw)
                        m_coord = re.search(r'["\']?(?:target_point|coordinate|point)["\']?\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', raw)
                        if m_id or m_coord:
                            decision = {
                                "target_dot_id": int(m_id.group(1)) if m_id else None,
                                "action": m_act.group(1) if m_act else "click",
                                "x": float(m_x.group(1)) if m_x else (float(m_coord.group(1)) if m_coord else None),
                                "y": float(m_y.group(1)) if m_y else (float(m_coord.group(2)) if m_coord else None),
                                "reasoning": raw[:160].replace("\n", " ")
                            }

                    if decision is not None:
                        decision['model_used'] = model_name
                        decision["provider"] = str(entry.get("provider", "groq"))
                        decision['key_used'] = f"Key #{active_idx + 1} ({masked_key})"
                        decision['manifest_candidate_ids'] = kwargs.get("_prepared_manifest_ids", [d.id for d in all_candidates])

                        target_id = decision.get("target_dot_id") or decision.get("target_id") or decision.get("dot_id") or decision.get("id")
                        if isinstance(target_id, list) and len(target_id) > 0:
                            target_id = target_id[0]
                        elif isinstance(target_id, str):
                            m = re.search(r'\d+', target_id)
                            target_id = int(m.group()) if m else None

                        cand_ids_set = {d.id for d in key_nodes}
                        args = decision.get("arguments", {}) if isinstance(decision.get("arguments"), dict) else {}
                        coord = (
                            decision.get("target_point")
                            or decision.get("coordinate")
                            or decision.get("point")
                            or args.get("target_point")
                            or args.get("coordinate")
                            or args.get("point")
                        )
                        if target_id is None and isinstance(coord, (list, tuple)) and len(coord) == 1:
                            try:
                                c0 = int(coord[0])
                                if c0 in cand_ids_set:
                                    target_id = c0
                            except (ValueError, TypeError):
                                pass

                        if target_id is None:
                            delib = decision.get("deliberation", {})
                            plan_text = str(delib.get("action_plan", "")) + " " + str(decision.get("reasoning", "")) + " " + str(delib.get("candidate_evaluation", "")) + " " + raw
                            plan_mentions = [int(m) for m in re.findall(r'(?:\[|Dot\s*|#)(\d+)(?:\]|\b)', plan_text, re.IGNORECASE) if int(m) in cand_ids_set]
                            if plan_mentions:
                                target_id = plan_mentions[0]

                        raw_px = decision.get("x") or args.get("x")
                        raw_py = decision.get("y") or args.get("y")
                        if raw_px is None and isinstance(coord, (list, tuple)) and len(coord) == 2:
                            raw_px, raw_py = coord[0], coord[1]

                        target_droplet = (
                            next((d for d in all_candidates if d.id == target_id), None)
                            or next((d for d in key_nodes if d.id == target_id), None)
                        ) if target_id is not None else None

                        if target_droplet is not None:
                            decision["target_x"] = float(target_droplet.center_x)
                            decision["target_y"] = float(target_droplet.center_y)
                            decision["target_droplet_id"] = int(target_droplet.id)
                            decision["target_dot_id"] = int(target_droplet.id)
                        elif raw_px is not None and raw_py is not None:
                            try:
                                f_px, f_py = float(raw_px), float(raw_py)
                                candidate_points = [(f_px, f_py)]
                                if abs(scale - 1.0) > 1e-3:
                                    candidate_points.append((f_px / max(scale, 1e-9), f_py / max(scale, 1e-9)))
                                if 0 <= f_px <= 1000 and 0 <= f_py <= 1000:
                                    candidate_points.append((f_px * sw_screen / 1000.0, f_py * sh_screen / 1000.0))
                                best_snap = None
                                best_dist = 9999.0
                                for px, py in candidate_points:
                                    for d in key_nodes:
                                        dist = math.hypot(d.center_x - px, d.center_y - py)
                                        if dist <= 45.0 and dist < best_dist:
                                            best_dist = dist
                                            best_snap = d
                                if best_snap is not None:
                                    decision["target_x"] = float(best_snap.center_x)
                                    decision["target_y"] = float(best_snap.center_y)
                                    decision["target_droplet_id"] = int(best_snap.id)
                                    decision["target_dot_id"] = int(best_snap.id)
                                    target_droplet = best_snap
                            except (ValueError, TypeError):
                                pass

                        # Grounded manifest fallback: if response was ungrounded/null, fallback to top-ranked manifest candidate
                        if target_droplet is None and all_candidates:
                            top_cand = all_candidates[0]
                            decision["target_x"] = float(top_cand.center_x)
                            decision["target_y"] = float(top_cand.center_y)
                            decision["target_droplet_id"] = int(top_cand.id)
                            decision["target_dot_id"] = int(top_cand.id)
                            decision["selection_source"] = "manifest_fallback"
                            decision["reasoning"] = f"{decision.get('reasoning', '')} [Grounded fallback to top candidate]"
                            target_droplet = top_cand
                        elif target_droplet is None:
                            decision["target_x"] = None
                            decision["target_y"] = None
                            decision["target_droplet_id"] = None
                            decision["action_type"] = "error"
                            decision["confidence"] = 0.0
                            decision["reasoning"] = f"{decision.get('reasoning', '')} [Refused ungrounded response: target not in candidate manifest]"

                        if target_droplet is not None:
                            # Continuous and sub-character geometry interpolation
                            try:
                                from ..core.spatial_geometry import SpatialGeometryEngine
                                sub_coords = SpatialGeometryEngine.parse_and_resolve(instruction, target_droplet)
                                if sub_coords:
                                    decision["target_x"] = float(sub_coords[0])
                                    decision["target_y"] = float(sub_coords[1])
                                    decision["reasoning"] = f"{decision.get('reasoning', '')} [SubGeometry: ({sub_coords[0]:.1f}, {sub_coords[1]:.1f})]"
                            except Exception:
                                pass

                        if not decision.get("deliberation") or not isinstance(decision.get("deliberation"), dict):
                            t_id_str = f"Dot {decision.get('target_droplet_id')}" if decision.get('target_droplet_id') else "screen"
                            decision["deliberation"] = {
                                "intent_analysis": f"Execute {decision.get('action', 'click')} on {t_id_str}",
                                "spatial_anchoring": f"Target coordinates ({decision.get('target_x', 0)}, {decision.get('target_y', 0)})",
                                "candidate_evaluation": f"Selected target {t_id_str} over candidates",
                                "action_plan": f"Perform {decision.get('action', 'click')} on target {t_id_str}"
                            }

                        with self.lock:
                            self.current_idx = (active_idx + 1) % num_keys
                        return DecisionResult(decision)
                except Exception as e:
                    err_str = str(e)
                    print(f"[DEBUG ERROR] {model_name} failed: {err_str}")
                    last_err = f"Key #{active_idx + 1} ({masked_key}) [{model_name}]: {err_str}"
                    if "413" in err_str or "request too large" in err_str.lower():
                        # Dynamically compact user_content so the very next attempt fits within Groq's 7000 ITPM limit
                        lines_u = user_content.splitlines()
                        if len(lines_u) > 60:
                            user_content = "\n".join(lines_u[:int(len(lines_u) * 0.72)] + [lines_u[-1]])
                        continue
                    if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                        secs = self._parse_groq_retry_seconds(err_str)
                        self._key_cooldowns[(masked_key, model_name)] = time.time() + secs
                    elif "429" in err_str or "rate limit" in err_str.lower():
                        self._key_cooldowns[(masked_key, model_name)] = time.time() + 10.0
                    continue

        return DecisionResult({
            "error": f"All {num_keys} Groq keys in pool failed. Last detail: {last_err}",
            "target_x": None,
            "target_y": None,
            "target_droplet_id": None,
            "action_type": "error",
            "confidence": 0.0,
            "reasoning": f"Failures across all Groq keys: {last_err}"
        })

    def decide_turn(
        self,
        goal: str,
        memory: EpisodicTrajectoryMemory,
        droplets: List[DiagonalDroplet],
        win_ctx: Optional[WindowContext] = None,
        screen_awareness: Optional[ScreenAwarenessContext] = None,
        loop_intervention: Optional[str] = None,
        screenshot: Optional[Any] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Cognitive Multi-Turn Trajectory Decision Step.
        Evaluates overall goal progress, trajectory history, live screen awareness,
        and freshly sprayed droplets to return the next action with explicit semantic
        intention and predicted visual screen transition.
        """
        target_provider = str(provider or kwargs.get("provider") or os.environ.get("SPATIAL_PROVIDER") or "").strip().lower()
        if not self.key_pool and target_provider not in (
            "gemini", "gemini-pro", "google", "gemini_pro", "gemini-flash", "gemini-1.5-pro", "gemini-2.0-flash",
            "openai", "azure", "local", "ollama", "vllm", "sglang"
        ):
            return {"error": "No OpenAI/Groq clients available in pool"}

        win_info = "Desktop / Full Screen"
        has_win = bool(win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop")
        if has_win:
            win_info = f"\"{win_ctx.title}\" [Bounds: {win_ctx.rect}]"

        key_nodes = [d for d in droplets if d.category != 'surface_probe']
        max_ui_nodes = 36
        goal_words = set(re.findall(r'\w+', goal.lower()))

        # Check if an active modal dialog is present in screen awareness
        modal_rect = None
        if screen_awareness and screen_awareness.modal_context and screen_awareness.modal_context.rect:
            modal_rect = screen_awareness.modal_context.rect

        DIALOG_CHOICE_TOKENS = {
            # Confirmation & commit (universal dialog actions)
            "save", "select", "folder", "open", "choose", "confirm", "ok", "yes", "apply",
            "done", "finish", "proceed", "continue", "submit", "create", "insert", "upload", "pick",
            # Dismissal & cancel
            "cancel", "dont", "don't", "discard", "no", "abort", "close", "exit", "quit",
            # Recovery & alternatives
            "retry", "reload", "overwrite", "replace", "keep", "ignore", "skip", "next"
        }

        def relevance_score(d: DiagonalDroplet):
            lbl_lower = d.label.lower()
            label_words = set(re.findall(r'\w+', lbl_lower))

            # 1. Direct goal keyword matches
            kw_match = 200 if bool(goal_words.intersection(label_words)) else 0
            num_match = 300 if str(d.id) in goal_words else 0

            # 2. Modal dialog bounds containment
            is_modal_element = False
            if modal_rect:
                mx1, my1, mx2, my2 = modal_rect
                if (mx1 - 10 <= d.center_x <= mx2 + 10 and my1 - 10 <= d.center_y <= my2 + 10):
                    is_modal_element = True
            modal_score = 500 if is_modal_element else 0

            # 3. Actionable dialog buttons / confirmation choices (Save, Don't Save, Cancel, OK, etc.)
            dialog_score = 160 if (bool(DIALOG_CHOICE_TOKENS.intersection(label_words)) and (d.is_foreground or is_modal_element)) else 0

            # 4. Foreground window controls vs desktop background
            if has_win:
                fg_score = 50 if d.is_foreground else 0
                app_score = 10 if d.category == 'interactive_app' else 0
            else:
                fg_score = 10 if d.is_foreground else 0
                app_score = 60 if d.category == 'interactive_app' else 0

            return (modal_score, num_match, kw_match, dialog_score, fg_score, app_score)

        if len(key_nodes) > max_ui_nodes:
            key_nodes.sort(key=relevance_score, reverse=True)
            key_nodes = key_nodes[:max_ui_nodes]
        elif has_win:
            key_nodes.sort(key=lambda d: (not d.is_foreground, d.id))

        surface_nodes = [d for d in droplets if d.category == 'surface_probe'][:3]
        nodes_for_ai = key_nodes + surface_nodes
        nodes_for_ai.sort(key=lambda d: d.id)

        manifest_lines = []
        for d in nodes_for_ai:
            is_in_modal = False
            if modal_rect:
                mx1, my1, mx2, my2 = modal_rect
                is_in_modal = (mx1 - 10 <= d.center_x <= mx2 + 10 and my1 - 10 <= d.center_y <= my2 + 10)
            ctx_tag = "MODAL" if is_in_modal else ("APP" if d.is_foreground else "DESK")
            role_tag = f"Role={getattr(d, 'semantic_role', 'btn')[:4].upper()} " if getattr(d, 'semantic_role', '') else ""
            lbl_compact = d.label[:36].replace('"', "'")
            manifest_lines.append(
                f"• [{d.id}] ({d.center_x},{d.center_y}) {role_tag}{ctx_tag} \"{lbl_compact}\""
            )
        manifest = "\n".join(manifest_lines)
        sw_turn = 1920
        sh_turn = 1080
        if screenshot is not None and hasattr(screenshot, 'shape') and len(screenshot.shape) >= 2:
            sh_turn, sw_turn = screenshot.shape[:2]

        win_rect = None
        if win_ctx and win_ctx.is_valid and win_ctx.rect and len(win_ctx.rect) == 4 and getattr(win_ctx, 'title', '') != "Desktop":
            win_rect = win_ctx.rect
        elif screen_awareness and screen_awareness.active_window_rect:
            win_rect = screen_awareness.active_window_rect

        spatial_graph = self._build_spatial_layout_graph(nodes_for_ai, int(sw_turn), int(sh_turn), modal_rect, win_rect=win_rect)
        trajectory_history = memory.format_history_for_prompt()

        grounding_info = get_system_grounding_context()
        grounding_text = grounding_info["text"]
        user_name = grounding_info["user_name"]
        docs_dir = grounding_info["docs_dir"]
        downloads_dir = grounding_info["downloads_dir"]

        awareness_block = ""
        if screen_awareness:
            modal_desc = f"⚠️ ACTIVE MODAL DIALOG: '{screen_awareness.modal_context.title}'" if screen_awareness.modal_context else "None"
            zones = screen_awareness.layout_zones
            nav_s = ", ".join(zones.get("navigation", [])) or "None"
            act_s = ", ".join(zones.get("action_controls", [])) or "None"
            stat_s = ", ".join(zones.get("status_info", [])) or "None"
            awareness_block = f"""
LIVE SCREEN AWARENESS:
• Application: {screen_awareness.app_name} [Scene Type: {screen_awareness.scene_type}]
• Active Modal: {modal_desc}
• Summary: {screen_awareness.summary}
• Key Layout Zones:
  - Navigation: [{nav_s}]
  - Action Controls: [{act_s}]
  - Status/Info: [{stat_s}]
"""

        system_instruction = f"""<system_capability>
Role: Autonomous Cognitive Spatial Desktop Automation Brain controlling a Windows PC.
Architecture: Middle Router between Foundation Models, Trajectory Orchestrator, and OS Input Dispatch.
Grounding: Spatial Droplet Set-of-Marks live visual grounding, 2D Visual Spatial Layout & Connectivity Graph, and Windows UIA sensory state.
Active Foreground Window: {win_info}
</system_capability>

<constitution>
Operational Invariants (NON-NEGOTIABLE):
1. 5-STAGE DEEP DELIBERATION: Every turn requires deep autoregressive deliberation across:
   - user_intent_and_success_criteria: Target application, goal requirements, implicit constraints, visual completion proof.
   - screen_state_and_visual_hierarchy: Foreground window focus, blocking modals, 2D functional screen zones ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP]).
   - mistake_critique_and_loop_diagnosis: Prior action critique, root-cause failure diagnosis, avoidance of repeated traps.
   - end_to_end_master_plan: Milestone plan tracking across turns ([Done], [Active], [Pending]).
   - tactical_action_justification: Step-by-step justification for chosen action & Dot ID, with explicit distractor elimination.
2. STEP CRITIQUE & LOOP BREAKING:
   - If previous action had Outcome=NO_EFFECT, EXECUTION_FAILED, or PARTIAL_CHANGE, repeating the identical action or clicking the same element is STRICTLY FORBIDDEN.
   - You MUST diagnose the failure in "mistake_critique_and_loop_diagnosis" and switch tactical modality in "end_to_end_master_plan" (e.g. hotkey, double-click, direct "open_path", keyboard Enter, minimize, maximize).
   - NEVER alternate between two actions in a ping-pong loop.
3. GROUNDED WINDOWS PATHS & WINDOW CONTROLS:
   - Current Windows User: "{user_name}". Never guess generic accounts.
   - Folder/Drive Navigation: Use action "open_path" with input_text (e.g. "{docs_dir}", "{downloads_dir}", "D:\\") or action "hotkey" with key_combo ["win", "e"].
   - Application Launch & Window Switching: Use action "launch_app" with input_text (e.g. "calc", "notepad", "camera", "powershell") or action "switch_window".
   - Window State: To minimize current window, use action "minimize" or click the titlebar minimize button. To maximize, use action "maximize".
4. MODAL DIALOG PRECEDENCE:
   - In modals/prompts ("Save Changes", "Confirm Exit", "Save As"), interact directly with dialog choices (Save, Don't Save, Cancel) instead of clicking the main window Close button.
5. GOAL COMPLETION: When visual evidence confirms the user's overall goal is accomplished, output action "stop" and status "completed".
6. FLUID HUMAN ACTIONS:
   - Text replacement: "clear_and_type" with input_text. Form submit: "type_and_enter". Shortcuts: "copy", "paste", "select_all", "undo", "redo", "save", "find", "minimize", "maximize", "win_run".
7. ZERO PASSIVE TEXT DELIBERATION:
   - Every single turn MUST dispatch a verified, grounded desktop action. You MUST NEVER return conversational text or deliberation without an action. Zero turns terminate in passive commentary.
</constitution>

<screen_context>
Foreground Window: {win_info}
{awareness_block}
</screen_context>

<few_shot_exemplars>
--- EXEMPLAR 1: In-App UI Interaction & Input ---
User Goal: "type meeting notes into notepad"
Observation: Notepad active in foreground. Editor canvas anchored at Dot [14] in [WORKSPACE].
Decision:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "Enter text 'meeting notes' into Notepad editor canvas.",
    "screen_state_and_visual_hierarchy": "Notepad editor canvas visible at Dot [14] in [WORKSPACE].",
    "mistake_critique_and_loop_diagnosis": "Initial turn; editor focused as expected.",
    "end_to_end_master_plan": [
      "[Phase 1] Type meeting notes into editor canvas Dot [14] (ACTIVE)",
      "[Phase 2] Verify text entered and complete (PENDING)"
    ],
    "tactical_action_justification": "Use click_and_type on Dot [14] to insert text directly into document editor."
  }},
  "status": "in_progress",
  "next_subgoal": "Type meeting notes into document",
  "action": "click_and_type",
  "target_dot_id": 14,
  "input_text": "meeting notes",
  "semantic_intention": "Type requested text into document editor",
  "expected_transition": "Text 'meeting notes' appears in Notepad editor",
  "reasoning": "Directly typing into the editor canvas inserts the user's text."
}}

--- EXEMPLAR 2: Goal Completion ---
User Goal: "open camera"
Observation: Windows Camera application is active in the foreground with live camera preview feed verified.
Decision:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "Open Windows Camera app. Success Criteria: Camera viewfinder active.",
    "screen_state_and_visual_hierarchy": "Camera app is running and visible in foreground [WORKSPACE].",
    "mistake_critique_and_loop_diagnosis": "Camera launched successfully in previous turn.",
    "end_to_end_master_plan": [
      "[Phase 1] Launch camera (DONE)",
      "[Phase 2] Verify viewfinder & complete (DONE)"
    ],
    "tactical_action_justification": "Goal conditions completely satisfied. Terminate trajectory successfully."
  }},
  "status": "completed",
  "next_subgoal": "Complete task",
  "action": "stop",
  "target_dot_id": null,
  "input_text": null,
  "semantic_intention": "Finish trajectory execution",
  "expected_transition": "None; task complete",
  "reasoning": "The Camera application is open and active in the foreground, satisfying all requirements of the user's goal."
}}
</few_shot_exemplars>

<action_dispatch>
You MUST respond with valid JSON where the "deliberation" object is generated FIRST:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "<query decomposition: target application, explicit & implicit constraints, completion proof>",
    "screen_state_and_visual_hierarchy": "<active window, focus state, blocking modals/spinners, 2D functional screen zones ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP])>",
    "mistake_critique_and_loop_diagnosis": "<critique of previous turn outcome (NO_EFFECT, EXECUTION_FAILED, PARTIAL_CHANGE); diagnose root cause, forbid repeating mistakes>",
    "end_to_end_master_plan": [
      "[Phase 1] <milestone description> (DONE/ACTIVE/PENDING)",
      "[Phase 2] <milestone description> (PENDING)"
    ],
    "tactical_action_justification": "<step-by-step rationale for why chosen action & Dot ID are optimal, with explicit distractor elimination>"
  }},
  "status": "in_progress" | "completed" | "failed",
  "next_subgoal": "<brief description of what this single turn achieves>",
  "action": "click" | "double_click" | "triple_click" | "right_click" | "middle_click"
          | "ctrl_click" | "shift_click" | "alt_click"
          | "clear_and_type" | "type_and_enter" | "click_and_type" | "type"
          | "press" | "hotkey" | "copy" | "paste" | "select_all" | "undo" | "redo" | "save" | "find"
          | "hover" | "hover_and_wait" | "drag" | "scroll" | "open_path" | "launch_app" | "switch_window" | "minimize" | "maximize" | "win_run" | "wait" | "stop",
  "target_dot_id": <int or null (Required if interacting with a specific UI element)>,
  "x": <int or null (Absolute X coordinate for start/target if dot ID not used)>,
  "y": <int or null (Absolute Y coordinate for start/target if dot ID not used)>,
  "drag_destination_dot_id": <int or null (Required if action is 'drag' and target is an element)>,
  "drag_dest_x": <int or null (Absolute X coordinate for drag destination if dot ID not used)>,
  "drag_dest_y": <int or null (Absolute Y coordinate for drag destination if dot ID not used)>,
  "press_enter": <boolean (optional, if true commits text with Enter in clear_and_type or click_and_type)>,
  "key": "<single key name (e.g. enter, esc, tab, backspace, up, down, left, right)>" or null,
  "key_combo": <list of strings (e.g. ["ctrl", "c"], ["alt", "tab"]) or null>,
  "scroll_direction": "vertical" | "horizontal",
  "scroll_amount": <int or null>,
  "input_text": <string or null (Text to type or command to run)>,
  "semantic_intention": "<1-sentence clear statement of intent and purpose>",
  "expected_transition": "<exact predicted visual/text transition on screen>",
  "reasoning": "<1-sentence rationale connecting history and screen awareness to this action>"
}}
</action_dispatch>"""

        turn_alert = ""
        if loop_intervention:
            turn_alert = f"""<critical_loop_intervention>
{loop_intervention}
</critical_loop_intervention>
"""
        elif memory and memory.steps:
            last = memory.steps[-1]
            if not last.verified or last.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE"):
                turn_alert = f"""<critical_step_alert>
Previous Action: {last.action.upper()}{f" on '{last.target_label}'" if last.target_label else ""}
Outcome: {last.verdict} ({last.verification_reason})
MANDATE: The previous action failed to make progress. Repeating this action is CONSTITUTIONALLY FORBIDDEN.
Diagnose the failure in "mistake_critique_and_loop_diagnosis" and pivot to an alternative tactical modality in "end_to_end_master_plan".
</critical_step_alert>
"""

        user_content = f"""<overall_user_goal>
{goal}
</overall_user_goal>

<system_grounding>
{grounding_text}
</system_grounding>

<trajectory_history>
{trajectory_history}
</trajectory_history>
{turn_alert}
<live_screen_observation>
Active Foreground Window: {win_info}

<spatial_layout_graph>
{spatial_graph}
</spatial_layout_graph>
</live_screen_observation>

<instruction>
Autoregressively evaluate the screen state and previous step critique in <deliberation>, and dispatch the single next action.
</instruction>"""

        # Strict Prompt Token Budget Guard (<4,800 tokens hard threshold, <5,000 tokens total budget)
        from .token_guard import TokenBudgetGuard
        system_instruction, user_content, nodes_for_ai = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=nodes_for_ai,
            max_tokens=4800,
            goal=goal,
            memory=memory,
            sw=int(sw_turn),
            sh=int(sh_turn),
            modal_rect=modal_rect,
            win_rect=win_rect,
            win_info=win_info,
            grounding_text=grounding_text,
            turn_alert=turn_alert,
        )

        # Gemini Multimodal Provider Routing
        target_provider = str(provider or kwargs.get("provider") or os.environ.get("SPATIAL_PROVIDER") or "").strip().lower()
        if target_provider in ("gemini", "gemini-pro", "google", "gemini_pro", "gemini-flash", "gemini-1.5-pro", "gemini-2.0-flash"):
            try:
                from .providers.registry import get_grounding_provider
                gemini_p = get_grounding_provider(target_provider, **kwargs)
                if gemini_p and gemini_p.is_available():
                    som_img_url = None
                    if screenshot is not None and hasattr(screenshot, 'shape') and len(screenshot.shape) >= 2:
                        som_img_url, _, _ = self._render_unoccluded_som(screenshot, nodes_for_ai, max_image_side=1920, jpeg_quality=85)

                    res = gemini_p.ground_sync(
                        instruction=goal,
                        manifest_text=spatial_graph,
                        all_candidates=nodes_for_ai,
                        som_image_url=som_img_url,
                        win_info=win_info,
                        system_instruction=system_instruction,
                        user_content=user_content,
                        **kwargs
                    )
                    if res and not res.error:
                        res_dict = res.to_dict()
                        res_dict["model_used"] = res.model_used or "gemini-1.5-pro"
                        res_dict["provider_used"] = "gemini"
                        return self._clean_decision(res_dict)
                    else:
                        print(f"[Rotator Trajectory] Gemini returned error ({res.error if res else 'None'}); falling back to 25-key Groq pool.")
            except Exception as gemini_err:
                print(f"[Rotator Trajectory] Gemini dispatch failed ({gemini_err}); falling back to 25-key Groq pool.")

        # Sync with live environment / UI model selector if changed
        env_use_ngrok = os.environ.get("USE_NGROK")
        if env_use_ngrok == "0":
            self.use_ngrok = False
        elif env_use_ngrok == "1" and os.environ.get("KAGGLE_QWEN_MODEL"):
            if not self.use_ngrok or self.ngrok_model != os.environ.get("KAGGLE_QWEN_MODEL"):
                self.set_active_model(True, os.environ.get("KAGGLE_QWEN_MODEL"))
        if os.environ.get("GROQ_PRIMARY_MODEL"):
            self.preferred_groq_model = os.environ.get("GROQ_PRIMARY_MODEL")

        # Primary Priority: Dedicated Kaggle Qwen endpoint if enabled
        if getattr(self, "use_ngrok", False) and self.ngrok_client is not None:
            try:
                user_msg_payload: Any = user_content
                if screenshot is not None:
                    try:
                        som_img_url, _, _ = self._render_unoccluded_som(screenshot, nodes_for_ai, max_image_side=1280, jpeg_quality=82)
                        if som_img_url:
                            user_msg_payload = [
                                {"type": "text", "text": user_content},
                                {"type": "image_url", "image_url": {"url": som_img_url}}
                            ]
                    except Exception:
                        user_msg_payload = user_content

                call_kwargs = {
                    "model": self.ngrok_model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_msg_payload}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1200,
                    "timeout": 45.0
                }
                response = self.ngrok_client.chat.completions.create(**call_kwargs)
                msg = response.choices[0].message
                raw = (msg.content or getattr(msg, "reasoning", "") or '').strip()
                decision = self._extract_json(raw)
                if decision:
                    decision['model_used'] = f"ngrok/{self.ngrok_model}"
                    decision['key_used'] = "Kaggle Qwen GPU"
                    try:
                        print(f"[Rotator Trajectory] Turn resolved via Kaggle Qwen ({self.ngrok_model})")
                    except Exception:
                        pass
                    return DecisionResult(decision)
            except Exception as ngrok_err:
                print(f"[Rotator Trajectory] Kaggle Qwen primary failed ({type(ngrok_err).__name__}: {str(ngrok_err)[:120]}); falling back to key pool.")

        num_keys = len(self.key_pool)
        if num_keys == 0:
            return {
                "error": "No OpenAI/Groq clients available in pool",
                "action": "error",
                "reasoning": "Primary provider failed and key_pool is empty; unable to rotate to fallback clients."
            }

        last_err = ""
        with self.lock:
            start_idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % num_keys

        pref_model = getattr(self, "preferred_groq_model", None) or os.environ.get("GROQ_PRIMARY_MODEL", "qwen/qwen3.8-27b")

        for offset in range(num_keys):
            active_idx = (start_idx + offset) % num_keys
            entry = self.key_pool[active_idx]
            client = entry["client"]
            masked_key = entry["masked"]

            base_models = list(entry.get("models", GROQ_MODELS))
            if pref_model:
                models_to_try = [pref_model] + [m for m in base_models if m != pref_model]
            else:
                models_to_try = base_models
            for model_name in models_to_try:
                try:
                    print(f"[*] Calling Groq Rotator Key #{active_idx + 1} ({masked_key}) with model '{model_name}'...")
                    call_kwargs = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    if "gpt-oss" in model_name:
                        call_kwargs["reasoning_effort"] = "low"
                    response = client.chat.completions.create(**call_kwargs)
                    raw = response.choices[0].message.content or ''
                    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
                    if think_match:
                        think_text = think_match.group(1).strip()
                        if think_text:
                            print(f"\n🧠 [Groq Qwen Thinking Trace]:\n{think_text}\n")
                    decision = self._extract_json(raw)
                    if decision:
                        decision['model_used'] = model_name
                        decision['key_used'] = f"Key #{active_idx + 1} ({masked_key})"
                        try:
                            print(f"[Rotator Trajectory] Turn resolved via Groq Key #{active_idx + 1} ({masked_key}) using {model_name}")
                        except Exception:
                            pass
                        return DecisionResult(decision)
                except Exception as e:
                    err_str = str(e)
                    last_err = f"Key #{active_idx + 1} ({masked_key}) [{model_name}]: {err_str}"
                    if "413" in err_str or "request too large" in err_str.lower() or "itpm" in err_str.lower():
                        # Immediately compact user_content so subsequent keys succeed without failing the entire pool
                        print(f"[*] Prompt exceeded Groq rate limit ({err_str[:90]}); dynamically pruning prompt...")
                        lines_u = user_content.splitlines()
                        if len(lines_u) > 25:
                            user_content = "\n".join(lines_u[:int(len(lines_u) * 0.60)] + [lines_u[-1]])
                        continue
                    if any(k in err_str.lower() for k in ("429", "rate_limit", "tpm", "quota")):
                        break

        return DecisionResult({
            "error": f"All {num_keys} Groq keys in pool failed. Last detail: {last_err}",
            "target_x": None,
            "target_y": None,
            "target_droplet_id": None,
            "target_dot_id": None,
            "action_type": "error",
            "confidence": 0.0,
            "reasoning": f"Failures across Groq keys: {last_err}"
        })

    decide_trajectory_turn = decide_turn
    decide_action = decide


# Module-level convenience aliases
_clean_decision = DiagonalDropletReasoner._clean_decision
_extract_json = DiagonalDropletReasoner._extract_json
classify_screen_zone = DiagonalDropletReasoner.classify_screen_zone
_build_spatial_layout_graph = DiagonalDropletReasoner._build_spatial_layout_graph
