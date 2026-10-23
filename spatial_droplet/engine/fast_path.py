"""
Spatial Droplet Fast-Path Router (Phase 4 & 5).
Dispatches direct numeric inputs, system shortcuts, window switching,
and unambiguous landmark clicks with zero AI latency (0ms).
"""

import ctypes
from ctypes import wintypes
import re
from typing import Optional, Tuple, List, Dict, Any

from ..core.models import DiagonalDroplet, WindowContext


class FastPathRouter:
    """
    Sub-millisecond fast-path router.
    Bypasses LLM reasoning when commands can be deterministically resolved:
      - Direct numeric dot index (e.g. "3", "dot 12")
      - System operations (volume, scroll, clipboard, minimize, maximize)
      - Window switching via Win32 EnumWindows (e.g. "switch to chrome")
      - Unambiguous landmark exact/substring matches (e.g. "click Save")
    """

    @staticmethod
    def find_window_by_name(query: str) -> Optional[Tuple[int, str]]:
        try:
            from live_copilot import IntentScopeRouter
            return IntentScopeRouter.find_window_by_name(query)
        except Exception:
            return None

    ENABLED = True

    @classmethod
    def route(cls, cmd: str, droplets: List[DiagonalDroplet], win_ctx: Optional[WindowContext]) -> Optional[Dict[str, Any]]:
        if not getattr(cls, 'ENABLED', True):
            return None

        raw = cmd.strip()
        low = raw.lower()

        # 0. OS Fast-Path Router (Lifecycle: Open/Launch, Switch, Window Management)
        try:
            from live_copilot import IntentScopeRouter
            os_route = IntentScopeRouter.route(cmd, win_ctx)
            if os_route:
                act = os_route.get("action")
                if act == "launch_app":
                    return {
                        "target_dot_id": None,
                        "action": "system_launch_app",
                        "target": os_route.get("target"),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path OS (0ms)"
                    }
                elif act == "window_switch":
                    return {
                        "target_dot_id": None,
                        "action": "system_window_switch",
                        "target_hwnd": os_route.get("target_hwnd"),
                        "target_title": os_route.get("target_title", os_route.get("target", "Window")),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path Win32 (0ms)"
                    }
                elif act == "window_op":
                    return {
                        "target_dot_id": None,
                        "action": "system_window_op",
                        "op": os_route.get("op"),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path Win32 (0ms)"
                    }
                elif act == "system_hotkey":
                    return {
                        "target_dot_id": None,
                        "action": "system_hotkey",
                        "combo": os_route.get("combo"),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path System (0ms)"
                    }
                elif act == "system_open_path":
                    return {
                        "target_dot_id": None,
                        "action": "system_open_path",
                        "path": os_route.get("path"),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path Shell (0ms)"
                    }
                elif act == "system_open_url":
                    # Automatic fixed browser URL launching disabled per specification: routes through cognitive reasoner
                    return None
                elif act == "system_compound":
                    return {
                        "target_dot_id": None,
                        "action": "system_compound",
                        "steps": os_route.get("steps", []),
                        "reasoning": f"Fast-Path OS: {os_route.get('summary', '')}",
                        "model_used": "Fast-Path OS (0ms)"
                    }
        except Exception:
            pass

        # 1. Direct Numeric Shortcut: e.g. "3", "click 3", "dot 3", "double click 3"
        m_num = re.match(r'^(?:click\s+|double\s*click\s+|dot\s+)?(\d+)$', raw, re.IGNORECASE)
        if m_num:
            target_id = int(m_num.group(1))
            node = next((d for d in droplets if d.id == target_id), None)
            if node:
                is_dbl = 'double' in low or (node.category == 'interactive_app')
                return {
                    "target_dot_id": target_id,
                    "action": "double_click" if is_dbl else "click",
                    "input_text": None,
                    "steps": [],
                    "reasoning": f"Fast-Path: Direct Numeric Dot [{target_id}] ({node.label})",
                    "model_used": "Fast-Path Number (0ms)"
                }

        # 2. System Hotkeys & Window Management
        # Volume
        if low in ("mute", "unmute", "volume mute", "toggle sound"):
            return {
                "target_dot_id": None, "action": "system_hotkey", "key": "volumemute",
                "reasoning": "Fast-Path: Toggle volume mute", "model_used": "Fast-Path System (0ms)"
            }
        elif low in ("volume up", "increase volume", "louder"):
            return {
                "target_dot_id": None, "action": "system_hotkey", "key": "volumeup", "presses": 3,
                "reasoning": "Fast-Path: Volume up", "model_used": "Fast-Path System (0ms)"
            }
        elif low in ("volume down", "decrease volume", "quieter"):
            return {
                "target_dot_id": None, "action": "system_hotkey", "key": "volumedown", "presses": 3,
                "reasoning": "Fast-Path: Volume down", "model_used": "Fast-Path System (0ms)"
            }

        # Scroll
        if low in ("scroll down", "page down", "down"):
            return {
                "target_dot_id": None, "action": "system_scroll", "amount": -600,
                "reasoning": "Fast-Path: Scroll down", "model_used": "Fast-Path System (0ms)"
            }
        elif low in ("scroll up", "page up", "up"):
            return {
                "target_dot_id": None, "action": "system_scroll", "amount": 600,
                "reasoning": "Fast-Path: Scroll up", "model_used": "Fast-Path System (0ms)"
            }

        # Screen Rescan / Refresh
        if low in ("rescan", "scan", "refresh", "sync", "sync screen", "reload"):
            return {
                "target_dot_id": None, "action": "system_rescan",
                "reasoning": "Fast-Path: Screen rescan and droplet sync", "model_used": "Fast-Path System (0ms)"
            }

        # Drive Navigation: e.g. "go to drive d", "open drive d", "drive c", "d:", "can you go to drive e"
        m_drive = re.match(r'^(?:can\s+you\s+|please\s+)?(?:go\s+to\s+|open\s+|explore\s+)?(?:drive\s+)?([a-zA-Z])(?::|:\\)?$', low)
        if m_drive and ("drive" in low or ":" in low or len(low.split()) <= 4):
            dl = m_drive.group(1).upper()
            return {
                "target_dot_id": None, "action": "system_open_path", "path": f"{dl}:\\",
                "reasoning": f"Fast-Path: Open Drive {dl}:\\ in File Explorer", "model_used": "Fast-Path Shell (0ms)"
            }

        # Standard User Folders: e.g. "open documents", "documents", "downloads", "open desktop"
        m_folder = re.match(r'^(?:can\s+you\s+|please\s+)?(?:go\s+to\s+|open\s+|explore\s+)?(?:the\s+|my\s+)?(documents|downloads|desktop|pictures|music|videos|home)(?:\s+folder)?$', low)
        if m_folder:
            f_name = m_folder.group(1).lower()
            import pathlib
            home = pathlib.Path.home()
            folder_map = {
                "documents": home / "Documents",
                "downloads": home / "Downloads",
                "desktop": home / "Desktop",
                "pictures": home / "Pictures",
                "music": home / "Music",
                "videos": home / "Videos",
                "home": home
            }
            target_path = str(folder_map.get(f_name, home))
            return {
                "target_dot_id": None, "action": "system_open_path", "path": target_path,
                "reasoning": f"Fast-Path: Open {f_name.title()} ({target_path}) in File Explorer", "model_used": "Fast-Path Shell (0ms)"
            }

        # Window Controls
        if low in ("minimize", "minimize window"):
            return {
                "target_dot_id": None, "action": "system_window_op", "op": "minimize",
                "reasoning": "Fast-Path: Minimize active window", "model_used": "Fast-Path System (0ms)"
            }
        elif low in ("maximize", "maximize window"):
            return {
                "target_dot_id": None, "action": "system_window_op", "op": "maximize",
                "reasoning": "Fast-Path: Maximize active window", "model_used": "Fast-Path System (0ms)"
            }

        # Clipboard
        if low == "copy":
            return {
                "target_dot_id": None, "action": "system_hotkey", "combo": ["ctrl", "c"],
                "reasoning": "Fast-Path: Clipboard Copy (Ctrl+C)", "model_used": "Fast-Path System (0ms)"
            }
        elif low == "paste":
            return {
                "target_dot_id": None, "action": "system_hotkey", "combo": ["ctrl", "v"],
                "reasoning": "Fast-Path: Clipboard Paste (Ctrl+V)", "model_used": "Fast-Path System (0ms)"
            }
        elif low in ("select all", "select-all"):
            return {
                "target_dot_id": None, "action": "system_hotkey", "combo": ["ctrl", "a"],
                "reasoning": "Fast-Path: Select All (Ctrl+A)", "model_used": "Fast-Path System (0ms)"
            }

        # 3. Window Switch / Focus: e.g. "switch to chrome", "focus terminal", "open code"
        m_switch = re.match(r'^(?:switch\s+to|focus|bring\s+up|activate)\s+(.+)$', low)
        if m_switch:
            app_query = m_switch.group(1).strip()
            win_match = cls.find_window_by_name(app_query)
            if win_match:
                target_hwnd, target_title = win_match
                return {
                    "target_dot_id": None,
                    "action": "system_window_switch",
                    "target_hwnd": target_hwnd,
                    "target_title": target_title,
                    "reasoning": f"Fast-Path: Activate window '{target_title[:30]}'",
                    "model_used": "Fast-Path Win32 (0ms)"
                }

        # 4. Unambiguous UI Landmark Fast-Match (0ms)
        clean_cmd = raw.strip().lower()
        # Strip conversational/polite openers
        clean_cmd = re.sub(r'^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|go\s+ahead\s+and\s+)', '', clean_cmd)

        # Check action type
        is_double = bool(re.match(r'^(?:double[\s-]?click)\s+', clean_cmd))
        is_right = bool(re.match(r'^(?:right[\s-]?click)\s+', clean_cmd))

        # Strip action verb & prepositions: "click on", "press", "tap on", "hit", "select", "activate", "open"
        norm_target = re.sub(r'^(?:double[\s-]?click|right[\s-]?click|click|press|tap|hit|select|activate|open)\s+(?:on\s+)?(?:the\s+|a\s+|an\s+)?', '', clean_cmd).strip()
        # Strip trailing UI entity nouns: "button", "icon", "tab", "link", "menu", "item", "option", "shortcut", "widget"
        norm_target_clean = re.sub(r'\s+(?:button|icon|tab|link|menu|item|option|shortcut|widget)$', '', norm_target).strip()

        candidate_tokens = []
        for t in (norm_target_clean, norm_target):
            if len(t) >= 2 and t not in candidate_tokens:
                candidate_tokens.append(t)

        if candidate_tokens:
            ui_nodes = [d for d in droplets if d.category != 'surface_probe']
            fg_nodes = [d for d in ui_nodes if d.is_foreground]
            search_pool = fg_nodes if fg_nodes else ui_nodes

            for target_token in candidate_tokens:
                # Exact match first
                exact_matches = [
                    d for d in search_pool
                    if d.label.strip().lower() == target_token
                    or d.label.strip().lower().replace("[btn] ", "").replace("[icon] ", "").strip() == target_token
                    or getattr(d, 'display_subtext', '').strip().lower() == target_token
                ]
                if len(exact_matches) == 1:
                    node = exact_matches[0]
                    act = "double_click" if (is_double or node.category == 'interactive_app') else ("right_click" if is_right else "click")
                    return {
                        "target_dot_id": node.id,
                        "action": act,
                        "input_text": None,
                        "steps": [],
                        "reasoning": f"Fast-Path: 100% exact match on '{node.label}' (Dot [{node.id}])",
                        "model_used": "Fast-Path Exact Match (0ms)"
                    }

                # Substring match (only if unambiguous 1-to-1)
                sub_matches = [
                    d for d in search_pool
                    if target_token in d.label.lower()
                    or target_token in getattr(d, 'display_subtext', '').lower()
                ]
                if len(sub_matches) == 1:
                    node = sub_matches[0]
                    act = "double_click" if (is_double or node.category == 'interactive_app') else ("right_click" if is_right else "click")
                    return {
                        "target_dot_id": node.id,
                        "action": act,
                        "input_text": None,
                        "steps": [],
                        "reasoning": f"Fast-Path: Unique substring match on '{node.label}' (Dot [{node.id}])",
                        "model_used": "Fast-Path Substring (0ms)"
                    }

        return None
