"""
================================================================================
     DROPLET-AI — AUTONOMOUS DESKTOP COPILOT & SPATIAL HUD OVERLAY
================================================================================
Floating draggable copilot bar with live transparent Set-of-Marks HUD reticles,
OS Fast-Path Intent & Scope Router, pure Droplet sub-40ms deterministic actuation,
asynchronous non-blocking UIA engine, and real-time visual diff closed loop.
================================================================================
"""
import sys
import os
import re
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if os.path.exists(PROJECT_ROOT) and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# PHASE 1: INTENT & SCOPE ROUTER / OS FAST-PATH ROUTER (< 25ms Execution)
# ==============================================================================

class IntentScopeRouter:
    """
    OS Fast-Path Intent & Scope Router.
    Dispatches application lifecycle, window switching, and window management
    deterministically via Win32 APIs in < 25ms, completely bypassing the VLM
    grounding pipeline to eliminate destructive mis-clicks inside active windows.
    """

    # Common Windows application aliases
    APP_ALIASES: Dict[str, str] = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "browser": "chrome",
        "notepad": "notepad",
        "text editor": "notepad",
        "code": "code",
        "vs code": "code",
        "vscode": "code",
        "visual studio code": "code",
        "calc": "calc",
        "calculator": "calc",
        "edge": "msedge",
        "microsoft edge": "msedge",
        "powershell": "powershell",
        "terminal": "powershell",
        "windows terminal": "wt",
        "cmd": "cmd",
        "command prompt": "cmd",
        "explorer": "explorer",
        "file explorer": "explorer",
        "files": "explorer",
        "paint": "mspaint",
        "mspaint": "mspaint",
        "settings": "ms-settings:",
        "control panel": "control",
        "task manager": "taskmgr",
        "taskmgr": "taskmgr",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "slack": "slack",
        "discord": "discord",
        "spotify": "spotify",
        "camera": "microsoft.windows.camera:",
        "webcam": "microsoft.windows.camera:",
    }

    WEB_SERVICES: Dict[str, str] = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "github": "https://github.com",
        "reddit": "https://www.reddit.com",
        "twitter": "https://www.twitter.com",
        "x": "https://www.x.com",
        "spotify": "https://open.spotify.com",
        "netflix": "https://www.netflix.com",
        "chatgpt": "https://chatgpt.com",
        "gmail": "https://mail.google.com",
        "wikipedia": "https://www.wikipedia.org",
        "amazon": "https://www.amazon.com",
    }

    # In-window GUI keywords that must NEVER be treated as OS application launches
    GUI_CONTROL_KEYWORDS = {
        "dropdown", "drop down", "filter", "menu", "tab", "dialog", "panel",
        "sidebar", "drawer", "button", "btn", "icon", "slider", "combobox",
        "swatch", "slide", "star", "checkbox", "radio", "picker", "popup",
        "link", "accordion", "item", "option", "search filter", "search box",
        "input", "field", "view", "section", "canvas", "toggle", "color", "colour",
        "mode", "dark mode", "light mode", "search", "box", "toolbar", "arrow",
        "spinbox", "stepper", "cell", "row", "column", "first", "second", "third",
        "next", "previous", "edit", "editor", "text", "form", "submit", "apply",
        "card", "thumbnail", "palette", "page"
    }

    @classmethod
    def find_window_by_name(cls, query: str) -> Optional[Tuple[int, str]]:
        """
        Searches all top-level running windows for a title or process matching query.
        Attaches to the active input desktop to discover interactive desktop windows across processes.
        Returns (hwnd, window_title) or None.
        """
        query = query.strip().lower()
        if not query:
            return None

        # 1. Attach to active input desktop so EnumWindows sees user interactive windows
        try:
            from spatial_droplet.platform.desktop import DesktopThreadManager
            DesktopThreadManager.attach_input_desktop()
        except Exception:
            pass

        # 2. Query running windows via full desktop window enumeration
        try:
            from spatial_droplet.platform.window import enumerate_all_windows
            wins = enumerate_all_windows(include_minimized=True)
            alias_target = cls.APP_ALIASES.get(query, query).lower()

            best_win = None
            best_score = -1

            for w in wins:
                title = (w.get("title") or "").strip()
                t_lower = title.lower()
                proc = (w.get("process") or "").strip()
                proc_base = proc.lower().replace(".exe", "")
                cname = (w.get("class") or "").strip()

                score = 0
                if query == t_lower:
                    score = 100
                elif alias_target == proc_base or query == proc_base:
                    score = 90
                elif t_lower.startswith(query) or (alias_target and t_lower.startswith(alias_target)):
                    score = 80
                elif query in t_lower or (alias_target and alias_target in t_lower):
                    score = 65
                elif query in proc_base or (alias_target and alias_target in proc_base):
                    score = 55
                elif query in cname.lower() and cname.lower() not in ("chrome_widgetwin_1", "workerw", "progman"):
                    score = 20

                if score > best_score:
                    best_score = score
                    best_win = (w["hwnd"], title or proc or cname)

            if best_win and best_score >= 50:
                return best_win
        except Exception:
            pass

        # Fallback to direct Win32 EnumWindows if window module unavailable
        user32 = ctypes.windll.user32
        matches: List[Tuple[int, str, int]] = []

        def enum_win_cb(hwnd, lp):
            l = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if l > 0:
                tbuf = ctypes.create_unicode_buffer(l + 1)
                user32.GetWindowTextW(hwnd, tbuf, l + 1)
                title = tbuf.value.strip()

            cbuf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cbuf, 256)
            cname = cbuf.value.strip()

            full_str = f"{title} {cname}".lower()
            if title and (query in full_str or cls.APP_ALIASES.get(query, query) in full_str):
                score = 100 if query == title.lower() else (50 if title.lower().startswith(query) else 10)
                matches.append((hwnd, title or cname, score))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        try:
            user32.EnumWindows(WNDENUMPROC(enum_win_cb), 0)
        except Exception:
            pass

        if matches:
            matches.sort(key=lambda m: m[2], reverse=True)
            return (matches[0][0], matches[0][1])
        return None

    @classmethod
    def launch_or_activate_app(cls, app_query: str, parameters: Optional[str] = None) -> Dict[str, Any]:
        """
        Fast-paths application launching:
        1. If parameters provided (e.g. URL) and app is browser, launches with parameters via ShellExecuteW.
        2. If an existing window matching app_query is already open, activates it instantly.
        3. If not running, launches via Win32 ShellExecuteW.
        """
        app_clean = re.sub(r'[\s.,!?]+$', '', app_query.strip().lower())
        target_exe = cls.APP_ALIASES.get(app_clean, app_clean)

        # 1. If parameters provided (such as URL), launch with parameters to open new tab directly
        if parameters:
            try:
                shell32 = ctypes.windll.shell32
                ret = shell32.ShellExecuteW(None, "open", target_exe, parameters, None, 1)
                if ret > 32:
                    return {
                        "success": True,
                        "action": "launch_app",
                        "app": target_exe,
                        "parameters": parameters,
                        "summary": f"Opened '{parameters}' in {target_exe}",
                        "latency_ms": 12
                    }
            except Exception:
                pass

        # 2. Search existing running windows
        existing_win = cls.find_window_by_name(app_clean)
        if not existing_win and target_exe != app_clean:
            existing_win = cls.find_window_by_name(target_exe)

        if existing_win:
            hwnd, title = existing_win
            cls._activate_hwnd(hwnd)
            return {
                "success": True,
                "action": "window_switch",
                "hwnd": hwnd,
                "title": title,
                "summary": f"Switched to '{title[:30]}'",
                "latency_ms": 5
            }

        # 3. Launch new instance via Win32 ShellExecute
        try:
            shell32 = ctypes.windll.shell32
            # ShellExecuteW(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd)
            ret = shell32.ShellExecuteW(None, "open", target_exe, parameters, None, 1)  # SW_SHOWNORMAL = 1
            if ret > 32:
                return {
                    "success": True,
                    "action": "launch_app",
                    "app": target_exe,
                    "summary": f"Launched application '{target_exe}'",
                    "latency_ms": 12
                }
            # Fallback to os.startfile
            if hasattr(os, "startfile") and not parameters:
                os.startfile(target_exe)
                return {
                    "success": True,
                    "action": "launch_app",
                    "app": target_exe,
                    "summary": f"Launched application '{target_exe}' via startfile",
                    "latency_ms": 15
                }
        except Exception as e:
            # Fallback to subprocess
            try:
                import subprocess
                cmd_args = f"{target_exe} {parameters}" if parameters else target_exe
                subprocess.Popen(cmd_args, shell=True)
                return {
                    "success": True,
                    "action": "launch_app",
                    "app": target_exe,
                    "summary": f"Launched application '{target_exe}' via subprocess",
                    "latency_ms": 20
                }
            except Exception as se:
                return {
                    "success": False,
                    "error": str(se),
                    "summary": f"Failed to launch '{target_exe}': {se}"
                }

        return {
            "success": True,
            "action": "launch_app",
            "app": target_exe,
            "summary": f"Dispatched launch for '{target_exe}'",
            "latency_ms": 15
        }

    @classmethod
    def _activate_hwnd(cls, hwnd: int):
        """Restores and brings a window to the foreground reliably bypassing OS focus lock."""
        try:
            from spatial_droplet.platform.window import restore_and_focus_window
            if restore_and_focus_window(hwnd):
                return
        except Exception:
            pass

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        cur_tid = kernel32.GetCurrentThreadId()
        fg_hwnd = user32.GetForegroundWindow()
        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0

        attached = False
        if fg_tid and fg_tid != cur_tid:
            try:
                attached = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
            except Exception:
                attached = False

        try:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE = 9
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            user32.SetFocus(hwnd)
        finally:
            if attached:
                try:
                    user32.AttachThreadInput(cur_tid, fg_tid, False)
                except Exception:
                    pass

    @classmethod
    def route(cls, cmd: str, win_ctx: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Fast-Path Intent & Scope classification.
        Returns a structured decision dict if cmd is an OS-level operation,
        or None if cmd is an in-window GUI grounding task.
        """
        raw = cmd.strip()
        low = raw.lower()

        # Clean conversational polite openers and trailing punctuation
        clean = re.sub(
            r'^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|go\s+ahead\s+and\s+|i\s+want\s+to\s+)',
            '',
            low
        ).strip()
        clean = re.sub(r'[\s.,!?]+$', '', clean).strip()

        # ----------------------------------------------------------------------
        # 1. Window Management Shortcuts (Direct Win32 / Hotkey)
        # ----------------------------------------------------------------------
        if clean in ("minimize", "minimize window", "minimize app", "minimise", "minimise window"):
            return {
                "intent": "os_fast_path",
                "action": "window_op",
                "op": "minimize",
                "summary": "Minimize active window"
            }
        if clean in ("maximize", "maximize window", "maximize app", "maximise", "maximise window", "full screen", "fullscreen"):
            return {
                "intent": "os_fast_path",
                "action": "window_op",
                "op": "maximize",
                "summary": "Maximize active window"
            }
        if clean in ("restore", "restore window", "unmaximize"):
            return {
                "intent": "os_fast_path",
                "action": "window_op",
                "op": "restore",
                "summary": "Restore active window"
            }
        if clean in ("close window", "close app", "exit window", "exit app", "quit window"):
            return {
                "intent": "os_fast_path",
                "action": "window_op",
                "op": "close",
                "summary": "Close active window"
            }
        if clean in ("close tab", "close the tab", "shut tab"):
            return {
                "intent": "os_fast_path",
                "action": "system_hotkey",
                "combo": ["ctrl", "w"],
                "summary": "Close current tab (Ctrl+W)"
            }
        if clean in ("new tab", "open new tab", "open a new tab"):
            return {
                "intent": "os_fast_path",
                "action": "system_hotkey",
                "combo": ["ctrl", "t"],
                "summary": "Open new tab (Ctrl+T)"
            }

        # ----------------------------------------------------------------------
        # 1.3 Web Search Fast-Path (e.g. "search <query> on youtube / google")
        # ----------------------------------------------------------------------
        m_search = re.match(
            r'^(?:search(?:\s+for)?|google)\s+(.+?)(?:\s+(?:on|in)\s+(google|youtube|bing|github))?$',
            clean
        )
        if m_search:
            query = m_search.group(1).strip()
            engine = (m_search.group(2) or "google").strip().lower()
            gui_query_guards = ("filter", "box", "input", "dropdown", "button", "bar", "field", "dialog", "menu")
            if query and not any(k in query.lower() for k in gui_query_guards):
                import urllib.parse
                encoded = urllib.parse.quote_plus(query)
                if engine == "youtube":
                    search_url = f"https://www.youtube.com/results?search_query={encoded}"
                elif engine == "github":
                    search_url = f"https://github.com/search?q={encoded}"
                else:
                    search_url = f"https://www.google.com/search?q={encoded}"
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_url",
                    "url": search_url,
                    "summary": f"Search '{query}' on {engine.title()}"
                }

        # ----------------------------------------------------------------------
        # 1.4 Compound App + Web / Search Intent
        # Pattern A: "go to chrome and open youtube", "open chrome to open youtube", "open chrome and search cute cats"
        # Pattern B: "open youtube in chrome", "open google on chrome", "play lofi on youtube"
        # ----------------------------------------------------------------------
        WEB_SERVICES = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://github.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://www.twitter.com",
            "x": "https://www.x.com",
            "spotify": "https://open.spotify.com",
            "netflix": "https://www.netflix.com",
            "chatgpt": "https://chatgpt.com",
            "gmail": "https://mail.google.com",
            "wikipedia": "https://www.wikipedia.org",
            "amazon": "https://www.amazon.com",
        }

        m_compound_a = re.match(
            r'^(?:open|go\s+to|launch|start|run|switch\s+to)\s+([a-zA-Z0-9\s]+?)\s+(?:and|to|then)\s+(?:open\s+(?:up\s+)?|go\s+to\s+|visit\s+|browse(?:\s+to)?\s+|launch\s+|search(?:\s+for)?\s+|play\s+)?(.+)$',
            clean
        )
        m_compound_b = re.match(
            r'^(?:open|go\s+to|visit|browse(?:\s+to)?|play)\s+(.+?)\s+(?:in|on|using|with)\s+([a-zA-Z0-9\s]+)$',
            clean
        )

        first_app = None
        second_target = None

        if m_compound_a:
            first_app = re.sub(r'[\s.,!?]+$', '', m_compound_a.group(1).strip().lower())
            second_target = re.sub(r'[\s.,!?]+$', '', m_compound_a.group(2).strip())
        elif m_compound_b:
            second_target = re.sub(r'[\s.,!?]+$', '', m_compound_b.group(1).strip())
            first_app = re.sub(r'[\s.,!?]+$', '', m_compound_b.group(2).strip().lower())

        if first_app and second_target:
            app_id = cls.APP_ALIASES.get(first_app, first_app)
            is_valid_app = (app_id in cls.APP_ALIASES.values() or first_app in cls.APP_ALIASES or app_id in WEB_SERVICES)

            if is_valid_app:
                target_low = re.sub(r'[\s.,!?]+$', '', second_target.lower().strip())
                target_url = None

                # 1. Check for generic intent: "some stuff", "some stuff like that"
                if any(k in target_low for k in ("some stuff", "stuff", "something")):
                    return {
                        "intent": "os_fast_path",
                        "action": "launch_app",
                        "target": app_id,
                        "summary": f"Launch or activate '{first_app}'"
                    }

                # 2. Check for search intent inside target: e.g. "youtube for music", "search on youtube", "lofi on youtube"
                m_sub_search = re.match(r'^(?:search\s+)?(?:on\s+|in\s+)?(youtube|google|github)\s+(?:for\s+)?(.+)$', target_low)
                if m_sub_search:
                    sub_engine = m_sub_search.group(1)
                    sub_q = m_sub_search.group(2).strip()
                    import urllib.parse
                    enc = urllib.parse.quote_plus(sub_q)
                    if sub_engine == "youtube":
                        target_url = f"https://www.youtube.com/results?search_query={enc}"
                    elif sub_engine == "github":
                        target_url = f"https://github.com/search?q={enc}"
                    else:
                        target_url = f"https://www.google.com/search?q={enc}"
                elif app_id == "youtube" or first_app == "youtube":
                    # E.g. "play lofi on youtube"
                    import urllib.parse
                    enc = urllib.parse.quote_plus(second_target)
                    target_url = f"https://www.youtube.com/results?search_query={enc}"
                    app_id = "chrome"  # Open in Chrome/browser
                elif target_low in WEB_SERVICES:
                    target_url = WEB_SERVICES[target_low]
                elif target_low.startswith(("http://", "https://", "www.")) or re.search(r'\.(?:com|org|net|io|ai|dev|edu|gov)$', target_low):
                    target_url = target_low if target_low.startswith("http") else f"https://{target_low}"
                else:
                    if app_id in ("chrome", "msedge", "firefox", "browser"):
                        import urllib.parse
                        encoded = urllib.parse.quote_plus(second_target)
                        target_url = f"https://www.google.com/search?q={encoded}"

                if target_url:
                    return {
                        "intent": "os_fast_path",
                        "action": "system_compound",
                        "steps": [
                            {"action": "launch_app", "target": app_id, "parameters": target_url},
                            {"action": "system_open_url", "url": target_url}
                        ],
                        "summary": f"Open {first_app.title()} ➜ {target_url}"
                    }

        # ----------------------------------------------------------------------
        # 1.5 Web URLs & Popular Web Destinations Fast-Path
        # ----------------------------------------------------------------------
        WEB_SERVICES = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://www.github.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://www.twitter.com",
            "x": "https://www.x.com",
            "spotify": "https://open.spotify.com",
            "netflix": "https://www.netflix.com",
            "chatgpt": "https://chatgpt.com",
            "gmail": "https://mail.google.com",
            "wikipedia": "https://www.wikipedia.org",
            "amazon": "https://www.amazon.com",
        }
        m_web = re.match(
            r'^(?:go\s+(?:over\s+)?to|open|navigate\s+to|visit|browse\s+to)\s+(https?://\S+|www\.\S+|\S+\.(?:com|org|net|io|edu|gov|ai|dev)|[a-zA-Z0-9_-]+)$',
            clean
        )
        if m_web:
            web_target = m_web.group(1).strip().lower()
            if web_target in WEB_SERVICES:
                url = WEB_SERVICES[web_target]
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_url",
                    "url": url,
                    "summary": f"Open {web_target.title()} ({url})"
                }
            elif web_target.startswith(("http://", "https://", "www.")) or re.search(r'\.(?:com|org|net|io|ai|dev|edu|gov)$', web_target):
                url = web_target if web_target.startswith("http") else f"https://{web_target}"
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_url",
                    "url": url,
                    "summary": f"Open {url}"
                }

        # ----------------------------------------------------------------------
        # 2. Window Switching / Focus (including "go to <app>")
        # ----------------------------------------------------------------------
        m_switch = re.match(
            r'^(?:switch\s+(?:over\s+)?to|focus(?:\s+on)?|bring\s+up|activate|go\s+to(?:\s+window)?)\s+(.+)$',
            clean
        )
        if m_switch:
            target_query = m_switch.group(1).strip()

            # Guard: If target specifies an in-window GUI element, proceed to visual grounding!
            if any(k in target_query.lower() for k in cls.GUI_CONTROL_KEYWORDS):
                return None

            # If target specifies a drive or folder, route to path open
            if re.match(r'^(?:drive\s+)?[a-zA-Z](?::|:\\)?$', target_query):
                dl = target_query.replace("drive", "").replace(":", "").replace("\\", "").strip().upper()
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_path",
                    "path": f"{dl}:\\",
                    "summary": f"Open Drive {dl}:\\"
                }
            win_match = cls.find_window_by_name(target_query)
            if win_match:
                target_hwnd, target_title = win_match
                return {
                    "intent": "os_fast_path",
                    "action": "window_switch",
                    "target_hwnd": target_hwnd,
                    "target_title": target_title,
                    "summary": f"Switch to window '{target_title[:30]}'"
                }
            elif target_query.lower() in cls.APP_ALIASES:
                # If window isn't currently running but is a recognized application, treat as launch_app
                return {
                    "intent": "os_fast_path",
                    "action": "launch_app",
                    "target": target_query,
                    "summary": f"Launch '{target_query}'"
                }
            else:
                return {
                    "intent": "os_fast_path",
                    "action": "window_switch",
                    "target": target_query,
                    "summary": f"Switch to window '{target_query}'"
                }

        # ----------------------------------------------------------------------
        # 3. Application Lifecycle: Open / Launch / Start / Run
        # ----------------------------------------------------------------------
        m_launch = re.match(
            r'^(?:open|launch|start|run)\s+(.+)$',
            clean
        )
        if m_launch:
            target_name = m_launch.group(1).strip()

            # Guard: check if this is an in-window GUI element rather than an app!
            words = set(re.findall(r'[a-zA-Z0-9]+', target_name.lower()))
            if any(k in target_name.lower() for k in cls.GUI_CONTROL_KEYWORDS):
                # E.g. "open dropdown", "open filter menu", "open settings dialog", "open search filter"
                # This is an IN-WINDOW GUI task, proceed to visual grounding pipeline!
                return None

            # Drive or folder navigation
            if re.match(r'^(?:drive\s+)?[a-zA-Z](?::|:\\)?$', target_name):
                dl = target_name.replace("drive", "").replace(":", "").replace("\\", "").strip().upper()
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_path",
                    "path": f"{dl}:\\",
                    "summary": f"Open Drive {dl}:\\"
                }

            # Standard Windows folders
            folder_map = {
                "documents": Path.home() / "Documents",
                "downloads": Path.home() / "Downloads",
                "desktop": Path.home() / "Desktop",
                "pictures": Path.home() / "Pictures",
                "music": Path.home() / "Music",
                "videos": Path.home() / "Videos",
            }
            if target_name.lower() in folder_map:
                p = str(folder_map[target_name.lower()])
                return {
                    "intent": "os_fast_path",
                    "action": "system_open_path",
                    "path": p,
                    "summary": f"Open folder '{p}'"
                }

            # Otherwise, treat as application lifecycle command!
            return {
                "intent": "os_fast_path",
                "action": "launch_app",
                "target": target_name,
                "summary": f"Launch or activate '{target_name}'"
            }

        # ----------------------------------------------------------------------
        # 4. In-Window GUI Tasks (Route to Visual Grounding Pipeline)
        # ----------------------------------------------------------------------
        # Commands such as "click search filter", "star slide", "select color",
        # "type hello", "submit form", etc. return None here so they proceed
        # directly to the visual grounding engine without destructive interference.
        return None

    @classmethod
    def execute(cls, decision: Dict[str, Any], current_win_ctx: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes the OS Fast-Path decision synchronously in < 25ms.
        """
        action = decision.get("action")
        user32 = ctypes.windll.user32

        if action == "launch_app":
            target = decision.get("target", "")
            parameters = decision.get("parameters")
            return cls.launch_or_activate_app(target, parameters=parameters)

        elif action == "window_switch":
            target_hwnd = decision.get("target_hwnd")
            if target_hwnd:
                cls._activate_hwnd(target_hwnd)
                return {"success": True, "action": "window_switch", "summary": decision.get("summary", "Window switched")}
            target = decision.get("target", "")
            return cls.launch_or_activate_app(target)

        elif action == "window_op":
            op = decision.get("op")
            active_hwnd = getattr(current_win_ctx, "hwnd", 0) if current_win_ctx else user32.GetForegroundWindow()
            if active_hwnd:
                if op == "minimize":
                    user32.ShowWindow(active_hwnd, 6)  # SW_MINIMIZE = 6
                elif op == "maximize":
                    user32.ShowWindow(active_hwnd, 3)  # SW_MAXIMIZE = 3
                elif op == "restore":
                    user32.ShowWindow(active_hwnd, 9)  # SW_RESTORE = 9
                elif op == "close":
                    WM_CLOSE = 0x0010
                    user32.PostMessageW(active_hwnd, WM_CLOSE, 0, 0)
                return {"success": True, "action": "window_op", "op": op, "summary": f"Window {op}"}

        elif action == "system_hotkey":
            combo = decision.get("combo")
            try:
                import pyautogui
                if combo:
                    pyautogui.hotkey(*combo)
                    return {"success": True, "action": "system_hotkey", "combo": combo, "summary": f"Dispatched {combo}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == "system_open_path":
            p = decision.get("path", "")
            try:
                os.startfile(p)
                return {"success": True, "action": "system_open_path", "path": p, "summary": f"Opened {p}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == "system_open_url":
            url = decision.get("url", "")
            try:
                import webbrowser
                webbrowser.open(url)
                return {"success": True, "action": "system_open_url", "url": url, "summary": f"Opened {url}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == "system_compound":
            steps = decision.get("steps", [])
            last_res = {"success": True, "action": "system_compound"}
            for step in steps:
                last_res = cls.execute(step, current_win_ctx)
                time.sleep(0.08)
            last_res["summary"] = decision.get("summary", "Compound action executed")
            return last_res

        return {"success": False, "error": f"Unknown OS action '{action}'"}


# Aliases for backward compatibility
OSFastPathRouter = IntentScopeRouter


# ==============================================================================
# PHASE 4: REAL-TIME VISUAL DIFF CLOSED-LOOP VERIFICATION (cv2.absdiff)
# ==============================================================================

def compute_visual_diff(pre_crop: np.ndarray, post_crop: np.ndarray, threshold: int = 20) -> int:
    """
    Computes real-time visual delta between pre-click and post-click crops using cv2.absdiff.
    Returns the count of pixels that changed by more than threshold.
    """
    if pre_crop is None or post_crop is None:
        return 0
    if pre_crop.shape != post_crop.shape:
        return 0

    diff = cv2.absdiff(pre_crop, post_crop)
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff

    changed_pixels = int(np.count_nonzero(diff_gray > threshold))
    return changed_pixels


def verify_visual_transition(
    pre_crop: np.ndarray,
    post_crop: np.ndarray,
    diff_threshold: int = 20,
    min_changed_pixels: int = 15
) -> Tuple[bool, int]:
    """
    Evaluates whether a physical action accomplished a visible UI state change.
    Returns (is_verified, changed_pixel_count).
    """
    changed = compute_visual_diff(pre_crop, post_crop, threshold=diff_threshold)
    is_verified = changed >= min_changed_pixels
    return is_verified, changed


def execute_click_with_visual_diff(
    target_x: int,
    target_y: int,
    patch_size: int = 120,
    action: str = "click",
    min_changed_pixels: int = 15
) -> Dict[str, Any]:
    """
    Phase 4 Closed-Loop Actuation:
    1. Capture a 120x120px pre-click patch around (target_x, target_y).
    2. Actuate hardware mouse click.
    3. Sleep 150-200ms and capture the post-click patch.
    4. Compute delta = cv2.absdiff(pre_crop, post_crop).
    5. If changed_pixels > threshold, confirm verified visual transition.
    6. If zero change, execute an automatic localized micro-nudge retry.
    """
    from spatial_droplet.vision.grabber import FastScreenGrabber
    from spatial_droplet.platform.actuator import Win32HardwareActuator

    grabber = FastScreenGrabber.get_instance()
    sw, sh = grabber.sw, grabber.sh

    half = patch_size // 2
    x1 = max(0, min(sw - patch_size, target_x - half))
    y1 = max(0, min(sh - patch_size, target_y - half))

    # 1. Pre-click crop
    pre_crop = grabber.grab_box(x1, y1, patch_size, patch_size)

    # 2. Actuate hardware click
    if action == "double_click":
        Win32HardwareActuator.double_click(target_x, target_y)
    elif action == "right_click":
        Win32HardwareActuator.right_click(target_x, target_y)
    else:
        Win32HardwareActuator.click(target_x, target_y)

    # 3. Dwell 160ms for UI transition / repaint
    time.sleep(0.16)

    # 4. Post-click crop
    post_crop = grabber.grab_box(x1, y1, patch_size, patch_size)

    # 5. Visual delta verification
    is_verified, changed_px = verify_visual_transition(
        pre_crop, post_crop, diff_threshold=20, min_changed_pixels=min_changed_pixels
    )

    if is_verified:
        return {
            "verified": True,
            "changed_pixels": changed_px,
            "retry_executed": False,
            "type": "VISUAL_DIFF_CONFIRMED"
        }

    # 6. Automatic Localized Micro-Nudge Retry (+5px offset)
    print(f"[Visual Diff Closed-Loop] Zero change detected (Δ={changed_px}px). Executing localized micro-nudge retry...")
    nudge_x = target_x + 5
    nudge_y = target_y + 4

    Win32HardwareActuator.move_cursor(nudge_x, nudge_y)
    time.sleep(0.04)

    if action == "double_click":
        Win32HardwareActuator.double_click(nudge_x, nudge_y)
    elif action == "right_click":
        Win32HardwareActuator.right_click(nudge_x, nudge_y)
    else:
        Win32HardwareActuator.click(nudge_x, nudge_y)

    time.sleep(0.16)
    post_retry_crop = grabber.grab_box(x1, y1, patch_size, patch_size)
    is_verified_retry, retry_px = verify_visual_transition(
        pre_crop, post_retry_crop, diff_threshold=20, min_changed_pixels=min_changed_pixels
    )

    return {
        "verified": is_verified_retry,
        "changed_pixels": retry_px,
        "retry_executed": True,
        "type": "VISUAL_DIFF_RETRY" if is_verified_retry else "VISUAL_DIFF_UNVERIFIED"
    }


# ==============================================================================
# MAIN COPILOT LAUNCHER
# ==============================================================================

import argparse
from spatial_droplet.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Droplet Live Desktop Copilot")
    parser.add_argument("--ngrok", action="store_true", help="Force Kaggle/Ngrok VLM")
    parser.add_argument("--groq", action="store_true", help="Force Groq 25-key pool")
    args, _ = parser.parse_known_args()

    if args.groq:
        os.environ["USE_NGROK"] = "0"
        provider_display = "⚡ Groq 25-Key Rotator (qwen/qwen3.8-27b) [Forced]"
    else:
        # Auto-detect if Kaggle ngrok endpoint is live
        detected_model = "qwen2.5vl:7b"
        is_ngrok_live = False
        ngrok_endpoint = os.environ.get("NGROK_URL", "https://greedy-porcupine-precut.ngrok-free.dev")
        try:
            import urllib.request
            import json
            tags_url = ngrok_endpoint.rstrip("/").replace("/v1", "") + "/api/tags"
            req = urllib.request.Request(tags_url, headers={"ngrok-skip-browser-warning": "true"})
            with urllib.request.urlopen(req, timeout=1.8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                if models:
                    is_ngrok_live = True
                    for pref in ["qwen2.5vl", "qwen3-vl", "ui-tars", "qwen"]:
                        for m in models:
                            if pref in m.lower():
                                detected_model = m
                                break
                        if detected_model:
                            break
        except Exception:
            is_ngrok_live = False

        if is_ngrok_live or args.ngrok:
            os.environ["USE_NGROK"] = "1"
            os.environ["KAGGLE_QWEN_MODEL"] = detected_model
            os.environ["ENABLE_KAGGLE_QWEN"] = "1"
            provider_display = f"⚡ Kaggle Dual-T4 Qwen ({detected_model} via ngrok) [Fallback: Groq 25-Key]"
        else:
            os.environ["USE_NGROK"] = "0"
            provider_display = "⚡ Groq 25-Key Rotator (qwen/qwen3.8-27b) [Kaggle Offline]"

    print("\n" + "=" * 74)
    print(" 🚀 LAUNCHING SPATIAL DROPLET LIVE COPILOT HUD & CHAT CONTROLLER")
    print("=" * 74)
    print(f" • Primary Reasoner:  {provider_display}")
    print(" • Voice Control:     Click '🎙️ Mic' or press Alt+V, Ctrl+Shift+V, or F8 to speak")
    print(" • Speech Engine:     ⚡ Groq Whisper Large V3 Turbo (<200ms latency)")
    print(" • Floating Chat Bar: Draggable anywhere on screen (Top of display)")
    print(" • Cognitive Engine:  100% Autonomous Reasoning via Groq Qwen 27B / ngrok")
    print(" • Transparent HUD:   Full-bleed multi-color context droplet pill badges")
    print(" • Direct Dot Clicks: Type Dot ID (e.g. 5, click 12) for instant 0ms clicks")
    print(" • Real-Time Diff:    120x120px cv2.absdiff closed loop with micro-nudge retry")
    print(" • Vimium Mode:       Press Alt+Space or click '⚡ Quick' in the bar")
    print(" • Mesh Dots Toggle:  Click '✦ Dots' to toggle full 3,726 background mesh")
    print(" • Emergency Abort:   Press ESC key at any time to abort automation")
    print("=" * 74 + "\n")
    main()
