"""
Cognitive Screen State & Layout Parser.
Extracts high-level semantic awareness of the active window, layout zones,
child/modal dialogs, and active screen state.
"""
import time
import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional, Any

from ..core.models import WindowContext, ModalDialogContext, ScreenAwarenessContext, is_own_window

class ScreenStateParser:
    """
    Cognitive Screen State & Layout Parser (Phase 6 & 8).
    Extracts high-level semantic awareness of the active window, layout zones,
    child/modal dialogs, and active screen state to break the perception ceiling.
    """
    @staticmethod
    def detect_modal_dialog(active_win: Optional[WindowContext]) -> Optional[ModalDialogContext]:
        if not active_win or not active_win.is_valid or active_win.title == "Desktop":
            return None

        user32 = ctypes.windll.user32
        modals: List[ModalDialogContext] = []

        # 1. Check if active_win ITSELF is a modal dialog
        cname_active = getattr(active_win, 'class_name', '')
        title_active = getattr(active_win, 'title', '')
        t_low = title_active.lower()

        is_active_dialog = (
            cname_active in ("#32770", "OperationStatusWindow", "PromptDialog") or
            any(k in t_low for k in ("save as", "save changes", "confirm", "confirm exit", "do you want to save", "dialog", "prompt"))
        )
        if is_active_dialog and active_win.rect:
            wx1, wy1, wx2, wy2 = active_win.rect
            w = wx2 - wx1
            h = wy2 - wy1
            if w >= 80 and h >= 50:
                GW_OWNER = 4
                owner_hwnd = user32.GetWindow(active_win.hwnd, GW_OWNER)
                return ModalDialogContext(
                    hwnd=active_win.hwnd,
                    title=title_active,
                    class_name=cname_active,
                    rect=active_win.rect,
                    width=w,
                    height=h,
                    is_modal=True,
                    owner_hwnd=owner_hwnd or active_win.hwnd,
                    dialog_type=cname_active
                )

        # 2. Check for active modal popup owned by active_win via Win32 API
        try:
            if not (user32 and user32.IsWindow(active_win.hwnd)):
                return None
            popup = user32.GetLastActivePopup(active_win.hwnd)
            if popup and popup != active_win.hwnd and user32.IsWindow(popup) and user32.IsWindowVisible(popup) and not is_own_window(popup):
                rect = wintypes.RECT()
                user32.GetWindowRect(popup, ctypes.byref(rect))
                pw = rect.right - rect.left
                ph = rect.bottom - rect.top
                if 80 <= pw <= 1920 and 50 <= ph <= 1080:
                    cbuf = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(popup, cbuf, 256)
                    cname = cbuf.value.strip()
                    if not any(nm in cname.lower() for nm in ("tooltip", "sysshadow", "droptarget", "ghost")):
                        l = user32.GetWindowTextLengthW(popup)
                        tbuf = ctypes.create_unicode_buffer(l + 1)
                        user32.GetWindowTextW(popup, tbuf, l + 1)
                        ptitle = tbuf.value.strip() or f"Dialog ({cname})"
                        return ModalDialogContext(
                            hwnd=popup,
                            title=ptitle,
                            class_name=cname,
                            rect=(rect.left, rect.top, rect.right, rect.bottom),
                            width=pw,
                            height=ph,
                            is_modal=True,
                            owner_hwnd=active_win.hwnd,
                            dialog_type=cname
                        )
        except Exception:
            pass

        NON_MODAL_CLASSES = {
            "popuphost", "auto-suggest dropdown", "dropdown", "combolbox",
            "#32768", "tooltips_class32", "sysshadow", "cabinetwclass",
            "windows.ui.core.corecomponentinputsource", "toplevelwindowforoverflowxamlisland",
            "xaml_windowedpopupclass", "shell_traywnd", "notifyiconoverflowwindow",
            "msgina: tooltips", "applicationframewindow"
        }

        def enum_thread_win(hwnd, lparam):
            if is_own_window(hwnd) or hwnd == active_win.hwnd:
                return True
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w < 100 or h < 60 or w > 1920 or h > 1080:
                return True

            cbuf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cbuf, 256)
            cname = cbuf.value.strip()
            cname_low = cname.lower()

            # Ignore transient popups, dropdowns, tooltips, menus, and explorer frames
            if any(nm in cname_low for nm in NON_MODAL_CLASSES):
                return True

            l = user32.GetWindowTextLengthW(hwnd)
            tbuf = ctypes.create_unicode_buffer(l + 1)
            user32.GetWindowTextW(hwnd, tbuf, l + 1)
            title = tbuf.value.strip()

            style = user32.GetWindowLongW(hwnd, -16)      # GWL_STYLE
            ex_style = user32.GetWindowLongW(hwnd, -20)   # GWL_EXSTYLE
            WS_CAPTION = 0x00C00000
            WS_EX_DLGMODALFRAME = 0x00000001

            is_known_dlg_class = (cname in ("#32770", "OperationStatusWindow", "PromptDialog"))
            has_dlg_frame = bool(ex_style & WS_EX_DLGMODALFRAME)
            has_caption = bool((style & WS_CAPTION) == WS_CAPTION)

            # Check if active window is disabled by this modal (standard Win32 modal behavior)
            owner_disabled = (user32.IsWindowEnabled(active_win.hwnd) == 0)

            # Check if this window is owned by active_win
            GW_OWNER = 4
            owner_hwnd = user32.GetWindow(hwnd, GW_OWNER)
            is_owned = (owner_hwnd == active_win.hwnd)

            wx1, wy1, wx2, wy2 = active_win.rect
            cx = (rect.left + rect.right) // 2
            cy = (rect.top + rect.bottom) // 2
            is_inside_active = (wx1 <= cx <= wx2 and wy1 <= cy <= wy2)

            is_genuine_dialog = (
                (is_known_dlg_class and (title or is_inside_active)) or
                (has_dlg_frame and title) or
                (owner_disabled and (title or is_inside_active)) or
                (is_owned and has_caption and title and is_inside_active)
            )

            if is_genuine_dialog:
                modals.append(ModalDialogContext(
                    hwnd=hwnd,
                    title=title or f"Dialog ({cname})",
                    class_name=cname,
                    rect=(rect.left, rect.top, rect.right, rect.bottom),
                    width=w,
                    height=h,
                    is_modal=True,
                    owner_hwnd=active_win.hwnd,
                    dialog_type=cname
                ))
            return True

        try:
            tid = user32.GetWindowThreadProcessId(active_win.hwnd, None)
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            user32.EnumThreadWindows(tid, WNDENUMPROC(enum_thread_win), 0)
        except Exception:
            pass

        return modals[0] if modals else None

    @classmethod
    def classify_scene_type(cls, win_ctx: Optional[WindowContext], text_lines: List[str], active_modal: Optional[ModalDialogContext]) -> str:
        if active_modal:
            return "MODAL_DIALOG"

        if not win_ctx or not win_ctx.is_valid or win_ctx.title == "Desktop":
            return "DESKTOP"

        title_lower = win_ctx.title.lower()
        full_text_lower = " ".join(text_lines).lower()

        # Game detection
        if any(k in title_lower for k in ("roblox", "minecraft", "unity", "unreal", "steam", "game")):
            if any(w in full_text_lower for w in ("play", "quests", "inventory", "gear up", "battlepass", "lobby")):
                return "GAME_LOBBY"
            return "GAME_CANVAS"

        # Code editor / IDE detection
        if any(k in title_lower for k in ("visual studio code", "vscode", "pycharm", "sublime", "notepad", ".py", ".ts", ".cpp", "editor")):
            return "CODE_EDITOR"

        # Browser
        if any(k in title_lower for k in ("chrome", "edge", "firefox", "brave", "browser")):
            if any(w in full_text_lower for w in ("login", "sign in", "password", "username", "submit", "register")):
                return "FORM_INPUT"
            return "WEB_APP"

        # Settings
        if "settings" in title_lower or "properties" in title_lower or "control panel" in title_lower:
            return "SETTINGS_VIEW"

        return "DESKTOP_APP"

    @classmethod
    def parse_screen_awareness(
        cls,
        win_ctx: Optional[WindowContext],
        all_ocr_lines: Optional[List[str]] = None,
        droplets: Optional[List[Any]] = None
    ) -> ScreenAwarenessContext:
        all_ocr_lines = all_ocr_lines or []
        droplets = droplets or []
        active_modal = cls.detect_modal_dialog(win_ctx)
        if not active_modal and all_ocr_lines:
            modal_triggers = (
                "do you want to save", "save changes", "unsaved changes",
                "are you sure", "confirm exit", "discard changes",
                "want to save the changes"
            )
            has_prompt = any(any(trig in l.lower() for trig in modal_triggers) for l in all_ocr_lines)
            if has_prompt and win_ctx and win_ctx.rect:
                active_modal = ModalDialogContext(
                    hwnd=win_ctx.hwnd,
                    title=f"Confirmation Sheet ({win_ctx.title[:24]})",
                    class_name="InWindowModalSheet",
                    rect=win_ctx.rect,
                    width=win_ctx.width,
                    height=win_ctx.height,
                    is_modal=True,
                    owner_hwnd=win_ctx.hwnd,
                    dialog_type="InWindowSheet"
                )
        scene_type = cls.classify_scene_type(win_ctx, all_ocr_lines, active_modal)

        nav_items: List[str] = []
        action_items: List[str] = []
        status_items: List[str] = []
        content_lines: List[str] = []

        for d in droplets:
            if d.category == 'surface_probe':
                continue
            lbl = d.label.strip()
            if not lbl:
                continue

            if d.center_y < 120 or (win_ctx and win_ctx.rect and d.center_y < win_ctx.rect[1] + 120):
                if lbl not in nav_items and len(nav_items) < 8:
                    nav_items.append(lbl)
            elif any(act in lbl.lower() for act in ("play", "submit", "run", "save", "ok", "cancel", "next", "search", "btn", "open")):
                if lbl not in action_items and len(action_items) < 8:
                    action_items.append(f"{lbl} [Dot {d.id}]")
            else:
                if lbl not in action_items and len(action_items) < 6:
                    action_items.append(f"{lbl} [Dot {d.id}]")

        for line in all_ocr_lines[:20]:
            clean_l = line.strip()
            if not clean_l:
                continue
            if any(s in clean_l.lower() for s in ("xp:", "level", "joined", "status:", "connected", "ready", "error", "warning")):
                if clean_l not in status_items and len(status_items) < 4:
                    status_items.append(clean_l)
            elif len(content_lines) < 6:
                content_lines.append(clean_l)

        zones = {
            "navigation": nav_items[:6],
            "action_controls": action_items[:8],
            "status_info": status_items[:4],
            "key_content": content_lines[:4]
        }

        app_name = win_ctx.title if (win_ctx and win_ctx.is_valid) else "Desktop"
        modal_str = f" [Active Modal Dialog: '{active_modal.title[:24]}']" if active_modal else ""
        compact_nav = [n[:20] for n in nav_items[:4]]
        compact_actions = [a[:28] for a in action_items[:4]]
        nav_str = f" Nav: [{', '.join(compact_nav)}]" if compact_nav else ""
        actions_str = f" Actions: [{', '.join(compact_actions)}]" if compact_actions else ""
        summary = f"{scene_type} in '{app_name[:32]}'.{modal_str}{nav_str}{actions_str}"
        if len(summary) > 280:
            summary = summary[:277] + "..."

        win_rect = win_ctx.rect if (win_ctx and win_ctx.is_valid and win_ctx.rect) else None

        return ScreenAwarenessContext(
            app_name=app_name,
            window_title=app_name,
            scene_type=scene_type,
            active_modal=active_modal,
            modal_context=active_modal,
            layout_zones=zones,
            total_text_tokens=len(" ".join(all_ocr_lines).split()),
            summary=summary,
            active_window_rect=win_rect,
            window_rect=win_rect
        )

    # Class method alias for compatibility across survey and milestone specifications
    extract_screen_awareness = parse_screen_awareness
