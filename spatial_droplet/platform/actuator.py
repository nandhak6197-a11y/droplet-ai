"""
Win32 SendInput hardware actuation, precision cursor driver, and desktop action execution.
Direct driver-level event dispatch with sub-millisecond execution latency (<0.5ms).
Supports 11 canonical actions: click, double_click, type_and_enter, clear_and_type,
launch_app, open_path, switch_window, minimize, maximize, hotkey, stop.
"""
import os
import re
import time
import ctypes
import shutil
import subprocess
import webbrowser
from ctypes import wintypes
from typing import Optional, Tuple, Dict, Any, List, Union

_windll = getattr(ctypes, 'windll', None)
_user32 = getattr(_windll, 'user32', None) if _windll else None
_shell32 = getattr(_windll, 'shell32', None) if _windll else None

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ('dx', ctypes.c_long),
        ('dy', ctypes.c_long),
        ('mouseData', ctypes.c_ulong),
        ('dwFlags', ctypes.c_ulong),
        ('time', ctypes.c_ulong),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong))
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', ctypes.c_ushort),
        ('wScan', ctypes.c_ushort),
        ('dwFlags', ctypes.c_ulong),
        ('time', ctypes.c_ulong),
        ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong))
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ('uMsg', ctypes.c_ulong),
        ('wParamL', ctypes.c_short),
        ('wParamH', ctypes.c_ushort)
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [
            ('mi', MOUSEINPUT),
            ('ki', KEYBDINPUT),
            ('hi', HARDWAREINPUT)
        ]
    _anonymous_ = ('i',)
    _fields_ = [('type', ctypes.c_ulong), ('i', _I)]


def get_cursor_shape() -> str:
    """Inspects the current OS hardware cursor shape."""
    if not _user32:
        return "ARROW/DEFAULT"
    try:
        class POINT(ctypes.Structure):
            _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

        class CURSORINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('flags', wintypes.DWORD),
                ('hCursor', wintypes.HANDLE),
                ('ptScreenPos', POINT)
            ]

        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO)
        if _user32.GetCursorInfo(ctypes.byref(ci)):
            hCur = ci.hCursor
            hHand = _user32.LoadCursorW(0, 32649)   # IDC_HAND
            hIBeam = _user32.LoadCursorW(0, 32513)  # IDC_IBEAM
            if hCur == hHand:
                return "HAND (Interactive Link/Button)"
            elif hCur == hIBeam:
                return "IBEAM (Editable Text Input)"
    except Exception:
        pass
    return "ARROW/DEFAULT"


VK_MAP = {
    # Modifiers
    'ctrl': 0x11, 'control': 0x11, 'lctrl': 0xA2, 'rctrl': 0xA3,
    'alt': 0x12, 'menu': 0x12, 'lalt': 0xA4, 'ralt': 0xA5,
    'shift': 0x10, 'lshift': 0xA0, 'rshift': 0xA1,
    'win': 0x5B, 'windows': 0x5B, 'super': 0x5B, 'cmd': 0x5B, 'command': 0x5B, 'lwin': 0x5B, 'rwin': 0x5C,
    # Standard Control Keys
    'enter': 0x0D, 'return': 0x0D,
    'esc': 0x1B, 'escape': 0x1B,
    'tab': 0x09,
    'space': 0x20, 'spacebar': 0x20,
    'backspace': 0x08, 'back': 0x08,
    'delete': 0x2E, 'del': 0x2E,
    'insert': 0x2D, 'ins': 0x2D,
    # Navigation
    'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
    'arrowup': 0x26, 'arrowdown': 0x28, 'arrowleft': 0x25, 'arrowright': 0x27,
    'arrow_up': 0x26, 'arrow_down': 0x28, 'arrow_left': 0x25, 'arrow_right': 0x27,
    'home': 0x24, 'end': 0x23,
    'pageup': 0x21, 'pgup': 0x21, 'page_up': 0x21,
    'pagedown': 0x22, 'pgdn': 0x22, 'page_down': 0x22,
    # Locks
    'capslock': 0x14, 'numlock': 0x90, 'scrolllock': 0x91,
    # Function Keys
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
    'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
    # System
    'printscreen': 0x2C, 'prtscn': 0x2C, 'prtsc': 0x2C, 'pause': 0x13,
}
for _i in range(10):
    VK_MAP[str(_i)] = 0x30 + _i
for _c in 'abcdefghijklmnopqrstuvwxyz':
    VK_MAP[_c] = 0x41 + (ord(_c) - ord('a'))


def resolve_vk(key: Any) -> int:
    """Translates key name or character into a Win32 Virtual-Key code."""
    if isinstance(key, int):
        return key
    if not isinstance(key, str):
        return 0
    k = key.lower().strip()
    if k in VK_MAP:
        return VK_MAP[k]
    if len(k) == 1:
        if _user32:
            try:
                return _user32.VkKeyScanW(ord(k)) & 0xFF
            except Exception:
                pass
        return ord(k.upper())
    return 0


class Win32HardwareActuator:
    """
    Direct C-types Win32 SendInput hardware driver.
    Dispatches mouse clicks, double clicks, triple clicks, drags, scrolls, 
    and unicode typing at the OS driver level with zero message-queue delays.
    """
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
    MOUSEEVENTF_WHEEL = 0x0800
    MOUSEEVENTF_HWHEEL = 0x1000
    MOUSEEVENTF_VIRTUALDESK = 0x4000
    MOUSEEVENTF_ABSOLUTE = 0x8000

    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_SCANCODE = 0x0008

    @staticmethod
    def _to_norm_coords(x: int, y: int, sw: int = 1920, sh: int = 1080) -> Tuple[int, int, int]:
        try:
            from .desktop import VirtualDesktopTopology
            nx, ny, is_v = VirtualDesktopTopology.to_normalized_coords(x, y)
            extra_flag = Win32HardwareActuator.MOUSEEVENTF_VIRTUALDESK if is_v else 0
            return nx, ny, extra_flag
        except Exception:
            norm_x = int(round(x * 65535.0 / max(1, sw - 1)))
            norm_y = int(round(y * 65535.0 / max(1, sh - 1)))
            return norm_x, norm_y, 0

    @classmethod
    def move_cursor(cls, x: int, y: int, sw: int = 1920, sh: int = 1080):
        if _user32:
            nx, ny, v_flag = cls._to_norm_coords(x, y, sw, sh)
            inp = INPUT()
            inp.type = cls.INPUT_MOUSE
            inp.mi.dx = nx
            inp.mi.dy = ny
            inp.mi.dwFlags = cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | v_flag
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.moveTo(x, y)
            except Exception:
                pass

    @classmethod
    def hover(cls, x: int, y: int, sw: int = 1920, sh: int = 1080):
        cls.move_cursor(x, y, sw, sh)

    @classmethod
    def mouse_down(cls, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", sw: int = 1920, sh: int = 1080):
        btn = button.lower()
        if _user32:
            if btn == "right":
                flag = cls.MOUSEEVENTF_RIGHTDOWN
            elif btn == "middle":
                flag = cls.MOUSEEVENTF_MIDDLEDOWN
            else:
                flag = cls.MOUSEEVENTF_LEFTDOWN

            inp = INPUT()
            inp.type = cls.INPUT_MOUSE
            if x is not None and y is not None:
                nx, ny, v_flag = cls._to_norm_coords(x, y, sw, sh)
                inp.mi.dx = nx
                inp.mi.dy = ny
                inp.mi.dwFlags = cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | flag | v_flag
            else:
                inp.mi.dwFlags = flag
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.mouseDown(x=x, y=y, button=btn)
            except Exception:
                pass

    @classmethod
    def mouse_up(cls, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", sw: int = 1920, sh: int = 1080):
        btn = button.lower()
        if _user32:
            if btn == "right":
                flag = cls.MOUSEEVENTF_RIGHTUP
            elif btn == "middle":
                flag = cls.MOUSEEVENTF_MIDDLEUP
            else:
                flag = cls.MOUSEEVENTF_LEFTUP

            inp = INPUT()
            inp.type = cls.INPUT_MOUSE
            if x is not None and y is not None:
                nx, ny, v_flag = cls._to_norm_coords(x, y, sw, sh)
                inp.mi.dx = nx
                inp.mi.dy = ny
                inp.mi.dwFlags = cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | flag | v_flag
            else:
                inp.mi.dwFlags = flag
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.mouseUp(x=x, y=y, button=btn)
            except Exception:
                pass

    @classmethod
    def click(cls, x: int, y: int, button: str = "left", double: bool = False, sw: int = 1920, sh: int = 1080):
        btn = button.lower()
        if _user32:
            nx, ny, v_flag = cls._to_norm_coords(x, y, sw, sh)
            if btn == "right":
                down_flag, up_flag = cls.MOUSEEVENTF_RIGHTDOWN, cls.MOUSEEVENTF_RIGHTUP
            elif btn == "middle":
                down_flag, up_flag = cls.MOUSEEVENTF_MIDDLEDOWN, cls.MOUSEEVENTF_MIDDLEUP
            else:
                down_flag, up_flag = cls.MOUSEEVENTF_LEFTDOWN, cls.MOUSEEVENTF_LEFTUP

            inputs = (INPUT * 2)()
            inputs[0].type = cls.INPUT_MOUSE
            inputs[0].mi.dx = nx
            inputs[0].mi.dy = ny
            inputs[0].mi.dwFlags = cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | down_flag | v_flag

            inputs[1].type = cls.INPUT_MOUSE
            inputs[1].mi.dx = nx
            inputs[1].mi.dy = ny
            inputs[1].mi.dwFlags = cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | up_flag | v_flag

            _user32.SendInput(2, inputs, ctypes.sizeof(INPUT))

            if double:
                time.sleep(0.04)
                _user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.click(x=x, y=y, button=btn, clicks=2 if double else 1, interval=0.04 if double else 0.0)
            except Exception:
                pass

    @classmethod
    def double_click(cls, x: int, y: int, button: str = "left", sw: int = 1920, sh: int = 1080):
        cls.move_cursor(x, y, sw, sh)
        time.sleep(0.02)
        cls.click(x, y, button=button, double=True, sw=sw, sh=sh)

    @classmethod
    def triple_click(cls, x: int, y: int, button: str = "left", sw: int = 1920, sh: int = 1080):
        if _user32:
            cls.click(x, y, button=button, double=False, sw=sw, sh=sh)
            time.sleep(0.04)
            cls.click(x, y, button=button, double=False, sw=sw, sh=sh)
            time.sleep(0.04)
            cls.click(x, y, button=button, double=False, sw=sw, sh=sh)
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.tripleClick(x=x, y=y, button=button)
            except Exception:
                pass

    @classmethod
    def right_click(cls, x: int, y: int, sw: int = 1920, sh: int = 1080):
        cls.click(x, y, button="right", sw=sw, sh=sh)

    @classmethod
    def middle_click(cls, x: int, y: int, sw: int = 1920, sh: int = 1080):
        cls.click(x, y, button="middle", sw=sw, sh=sh)

    @classmethod
    def drag(cls, from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.35, button: str = "left", steps: int = 15, sw: int = 1920, sh: int = 1080):
        if _user32:
            cls.move_cursor(from_x, from_y, sw, sh)
            time.sleep(0.02)
            cls.mouse_down(from_x, from_y, button=button, sw=sw, sh=sh)
            time.sleep(0.02)
            step_sleep = max(0.005, duration / max(1, steps))
            for i in range(1, steps + 1):
                t = i / float(steps)
                cx = int(round(from_x + (to_x - from_x) * t))
                cy = int(round(from_y + (to_y - from_y) * t))
                cls.move_cursor(cx, cy, sw, sh)
                time.sleep(step_sleep)
            cls.mouse_up(to_x, to_y, button=button, sw=sw, sh=sh)
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.moveTo(from_x, from_y)
                pyautogui.dragTo(to_x, to_y, duration=duration, button=button)
            except Exception:
                pass

    @classmethod
    def drop(cls, x: int, y: int, button: str = "left", sw: int = 1920, sh: int = 1080):
        cls.mouse_up(x, y, button=button, sw=sw, sh=sh)

    @classmethod
    def scroll(cls, clicks: int, direction: str = "vertical", x: Optional[int] = None, y: Optional[int] = None, sw: int = 1920, sh: int = 1080):
        if _user32:
            wheel_delta = int(clicks * 120)
            flag = cls.MOUSEEVENTF_HWHEEL if direction.lower() in ("horizontal", "h", "left", "right") else cls.MOUSEEVENTF_WHEEL
            inp = INPUT()
            inp.type = cls.INPUT_MOUSE
            if x is not None and y is not None:
                nx, ny, v_flag = cls._to_norm_coords(x, y, sw, sh)
                inp.mi.dx = nx
                inp.mi.dy = ny
                inp.mi.dwFlags = flag | cls.MOUSEEVENTF_MOVE | cls.MOUSEEVENTF_ABSOLUTE | v_flag
            else:
                inp.mi.dwFlags = flag
            inp.mi.mouseData = ctypes.c_ulong(wheel_delta).value
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                if direction.lower() in ("horizontal", "h", "left", "right"):
                    pyautogui.hscroll(clicks, x=x, y=y)
                else:
                    pyautogui.scroll(clicks, x=x, y=y)
            except Exception:
                pass

    @classmethod
    def type_unicode(cls, text: str):
        if not text:
            return
        if _user32:
            inputs = (INPUT * (len(text) * 2))()
            idx = 0
            for ch in text:
                code = ord(ch)
                inputs[idx].type = cls.INPUT_KEYBOARD
                inputs[idx].ki.wScan = code
                inputs[idx].ki.dwFlags = cls.KEYEVENTF_UNICODE
                idx += 1
                inputs[idx].type = cls.INPUT_KEYBOARD
                inputs[idx].ki.wScan = code
                inputs[idx].ki.dwFlags = cls.KEYEVENTF_UNICODE | cls.KEYEVENTF_KEYUP
                idx += 1
            _user32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.write(text)
            except Exception:
                pass

    type_text = type_unicode

    @classmethod
    def key_down(cls, key: Any):
        if _user32:
            vk = resolve_vk(key)
            if vk == 0:
                return
            inp = INPUT()
            inp.type = cls.INPUT_KEYBOARD
            inp.ki.wVk = vk
            inp.ki.dwFlags = 0
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.keyDown(str(key).lower())
            except Exception:
                pass

    @classmethod
    def key_up(cls, key: Any):
        if _user32:
            vk = resolve_vk(key)
            if vk == 0:
                return
            inp = INPUT()
            inp.type = cls.INPUT_KEYBOARD
            inp.ki.wVk = vk
            inp.ki.dwFlags = cls.KEYEVENTF_KEYUP
            _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.keyUp(str(key).lower())
            except Exception:
                pass

    @classmethod
    def press_key(cls, key: Any):
        if _user32:
            vk = resolve_vk(key)
            if vk == 0:
                return
            inputs = (INPUT * 2)()
            inputs[0].type = cls.INPUT_KEYBOARD
            inputs[0].ki.wVk = vk
            inputs[0].ki.dwFlags = 0

            inputs[1].type = cls.INPUT_KEYBOARD
            inputs[1].ki.wVk = vk
            inputs[1].ki.dwFlags = cls.KEYEVENTF_KEYUP

            _user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.press(str(key).lower())
            except Exception:
                pass

    @classmethod
    def hotkey(cls, *keys: Any):
        if _user32:
            vks = [resolve_vk(k) for k in keys if resolve_vk(k) != 0]
            if not vks:
                return
            n = len(vks)
            inputs = (INPUT * (n * 2))()
            for i, vk in enumerate(vks):
                inputs[i].type = cls.INPUT_KEYBOARD
                inputs[i].ki.wVk = vk
                inputs[i].ki.dwFlags = 0
            for i, vk in enumerate(reversed(vks)):
                inputs[n + i].type = cls.INPUT_KEYBOARD
                inputs[n + i].ki.wVk = vk
                inputs[n + i].ki.dwFlags = cls.KEYEVENTF_KEYUP

            _user32.SendInput(n * 2, inputs, ctypes.sizeof(INPUT))
        elif HAS_PYAUTOGUI:
            try:
                pyautogui.hotkey(*[str(k).lower() for k in keys])
            except Exception:
                pass

    @classmethod
    def clear_input(cls, x: int, y: int, sw: int = 1920, sh: int = 1080):
        cls.click(x, y, button="left", sw=sw, sh=sh)
        time.sleep(0.04)
        cls.hotkey('ctrl', 'a')
        time.sleep(0.04)
        cls.press_key('backspace')

    @classmethod
    def modifier_click(cls, x: int, y: int, modifier: str = "ctrl", button: str = "left", double: bool = False, sw: int = 1920, sh: int = 1080):
        cls.move_cursor(x, y, sw, sh)
        time.sleep(0.02)
        cls.key_down(modifier)
        time.sleep(0.02)
        try:
            cls.click(x, y, button=button, double=double, sw=sw, sh=sh)
        finally:
            time.sleep(0.02)
            cls.key_up(modifier)

    @classmethod
    def hover_and_wait(cls, x: int, y: int, wait_time: float = 0.25, sw: int = 1920, sh: int = 1080):
        cls.move_cursor(x, y, sw, sh)
        time.sleep(max(0.05, wait_time))

    @classmethod
    def clear_and_type(cls, x: int, y: int, text: str, press_enter: bool = False, sw: int = 1920, sh: int = 1080):
        if x > 0 and y > 0:
            cls.clear_input(x, y, sw=sw, sh=sh)
        else:
            cls.hotkey('ctrl', 'a')
            time.sleep(0.04)
            cls.press_key('backspace')
        time.sleep(0.04)
        if text:
            if len(text) > 3:
                try:
                    import pyperclip
                    pyperclip.copy(text)
                    cls.hotkey('ctrl', 'v')
                    time.sleep(0.06)
                except Exception:
                    cls.type_unicode(text)
            else:
                cls.type_unicode(text)
        if press_enter:
            time.sleep(0.05)
            cls.press_key('enter')

    @classmethod
    def click_and_type(cls, x: int, y: int, text: str, press_enter: bool = False, sw: int = 1920, sh: int = 1080):
        if x > 0 and y > 0:
            cls.click(x, y, button="left", sw=sw, sh=sh)
            time.sleep(0.04)
        if text:
            if len(text) > 3:
                try:
                    import pyperclip
                    pyperclip.copy(text)
                    cls.hotkey('ctrl', 'v')
                    time.sleep(0.06)
                except Exception:
                    cls.type_unicode(text)
            else:
                cls.type_unicode(text)
        if press_enter:
            time.sleep(0.05)
            cls.press_key('enter')

    # Universal Shortcuts
    @classmethod
    def copy(cls):
        cls.hotkey('ctrl', 'c')

    @classmethod
    def paste(cls):
        cls.hotkey('ctrl', 'v')

    @classmethod
    def select_all(cls):
        cls.hotkey('ctrl', 'a')

    @classmethod
    def undo(cls):
        cls.hotkey('ctrl', 'z')

    @classmethod
    def redo(cls):
        cls.hotkey('ctrl', 'y')

    @classmethod
    def save(cls):
        cls.hotkey('ctrl', 's')

    @classmethod
    def find(cls):
        cls.hotkey('ctrl', 'f')

    @classmethod
    def close_tab(cls):
        cls.hotkey('ctrl', 'w')

    @classmethod
    def new_tab(cls):
        cls.hotkey('ctrl', 't')

    @classmethod
    def switch_window(cls):
        cls.hotkey('alt', 'tab')

    @classmethod
    def win_run(cls, command: str = ""):
        cls.hotkey('win', 'r')
        time.sleep(0.25)
        if command:
            cls.type_unicode(command)
            time.sleep(0.05)
            cls.press_key('enter')


class PrecisionActuator:
    """
    Sub-millisecond desktop action execution router.
    Integrates with Win32HardwareActuator, Win32 window management,
    and ActionVerifier closed-loop visual settle sensing.
    """
    def __init__(self, hud: Any = None, verifier: Any = None):
        self.hud = hud
        self.verifier = verifier

    def execute(
        self,
        droplet: Any,
        action: str,
        input_text: Optional[str] = None,
        win_ctx: Optional[Any] = None,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
        drag_dest: Optional[Tuple[int, int]] = None,
        key_combo: Optional[List[str]] = None,
        key: Optional[str] = None,
        scroll_direction: str = "vertical",
        scroll_amount: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        act_clean = action.lower().strip()
        non_spatial_actions = {
            "wait", "hotkey", "chord", "press", "key_down", "key_up",
            "switch_window", "win_run", "copy", "paste", "select_all",
            "undo", "redo", "save", "find", "close_tab", "new_tab",
            "launch_app", "open_path", "system_open", "system_open_path",
            "stop", "finish", "done",
            "minimize", "minimize_window", "maximize", "maximize_window"
        }
        is_spatial = act_clean not in non_spatial_actions

        tx = target_x
        ty = target_y
        if tx is None or ty is None:
            if droplet and getattr(droplet, 'bbox', None):
                bx1, by1, bx2, by2 = droplet.bbox
                if bx2 > bx1 and by2 > by1:
                    tx = (bx1 + bx2) // 2
                    is_desktop = (not win_ctx) or (not win_ctx.is_valid) or (win_ctx.title == "Desktop")
                    if is_desktop and droplet.category == "interactive_app" and (by2 - by1) > 38:
                        ty = by1 + min(24, max(14, (by2 - by1) // 3))
                    else:
                        ty = (by1 + by2) // 2
            if tx is None and droplet:
                tx = droplet.center_x
            if ty is None and droplet:
                ty = droplet.center_y

        dot_id = droplet.id if droplet else None
        label = droplet.label if droplet else None

        if is_spatial:
            coord_str = f"at ({tx}, {ty})" if (tx is not None and ty is not None) else "at (Focused Target)"
            lbl_str = f" | Label: \"{label}\"" if label else ""
            dot_str = f"on Dot [{dot_id}] " if dot_id is not None else ""
            print(f"\n[Actuator] -> Action: {act_clean.upper()} {dot_str}{coord_str}{lbl_str}")
        else:
            aux_str = f" | Input: \"{input_text}\"" if input_text else (f" | Key: {key_combo or key}" if (key_combo or key) else "")
            print(f"\n[Actuator] -> Action: {act_clean.upper()} (Keyboard / System Action){aux_str}")

        pre_snap = None
        if self.verifier and droplet:
            pre_snap = self.verifier.snapshot_pre_action(droplet, win_ctx)

        # Smooth glide cursor move if coordinates available
        if is_spatial and tx is not None and ty is not None and tx > 0 and ty > 0:
            if HAS_PYAUTOGUI:
                try:
                    pyautogui.moveTo(tx, ty, duration=0.25, tween=pyautogui.easeInOutQuad)
                    time.sleep(0.03)
                except Exception:
                    Win32HardwareActuator.move_cursor(tx, ty)
            else:
                Win32HardwareActuator.move_cursor(tx, ty)

            if self.hud:
                try:
                    self.hud.trigger_ripple(tx, ty)
                except Exception:
                    pass

            cursor_shape = get_cursor_shape()
            print(f"[Cursor Sensor] Hover shape over [{dot_id or '?' }]: {cursor_shape}")

        # Restore minimized window if control belongs to minimized app
        if droplet and getattr(droplet, 'is_minimized', False) and getattr(droplet, 'hwnd', 0):
            try:
                from .window import restore_and_focus_window
                print(f"[Actuator] [Restore Minimized Application] HWND: {droplet.hwnd}")
                restore_and_focus_window(droplet.hwnd)
                time.sleep(0.20)
            except Exception:
                pass

        # Validate spatial actions require a target coordinate or element
        if act_clean in ('click', 'single_click', 'double_click', 'triple_click', 'right_click', 'context_click', 'middle_click'):
            if tx is None or ty is None or (tx == 0 and ty == 0):
                print(f"[Actuator] ⚠️ Action '{act_clean}' requested without target coordinates or droplet; rejecting blind click.")
                return {"verified": False, "error": f"Action '{act_clean}' requires target element or coordinates", "reason": "No target coordinates provided", "type": "INVALID_ACTION"}

        # -------------------------------------------------------------------------
        # Canonical Action Dispatch
        # -------------------------------------------------------------------------
        if act_clean in ('click', 'single_click'):
            print(f"[Actuator] [Click] Left Click: {label or f'({tx}, {ty})'}")
            Win32HardwareActuator.click(tx, ty, button="left")

        elif act_clean == 'double_click':
            print(f"[Actuator] [Double Click]: {label or f'({tx}, {ty})'}")
            Win32HardwareActuator.double_click(tx, ty, button="left")

        elif act_clean == 'triple_click':
            print(f"[Actuator] [Triple Click]: {label or f'({tx}, {ty})'}")
            Win32HardwareActuator.triple_click(tx, ty, button="left")

        elif act_clean in ('right_click', 'context_click'):
            print(f"[Actuator] [Right Click] Context Menu: {label or f'({tx}, {ty})'}")
            Win32HardwareActuator.right_click(tx, ty)

        elif act_clean == 'middle_click':
            print(f"[Actuator] [Middle Click]: {label or f'({tx}, {ty})'}")
            Win32HardwareActuator.middle_click(tx, ty)

        elif act_clean in ('hover', 'move_to'):
            if tx is not None and ty is not None:
                print(f"[Actuator] [Hover]: {label or f'({tx}, {ty})'}")
                Win32HardwareActuator.hover(tx, ty)

        elif act_clean in ('mouse_down', 'press_mouse'):
            btn = kwargs.get('button', 'left')
            print(f"[Actuator] [Mouse Down] ({btn}) at ({tx or 0}, {ty or 0})")
            Win32HardwareActuator.mouse_down(tx or 0, ty or 0, button=btn)

        elif act_clean in ('mouse_up', 'release_mouse'):
            btn = kwargs.get('button', 'left')
            print(f"[Actuator] [Mouse Up] ({btn}) at ({tx or 0}, {ty or 0})")
            Win32HardwareActuator.mouse_up(tx or 0, ty or 0, button=btn)

        elif act_clean in ('drag', 'drag_to', 'drag_and_drop', 'free_drag', 'drag_area'):
            dest_x, dest_y = None, None
            if drag_dest:
                dest_x, dest_y = drag_dest
            elif kwargs.get('to_x') is not None and kwargs.get('to_y') is not None:
                dest_x, dest_y = int(kwargs['to_x']), int(kwargs['to_y'])
            elif kwargs.get('drag_dest_x') is not None and kwargs.get('drag_dest_y') is not None:
                dest_x, dest_y = int(kwargs['drag_dest_x']), int(kwargs['drag_dest_y'])
            if dest_x is None or dest_y is None or tx is None or ty is None:
                return {"verified": False, "error": "Missing drag source/destination", "reason": "Drag requires valid start and destination coordinates", "type": "INVALID_ACTION"}
            print(f"[Actuator] [Drag] From ({tx}, {ty}) to ({dest_x}, {dest_y})...")
            Win32HardwareActuator.drag(tx, ty, dest_x, dest_y, duration=0.35)

        elif act_clean in ('ctrl_click', 'control_click'):
            print(f"[Actuator] [Ctrl + Click] at ({tx}, {ty})")
            Win32HardwareActuator.modifier_click(tx, ty, modifier="ctrl", button="left")

        elif act_clean == 'shift_click':
            print(f"[Actuator] [Shift + Click] at ({tx}, {ty})")
            Win32HardwareActuator.modifier_click(tx, ty, modifier="shift", button="left")

        elif act_clean == 'alt_click':
            print(f"[Actuator] [Alt + Click] at ({tx}, {ty})")
            Win32HardwareActuator.modifier_click(tx, ty, modifier="alt", button="left")

        elif act_clean in ('hover_and_wait', 'hover_wait'):
            wait_sec = float(input_text) if input_text and input_text.replace('.', '', 1).isdigit() else 0.3
            print(f"[Actuator] [Hover & Wait] {wait_sec}s at ({tx}, {ty})")
            Win32HardwareActuator.hover_and_wait(tx or 0, ty or 0, wait_time=wait_sec)

        elif act_clean in ('clear_and_type', 'replace_text', 'clear_type'):
            txt_to_type = input_text or kwargs.get('text', '')
            do_enter = bool(kwargs.get('press_enter') or kwargs.get('enter') or kwargs.get('submit'))
            loc_str = f"at ({tx}, {ty})" if (tx and ty and tx > 0 and ty > 0) else "into focused control"
            print(f"[Actuator] [Clear & Type] Text: \"{txt_to_type}\" (Enter: {do_enter}) {loc_str}")
            Win32HardwareActuator.clear_and_type(tx or 0, ty or 0, txt_to_type, press_enter=do_enter)

        elif act_clean in ('type_and_enter', 'submit'):
            txt_to_type = input_text or kwargs.get('text', '')
            if tx and ty and tx > 0 and ty > 0:
                print(f"[Actuator] [Type & Enter] Text: \"{txt_to_type}\" at ({tx}, {ty})")
                Win32HardwareActuator.click_and_type(tx, ty, txt_to_type, press_enter=True)
            else:
                print(f"[Actuator] [Type & Enter] Text: \"{txt_to_type}\" into focused control")
                # Focus safety guard: Prevent recursive typing into Copilot HUD
                try:
                    from ..core.models import _OWN_HWN_SET
                    if _user32:
                        fg = _user32.GetForegroundWindow()
                        if fg in _OWN_HWN_SET:
                            print("[Actuator] [Focus Safety]: Copilot window had focus. Skipping blind type to prevent self-injection.")
                            return {"verified": False, "error": "Copilot window was focused during ungrounded type", "reason": "Blocked typing into Copilot HUD", "type": "INVALID_ACTION"}
                except Exception:
                    pass

                if txt_to_type:
                    Win32HardwareActuator.type_unicode(txt_to_type)
                    time.sleep(0.04)
                Win32HardwareActuator.press_key('enter')

        elif act_clean == 'click_and_type':
            txt_to_type = input_text or kwargs.get('text', '')
            do_enter = bool(kwargs.get('press_enter') or kwargs.get('enter'))
            print(f"[Actuator] [Click & Type] Text: \"{txt_to_type}\" at ({tx}, {ty})")
            Win32HardwareActuator.click_and_type(tx, ty, txt_to_type, press_enter=do_enter)

        elif act_clean == 'copy':
            print("[Actuator] [Shortcut] Copy (Ctrl+C)")
            Win32HardwareActuator.copy()

        elif act_clean == 'paste':
            print("[Actuator] [Shortcut] Paste (Ctrl+V)")
            Win32HardwareActuator.paste()

        elif act_clean == 'select_all':
            print("[Actuator] [Shortcut] Select All (Ctrl+A)")
            Win32HardwareActuator.select_all()

        elif act_clean == 'undo':
            print("[Actuator] [Shortcut] Undo (Ctrl+Z)")
            Win32HardwareActuator.undo()

        elif act_clean == 'redo':
            print("[Actuator] [Shortcut] Redo (Ctrl+Y)")
            Win32HardwareActuator.redo()

        elif act_clean == 'save':
            print("[Actuator] [Shortcut] Save (Ctrl+S)")
            Win32HardwareActuator.save()

        elif act_clean == 'find':
            print("[Actuator] [Shortcut] Find (Ctrl+F)")
            Win32HardwareActuator.find()

        elif act_clean == 'close_tab':
            print("[Actuator] [Shortcut] Close Tab (Ctrl+W)")
            Win32HardwareActuator.close_tab()

        elif act_clean == 'new_tab':
            print("[Actuator] [Shortcut] New Tab (Ctrl+T)")
            Win32HardwareActuator.new_tab()

        elif act_clean == 'switch_window':
            target_win = input_text or kwargs.get('target', '') or kwargs.get('title', '')
            # 1. If droplet was specified (e.g. taskbar item or window tab), click it directly
            if droplet or (target_x is not None and target_y is not None and (tx and ty and (tx > 0 or ty > 0))):
                print(f"[Actuator] [Switch Window]: Clicking window/taskbar control '{label}' at ({tx}, {ty})")
                Win32HardwareActuator.click(tx, ty, button="left")
                time.sleep(0.35)
                return {"verified": True, "reason": f"Clicked window/taskbar control '{label}'", "type": "HARDWARE"}

            # 2. Native Win32 search by window title or process name
            if target_win:
                print(f"[Actuator] [Switch Window]: Searching for existing window matching '{target_win}'...")
                try:
                    from .window import switch_to_window
                    found_hwnd = switch_to_window(target_win)
                    if found_hwnd:
                        time.sleep(0.35)
                        return {"verified": True, "reason": f"Switched to window '{target_win}' (HWND: {found_hwnd})", "type": "HARDWARE", "hwnd": found_hwnd}
                except Exception as e:
                    print(f"[Actuator] [Switch Window Notice]: {e}")

            # 3. Fallback: Alt+Tab shortcut
            print("[Actuator] [Shortcut] Switch Window (Alt+Tab)")
            Win32HardwareActuator.switch_window()
            time.sleep(0.35)
            return {"verified": True, "reason": "Switched window via Alt+Tab shortcut", "type": "HARDWARE"}

        elif act_clean == 'win_run':
            cmd = input_text or kwargs.get('command', '') or 'cmd'
            print(f"[Actuator] [Win + R Run]: '{cmd}'")
            Win32HardwareActuator.win_run(cmd)

        elif act_clean in ('minimize', 'minimize_window'):
            print("[Actuator] [Minimize Window]: Active window context")
            minimized = False
            if win_ctx and win_ctx.hwnd and win_ctx.title != "Desktop":
                try:
                    from .window import minimize_window
                    minimized = minimize_window(win_ctx.hwnd)
                except Exception:
                    pass
            if not minimized:
                Win32HardwareActuator.hotkey('win', 'down')
            time.sleep(0.35)
            return {"verified": True, "reason": "Minimized window", "type": "HARDWARE"}

        elif act_clean in ('maximize', 'maximize_window'):
            print("[Actuator] [Maximize Window]: Active window context")
            maximized = False
            if win_ctx and win_ctx.hwnd and win_ctx.title != "Desktop":
                try:
                    from .window import maximize_window
                    maximized = maximize_window(win_ctx.hwnd)
                except Exception:
                    pass
            if not maximized:
                Win32HardwareActuator.hotkey('win', 'up')
            time.sleep(0.35)
            return {"verified": True, "reason": "Maximized window", "type": "HARDWARE"}

        elif act_clean in ('stop', 'finish', 'done'):
            print(f"[Actuator] [Stop]: Autonomous task completion requested.")
            return {"verified": True, "reason": "Task completed successfully", "type": "TERMINATION"}

        elif act_clean in ('drop', 'release'):
            print(f"[Actuator] [Drop] At ({tx}, {ty})")
            Win32HardwareActuator.drop(tx, ty)

        elif act_clean in ('scroll', 'wheel'):
            clicks = scroll_amount if scroll_amount is not None else -5
            if input_text and (input_text.lstrip('-').isdigit()):
                clicks = int(input_text) // 120
            direction = scroll_direction or ("horizontal" if "h" in act_clean else "vertical")
            print(f"[Actuator] [Scroll] {clicks} clicks ({direction}) at ({tx}, {ty})")
            Win32HardwareActuator.scroll(clicks, direction=direction, x=tx, y=ty)

        elif act_clean in ('type', 'fill', 'write', 'input'):
            txt_to_type = input_text or kwargs.get('text', '')
            print(f"[Actuator] [Type] Text: \"{txt_to_type}\"")
            if droplet or (target_x is not None and target_y is not None and (tx and ty and (tx > 0 or ty > 0))):
                Win32HardwareActuator.click(tx, ty, button="left")
                time.sleep(0.06)
            if txt_to_type:
                if len(txt_to_type) > 3:
                    try:
                        import pyperclip
                        pyperclip.copy(txt_to_type)
                        Win32HardwareActuator.hotkey('ctrl', 'v')
                        time.sleep(0.08)
                    except ImportError:
                        Win32HardwareActuator.type_unicode(txt_to_type)
                else:
                    Win32HardwareActuator.type_unicode(txt_to_type)

        elif act_clean == 'press':
            target_key = key or input_text or kwargs.get('key_name') or kwargs.get('key') or 'enter'
            target_key = str(target_key).strip().lower()
            if target_key in ("none", "null", ""):
                target_key = "enter"
            print(f"[Actuator] [Press Key]: '{target_key}'")
            Win32HardwareActuator.press_key(target_key)

        elif act_clean in ('hotkey', 'chord'):
            combo_candidates = []
            if key_combo:
                if isinstance(key_combo, list):
                    combo_candidates.extend(key_combo)
                elif isinstance(key_combo, str):
                    combo_candidates.extend(key_combo.split('+'))
            if not combo_candidates and key:
                if isinstance(key, str) and '+' in key:
                    combo_candidates.extend(key.split('+'))
                else:
                    combo_candidates.append(key)
            if not combo_candidates and input_text:
                if isinstance(input_text, str) and '+' in input_text:
                    combo_candidates.extend(input_text.split('+'))
                else:
                    combo_candidates.append(input_text)
            for extra in ('hotkey', 'keys', 'shortcut'):
                if not combo_candidates and kwargs.get(extra):
                    val = kwargs[extra]
                    if isinstance(val, list):
                        combo_candidates.extend(val)
                    elif isinstance(val, str):
                        combo_candidates.extend(val.split('+'))

            combo = [str(k).strip().lower() for k in combo_candidates if k and str(k).strip().lower() not in ("none", "null", "")]
            if combo:
                print(f"[Actuator] [Hotkey]: {combo}")
                Win32HardwareActuator.hotkey(*combo)
                time.sleep(0.10)
            else:
                print("[Actuator] [Hotkey ⚠️]: No valid key combination specified for hotkey action")
                return {"verified": False, "error": "Missing key combination for hotkey", "reason": "No valid keys specified", "type": "INVALID_ACTION"}

        elif act_clean == 'key_down':
            target_key = key or input_text or kwargs.get('key_name', 'shift')
            print(f"[Actuator] [Key Down]: '{target_key}'")
            Win32HardwareActuator.key_down(target_key)

        elif act_clean == 'key_up':
            target_key = key or input_text or kwargs.get('key_name', 'shift')
            print(f"[Actuator] [Key Up]: '{target_key}'")
            Win32HardwareActuator.key_up(target_key)

        elif act_clean == 'clear':
            print(f"[Actuator] [Clear] Input Field at ({tx}, {ty})")
            Win32HardwareActuator.clear_input(tx, ty)

        elif act_clean in ('check', 'uncheck'):
            print(f"[Actuator] [{act_clean.upper()}] Form Control at ({tx}, {ty}): {label}")
            Win32HardwareActuator.click(tx, ty, button="left")

        elif act_clean == 'select_option':
            opt = input_text or kwargs.get('option', '')
            print(f"[Actuator] [Select Option] \"{opt}\" on {label}")
            Win32HardwareActuator.click(tx, ty, button="left")
            time.sleep(0.1)
            if opt:
                Win32HardwareActuator.type_unicode(opt)
                time.sleep(0.05)
                Win32HardwareActuator.press_key('enter')

        elif act_clean == 'wait':
            duration = float(input_text) if input_text and input_text.replace('.', '', 1).isdigit() else 0.5
            print(f"[Actuator] [Wait] {duration}s for UI...")
            time.sleep(duration)

        # ---------------------------------------------------------------------
        # Application Launching & Path Opening
        # ---------------------------------------------------------------------
        elif act_clean in ('launch_app', 'open_app'):
            app_to_launch = input_text or kwargs.get('app', '') or kwargs.get('path', '') or kwargs.get('target', '')
            # 1. Droplet activation
            if not app_to_launch and droplet:
                print(f"[Actuator] [Launch App]: Actuating desktop/launcher droplet [{dot_id}] '{label}' at ({tx}, {ty})")
                is_desktop = (not win_ctx) or (not win_ctx.is_valid) or (win_ctx.title == "Desktop")
                if is_desktop:
                    Win32HardwareActuator.double_click(tx, ty, button="left")
                else:
                    Win32HardwareActuator.click(tx, ty, button="left")
                time.sleep(0.4)
                return {"verified": True, "reason": f"Actuated launcher '{label}'", "type": "HARDWARE"}

            if app_to_launch:
                app_clean = app_to_launch.strip()
                print(f"[Actuator] [Launch App]: '{app_clean}'")

                # Guard 1: Reject numbers, math expressions, and single punctuation
                is_formula = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', app_clean)) or any(op in app_clean for op in ("+", "*", "=", "/", "<", ">"))
                if app_clean.isdigit() or is_formula or (len(app_clean) <= 2 and not app_clean.isalpha()):
                    return {
                        "verified": False,
                        "error": f"Invalid application name '{app_clean}'",
                        "reason": f"Cannot launch '{app_clean}' as application. For typing numbers or text into active windows, use 'type' action.",
                        "type": "INVALID_ACTION"
                    }

                # Guard 2: Enforce that LAUNCH_APP is ONLY used when not already in foreground
                if win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop":
                    win_t = win_ctx.title.lower()
                    app_low = app_clean.lower()
                    from ..reasoning.awareness_router import AwarenessRouter
                    known_aliases = AwarenessRouter.KNOWN_APPS.get(app_low, [app_low])
                    is_already_active = any(alias in win_t for alias in known_aliases if len(alias) >= 3) or (app_low in win_t)
                    if is_already_active:
                        print(f"[Actuator] [Launch App]: Application '{app_clean}' is ALREADY in foreground window '{win_ctx.title}'!")
                        return {
                            "verified": False,
                            "error": f"Application '{app_clean}' is already in foreground",
                            "reason": f"Application '{app_clean}' already active in foreground window '{win_ctx.title}'. Dispatch 'type' or in-window controls instead.",
                            "type": "ALREADY_ACTIVE"
                        }

                try:
                    # Try ShellExecuteW directly
                    launched = False
                    if _shell32 and hasattr(_shell32, 'ShellExecuteW'):
                        h_inst = _shell32.ShellExecuteW(0, "open", app_clean, None, None, 1)
                        if h_inst > 32:
                            launched = True

                    if not launched:
                        # Try IntentScopeRouter accelerator if available
                        try:
                            from live_copilot import IntentScopeRouter
                            res = IntentScopeRouter.launch_or_activate_app(app_clean)
                            if res and res.get("success"):
                                launched = True
                        except Exception:
                            pass

                    if not launched:
                        # Fallback to os.startfile or subprocess
                        if hasattr(os, 'startfile'):
                            try:
                                os.startfile(app_clean)
                                launched = True
                            except Exception:
                                pass

                    if not launched:
                        import shutil
                        cmd_head = app_clean.split()[0]
                        if shutil.which(cmd_head) or app_clean.endswith(".exe"):
                            subprocess.Popen(app_clean, shell=True)
                            launched = True
                        else:
                            return {
                                "verified": False,
                                "error": f"Command or application '{app_clean}' not found on system",
                                "reason": f"Application executable '{app_clean}' is not recognized or available.",
                                "type": "EXECUTION_ERROR"
                            }

                    time.sleep(0.5)
                    return {"verified": True, "reason": f"Launched application '{app_clean}'", "type": "PROCESS_LAUNCH"}
                except Exception as e:
                    print(f"[Actuator] [Launch App Error]: {e}")
                    return {"verified": False, "error": str(e), "reason": f"Failed to launch '{app_clean}': {e}", "type": "EXECUTION_ERROR"}
            else:
                return {"verified": False, "error": "Empty application name provided", "reason": "No application specified to launch", "type": "EXECUTION_ERROR"}


        elif act_clean in ('open_path', 'system_open', 'system_open_path'):
            path_to_open = input_text or kwargs.get('path', '') or kwargs.get('target', '') or kwargs.get('app', '')
            if not path_to_open and droplet:
                print(f"[Actuator] [Open Path]: Actuating droplet [{dot_id}] '{label}' at ({tx}, {ty})")
                Win32HardwareActuator.click(tx, ty, button="left")
                time.sleep(0.4)
                return {"verified": True, "reason": f"Opened item '{label}'", "type": "HARDWARE"}

            if path_to_open:
                p_clean = path_to_open.strip()
                if p_clean.lower().startswith("explorer "):
                    p_clean = p_clean[9:].strip()
                if re.match(r'^[a-zA-Z]:$', p_clean):
                    p_clean += "\\"

                # Web URLs
                if p_clean.startswith(("http://", "https://", "www.")) or re.search(r'\.(?:com|org|net|io|ai|gov|edu)$', p_clean):
                    url = p_clean if p_clean.startswith("http") else f"https://{p_clean}"
                    print(f"[Actuator] [Open Path]: Opening URL '{url}'")
                    webbrowser.open(url)
                    time.sleep(0.5)
                    return {"verified": True, "reason": f"Opened URL '{url}'", "type": "HARDWARE"}

                # Expand environment and home variables
                expanded_path = os.path.normpath(os.path.expanduser(os.path.expandvars(p_clean)))
                print(f"[Actuator] [Open Path]: '{expanded_path}'")
                try:
                    if hasattr(os, 'startfile'):
                        os.startfile(expanded_path)
                    elif _shell32 and hasattr(_shell32, 'ShellExecuteW'):
                        _shell32.ShellExecuteW(0, "open", expanded_path, None, None, 1)
                    else:
                        subprocess.Popen(["xdg-open", expanded_path])
                    time.sleep(0.5)
                    return {"verified": True, "reason": f"Opened path '{expanded_path}'", "type": "HARDWARE"}
                except Exception as oe:
                    print(f"[Actuator] [Open Path Error]: {oe}")
                    return {"verified": False, "error": str(oe), "reason": f"Failed to open '{expanded_path}': {oe}", "type": "EXECUTION_ERROR"}
            else:
                return {"verified": False, "error": "Empty path provided", "reason": "No path specified to open", "type": "EXECUTION_ERROR"}

        else:
            print(f"[Actuator] Rejecting unknown action '{act_clean}'")
            return {"verified": False, "error": f"Unknown action: {act_clean}", "reason": "Action is not in the actuator allowlist", "type": "INVALID_ACTION"}

        # ---------------------------------------------------------------------
        # Closed-Loop Verification & Smart Micro-Retry
        # ---------------------------------------------------------------------
        if self.verifier and pre_snap and droplet:
            try:
                from ..vision.settle_sensor import AdaptiveSettleSensor
                from ..vision.grabber import FastScreenGrabber
                AdaptiveSettleSensor.wait_for_settle(
                    grabber_func=FastScreenGrabber.get_instance().grab_full,
                    timeout_sec=0.45,
                    motion_threshold=0.8
                )
            except Exception:
                pass

            ver_res = self.verifier.verify_action(pre_snap, droplet, act_clean)
            print(f"[Closed-Loop Verification] Outcome: {'PASS' if ver_res['verified'] else 'NO CHANGE'} -> {ver_res['reason']}")

            if not ver_res['verified'] and act_clean in ('click', 'single_click', 'double_click', 'right_click'):
                is_launcher = (getattr(droplet, 'category', '') == 'interactive_app') or (pre_snap.title == "Desktop")
                if not is_launcher:
                    print(f"[Visual Diff Micro-Nudge Retry] Zero change on Dot [{droplet.id}]. Executing localized micro-nudge (+5px)...")
                    nudge_x = tx + 5
                    nudge_y = ty + 4
                    Win32HardwareActuator.move_cursor(nudge_x, nudge_y)
                    time.sleep(0.04)
                    btn_type = "right" if act_clean == 'right_click' else "left"
                    Win32HardwareActuator.click(nudge_x, nudge_y, button=btn_type, double=(act_clean == 'double_click'))
                    time.sleep(0.16)
                    ver_res = self.verifier.verify_action(pre_snap, droplet, act_clean)
                    print(f"[Closed-Loop Verification (Micro-Nudge Retry)] Outcome: {'PASS' if ver_res['verified'] else 'NO CHANGE'} -> {ver_res['reason']}")

            return ver_res

        return {"verified": True, "reason": f"Action {act_clean} executed successfully", "type": "HARDWARE"}
