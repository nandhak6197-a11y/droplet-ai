"""
Global Hotkey Listener.
Registers system-wide hotkeys (Alt+Space, Ctrl+Shift+Space, Alt+Q)
to trigger Vimium quick-select mode from any background or foreground application.
"""

import ctypes
from ctypes import wintypes
import threading
import time
from typing import Callable, List, Optional


class GlobalHotkeyListener:
    """
    Background Win32 message loop listener for global hotkeys.
    Triggers callback on Alt+Space, Ctrl+Shift+Space, or Alt+Q.
    """

    def __init__(
        self,
        on_hotkey_callback: Callable[[], None],
        on_toggle_droplets_callback: Optional[Callable[[], None]] = None,
        on_voice_toggle_callback: Optional[Callable[[], None]] = None
    ):
        self.on_hotkey = on_hotkey_callback
        self.on_toggle_droplets = on_toggle_droplets_callback
        self.on_voice_toggle = on_voice_toggle_callback
        self.running = False
        self.thread = None
        self.registered_ids: List[int] = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)

    def _run_loop(self):
        user32 = ctypes.windll.user32
        MOD_ALT = 0x0001
        MOD_CONTROL = 0x0002
        MOD_SHIFT = 0x0004
        VK_SPACE = 0x20
        VK_F8 = 0x77

        # Register Alt+Space (101), Ctrl+Shift+Space (102), Alt+Q (103), Alt+H (104)
        if user32.RegisterHotKey(0, 101, MOD_ALT, VK_SPACE):
            self.registered_ids.append(101)
        if user32.RegisterHotKey(0, 102, MOD_CONTROL | MOD_SHIFT, VK_SPACE):
            self.registered_ids.append(102)
        if user32.RegisterHotKey(0, 103, MOD_ALT, ord('Q')):
            self.registered_ids.append(103)
        if user32.RegisterHotKey(0, 104, MOD_ALT, ord('H')):
            self.registered_ids.append(104)

        # Register Voice Control: Alt+V (105), Ctrl+Shift+V (106), F8 (107)
        if user32.RegisterHotKey(0, 105, MOD_ALT, ord('V')):
            self.registered_ids.append(105)
        if user32.RegisterHotKey(0, 106, MOD_CONTROL | MOD_SHIFT, ord('V')):
            self.registered_ids.append(106)
        if user32.RegisterHotKey(0, 107, 0, VK_F8):
            self.registered_ids.append(107)

        print(f"[*] Global Shortcuts active! Registered {len(self.registered_ids)} shortcuts (Alt+Space, Ctrl+Shift+Space, Alt+Q, Alt+H, Alt+V, F8)")

        msg = wintypes.MSG()
        while self.running:
            if user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 1):  # PM_REMOVE = 1
                if msg.message == 0x0312:  # WM_HOTKEY
                    try:
                        if msg.wParam == 104 and self.on_toggle_droplets:
                            self.on_toggle_droplets()
                        elif msg.wParam in (105, 106, 107) and self.on_voice_toggle:
                            self.on_voice_toggle()
                        else:
                            self.on_hotkey()
                    except Exception as e:
                        print(f"[Hotkey Callback Error] {e}")
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.03)

        for hid in self.registered_ids:
            try:
                user32.UnregisterHotKey(0, hid)
            except Exception:
                pass
