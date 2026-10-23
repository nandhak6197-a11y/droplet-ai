"""
Autonomous Visual & Structural Scene Tracker.
Zero-overhead hardware compositor thumbnail diffing (160x90 Direct GDI StretchBlt).
Monitors continuous pixel-delta motion, window geometry, focus transitions, and scrolling.
Debounces settling so full pristine scans fire at the exact moment UI transitions settle.
"""
import threading
import time
import ctypes
from ctypes import wintypes
from typing import Optional, Callable, Tuple
import cv2
import numpy as np

from ..platform.desktop import DesktopThreadManager
from ..platform.window import get_active_window_context, WindowContext

u32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Configure 64-bit safe ctypes signatures for all GDI functions
try:
    gdi32.CreateDCA.restype = wintypes.HDC
    gdi32.CreateDCA.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SetStretchBltMode.restype = ctypes.c_int
    gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]
    gdi32.StretchBlt.restype = wintypes.BOOL
    gdi32.StretchBlt.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.DWORD
    ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
        ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT
    ]
except Exception:
    pass


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)
    ]


class AutonomousSceneTracker:
    """
    Sub-millisecond hardware-accelerated scene watcher.
    Monitors:
      1. Pixel-level motion delta via 160x90 downsampled GDI StretchBlt (scrolling, typing, hovers, animations).
      2. Window geometry shifts (window moves, resizes, snaps).
      3. Focus transitions (foreground app changes, flyouts, menus).
    Debounces UI animation settling to guarantee razor-sharp rescan on settled states.
    """
    def __init__(
        self,
        on_settled_callback: Callable[[str, WindowContext], None],
        motion_threshold: float = 0.50,
        settle_delay: float = 0.12,
        streaming_interval: float = 0.35,
        sample_hz: float = 20.0
    ):
        self.on_settled = on_settled_callback
        self.motion_threshold = motion_threshold
        self.settle_delay = settle_delay
        self.streaming_interval = streaming_interval
        self.sample_delay = 1.0 / sample_hz

        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

        # State tracking
        self._in_motion = False
        self._last_motion_time = 0.0
        self._last_settled_time = 0.0
        self._motion_reason = "visual_change"

        # Window state
        self._last_hwnd = -1
        self._last_title = ""
        self._last_rect: Optional[Tuple[int, int, int, int]] = None
        self._current_win_ctx: Optional[WindowContext] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="SceneTrackerThread", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)

    def pause(self):
        """Temporarily suspend change triggers during programmatic mouse/keyboard actuation."""
        self._paused = True

    def resume(self):
        """Resume change triggers after actuation finishes."""
        self._paused = False
        self._in_motion = True
        self._last_motion_time = time.time()

    def _run_loop(self):
        DesktopThreadManager.attach_input_desktop()
        sw = u32.GetSystemMetrics(0) or 1920
        sh = u32.GetSystemMetrics(1) or 1080
        tw, th = 160, 90

        hdc = None
        memdc = None
        bmp = None
        old_bmp = None

        try:
            hdc = gdi32.CreateDCA(b"DISPLAY", None, None, None)
            if not hdc:
                return
            memdc = gdi32.CreateCompatibleDC(hdc)
            bmp = gdi32.CreateCompatibleBitmap(hdc, tw, th)
            old_bmp = gdi32.SelectObject(memdc, bmp)
            gdi32.SetStretchBltMode(memdc, 3)  # COLORONCOLOR

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = tw
            bmi.biHeight = -th
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            buf = np.zeros((th, tw, 4), dtype=np.uint8)

            def _sample_thumb():
                gdi32.StretchBlt(memdc, 0, 0, tw, th, hdc, 0, 0, sw, sh, 0x00CC0020)
                gdi32.GetDIBits(memdc, bmp, 0, th, buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(bmi), 0)
                return buf[:, :, 0].copy()

            prev_thumb = _sample_thumb()
            time.sleep(self.sample_delay)

            # Initialize initial window state
            self._current_win_ctx = get_active_window_context()
            if self._current_win_ctx:
                self._last_hwnd = self._current_win_ctx.hwnd
                self._last_title = self._current_win_ctx.title
                self._last_rect = self._current_win_ctx.rect

            while self._running:
                loop_start = time.time()

                if self._paused:
                    time.sleep(self.sample_delay)
                    continue

                # 1. Grab sub-millisecond hardware thumbnail
                curr_thumb = _sample_thumb()
                diff = cv2.absdiff(curr_thumb, prev_thumb)
                delta = float(np.mean(diff))
                prev_thumb = curr_thumb

                now = time.time()
                motion_detected = False
                detected_reason = "visual_change"

                # 2. Check pixel motion
                if delta > self.motion_threshold:
                    motion_detected = True
                    detected_reason = f"pixel_delta({delta:.1f})"

                # 3. Check window focus & geometry
                win_ctx = get_active_window_context()
                if win_ctx and win_ctx.is_valid:
                    self._current_win_ctx = win_ctx
                    if win_ctx.hwnd != self._last_hwnd or win_ctx.title != self._last_title:
                        motion_detected = True
                        detected_reason = f"window_switch('{win_ctx.title[:16]}')"
                        self._last_hwnd = win_ctx.hwnd
                        self._last_title = win_ctx.title
                        self._last_rect = win_ctx.rect
                    elif win_ctx.rect and self._last_rect:
                        # Detect window movement / resize (>= 12px shift)
                        rx_shift = abs(win_ctx.rect[0] - self._last_rect[0]) + abs(win_ctx.rect[1] - self._last_rect[1])
                        rw_shift = abs(win_ctx.rect[2] - self._last_rect[2]) + abs(win_ctx.rect[3] - self._last_rect[3])
                        if rx_shift >= 12 or rw_shift >= 12:
                            motion_detected = True
                            detected_reason = f"window_move({rx_shift}px)"
                            self._last_rect = win_ctx.rect

                # 4. Debounce and Settle State Machine
                if motion_detected:
                    self._in_motion = True
                    self._last_motion_time = now
                    self._motion_reason = detected_reason

                    # Streaming sync during prolonged motion (e.g. continuous scrolling > 600ms)
                    if now - self._last_settled_time >= self.streaming_interval:
                        self._last_settled_time = now
                        self._trigger_settle(f"streaming_{self._motion_reason}")
                else:
                    # No motion in this sample. Check if previous motion has settled
                    if self._in_motion and (now - self._last_motion_time >= self.settle_delay):
                        self._in_motion = False
                        self._last_settled_time = now
                        self._trigger_settle(self._motion_reason)

                # Maintain sampling rate
                elapsed = time.time() - loop_start
                sleep_time = max(0.01, self.sample_delay - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[AutonomousSceneTracker Notice]: {e}")
        finally:
            if memdc and old_bmp:
                gdi32.SelectObject(memdc, old_bmp)
            if bmp:
                gdi32.DeleteObject(bmp)
            if memdc:
                gdi32.DeleteDC(memdc)
            if hdc:
                gdi32.DeleteDC(hdc)

    def _trigger_settle(self, reason: str):
        try:
            if self.on_settled and not self._paused:
                ctx = self._current_win_ctx or get_active_window_context()
                self.on_settled(reason, ctx)
        except Exception as e:
            print(f"[SceneTracker Settle Error]: {e}")
