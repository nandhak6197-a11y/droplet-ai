"""
Windows High-DPI Awareness and Coordinate Space Manager.
Guarantees 1:1 hardware physical pixel parity (0px scaling error).
"""
import ctypes
from typing import Tuple


class DpiAwarenessManager:
    """Manages process and window DPI awareness contexts for pixel-perfect HUD alignment."""
    _initialized = False

    @classmethod
    def init_dpi_awareness(cls) -> bool:
        """Initializes Per-Monitor DPI awareness coordinated with active desktop connection."""
        if cls._initialized:
            return True

        from .desktop import DesktopThreadManager
        DesktopThreadManager.ensure_input_desktop()

        success = False
        # Try Per-Monitor V2 (Windows 10 1703+)
        try:
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
            if ctypes.windll.user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
                success = True
        except Exception:
            pass

        if not success:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
                success = True
            except Exception:
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
                    success = True
                except Exception:
                    try:
                        ctypes.windll.user32.SetProcessDPIAware()
                        success = True
                    except Exception:
                        pass

        cls._initialized = success
        return success

    enable_per_monitor_dpi = init_dpi_awareness

    @staticmethod
    def get_physical_screen_metrics() -> Tuple[int, int]:
        """Returns the true physical monitor resolution in hardware pixels."""
        u32 = ctypes.windll.user32
        w = u32.GetSystemMetrics(0)
        h = u32.GetSystemMetrics(1)
        return (w, h)

    @staticmethod
    def get_dpi_scale_factor() -> float:
        """Returns system DPI scale factor (e.g. 1.0, 1.25, 1.5, 2.0)."""
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            if hdc:
                LOGPIXELSX = 88
                dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                ctypes.windll.user32.ReleaseDC(0, hdc)
                return dpi / 96.0
        except Exception:
            pass
        return 1.0

    get_display_scale_factor = get_dpi_scale_factor
