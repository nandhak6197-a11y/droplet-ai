"""
Desktop thread, session management, and Virtual Desktop Topology (Pillar 2).
Supports multi-monitor setups with negative virtual coordinates and multi-DPI displays.
"""
import ctypes
from typing import Optional, Tuple

_windll = getattr(ctypes, 'windll', None)
_user32 = getattr(_windll, 'user32', None) if _windll else None

# Win32 Virtual Desktop Metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
SM_CMONITORS = 80


class DesktopThreadManager:
    """Manages connection to the Windows input desktop to prevent GDI capture failures."""

    @staticmethod
    def ensure_input_desktop() -> Optional[int]:
        """Opens active input desktop (or default) and attaches the calling thread."""
        if not _user32:
            return None
        try:
            hdesk = _user32.OpenInputDesktop(0, False, 0x01FF)
            if not hdesk:
                hdesk = _user32.OpenDesktopW("default", 0, False, 0x01FF)
            if hdesk:
                _user32.SetThreadDesktop(hdesk)
                return hdesk
        except Exception:
            pass
        return None

    attach_input_desktop = ensure_input_desktop

    @staticmethod
    def safe_set_thread_desktop(hdesk: int) -> bool:
        if hdesk and _user32:
            try:
                return bool(_user32.SetThreadDesktop(hdesk))
            except Exception:
                pass
        return False


class VirtualDesktopTopology:
    """
    Virtual Desktop Topology Sensor for multi-monitor, negative-coordinate,
    and multi-display workstation environments.
    """

    @staticmethod
    def get_virtual_bounds() -> Tuple[int, int, int, int]:
        """
        Returns (vx, vy, vw, vh) spanning all connected active monitors.
        vx and vy can be negative if a secondary monitor is positioned to the left or top.
        """
        if _user32:
            try:
                vx = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
                vy = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
                vw = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
                vh = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
                if vw > 0 and vh > 0:
                    return vx, vy, vw, vh
            except Exception:
                pass
        return 0, 0, 1920, 1080

    @staticmethod
    def get_monitor_count() -> int:
        """Returns the number of active display monitors."""
        if _user32:
            try:
                cnt = _user32.GetSystemMetrics(SM_CMONITORS)
                return max(1, cnt)
            except Exception:
                pass
        return 1

    @classmethod
    def is_multi_monitor(cls) -> bool:
        """Returns True if the system operates with multiple displays."""
        return cls.get_monitor_count() > 1

    @classmethod
    def to_normalized_coords(cls, x: int, y: int) -> Tuple[int, int, bool]:
        """
        Transforms screen coordinate (x, y) into Windows SendInput 0-65535 space.
        Returns: (norm_x, norm_y, is_virtual_desk_flag_needed).
        """
        vx, vy, vw, vh = cls.get_virtual_bounds()
        is_virtual = (vx != 0 or vy != 0 or cls.is_multi_monitor())

        if is_virtual and vw > 0 and vh > 0:
            norm_x = int(round((x - vx) * 65535.0 / max(1, vw - 1)))
            norm_y = int(round((y - vy) * 65535.0 / max(1, vh - 1)))
            return norm_x, norm_y, True

        # Standard primary display fallback
        norm_x = int(round(x * 65535.0 / max(1, vw - 1)))
        norm_y = int(round(y * 65535.0 / max(1, vh - 1)))
        return norm_x, norm_y, False
