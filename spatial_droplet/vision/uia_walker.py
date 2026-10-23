"""
Windows UI Automation (UIA) Control Walker with Asynchronous Non-Blocking Daemon.
Discovers semantic controls (Buttons, Tabs, Inputs, Icons, Menus, Bars)
directly from the OS accessibility tree (Confidence: 1.0, 100% immune to contrast/themes).

Phase 2 Architecture:
- Decoupled from the synchronous critical path into a dedicated background daemon thread.
- Thread-safe in-memory cache updated continuously.
- Main droplet spray checks cache with 0ms blocking.
- If fresh (< 1.5s), enriches droplets; if stale or walking a deep DOM tree (e.g. Chrome),
  proceeds immediately with pure CV lattice without tripping the 6-second blackout.
"""
import time
import threading
from typing import List, Tuple, Optional, Dict, Any

try:
    import uiautomation as auto
    HAS_UIA = True
except ImportError:
    HAS_UIA = False


class AsyncUiaBackgroundWorker:
    """
    Dedicated background worker thread for asynchronous UI Automation scans.
    Maintains a thread-safe in-memory cache with zero blocking on the critical path.
    """
    _instance: Optional["AsyncUiaBackgroundWorker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.cache: Dict[str, Any] = {
            "hwnd": 0,
            "title": "",
            "elements": [],
            "timestamp": 0.0,
            "is_scanning": False,
        }
        self.cache_lock = threading.Lock()
        self.scan_trigger = threading.Event()
        self.target_hwnd: int = 0
        self.running = True
        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncUiaWorker"
        )
        self.thread.start()

    @classmethod
    def get_instance(cls) -> "AsyncUiaBackgroundWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _worker_loop(self):
        while self.running:
            # Dwell or wait for trigger
            self.scan_trigger.wait(timeout=0.60)
            self.scan_trigger.clear()
            if not self.running:
                break

            target_hwnd = self.target_hwnd
            if not target_hwnd or target_hwnd == 0:
                try:
                    import ctypes
                    target_hwnd = ctypes.windll.user32.GetForegroundWindow()
                except Exception:
                    target_hwnd = 0

            with self.cache_lock:
                self.cache["is_scanning"] = True

            try:
                elements = UiaControlWalker._raw_walk_controls(hwnd=target_hwnd)
                with self.cache_lock:
                    self.cache["hwnd"] = target_hwnd
                    self.cache["elements"] = elements
                    self.cache["timestamp"] = time.time()
            except Exception:
                pass
            finally:
                with self.cache_lock:
                    self.cache["is_scanning"] = False

    def request_scan(self, hwnd: Optional[int] = None):
        """Asynchronously requests a fresh scan of the given or foreground window."""
        if hwnd is not None and hwnd != 0:
            self.target_hwnd = hwnd
        self.scan_trigger.set()

    def get_cached_controls(self, hwnd: Optional[int] = None, max_age_sec: float = 1.5) -> List[Tuple]:
        """
        Non-blocking 0ms check of the UIA cache.
        Returns cached controls if fresh (< max_age_sec), else triggers background scan
        and returns [] immediately so main thread never stalls.
        """
        now = time.time()
        with self.cache_lock:
            cached_hwnd = self.cache["hwnd"]
            cached_ts = self.cache["timestamp"]
            elements = list(self.cache["elements"])

        # Check freshness
        if (now - cached_ts) <= max_age_sec and elements:
            if hwnd is None or hwnd == 0 or hwnd == cached_hwnd:
                return elements

        # Cache is stale, empty, or for a different window -> trigger background scan
        self.request_scan(hwnd)
        # Return immediately with 0ms wait!
        return []


class UiaControlWalker:
    """
    Decoupled Asynchronous UIA control walker.
    Guarantees 0ms wait time for the main spray lattice pipeline.
    """
    TARGET_TYPES: Dict[str, str] = {
        "ButtonControl": "Btn",
        "TabItemControl": "Tab",
        "EditControl": "Edit",
        "MenuItemControl": "Menu",
        "ToggleButtonControl": "Toggle",
        "CheckBoxControl": "Check",
        "RadioButtonControl": "Radio",
        "HyperlinkControl": "Link",
        "SplitButtonControl": "Btn",
        "ImageControl": "Icon",
        "CustomControl": "Btn",
        "ToolBarControl": "Bar",
        "HeaderItemControl": "Header",
        "ListItemControl": "Item",
        "TreeItemControl": "Item",
        "DataItemControl": "Item",
        "ComboBoxControl": "Dropdown",
        "SpinnerControl": "Spin",
        "PaneControl": "Tile",
        "TextControl": "Text",
    }

    @classmethod
    def get_cached_controls(
        cls,
        hwnd: Optional[int] = None,
        max_age_sec: float = 1.5
    ) -> List[Tuple]:
        """
        0ms non-blocking check: returns cached UIA controls if fresh (< 1.5s),
        or empty list immediately without waiting.
        """
        if not HAS_UIA:
            return []
        worker = AsyncUiaBackgroundWorker.get_instance()
        return worker.get_cached_controls(hwnd=hwnd, max_age_sec=max_age_sec)

    @classmethod
    def _raw_walk_controls(
        cls,
        hwnd: Optional[int] = None,
        max_depth: int = 5,
        screen_w: int = 1920,
        screen_h: int = 1080,
        max_controls: int = 160
    ) -> List[Tuple]:
        """
        Low-level UIA tree walker executed in background thread.
        """
        if not HAS_UIA:
            return []

        results: List[Tuple] = []
        try:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF) or user32.OpenDesktopW("default", 0, False, 0x01FF)
                if hdesk:
                    user32.SetThreadDesktop(hdesk)
            except Exception:
                pass

            with auto.UIAutomationInitializerInThread():
                root_ctrls = []
                target_hwnd = hwnd
                if not target_hwnd or target_hwnd == 0:
                    try:
                        import ctypes
                        target_hwnd = ctypes.windll.user32.GetForegroundWindow()
                    except Exception:
                        target_hwnd = 0

                if target_hwnd and target_hwnd != 0:
                    # Check active modal popup/dialog
                    try:
                        import ctypes
                        popup_hwnd = ctypes.windll.user32.GetLastActivePopup(target_hwnd)
                        if popup_hwnd and popup_hwnd != target_hwnd and ctypes.windll.user32.IsWindow(popup_hwnd) and ctypes.windll.user32.IsWindowVisible(popup_hwnd):
                            p_ctrl = auto.ControlFromHandle(popup_hwnd)
                            if p_ctrl:
                                root_ctrls.append(p_ctrl)
                    except Exception:
                        pass

                    try:
                        wc = auto.ControlFromHandle(target_hwnd)
                        if wc and wc not in root_ctrls:
                            root_ctrls.append(wc)
                    except Exception:
                        pass

                # Focused popup or shell flyout
                try:
                    fc = auto.GetFocusedControl()
                    if fc and fc.NativeWindowHandle and fc.NativeWindowHandle != target_hwnd:
                        fw = auto.ControlFromHandle(fc.NativeWindowHandle)
                        if fw and fw not in root_ctrls:
                            root_ctrls.append(fw)
                except Exception:
                    pass

                walk_depth = max_depth
                if not root_ctrls:
                    root_ctrls.append(auto.GetRootControl())
                    walk_depth = min(3, max_depth)
                elif target_hwnd and target_hwnd != 0:
                    try:
                        import ctypes
                        buf = ctypes.create_unicode_buffer(256)
                        ctypes.windll.user32.GetClassNameW(target_hwnd, buf, 256)
                        cname = buf.value
                        if cname in ("Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"):
                            walk_depth = min(3, max_depth)
                    except Exception:
                        pass

                walk_start = time.time()
                total_walked = 0
                max_walk_sec = 1.25
                max_walked_nodes = 400

                for rctrl in root_ctrls:
                    if (time.time() - walk_start) >= max_walk_sec:
                        break
                    for ctrl, depth in auto.WalkControl(rctrl, maxDepth=walk_depth):
                        total_walked += 1
                        if total_walked >= max_walked_nodes or (time.time() - walk_start) >= max_walk_sec:
                            break
                        try:
                            ctype = ctrl.ControlTypeName
                            if ctype in cls.TARGET_TYPES:
                                r = ctrl.BoundingRectangle
                                if not r:
                                    continue
                                bw = r.width()
                                bh = r.height()
                                # Filter static document text
                                if ctype == "TextControl" and (bw > 450 or bh > 80 or not (ctrl.Name or "").strip()):
                                    continue
                                if ctype == "PaneControl" and (bw > 500 or bh > 250 or not (ctrl.Name or "").strip()):
                                    continue
                                if 6 <= bw <= 950 and 6 <= bh <= 450:
                                    cx = (r.left + r.right) // 2
                                    cy = (r.top + r.bottom) // 2
                                    if 0 <= cx <= screen_w and 0 <= cy <= screen_h:
                                        name = (ctrl.Name or "").strip()
                                        aid = (ctrl.AutomationId or "").strip()
                                        label = name or aid or f"{cls.TARGET_TYPES[ctype]} Action"
                                        prefix = cls.TARGET_TYPES[ctype]
                                        sub = label[:12]
                                        rx1 = min(int(r.left), int(r.right))
                                        rx2 = max(int(r.left), int(r.right))
                                        ry1 = min(int(r.top), int(r.bottom))
                                        ry2 = max(int(r.top), int(r.bottom))
                                        bbox = (rx1, ry1, rx2, ry2)
                                        results.append((cx, cy, f"[{prefix}] {label}", "interactive_widget", sub, bbox))
                                        if len(results) >= max_controls:
                                            return results
                        except Exception:
                            continue
        except Exception:
            pass
        return results

    @classmethod
    def walk_controls(
        cls,
        hwnd: Optional[int] = None,
        max_depth: int = 5,
        timeout_sec: float = 1.25,
        screen_w: int = 1920,
        screen_h: int = 1080,
        max_controls: int = 160
    ) -> List[Tuple]:
        """
        Non-blocking walk wrapper. Returns fresh cached controls with 0ms delay,
        or launches asynchronous scan in background daemon without blocking the caller.
        """
        cached = cls.get_cached_controls(hwnd=hwnd, max_age_sec=1.5)
        if cached:
            return cached

        # Trigger background daemon scan
        AsyncUiaBackgroundWorker.get_instance().request_scan(hwnd)
        return []

    walk_window_controls = walk_controls
