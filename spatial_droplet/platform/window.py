"""
Window enumeration, active context resolution, Z-order tracking, and native Win32 window manipulation.
Provides robust window management (minimize, maximize, restore, switch) without coordinate guesswork.
"""
import ctypes
from ctypes import wintypes
from typing import Optional, Dict, Tuple, List, Union

from ..core.models import WindowContext, is_own_window

DWMWA_CLOAKED = 14

# Win32 ShowWindow Constants
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_MAXIMIZE = 3
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_MINIMIZE = 6
SW_SHOWMINNOACTIVE = 7
SW_SHOWNA = 8
SW_RESTORE = 9

# Win32 SetWindowPos Constants
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
HWND_TOP = 0
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_SHOWWINDOW = 0x0040
SWP_NOACTIVATE = 0x0010


def is_window_cloaked(hwnd: int) -> bool:
    """Checks if a window is cloaked by Windows DWM (virtual desktop / UWP background)."""
    try:
        dwmapi = ctypes.windll.dwmapi
        cloaked = wintypes.DWORD()
        res = dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return res == 0 and cloaked.value != 0
    except Exception:
        return False


def get_process_name_for_hwnd(hwnd: int) -> str:
    """Retrieves the executable process name associated with an HWND."""
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return ""
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        hProcess = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hProcess:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(hProcess, 0, buf, ctypes.byref(size)):
                full_path = buf.value
                return full_path.split("\\")[-1]
        finally:
            kernel32.CloseHandle(hProcess)
    except Exception:
        pass
    return ""


def get_desktop_context() -> WindowContext:
    """Returns fallback desktop context representing the full virtual screen."""
    user32 = ctypes.windll.user32
    sw = user32.GetSystemMetrics(0) or 1920
    sh = user32.GetSystemMetrics(1) or 1080
    desk_hwnd = user32.GetDesktopWindow() or 0
    return WindowContext(
        hwnd=desk_hwnd,
        title="Desktop",
        rect=(0, 0, sw, sh),
        width=sw,
        height=sh,
        is_valid=True,
        class_name="Desktop"
    )


def get_window_details(hwnd: int) -> Optional[Dict]:
    """Inspects a window handle and returns its metadata, dimensions, and semantic title."""
    try:
        user32 = ctypes.windll.user32
        if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return None
        if is_own_window(hwnd) or is_window_cloaked(hwnd):
            return None

        l = user32.GetWindowTextLengthW(hwnd)
        title = ""
        if l > 0:
            tbuf = ctypes.create_unicode_buffer(l + 1)
            user32.GetWindowTextW(hwnd, tbuf, l + 1)
            title = tbuf.value.strip()

        cbuf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cbuf, 256)
        cname = cbuf.value.strip()

        SYSTEM_IGNORE_CLASSES = (
            "Shell_TrayWnd", "Progman", "WorkerW", "Button", "tooltips_class32",
            "TkTopLevel", "MSCTFIME UI", "IME", "SysShadow", "DropShadow",
            "NativeDropTarget", "EdgeUiInputTopWndClass", "GDI+ Hook Window Class",
            "DummyWindow", "Windows.UI.Core.CoreComponentInputSource",
            "TaskListThumbnailWnd", "NotifyIconOverflowWindow"
        )
        if cname in SYSTEM_IGNORE_CLASSES:
            return None
        c_lower = cname.lower()
        if any(bad in c_lower for bad in ("shadow", "tooltip", "ime", "ghost", "droptarget", "hook", "dummy", "caret")):
            if cname != "#32768":
                return None
        if title in ("Task Switching", "Windows Input Experience", "Program Manager", "Default IME", "MSCTFIME UI", "tk", "Tk"):
            return None

        if not title:
            proc = get_process_name_for_hwnd(hwnd)
            p_lower = proc.lower() if proc else ""
            if "startmenu" in p_lower:
                title = "Start Menu"
            elif "search" in p_lower:
                title = "Windows Search"
            elif "shellexperience" in p_lower:
                title = "Start Menu / Windows Shell"
            elif cname == "Windows.UI.Core.CoreWindow":
                if "start" in p_lower:
                    title = "Start Menu"
                elif "search" in p_lower:
                    title = "Windows Search"
                else:
                    title = "Windows Shell App"
            elif cname == "#32768":
                title = "Context Menu"
            elif cname in ("XamlExplorerHostIslandWindow", "Windows.UI.Input.InputSite.WindowClass"):
                if "search" in p_lower:
                    title = "Windows Search"
                elif "start" in p_lower:
                    title = "Start Menu"
                else:
                    title = "Windows Flyout"
            elif proc and p_lower not in ("explorer.exe", "svchost.exe", "dwm.exe", "systemsettings.exe"):
                title = f"[{proc.replace('.exe', '').capitalize()}]"
            else:
                return None

        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        min_dim = 40 if (cname in ("#32768", "Windows.UI.Core.CoreWindow") or "menu" in title.lower()) else 60
        if w < min_dim or h < min_dim:
            return None

        return {
            "hwnd": hwnd,
            "title": title,
            "class": cname,
            "rect": (rect.left, rect.top, rect.right, rect.bottom),
            "w": w,
            "h": h
        }
    except Exception:
        return None


def get_active_window_context(preferred_hwnd: Optional[int] = None) -> WindowContext:
    """Resolves the currently active foreground application window context."""
    user32 = ctypes.windll.user32
    hdesk = None
    try:
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF) or user32.OpenDesktopW("default", 0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass

    if preferred_hwnd and user32.IsWindow(preferred_hwnd):
        det = get_window_details(preferred_hwnd)
        if det:
            return WindowContext(
                hwnd=det["hwnd"], title=det["title"],
                rect=det["rect"], width=det["w"], height=det["h"], is_valid=True,
                class_name=det.get("class", "")
            )

    # 1. Primary: True Foreground Window
    try:
        fg = user32.GetForegroundWindow()
        if fg and user32.IsWindow(fg) and not is_own_window(fg):
            det = get_window_details(fg)
            if det:
                return WindowContext(
                    hwnd=det["hwnd"], title=det["title"],
                    rect=det["rect"], width=det["w"], height=det["h"], is_valid=True,
                    class_name=det.get("class", "")
                )
    except Exception:
        pass

    # 2. Secondary: Topmost desktop application window via Z-order EnumDesktopWindows
    topmost_apps = []
    def enum_cb(hwnd, lp):
        if is_own_window(hwnd) or not user32.IsWindow(hwnd):
            return True
        det = get_window_details(hwnd)
        if det:
            topmost_apps.append(det)
            return False
        return True

    try:
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        if hdesk:
            user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_cb), 0)
        else:
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    except Exception:
        pass

    if topmost_apps:
        det = topmost_apps[0]
        return WindowContext(
            hwnd=det["hwnd"], title=det["title"],
            rect=det["rect"], width=det["w"], height=det["h"], is_valid=True,
            class_name=det.get("class", "")
        )

    # 3. Fallback: Cursor position window
    try:
        pt = wintypes.POINT()
        if user32.GetCursorPos(ctypes.byref(pt)):
            chwnd = user32.WindowFromPoint(pt)
            if chwnd and user32.IsWindow(chwnd):
                root_hwnd = user32.GetAncestor(chwnd, 2)  # GA_ROOT
                if root_hwnd and user32.IsWindow(root_hwnd) and not is_own_window(root_hwnd):
                    det = get_window_details(root_hwnd)
                    if det:
                        return WindowContext(
                            hwnd=det["hwnd"], title=det["title"],
                            rect=det["rect"], width=det["w"], height=det["h"], is_valid=True,
                            class_name=det.get("class", "")
                        )
    except Exception:
        pass

    return get_desktop_context()


def get_taskbar_window_details() -> Optional[Dict]:
    """Locates and extracts the Windows Taskbar (Shell_TrayWnd) geometry."""
    try:
        user32 = ctypes.windll.user32
        tb_hwnd = user32.FindWindowW("Shell_TrayWnd", None)
        if tb_hwnd and user32.IsWindow(tb_hwnd) and user32.IsWindowVisible(tb_hwnd):
            rect = wintypes.RECT()
            user32.GetWindowRect(tb_hwnd, ctypes.byref(rect))
            return {
                "hwnd": tb_hwnd,
                "title": "Windows Taskbar",
                "class": "Shell_TrayWnd",
                "rect": (rect.left, rect.top, rect.right, rect.bottom),
                "w": rect.right - rect.left,
                "h": rect.bottom - rect.top
            }
    except Exception:
        pass
    return None


def enumerate_all_windows(include_minimized: bool = True) -> List[Dict]:
    """
    Discovers all top-level application windows on the desktop.
    Includes active, background, and minimized/iconic windows (IsIconic).
    """
    user32 = ctypes.windll.user32
    results = []
    hdesk = None
    try:
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF) or user32.OpenDesktopW("default", 0, False, 0x01FF)
    except Exception:
        pass

    def enum_all_cb(hwnd, lp):
        try:
            if is_own_window(hwnd) or not user32.IsWindow(hwnd):
                return True
            if is_window_cloaked(hwnd):
                return True

            is_min = bool(user32.IsIconic(hwnd))
            is_vis = bool(user32.IsWindowVisible(hwnd))
            if not include_minimized and is_min:
                return True
            if not (is_vis or is_min):
                return True

            l = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if l > 0:
                tbuf = ctypes.create_unicode_buffer(l + 1)
                user32.GetWindowTextW(hwnd, tbuf, l + 1)
                title = tbuf.value.strip()

            cbuf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cbuf, 256)
            cname = cbuf.value.strip()

            if cname in ("Progman", "WorkerW", "tooltips_class32", "TkTopLevel", "MSCTFIME UI", "IME"):
                return True
            if title in ("Task Switching", "Windows Input Experience", "Program Manager", "Default IME", "MSCTFIME UI"):
                return True

            proc = get_process_name_for_hwnd(hwnd)
            if not title:
                p_lower = proc.lower() if proc else ""
                if "startmenu" in p_lower:
                    title = "Start Menu"
                elif "search" in p_lower:
                    title = "Windows Search"
                elif "shellexperience" in p_lower or cname == "Windows.UI.Core.CoreWindow":
                    title = "Start Menu / Windows Shell"
                elif proc and p_lower not in ("explorer.exe",):
                    title = f"[{proc.replace('.exe', '').capitalize()}]"
                else:
                    return True

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            rw = rect.right - rect.left
            rh = rect.bottom - rect.top
            if rw < 50 or rh < 50:
                return True

            results.append({
                "hwnd": hwnd,
                "title": title,
                "process": proc,
                "class": cname,
                "is_minimized": is_min,
                "rect": (rect.left, rect.top, rect.right, rect.bottom),
                "w": rw,
                "h": rh
            })
        except Exception:
            pass
        return True

    try:
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        if hdesk:
            user32.EnumDesktopWindows(hdesk, WNDENUMPROC(enum_all_cb), 0)
        else:
            user32.EnumWindows(WNDENUMPROC(enum_all_cb), 0)
    except Exception:
        pass
    return results


def bring_to_foreground(hwnd: int) -> bool:
    """
    Reliably brings a target window to the active foreground.
    Bypasses Windows Foreground Lock Timeout using AttachThreadInput and BringWindowToTop.
    """
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not hwnd or not user32.IsWindow(hwnd):
            return False

        # 1. Un-minimize if iconic
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        else:
            user32.ShowWindow(hwnd, SW_SHOW)

        # 2. Check if already foreground
        fg = user32.GetForegroundWindow()
        if fg == hwnd:
            return True

        # 3. Thread input attachment to bypass foreground lock timeout
        cur_tid = kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)

        attached = False
        if fg_tid and fg_tid != cur_tid:
            try:
                user32.AttachThreadInput(cur_tid, fg_tid, True)
                attached = True
            except Exception:
                pass

        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                try:
                    user32.AttachThreadInput(cur_tid, fg_tid, False)
                except Exception:
                    pass

        # 4. Fallback: Topmost toggle dance if still not in foreground
        if user32.GetForegroundWindow() != hwnd:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            user32.SetForegroundWindow(hwnd)

        return user32.GetForegroundWindow() == hwnd or user32.IsWindowVisible(hwnd)
    except Exception:
        return False


def restore_and_focus_window(hwnd: int) -> bool:
    """Restores a minimized or background window and brings it to the active foreground."""
    return bring_to_foreground(hwnd)


def minimize_window(hwnd: Optional[int] = None) -> bool:
    """
    Minimizes a window cleanly via Win32 ShowWindow without coordinate guesswork.
    If hwnd is omitted or 0, targets the current foreground window.
    Returns False if hwnd is invalid or belongs to our own UI.
    """
    try:
        user32 = ctypes.windll.user32
        target_hwnd = hwnd

        # Defensive check against explicit invalid handles (e.g. -99999)
        if target_hwnd is not None and target_hwnd < 0:
            return False

        if not target_hwnd or not user32.IsWindow(target_hwnd):
            target_hwnd = user32.GetForegroundWindow()

        if not target_hwnd or not user32.IsWindow(target_hwnd) or is_own_window(target_hwnd):
            return False

        user32.ShowWindow(target_hwnd, SW_MINIMIZE)
        return True
    except Exception:
        return False


def maximize_window(hwnd: Optional[int] = None) -> bool:
    """
    Maximizes a window cleanly via Win32 ShowWindow without coordinate guesswork.
    If hwnd is omitted or 0, targets the current foreground window.
    Returns False if hwnd is invalid or belongs to our own UI.
    """
    try:
        user32 = ctypes.windll.user32
        target_hwnd = hwnd

        # Defensive check against explicit invalid handles (e.g. -99999)
        if target_hwnd is not None and target_hwnd < 0:
            return False

        if not target_hwnd or not user32.IsWindow(target_hwnd):
            target_hwnd = user32.GetForegroundWindow()

        if not target_hwnd or not user32.IsWindow(target_hwnd) or is_own_window(target_hwnd):
            return False

        user32.ShowWindow(target_hwnd, SW_MAXIMIZE)
        bring_to_foreground(target_hwnd)
        return True
    except Exception:
        return False


def close_window(hwnd: Optional[int] = None) -> bool:
    """
    Closes a window cleanly via Win32 WM_SYSCOMMAND SC_CLOSE / WM_CLOSE message.
    If hwnd is omitted or 0, targets the current foreground window.
    Returns False if hwnd is invalid or belongs to our own UI.
    """
    try:
        user32 = ctypes.windll.user32
        target_hwnd = hwnd

        if target_hwnd is not None and target_hwnd < 0:
            return False

        if not target_hwnd or not user32.IsWindow(target_hwnd):
            target_hwnd = user32.GetForegroundWindow()

        if not target_hwnd or not user32.IsWindow(target_hwnd) or is_own_window(target_hwnd):
            return False

        WM_SYSCOMMAND = 0x0112
        SC_CLOSE = 0xF060
        WM_CLOSE = 0x0010
        user32.PostMessageW(target_hwnd, WM_SYSCOMMAND, SC_CLOSE, 0)
        user32.PostMessageW(target_hwnd, WM_CLOSE, 0, 0)
        return True
    except Exception:
        return False


def switch_to_window(target: Union[str, int]) -> Optional[int]:
    """
    Searches running desktop windows by title, process name, or HWND,
    and brings the matching window to the active foreground.
    Returns the HWND of the activated window, or None if not found.
    """
    try:
        user32 = ctypes.windll.user32
        if isinstance(target, int):
            if user32.IsWindow(target) and bring_to_foreground(target):
                return target
            return None

        if not isinstance(target, str) or not target.strip():
            return None

        tgt_lower = target.lower().strip()
        wins = enumerate_all_windows(include_minimized=True)

        # 1. Exact title match
        for w in wins:
            if w.get("title", "").lower() == tgt_lower:
                if bring_to_foreground(w["hwnd"]):
                    return w["hwnd"]

        # 2. Substring title match
        for w in wins:
            title_l = w.get("title", "").lower()
            if tgt_lower in title_l or title_l in tgt_lower:
                if bring_to_foreground(w["hwnd"]):
                    return w["hwnd"]

        # 3. Process name match (e.g. 'chrome', 'notepad', 'calc')
        for w in wins:
            proc_l = w.get("process", "").lower()
            proc_base = proc_l.replace(".exe", "")
            if tgt_lower == proc_l or tgt_lower == proc_base:
                if bring_to_foreground(w["hwnd"]):
                    return w["hwnd"]

        return None
    except Exception:
        return None
