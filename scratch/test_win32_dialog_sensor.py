import sys
import ctypes
from ctypes import wintypes

sys.stdout.reconfigure(encoding='utf-8')

user32 = ctypes.windll.user32

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

def get_child_buttons(parent_hwnd):
    results = []
    
    def _enum_child(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        cname = class_buf.value
        
        # Win32 Buttons (Buttons, Checkboxes, Radio, SplitButton, DirectUI controls)
        if cname in ("Button", "CCtrlButton", "CtrlButton", "DirectUIHWND"):
            text_buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, text_buf, 256)
            txt = text_buf.value.strip()
            
            # Clean hotkey accelerators like "&Yes" -> "Yes", "&No" -> "No", "&Save" -> "Save"
            txt_clean = txt.replace("&", "").strip()
            
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 10 and h > 10:
                cx = (rect.left + rect.right) // 2
                cy = (rect.top + rect.bottom) // 2
                results.append((cx, cy, txt_clean or cname, cname, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumChildWindows(parent_hwnd, WNDENUMPROC(_enum_child), 0)
    return results

print("Win32 Dialog Button Sensor defined successfully.")
