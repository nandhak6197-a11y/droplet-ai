import os, sys

target_path = 'spatial_droplet/platform/window.py'
with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

funcs = '''

def minimize_window(hwnd: int) -> bool:
    """Minimizes a window cleanly via Win32 ShowWindow."""
    try:
        user32 = ctypes.windll.user32
        if not hwnd or not user32.IsWindow(hwnd):
            return False
        SW_MINIMIZE = 6
        user32.ShowWindow(hwnd, SW_MINIMIZE)
        return True
    except Exception:
        return False


def maximize_window(hwnd: int) -> bool:
    """Maximizes a window cleanly via Win32 ShowWindow."""
    try:
        user32 = ctypes.windll.user32
        if not hwnd or not user32.IsWindow(hwnd):
            return False
        SW_MAXIMIZE = 3
        user32.ShowWindow(hwnd, SW_MAXIMIZE)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False
'''

if 'def minimize_window' not in content:
    content = content.rstrip() + funcs + '\n'
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Added minimize_window and maximize_window')
else:
    print('Already present')
