import os

target_path = 'spatial_droplet/platform/actuator.py'
with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

target = "        elif act_clean in ('drop', 'release'):"
replacement = """        elif act_clean in ('minimize', 'minimize_window'):
            print(f"[Actuator] [Minimize Window]: Active window context")
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
            print(f"[Actuator] [Maximize Window]: Active window context")
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

        elif act_clean in ('drop', 'release'):"""

if "elif act_clean in ('minimize', 'minimize_window'):" not in content:
    assert target in content, "Target not found in actuator.py"
    content = content.replace(target, replacement, 1)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added minimize and maximize to actuator.py")
else:
    print("Already present in actuator.py")
