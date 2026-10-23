import os

target_path = 'spatial_droplet/engine/trajectory_engine.py'
with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """            elif action in ("type_text", "write_text"):
                action = "type" """
replacement1 = """            elif action in ("type_text", "write_text"):
                action = "type"
            elif action in ("minimize", "minimize_window"):
                action = "minimize"
            elif action in ("maximize", "maximize_window"):
                action = "maximize" """

target2 = """                "close_tab", "new_tab", "switch_window", "win_run"
            }"""
replacement2 = """                "close_tab", "new_tab", "switch_window", "win_run",
                "minimize", "minimize_window", "maximize", "maximize_window"
            }"""

if '"minimize"' not in content:
    assert target2 in content, "Target2 not found in trajectory_engine.py"
    content = content.replace(target2, replacement2, 1)
    if 'elif action in ("type_text", "write_text"):' in content:
        # replace cleanly
        old_part = '            elif action in ("type_text", "write_text"):\n                action = "type"'
        new_part = '            elif action in ("type_text", "write_text"):\n                action = "type"\n            elif action in ("minimize", "minimize_window"):\n                action = "minimize"\n            elif action in ("maximize", "maximize_window"):\n                action = "maximize"'
        content = content.replace(old_part, new_part, 1)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added minimize/maximize actions to trajectory_engine.py")
else:
    print("Already present in trajectory_engine.py")
