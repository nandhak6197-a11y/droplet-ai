with open('spatial_droplet/platform/actuator.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        non_spatial_actions = {
            "wait", "hotkey", "chord", "press", "key_down", "key_up",
            "switch_window", "win_run", "copy", "paste", "select_all",
            "undo", "redo", "save", "find", "close_tab", "new_tab",
            "launch_app", "open_path", "system_open", "system_open_path", "stop", "finish"
        }'''

replacement = '''        non_spatial_actions = {
            "wait", "hotkey", "chord", "press", "key_down", "key_up",
            "switch_window", "win_run", "copy", "paste", "select_all",
            "undo", "redo", "save", "find", "close_tab", "new_tab",
            "launch_app", "open_path", "system_open", "system_open_path", "stop", "finish",
            "minimize", "minimize_window", "maximize", "maximize_window"
        }'''

# Handle both CRLF and LF
if target not in content:
    target_crlf = target.replace('\n', '\r\n')
    replacement_crlf = replacement.replace('\n', '\r\n')
    assert target_crlf in content, "Target not found in actuator.py"
    new_content = content.replace(target_crlf, replacement_crlf, 1)
else:
    new_content = content.replace(target, replacement, 1)

with open('spatial_droplet/platform/actuator.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Successfully patched non_spatial_actions in actuator.py")
