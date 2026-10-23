import os
from pathlib import Path

with open("live_copilot.py", "r", encoding="utf-8") as f:
    content = f.read()

target = 'PROJECT_ROOT = r"c:\\Users\\NANDHA K\\Documents\\antigravity\\amazing-bose"'
replacement = 'PROJECT_ROOT = str(Path(__file__).resolve().parent)'

assert target in content, "Target PROJECT_ROOT line not found"
new_content = content.replace(target, replacement, 1)

new_content = new_content.replace(
    "SPATIAL DROPLET — LIVE DESKTOP COPILOT & TRANSPARENT HUD OVERLAY",
    "DROPLET-AI — AUTONOMOUS DESKTOP COPILOT & SPATIAL HUD OVERLAY",
    1
)

with open("droplet_ai.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Created droplet_ai.py ({len(new_content)} characters).")
