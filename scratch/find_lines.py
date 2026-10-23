import os
with open('spatial_droplet/reasoning/reasoner.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'def decide_turn(' in line or 'system_instruction = f"""<system_capability>' in line:
            print(f'{i}: {line.strip()[:80]}')
