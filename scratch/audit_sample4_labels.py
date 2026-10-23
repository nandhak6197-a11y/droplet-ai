import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext

with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    s = json.load(f)[3] # sample 4

img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
sh, sw = img.shape[:2]
win = WindowContext(hwnd=0, title='OSWorld-G', rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
sprayer = FullBleedLatticeSprayer(48)
droplets = asyncio.run(sprayer.spray(img, win_ctx=win, offline_frame=True))

print(f"Total droplets on Sample 4: {len(droplets)}")
# Count by label pattern
label_types = {}
for d in droplets:
    lbl = d.label
    if d.category == 'surface_probe':
        key = 'surface_probe'
    elif lbl.startswith('[Icon]'):
        # extract base icon type
        base = lbl.split()[1] if len(lbl.split()) > 1 else 'Icon'
        key = f'[Icon] {base}'
    elif lbl.startswith('[Btn]'):
        key = '[Btn]'
    else:
        key = 'OCR text'
    label_types[key] = label_types.get(key, 0) + 1

for k, v in sorted(label_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:<35}: {v}")
