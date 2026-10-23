import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
import numpy as np
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

toolbars = [d for d in droplets if d.label.startswith('[Icon] Toolbar Tool')]
print(f"Total generic toolbar icons: {len(toolbars)}")
# Inspect the first 10
for i, d in enumerate(toolbars[:15]):
    box = d.bbox
    bw, bh = box[2] - box[0], box[3] - box[1]
    # compute local contrast in bounding box
    patch = img[max(0, box[1]):min(sh, box[3]), max(0, box[0]):min(sw, box[2])]
    std = float(np.std(patch)) if patch.size > 0 else 0
    print(f"  #{i+1}: ID={d.id} at ({d.center_x}, {d.center_y}), size={bw}x{bh}, std={std:.1f}, label={d.label[:50]}")
