import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2, json, asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer

with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)
s91 = samples[90]
img = cv2.imread(str(Path('osworld_g_images') / s91['image_path']))
sprayer = FullBleedLatticeSprayer(48)
droplets = asyncio.run(sprayer.spray(img))
for d in droplets:
    if 100 <= d.center_y <= 250:
        print(f"({d.center_x}, {d.center_y}): \"{d.label}\"")
