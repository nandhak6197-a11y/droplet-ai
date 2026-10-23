import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def test_formatting_row():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[144]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    d71 = next(d for d in droplets if abs(d.center_x - 1159) < 10 and abs(d.center_y - 148) < 10)
    
    # We can temporarily patch decide's relevance_score
    # Let's inspect get_score of d71 with formatting row bonus
    print(f"d71 ID={d71.id} at ({d71.center_x},{d71.center_y})")

asyncio.run(test_formatting_row())
