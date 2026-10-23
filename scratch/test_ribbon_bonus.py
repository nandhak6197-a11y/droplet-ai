import asyncio
import cv2
import json
import sys
import re
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def test_bonus():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    # Test Sample 145
    s145 = samples[144]
    img145 = cv2.imread(str(Path('osworld_g_images') / s145['image_path']))
    d145 = await sprayer.spray(img145)
    win145 = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, img145.shape[1], img145.shape[0]), width=img145.shape[1], height=img145.shape[0], is_valid=True)
    
    # Test Sample 69
    s69 = samples[68]
    img69 = cv2.imread(str(Path('osworld_g_images') / s69['image_path']))
    d69 = await sprayer.spray(img69)
    win69 = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, img69.shape[1], img69.shape[0]), width=img69.shape[1], height=img69.shape[0], is_valid=True)

    print("Setup completed successfully.")

asyncio.run(test_bonus())
