import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def debug_why_not_in_manifest():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    # 1. Inspect Sample 69 (index 68)
    s69 = samples[68]
    img69 = cv2.imread(str(Path('osworld_g_images') / s69['image_path']))
    d69 = await sprayer.spray(img69)
    # Find droplet 228 or GT droplet: (747, 523)
    gt_d69 = next((d for d in d69 if 724 <= d.center_x <= 771 and 501 <= d.center_y <= 545), None)
    print(f"Sample 69 GT Droplet: ID {gt_d69.id if gt_d69 else None} Category: {gt_d69.category} Label: '{gt_d69.label}'")
    
    # 2. Inspect Sample 145 (index 144)
    s145 = samples[144]
    img145 = cv2.imread(str(Path('osworld_g_images') / s145['image_path']))
    d145 = await sprayer.spray(img145)
    gt_d145 = next((d for d in d145 if 1140 <= d.center_x <= 1180 and 130 <= d.center_y <= 165), None)
    print(f"Sample 145 GT Droplet: ID {gt_d145.id if gt_d145 else None} Category: {gt_d145.category} Label: '{gt_d145.label}'")
    
    # Let's inspect where they were ranked in sorted_nodes!
    # In reasoner.decide, let's see their scores
    # Why were they not in Pool A, Pool B, or Pool C?

asyncio.run(debug_why_not_in_manifest())
