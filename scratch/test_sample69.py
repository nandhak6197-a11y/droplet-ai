import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def test_sample():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Sample 69 is index 68
    s = samples[68]
    print(f"Sample 69 instruction: '{s['instruction']}'")
    
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    # Find droplet 269
    d269 = next((d for d in droplets if d.id == 269), None)
    print(f"Droplet 269 found: {d269.label} at ({d269.center_x}, {d269.center_y})")
    
    reasoner = DiagonalDropletReasoner()
    # We can inspect sorted_nodes by creating a dummy call or inspecting scoring
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, img.shape[1], img.shape[0]), width=img.shape[1], height=img.shape[0], is_valid=True)
    
    # Let's see what candidates get selected
    # We can run decide with a mock or inspect all_candidates
    # Let's inspect how droplets are scored
    active_droplets = [d for d in droplets if d.category != 'surface_probe']
    print(f"Total active droplets: {len(active_droplets)}")

asyncio.run(test_sample())
