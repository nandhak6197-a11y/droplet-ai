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

async def test_qwen_with_d269():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[68] # Sample 69
    print(f"Task: {s['instruction']}")
    
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    # Let's inspect what happens if D269 has a neutral label [Icon] Dialog Button / Question / Help / Tool
    d269 = next((d for d in droplets if d.id == 269), None)
    d269.label = "[Icon] Toolbar Tool / Visual Button / Dialog Icon"
    # And give it a boost or include it
    d269._score = 1500
    
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, img.shape[1], img.shape[0]), width=img.shape[1], height=img.shape[0], is_valid=True)
    reasoner = DiagonalDropletReasoner()
    
    res = await reasoner.decide(
        user_intent=s['instruction'],
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    print("Decision with D269 in manifest:")
    print(f"  Target: ({res.target_x}, {res.target_y}), Dot ID: {res.get('target_droplet_id')}")
    print(f"  Reasoning: {res.reasoning}")
    cands = res.get('manifest_candidate_ids') or []
    print(f"  Is 269 in manifest? {269 in cands}")
    if res.get('target_droplet_id') == 269:
        print("  🎉 PERFECT HIT ON DOT 269!")

asyncio.run(test_qwen_with_d269())
