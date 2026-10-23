import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def debug_scores():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[68]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    gt_d = next(d for d in droplets if 724 <= d.center_x <= 771 and 501 <= d.center_y <= 545)
    
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    
    # We can inspect inside decide by calling it with a dummy target_model
    class MockCompletions:
        def create(self, **kwargs):
            class Resp:
                choices = [type('Msg', (), {'message': type('Content', (), {'content': '{"target_dot_id": 1, "action": "click", "reasoning": "mock"}'})()})]
            return Resp()
    for entry in reasoner.key_pool:
        entry['client'].chat = type('Chat', (), {'completions': MockCompletions()})()

    # Let's inspect active_droplets
    active = [d for d in droplets if d.category != 'surface_probe']
    print(f"Total active droplets: {len(active)}")
    
    # Run decide
    res = await reasoner.decide(
        user_intent=s['instruction'],
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    cands = res.get('manifest_candidate_ids') or []
    print(f"Target Dot ID: {gt_d.id} at ({gt_d.center_x},{gt_d.center_y})")
    print(f"Is Target Dot in manifest? {gt_d.id in cands}")
    print(f"Manifest candidates ({len(cands)}):")
    for cid in cands:
        cd = next(d for d in droplets if d.id == cid)
        print(f"  • [{cd.id}] ({cd.center_x},{cd.center_y}) {cd.label}")

asyncio.run(debug_scores())
