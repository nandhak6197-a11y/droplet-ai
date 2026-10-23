import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def debug_pool_b():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[144]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    
    active = [d for d in droplets if d.category != 'surface_probe']
    d71 = next(d for d in active if d.id == 71)
    
    class MockCompletions:
        def create(self, **kwargs):
            class Resp:
                choices = [type('Msg', (), {'message': type('Content', (), {'content': '{"target_dot_id": 1, "action": "click", "reasoning": "mock"}'})()})]
            return Resp()
    for entry in reasoner.key_pool:
        entry['client'].chat = type('Chat', (), {'completions': MockCompletions()})()

    res = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, window_ctx=win, screenshot=img, mode='combined', target_model='qwen/qwen3.8-27b')
    cands = res.get('manifest_candidate_ids') or []
    
    print(f"Total candidates in manifest: {len(cands)}")
    near_d71 = [d for d in active if abs(d.center_x - 1159) < 200 and abs(d.center_y - 148) < 50]
    for nd in near_d71:
        print(f"nd [{nd.id}] ({nd.center_x},{nd.center_y}) in_cands={nd.id in cands} label={nd.label}")

asyncio.run(debug_pool_b())
