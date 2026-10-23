import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def trace_sample(s_idx):
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[s_idx]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    gt_coords = s['box_coordinates']
    if len(gt_coords) == 4:
        x1, y1, w, h = gt_coords
        gt_box = (x1, y1, x1 + w, y1 + h)
    else:
        xs = gt_coords[0::2]; ys = gt_coords[1::2]
        gt_box = (min(xs), min(ys), max(xs), max(ys))
    
    active = [d for d in droplets if d.category != 'surface_probe']
    gt_dots = [d for d in active if gt_box[0] <= d.center_x <= gt_box[2] and gt_box[1] <= d.center_y <= gt_box[3]]
    gt_d = gt_dots[0] if gt_dots else None
    
    print("=" * 80)
    print(f"Sample {s_idx + 1}: Task = '{s['instruction']}'")
    if not gt_d:
        print("  NO GT DOT FOUND IN SPRAY!")
        return
    print(f"  GT Dot ID={gt_d.id} at ({gt_d.center_x},{gt_d.center_y}), Label='{gt_d.label}', Archetype={getattr(gt_d, 'archetype', None)}")
    
    # We can inspect how the reasoner ranks it
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    
    # Capture candidates
    cands_captured = []
    class MockChoices:
        def __init__(self):
            self.message = type('Msg', (), {'content': '{"target_dot_id": 1, "action": "click", "reasoning": "mock"}'})()
    class MockResp:
        def __init__(self):
            self.choices = [MockChoices()]
    class MockCompletions:
        def create(self, **kwargs):
            return MockResp()
    class MockClient:
        chat = type('Chat', (), {'completions': MockCompletions()})()
    for entry in reasoner.key_pool:
        entry['client'] = MockClient()
        
    res = await reasoner.decide(
        user_intent=s['instruction'],
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    cands = res.get('manifest_candidate_ids') or []
    print(f"  GT Dot in manifest? {gt_d.id in cands} (Total cands: {len(cands)})")
    if gt_d.id not in cands:
        print(f"  First 10 cands in manifest: {cands[:10]}")

async def main():
    for s_idx in [68, 41, 57, 64, 144]:
        await trace_sample(s_idx)

asyncio.run(main())
