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

TEST_INDICES = [68, 41, 57, 64, 144]

async def test_pipeline():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    reasoner = DiagonalDropletReasoner()
    
    for s_idx in TEST_INDICES:
        s = samples[s_idx]
        print("=" * 80)
        print(f"Sample {s_idx + 1}: Task = '{s['instruction']}'")
        img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
        sh, sw = img.shape[:2]
        
        # Spray droplets with fresh ID counter per sample
        sprayer = FullBleedLatticeSprayer(48)
        droplets = await sprayer.spray(img)
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
        
        # Run Qwen
        res = await reasoner.decide(
            user_intent=s['instruction'],
            droplets=droplets,
            window_ctx=win,
            screenshot=img,
            mode="combined",
            target_model="qwen/qwen3.8-27b"
        )
        
        gt_coords = s['box_coordinates']
        if len(gt_coords) == 4:
            x1, y1, w, h = gt_coords
            gt_box = (x1, y1, x1 + w, y1 + h)
        else:
            xs = gt_coords[0::2]; ys = gt_coords[1::2]
            gt_box = (min(xs), min(ys), max(xs), max(ys))
            
        px, py = res.target_x, res.target_y
        cands = res.get('manifest_candidate_ids') or []
        
        # Find which droplet in active_droplets was in GT box
        active = [d for d in droplets if d.category != 'surface_probe']
        gt_dots = [d for d in active if gt_box[0] <= d.center_x <= gt_box[2] and gt_box[1] <= d.center_y <= gt_box[3]]
        gt_id = gt_dots[0].id if gt_dots else None
        
        is_in_manifest = any(d.id in cands for d in gt_dots) if gt_dots else False
        is_hit = (px is not None and gt_box[0] <= px <= gt_box[2] and gt_box[1] <= py <= gt_box[3])
        
        print(f"  Target: ({px}, {py}), Pred Dot: {res.get('target_droplet_id')}")
        print(f"  GT Dot ID: {gt_id}, Was in manifest? {is_in_manifest} (Manifest total: {len(cands)})")
        print(f"  Strict Hit: {is_hit}")
        print(f"  Reasoning: {res.reasoning[:120]}")

asyncio.run(test_pipeline())
