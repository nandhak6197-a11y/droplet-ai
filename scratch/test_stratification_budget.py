import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext
from run_qwen_200_diagnostic import convert_gt_box, distance_to_box

async def test():
    with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    # Check sample 9 (index 8), sample 12 (index 11), sample 16 (index 15), sample 17 (index 16), sample 18 (index 17)
    for s_idx in [8, 11, 15, 16, 17]:
        s = samples[s_idx]
        img_name = s["image_path"]
        img_p = Path("osworld_g_images") / img_name
        img = cv2.imread(str(img_p))
        sh, sw = img.shape[:2]
        gt_box = convert_gt_box(s["box_coordinates"])
        instruction = s["instruction"]
        
        droplets = await sprayer.spray(img)
        active_droplets = [d for d in droplets if d.category != 'surface_probe']
        
        closest_dot = min(active_droplets, key=lambda d: distance_to_box(d.center_x, d.center_y, gt_box))
        closest_dist = distance_to_box(closest_dot.center_x, closest_dot.center_y, gt_box)
        
        # Test current candidate selection with cand_budget 68 vs 110
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
        sw_screen, sh_screen = sw, sh
        key_nodes = active_droplets
        
        # Let's run decide with target_model
        res = await reasoner.decide(
            user_intent=instruction,
            droplets=droplets,
            window_ctx=win,
            mode="combined",
            target_model="qwen/qwen3.8-27b"
        )
        cand_ids = res.get("manifest_candidate_ids", [])
        print(f"Sample {s_idx+1}: '{instruction[:40]}'")
        print(f"   Closest Dot: ID={closest_dot.id} Dist={closest_dist:.1f}px | In Manifest (curr budget={len(cand_ids)})? {closest_dot.id in cand_ids}")

asyncio.run(test())
