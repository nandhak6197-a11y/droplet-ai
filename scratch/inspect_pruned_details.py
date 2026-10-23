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
from run_qwen_200_diagnostic import convert_gt_box, is_point_in_box, distance_to_box

async def main():
    with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    pruned_indices = [9, 12, 16, 17, 18] # 1-indexed in report: 9, 12, 16, 17, 18 -> 0-indexed: 8, 11, 15, 16, 17
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
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
        dots_in_gt = [d for d in active_droplets if is_point_in_box(d.center_x, d.center_y, gt_box)]
        
        closest_dot = min(active_droplets, key=lambda d: distance_to_box(d.center_x, d.center_y, gt_box))
        closest_dist = distance_to_box(closest_dot.center_x, closest_dot.center_y, gt_box)
        
        print(f"\n--- Sample {s_idx+1}: {instruction} ---")
        print(f"GT Box: {gt_box}")
        print(f"Closest Dot: ID={closest_dot.id} at ({closest_dot.center_x},{closest_dot.center_y}) Dist={closest_dist:.1f}px")
        print(f"   Label: '{closest_dot.label}' | Archetype: {closest_dot.archetype} | Role: {closest_dot.semantic_role}")
        
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
        # Mock decide to see manifest
        # We can run decide with dry_run or just call it
        res = await reasoner.decide(
            user_intent=instruction,
            droplets=droplets,
            window_ctx=win,
            mode="combined",
            target_model="qwen/qwen3.8-27b"
        )
        cand_ids = res.get("manifest_candidate_ids", [])
        print(f"   In Manifest (68 candidates)? {closest_dot.id in cand_ids}")
        if closest_dot.id not in cand_ids:
            print(f"   Top 10 Manifest IDs: {cand_ids[:10]}")
        print(f"   Total droplets: {len(droplets)}")

asyncio.run(main())
