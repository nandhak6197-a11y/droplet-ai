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
    
    sprayer = FullBleedLatticeSprayer(48)
    
    # We will test the 5 pruned samples: 9, 12, 16, 17, 18 (0-indexed: 8, 11, 15, 16, 17)
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
        
        print(f"Sample {s_idx+1}: '{instruction}'")
        print(f"   Closest Dot: ID={closest_dot.id} at ({closest_dot.center_x},{closest_dot.center_y}) Dist={closest_dist:.1f}px")
        print(f"   Label: '{closest_dot.label}'")

asyncio.run(main())
