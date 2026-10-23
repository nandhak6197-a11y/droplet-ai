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

async def main():
    with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    sprayer = FullBleedLatticeSprayer(48)
    
    for s_idx in [16, 17]: # Samples 17 and 18
        s = samples[s_idx]
        img_p = Path("osworld_g_images") / s["image_path"]
        img = cv2.imread(str(img_p))
        sh, sw = img.shape[:2]
        gt_box = convert_gt_box(s["box_coordinates"])
        instruction = s["instruction"]
        
        droplets = await sprayer.spray(img)
        active_droplets = [d for d in droplets if d.category != 'surface_probe']
        
        closest_dot = min(active_droplets, key=lambda d: distance_to_box(d.center_x, d.center_y, gt_box))
        
        # Calculate cell
        num_cols, num_rows = 4, 3
        cell_w = max(1, sw // num_cols)
        cell_h = max(1, sh // num_rows)
        gx = min(num_cols - 1, max(0, int(closest_dot.center_x) // cell_w))
        gy = min(num_rows - 1, max(0, int(closest_dot.center_y) // cell_h))
        
        print(f"\nSample {s_idx+1}: '{instruction}'")
        print(f"Closest Dot: ID={closest_dot.id} at ({closest_dot.center_x},{closest_dot.center_y})")
        print(f"   Cell: ({gx}, {gy}) in {num_cols}x{num_rows} grid")
        print(f"   Label: {closest_dot.label} | Archetype: {closest_dot.archetype} | Role: {closest_dot.semantic_role}")

asyncio.run(main())
