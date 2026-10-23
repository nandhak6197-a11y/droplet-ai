import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import numpy as np
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.vision.icon_detector import EnhancedIconDetector
from run_qwen_200_diagnostic import convert_gt_box, distance_to_box, is_point_in_box

async def diagnose_miss(sample_idx):
    with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    s = samples[sample_idx]
    img_name = s["image_path"]
    img_p = Path("osworld_g_images") / img_name
    img = cv2.imread(str(img_p))
    sh, sw = img.shape[:2]
    gt_box = convert_gt_box(s["box_coordinates"])
    instruction = s["instruction"]
    
    print("=" * 80)
    print(f"DIAGNOSING SAMPLE {sample_idx+1}: '{instruction}'")
    print(f"Image: {img_name} ({sw}x{sh})")
    print(f"GT Box: {gt_box} | Center: ({(gt_box[0]+gt_box[2])/2:.1f}, {(gt_box[1]+gt_box[3])/2:.1f})")
    print(f"GT Box Size: W={gt_box[2]-gt_box[0]:.1f} x H={gt_box[3]-gt_box[1]:.1f}")
    
    # Crop GT Box region with 40px margin
    x1, y1, x2, y2 = [int(v) for v in gt_box]
    pad = 40
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(sw, x2 + pad), min(sh, y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    
    crop_path = f"scratch/gt_crop_sample_{sample_idx+1}.png"
    cv2.imwrite(crop_path, crop)
    print(f"Saved GT Box visual crop to: {crop_path}")
    
    # Run sprayer step-by-step
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    active_droplets = [d for d in droplets if d.category != 'surface_probe']
    
    # Find all droplets within 100px
    nearby_dots = []
    for d in active_droplets:
        d_box = distance_to_box(d.center_x, d.center_y, gt_box)
        if d_box <= 100:
            nearby_dots.append((d, d_box))
    
    nearby_dots.sort(key=lambda x: x[1])
    print(f"\nDroplets within 100px of GT box ({len(nearby_dots)} found):")
    for d, dist in nearby_dots[:5]:
        print(f"   • Dot [{d.id}] at ({d.center_x}, {d.center_y}) | Dist: {dist:.1f}px | Label: '{d.label}' | Archetype: {d.archetype}")
    
    if not nearby_dots:
        print("   ❌ ZERO droplets found within 100px!")
        
    # Test EnhancedIconDetector directly on the region
    icons = EnhancedIconDetector.detect_icons(img)
    print(f"\nTotal raw icons detected across screen: {len(icons)}")
    gt_icons = [ic for ic in icons if (x1 - 20 <= ic[0] <= x2 + 20 and y1 - 20 <= ic[1] <= y2 + 20)]
    print(f"Raw icons detected within/near GT box: {len(gt_icons)}")
    for ic in gt_icons:
        print(f"   • Icon at ({ic[0]}, {ic[1]}) | Label: '{ic[2]}' | Sub: '{ic[4]}' | BBox: {ic[5]}")

import asyncio
async def main():
    for idx in [3, 4, 10]: # Samples 4, 5, 11 (0-indexed: 3, 4, 10)
        await diagnose_miss(idx)

if __name__ == "__main__":
    asyncio.run(main())
