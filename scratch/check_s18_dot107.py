import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from run_qwen_200_diagnostic import convert_gt_box, distance_to_box, is_point_in_box

async def test():
    with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    s = samples[17] # Sample 18
    img = cv2.imread(str(Path("osworld_g_images") / s["image_path"]))
    gt_box = convert_gt_box(s["box_coordinates"])
    
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    dot107 = next((d for d in droplets if d.id == 107), None)
    if dot107:
        dist_b = distance_to_box(dot107.center_x, dot107.center_y, gt_box)
        strict = is_point_in_box(dot107.center_x, dot107.center_y, gt_box)
        print(f"Sample 18 Dot 107: ({dot107.center_x}, {dot107.center_y})")
        print(f"GT Box: {gt_box}")
        print(f"Dist to box: {dist_b:.2f}px | Strict: {strict}")
        print(f"Label: {dot107.label}")

asyncio.run(test())
