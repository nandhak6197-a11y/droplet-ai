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
    
    for s_idx in [15, 17]: # Samples 16 and 18
        s = samples[s_idx]
        img_name = s["image_path"]
        img_p = Path("osworld_g_images") / img_name
        img = cv2.imread(str(img_p))
        sh, sw = img.shape[:2]
        gt_box = convert_gt_box(s["box_coordinates"])
        instruction = s["instruction"]
        
        droplets = await sprayer.spray(img)
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
        
        res = await reasoner.decide(
            user_intent=instruction,
            droplets=droplets,
            window_ctx=win,
            mode="combined",
            target_model="qwen/qwen3.8-27b"
        )
        print(f"\nSample {s_idx+1}: '{instruction}'")
        print(f"Error: {res.get('error')}")
        print(f"Model used: {res.get('model_used')}")
        print(f"Key used: {res.get('key_used')}")
        print(f"Target Dot ID: {res.target_droplet_id}")
        print(f"Manifest candidates count: {len(res.get('manifest_candidate_ids', []))}")

asyncio.run(test())
