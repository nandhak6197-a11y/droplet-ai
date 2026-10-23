import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
import numpy as np
import math
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext
from run_qwen_200_diagnostic import convert_gt_box, is_point_in_box, distance_to_box

with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
    samples = json.load(f)[:20]

print("Testing Saliency Filter on 20 samples with Affordance-Grounded Probes...", flush=True)

class SaliencySprayer(FullBleedLatticeSprayer):
    async def spray(self, frame_bgr, win_ctx=None, copilot_rect=None, offline_frame=False):
        droplets = await super().spray(frame_bgr, win_ctx=win_ctx, copilot_rect=copilot_rect, offline_frame=offline_frame)
        h, w = frame_bgr.shape[:2]
        
        # Saliency filtering on droplets:
        # Separate UI landmarks from surface probes
        ui_droplets = [d for d in droplets if d.category != 'surface_probe']
        probes = [d for d in droplets if d.category == 'surface_probe']
        
        # 1. Affordance-grounded probe culling:
        # Probes must be within an affordance radius of at least one interactive control
        affordance_radius = min(w, h) * 0.22  # ~230px on 1080p
        culled_probes = []
        for p in probes:
            # Check proximity to nearest UI control
            if ui_droplets:
                min_dist = min(math.hypot(p.center_x - u.center_x, p.center_y - u.center_y) for u in ui_droplets)
                if min_dist <= affordance_radius:
                    culled_probes.append(p)
            else:
                culled_probes.append(p)
                
        # Dynamically scale probe budget: if screen has rich UI (>=40 elements), keep at most 6 probes
        max_probes = 6 if len(ui_droplets) >= 40 else (12 if len(ui_droplets) >= 20 else 20)
        culled_probes = culled_probes[:max_probes]
        
        # 2. Icon Saliency Culling for Generic Fallback Icons
        # If an icon is classified as generic 'Toolbar Tool / Visual Button',
        # verify it has real local contrast and is not an isolated faint speck on blank canvas
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        culled_ui = []
        for d in ui_droplets:
            if d.label == "[Icon] Toolbar Tool / Visual Button":
                box = d.bbox
                if box and box[2] > box[0] and box[3] > box[1]:
                    patch = gray_frame[max(0, box[1]):min(h, box[3]), max(0, box[0]):min(w, box[2])]
                    if patch.size > 0:
                        std = float(np.std(patch))
                        # Suppress faint low-contrast noise artifacts
                        if std < 7.0:
                            continue
            culled_ui.append(d)
            
        return culled_ui + culled_probes

sprayer = SaliencySprayer(48)
counts = []
recalls = []

for idx, s in enumerate(samples):
    img = cv2.imread(str(Path("osworld_g_images") / s["image_path"]))
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
    gt_box = convert_gt_box(s["box_coordinates"])
    
    droplets = asyncio.run(sprayer.spray(img, win_ctx=win, offline_frame=True))
    counts.append(len(droplets))
    
    active = [d for d in droplets if d.category != 'surface_probe']
    in_gt = any(is_point_in_box(d.center_x, d.center_y, gt_box) for d in active)
    closest = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in active), default=999)
    hit = in_gt or (closest <= 24.0)
    recalls.append(hit)

print(f"Saliency Sprayer Mean Droplets: {np.mean(counts):.1f} (min={min(counts)}, max={max(counts)})", flush=True)
print(f"Saliency Sprayer Recall: {sum(recalls)}/{len(recalls)} ({sum(recalls)/len(recalls)*100:.1f}%)", flush=True)
