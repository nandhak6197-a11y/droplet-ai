import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
import numpy as np
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext
from run_qwen_200_diagnostic import convert_gt_box, is_point_in_box, distance_to_box

with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
    samples = json.load(f)[:20]

print("Evaluating Phase 1 Saliency Sprayer on 20 samples...", flush=True)
sprayer = FullBleedLatticeSprayer(48)

total_droplets = []
recalls = []
misses = []

for idx, s in enumerate(samples):
    img = cv2.imread(str(Path("osworld_g_images") / s["image_path"]))
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
    gt_box = convert_gt_box(s["box_coordinates"])
    
    droplets = asyncio.run(sprayer.spray(img, win_ctx=win, offline_frame=True))
    total_droplets.append(len(droplets))
    
    active = [d for d in droplets if d.category != 'surface_probe']
    in_gt = any(is_point_in_box(d.center_x, d.center_y, gt_box) for d in active)
    closest = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in active), default=999)
    hit = in_gt or (closest <= 24.0)
    recalls.append(hit)
    
    status = "✔ HIT" if hit else "❌ MISS"
    probes = sum(1 for d in droplets if d.category == 'surface_probe')
    ui_cnt = len(droplets) - probes
    print(f"[{idx+1:02d}/20] {status} | Droplets: {len(droplets):3d} (UI: {ui_cnt:3d}, Probes: {probes:2d}) | Dist: {closest:4.1f}px | '{s['instruction'][:45]}'", flush=True)
    if not hit:
        misses.append(idx+1)

print("-" * 80, flush=True)
print(f"Phase 1 Mean Droplets: {np.mean(total_droplets):.1f} (min={min(total_droplets)}, max={max(total_droplets)})", flush=True)
print(f"Phase 1 Proposal Recall: {sum(recalls)}/{len(recalls)} ({sum(recalls)/len(recalls)*100:.1f}%)", flush=True)
if misses:
    print(f"Missed samples: {misses}", flush=True)
else:
    print("🏆 PERFECT RECALL: 100.0% of ground-truth targets detected!", flush=True)
