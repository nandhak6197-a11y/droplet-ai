import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
import json
import math
import cv2
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner

def convert_gt_box(box_coords):
    if len(box_coords) == 4:
        x, y, w, h = box_coords
        return x, y, x + w, y + h
    xs = box_coords[0::2]
    ys = box_coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def distance_to_box(px, py, box):
    x1, y1, x2, y2 = box
    dx = max(0.0, x1 - px, px - x2)
    dy = max(0.0, y1 - py, py - y2)
    return math.hypot(dx, dy)

async def main():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)[:100]

    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()

    raw_strict = 0
    raw_tol = 0
    man_strict = 0
    man_tol = 0
    missed_by_sprayer = []
    missed_by_manifest = []

    for idx, s in enumerate(samples):
        img_path = Path('osworld_g_images') / s['image_path']
        img = cv2.imread(str(img_path))
        gt_box = convert_gt_box(s['box_coordinates'])
        droplets = await sprayer.spray(img, offline_frame=True)
        ui_droplets = [d for d in droplets if d.category != 'surface_probe']

        min_d_raw = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in ui_droplets), default=999.0)
        if min_d_raw == 0.0:
            raw_strict += 1
        if min_d_raw <= 24.0:
            raw_tol += 1
        else:
            missed_by_sprayer.append((idx + 1, min_d_raw, s['instruction']))

        trace = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, screenshot=img, candidate_trace_only=True)
        cand_ids = set(trace.get('manifest_candidate_ids', []))
        cand_droplets = [d for d in droplets if d.id in cand_ids]
        min_d_man = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in cand_droplets), default=999.0)
        if min_d_man == 0.0:
            man_strict += 1
        if min_d_man <= 24.0:
            man_tol += 1
        elif min_d_raw <= 24.0:
            missed_by_manifest.append((idx + 1, min_d_raw, min_d_man, s['instruction']))

    print(f"=== 100-SAMPLE ORACLE RECALL AUDIT ===")
    print(f"Sprayer Raw Strict Recall (0px to GT box):       {raw_strict}/100")
    print(f"Sprayer Raw Tolerant Recall (<=24px to GT box):  {raw_tol}/100")
    print(f"96-Cand Manifest Strict Recall (0px to GT box):  {man_strict}/100")
    print(f"96-Cand Manifest Tolerant Recall (<=24px):       {man_tol}/100")
    print(f"\nMissed by Sprayer (>24px): {len(missed_by_sprayer)}")
    for item in missed_by_sprayer:
        print(f"  Sample {item[0]:03d}: raw_dist={item[1]:.1f}px | {item[2][:65]}")
    print(f"\nDetected by Sprayer (<=24px) but Pruned by 96-Cand Manifest: {len(missed_by_manifest)}")
    for item in missed_by_manifest:
        print(f"  Sample {item[0]:03d}: raw_dist={item[1]:.1f}px -> man_dist={item[2]:.1f}px | {item[3][:65]}")

asyncio.run(main())
