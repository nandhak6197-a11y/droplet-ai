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
        samples = json.load(f)[:25]  # Fast sample of 25 to check sprayer recall

    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()

    raw_strict_recall = 0
    raw_tol_recall = 0
    manifest_strict_recall = 0
    manifest_tol_recall = 0

    for idx, s in enumerate(samples):
        img_path = Path('osworld_g_images') / s['image_path']
        img = cv2.imread(str(img_path))
        gt_box = convert_gt_box(s['box_coordinates'])
        droplets = await sprayer.spray(img, offline_frame=True)
        ui_droplets = [d for d in droplets if d.category != 'surface_probe']

        # Check if any ui_droplet is inside or <=24px of gt_box
        min_d_raw = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in ui_droplets), default=999.0)
        if min_d_raw == 0.0:
            raw_strict_recall += 1
        if min_d_raw <= 24.0:
            raw_tol_recall += 1

        # Check manifest candidate trace
        trace = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, screenshot=img, candidate_trace_only=True)
        cand_ids = set(trace.get('manifest_candidate_ids', []))
        cand_droplets = [d for d in droplets if d.id in cand_ids]
        min_d_man = min((distance_to_box(d.center_x, d.center_y, gt_box) for d in cand_droplets), default=999.0)
        if min_d_man == 0.0:
            manifest_strict_recall += 1
        if min_d_man <= 24.0:
            manifest_tol_recall += 1

        print(f"Sample {idx+1:02d}: UI_Count={len(ui_droplets):3d} | RawMinDist={min_d_raw:5.1f}px | ManifestMinDist={min_d_man:5.1f}px | Task: {s['instruction'][:50]}")

    print(f"\n--- RECALL SUMMARY ON 25 SAMPLES ---")
    print(f"Sprayer Raw Strict Recall (in GT box):      {raw_strict_recall}/25 ({raw_strict_recall*4}%)")
    print(f"Sprayer Raw Tolerant Recall (<=24px):       {raw_tol_recall}/25 ({raw_tol_recall*4}%)")
    print(f"96-Cand Manifest Strict Recall (in GT box): {manifest_strict_recall}/25 ({manifest_strict_recall*4}%)")
    print(f"96-Cand Manifest Tolerant Recall (<=24px):  {manifest_tol_recall}/25 ({manifest_tol_recall*4}%)")

asyncio.run(main())
