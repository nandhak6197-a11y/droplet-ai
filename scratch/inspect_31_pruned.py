import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')
import asyncio, json, math, cv2
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner

def convert_gt_box(box_coords):
    if len(box_coords) == 4:
        x, y, w, h = box_coords
        return x, y, x + w, y + h
    xs, ys = box_coords[0::2], box_coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def distance_to_box(px, py, box):
    x1, y1, x2, y2 = box
    return math.hypot(max(0.0, x1 - px, px - x2), max(0.0, y1 - py, py - y2))

async def main():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)[:100]
    sprayer = FullBleedLatticeSprayer(48)
    target_indices = [32, 37, 42, 43, 47, 48, 54, 57, 58, 59, 60, 62, 69, 70, 72, 76, 77, 78, 79, 80, 81, 84, 86, 88, 89, 91, 92, 93, 94, 95, 98]
    for s_idx in target_indices:
        s = samples[s_idx - 1]
        img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
        gt_box = convert_gt_box(s['box_coordinates'])
        droplets = await sprayer.spray(img, offline_frame=True)
        ui_droplets = [d for d in droplets if d.category != 'surface_probe']
        best_d = min(ui_droplets, key=lambda d: distance_to_box(d.center_x, d.center_y, gt_box))
        dist = distance_to_box(best_d.center_x, best_d.center_y, gt_box)
        print(f"[{s_idx:03d}] Dist={dist:4.1f} | ID={best_d.id:3d} ({best_d.center_x},{best_d.center_y}) | Cat={best_d.category} | Arch={getattr(best_d,'archetype','')} | Lbl='{best_d.label}' | Att='{getattr(best_d,'attached_label','')}'")
        print(f"      Task: {s['instruction']}")

asyncio.run(main())
