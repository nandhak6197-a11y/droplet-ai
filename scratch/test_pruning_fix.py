import sys, os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2, json, asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def test_samples():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Previously pruned samples: 19 (idx 18), 22 (idx 21), 26 (idx 25), 30 (idx 29)
    test_indices = [18, 21, 25, 29]
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()

    for idx in test_indices:
        s = samples[idx]
        img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
        sh, sw = img.shape[:2]
        gt_box = s['box_coordinates']
        if len(gt_box) == 4:
            gx1, gy1, gx2, gy2 = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
        else:
            gx1, gy1, gx2, gy2 = min(gt_box[0::2]), min(gt_box[1::2]), max(gt_box[0::2]), max(gt_box[1::2])
        gt_cx, gt_cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0

        print(f"\n[Sample {idx+1:03d}] \"{s['instruction']}\"")
        droplets = await sprayer.spray(img)
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
        dec = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, window_ctx=win, memory=None, screenshot=img, mode="combined", target_model="qwen/qwen3.8-27b")

        px, py = dec.target_x, dec.target_y
        dist = ((px - gt_cx)**2 + (py - gt_cy)**2)**0.5 if px is not None else 999.0
        strict = (gx1 <= px <= gx2) and (gy1 <= py <= gy2) if px is not None else False
        hit_tag = "✔ STRICT HIT" if strict else f"❌ Miss ({dist:.1f}px)"
        print(f"  Result: {hit_tag} | Pred: ({px}, {py}) | GT Centroid: ({gt_cx:.1f}, {gt_cy:.1f}) | Picked Dot: {dec.get('target_dot_id')}")
        print(f"  Reasoning: {dec.reasoning[:90]}")

if __name__ == "__main__":
    asyncio.run(test_samples())
