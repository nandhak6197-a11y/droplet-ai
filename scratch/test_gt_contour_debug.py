import cv2
import numpy as np
import json
from pathlib import Path
from spatial_droplet.vision.clahe import AdaptiveContrastNormalizer
from run_qwen_200_diagnostic import convert_gt_box

with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

for idx in [3, 4, 10]:
    s = samples[idx]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    gt_box = convert_gt_box(s['box_coordinates'])
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    print(f"=== Sample {idx+1}: {s['instruction']} ===")
    print(f"GT Box: {gt_box}, W={gx2-gx1}, H={gy2-gy1}")
    
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = AdaptiveContrastNormalizer.enhance_grayscale(gray_raw)
    edges = cv2.Canny(gray, 24, 85)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close)
    all_contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    found_overlapping = []
    for cnt in all_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if max(gx1-10, bx) < min(gx2+10, bx+bw) and max(gy1-10, by) < min(gy2+10, by+bh):
            patch = edges[by:by+bh, bx:bx+bw]
            density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
            found_overlapping.append((bx, by, bw, bh, density))
            
    print(f"Found {len(found_overlapping)} overlapping contours:")
    for bx, by, bw, bh, density in found_overlapping:
        aspect = bw / float(bh) if bh > 0 else 0
        print(f"   bbox: ({bx}, {by}, {bw}, {bh}), aspect={aspect:.2f}, density={density:.3f}")
