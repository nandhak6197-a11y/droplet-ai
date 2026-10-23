import asyncio
import cv2
import json
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

# Test on 5 difficult previously pruned samples:
# Sample 69 (index 68): "Click on the question icon of the 'save' dialog."
# Sample 42 (index 41): "Click the bucket fill tool"
# Sample 58 (index 57): "Click on the checkbox of 'Current layer only'"
# Sample 65 (index 64): "Click on the checkbox of Split View"
# Sample 145 (index 144): "Add Decimal Place for the current cell"

TEST_INDICES = [68, 41, 57, 64, 144]

async def test_batch():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    for s_idx in TEST_INDICES:
        s = samples[s_idx]
        print("=" * 80)
        print(f"Sample {s_idx + 1}: Task = '{s['instruction']}'")
        img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
        
        # Spray droplets
        droplets = await sprayer.spray(img)
        active_droplets = [d for d in droplets if d.category != 'surface_probe']
        
        # Check if GT has droplets
        gt_coords = s['box_coordinates']
        if len(gt_coords) == 4:
            x1, y1, w, h = gt_coords
            gt_box = (x1, y1, x1 + w, y1 + h)
        else:
            xs = gt_coords[0::2]; ys = gt_coords[1::2]
            gt_box = (min(xs), min(ys), max(xs), max(ys))
            
        dots_in_gt = [d for d in active_droplets if (gt_box[0] <= d.center_x <= gt_box[2] and gt_box[1] <= d.center_y <= gt_box[3])]
        print(f"  Active droplets: {len(active_droplets)} | Droplets inside GT box: {len(dots_in_gt)}")
        if dots_in_gt:
            for d in dots_in_gt[:3]:
                print(f"    GT Dot ID {d.id} at ({d.center_x}, {d.center_y}): Label='{d.label}'")

asyncio.run(test_batch())
