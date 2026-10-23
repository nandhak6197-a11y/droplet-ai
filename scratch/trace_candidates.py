import asyncio
import cv2
import json
import sys
import re
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def trace_candidate_selection():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    for s_idx in [68, 144]:
        s = samples[s_idx]
        img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
        sh, sw = img.shape[:2]
        droplets = await sprayer.spray(img)
        
        # Find GT droplet
        gt_coords = s['box_coordinates']
        if len(gt_coords) == 4:
            x1, y1, w, h = gt_coords
            gt_box = (x1, y1, x1 + w, y1 + h)
        else:
            xs = gt_coords[0::2]; ys = gt_coords[1::2]
            gt_box = (min(xs), min(ys), max(xs), max(ys))
            
        active = [d for d in droplets if d.category != 'surface_probe']
        gt_d = next(d for d in active if gt_box[0] <= d.center_x <= gt_box[2] and gt_box[1] <= d.center_y <= gt_box[3])
        print("=" * 80)
        print(f"Sample {s_idx + 1}: Task='{s['instruction']}'")
        print(f"GT Droplet ID {gt_d.id} at ({gt_d.center_x}, {gt_d.center_y}) Label='{gt_d.label}'")
        
        # Now let's trace scoring of all active droplets
        # Let's inspect where gt_d lands in sorted_nodes
        # We can simulate relevance_score exactly as in reasoner.py
        UI_STOPWORDS = {
            "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "by", "from", "with",
            "click", "open", "select", "want", "current", "this", "that", "it", "is", "be",
            "window", "screen", "panel", "slide", "box", "text", "button", "area", "page",
            "value", "item", "icon", "setting", "settings", "function"
        }
        prompt_words = set(re.findall(r'\w+', s['instruction'].lower()))
        rare_prompt_words = set(w for w in prompt_words if len(w) >= 3 and w not in UI_STOPWORDS)
        
        # Count how many visual tools are in active droplets
        visual_tools = [d for d in active if d.label.startswith("[Icon]") or getattr(d, 'archetype', '') in ('button', 'icon', 'control', 'slider', 'tab')]
        print(f"Total visual tools in active droplets: {len(visual_tools)}")
        gt_in_vt = any(d.id == gt_d.id for d in visual_tools)
        print(f"Is GT droplet in visual_tools? {gt_in_vt}")
        
        # Find index of gt_d in visual_tools
        if gt_in_vt:
            vt_idx = next(i for i, d in enumerate(visual_tools) if d.id == gt_d.id)
            print(f"Index of GT droplet in visual_tools: {vt_idx} (out of {len(visual_tools)})")

asyncio.run(trace_candidate_selection())
