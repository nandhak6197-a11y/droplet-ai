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

async def inspect_sample69_manifest():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[68]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    # GT box: x: 724 to 771, y: 501 to 545
    gt_d = next(d for d in droplets if 724 <= d.center_x <= 771 and 501 <= d.center_y <= 545)
    print(f"Sample 69 GT Droplet: ID {gt_d.id} at ({gt_d.center_x}, {gt_d.center_y}) Label='{gt_d.label}'")
    
    # Let's inspect the decision process and candidates in reasoner
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    
    res = await reasoner.decide(
        user_intent=s['instruction'],
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    cands = res.get('manifest_candidate_ids') or []
    print(f"Total candidates in manifest: {len(cands)}")
    print(f"Is gt_d.id ({gt_d.id}) in manifest? {gt_d.id in cands}")
    print(f"gt_d score: {getattr(gt_d, '_score', 'N/A')}")

asyncio.run(inspect_sample69_manifest())
