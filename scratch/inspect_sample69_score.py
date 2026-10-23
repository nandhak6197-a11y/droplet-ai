import asyncio
import cv2
import json
import sys
import re
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def inspect_rank():
    img = cv2.imread(str(Path('osworld_g_images') / 'GDZVK92HHZ.png'))
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    
    gt_d = next(d for d in droplets if 724 <= d.center_x <= 771 and 501 <= d.center_y <= 545)
    print(f"Target Droplet: ID {gt_d.id} at ({gt_d.center_x}, {gt_d.center_y}) Label='{gt_d.label}'")
    
    # We can inspect the candidate selection logic by calling a modified method or calling decide with dry_run
    # Let's inspect how reasoner scores all droplets:
    reasoner = DiagonalDropletReasoner()
    
    # Let's run decide with mock target_model so it doesn't call Groq if we just want to see candidates
    # Or let's see which candidates are in all_candidates
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, img.shape[1], img.shape[0]), width=img.shape[1], height=img.shape[0], is_valid=True)
    res = await reasoner.decide(
        user_intent='Click on the question icon of the "save" dialog.',
        droplets=droplets,
        window_ctx=win,
        screenshot=img,
        mode="combined",
        target_model="qwen/qwen3.8-27b"
    )
    cands = res.get('manifest_candidate_ids') or []
    print(f"Total candidates in manifest: {len(cands)}")
    print(f"Is target droplet {gt_d.id} in manifest? {gt_d.id in cands}")
    if gt_d.id not in cands:
        # Find where gt_d was ranked among active droplets
        # Let's check gt_d._score if set
        print(f"Target droplet score: {getattr(gt_d, '_score', 'NOT SET')}")

asyncio.run(inspect_rank())
