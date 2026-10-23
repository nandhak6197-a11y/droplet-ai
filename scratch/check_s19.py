import sys, os
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2, json, asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def check():
    samples = json.load(open('osworld_g_cache.json', encoding='utf-8'))
    s = samples[18]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    reasoner = DiagonalDropletReasoner()
    win = WindowContext(hwnd=0, title='OSWorld-G', rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
    dec = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, window_ctx=win, memory=None, screenshot=img, mode='combined', target_model='qwen/qwen3.8-27b')
    print('Picked:', dec.get('target_dot_id'), (dec.target_x, dec.target_y))
    print('Deliberation intent:', dec.deliberation.get('intent_analysis'))
    print('Deliberation candidate_eval:', dec.deliberation.get('candidate_evaluation'))

if __name__ == "__main__":
    asyncio.run(check())
