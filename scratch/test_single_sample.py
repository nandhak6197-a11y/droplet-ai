import sys, os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio, cv2, json
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

async def test_one():
    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    s = samples[0]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    print(f"Testing Sample 1: \"{s['instruction']}\"")
    
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    droplets = await sprayer.spray(img)
    print(f"Sprayed {len(droplets)} droplets.")
    
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0,0,img.shape[1],img.shape[0]), width=img.shape[1], height=img.shape[0], is_valid=True)
    dec = await reasoner.decide(user_intent=s['instruction'], droplets=droplets, window_ctx=win, memory=None, screenshot=img, mode="combined", target_model="qwen/qwen3.8-27b")
    
    print("Decision:")
    print("  • Target Dot ID:", dec.target_droplet_id)
    print("  • Action:", dec.action)
    print("  • Coordinates:", (dec.target_x, dec.target_y))
    print("  • Model Used:", dec.get("model_used"))
    print("  • Key Used:", dec.get("key_used"))
    print("Deliberation:", dec.deliberation)

if __name__ == "__main__":
    asyncio.run(test_one())
