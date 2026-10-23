import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import json
import cv2
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext
import asyncio

with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    s = json.load(f)[0]

img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
sh, sw = img.shape[:2]
win = WindowContext(hwnd=0, title='OSWorld-G', rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)

sprayer = FullBleedLatticeSprayer(48)
droplets = asyncio.run(sprayer.spray(img, win_ctx=win, offline_frame=True))

reasoner = DiagonalDropletReasoner()
reasoner.use_ngrok = False

t0 = time.time()
dec = asyncio.run(reasoner.decide(
    user_intent=s['instruction'],
    droplets=droplets,
    window_ctx=win,
    screenshot=img,
    mode='combined',
    target_model='qwen/qwen3.8-27b',
    provider='groq'
))
lat = time.time() - t0
print(f"Done in {lat:.2f}s! Provider: {dec.get('provider')}, Dot: {dec.target_dot_id}, Point: ({dec.target_x}, {dec.target_y})")
print("Reasoning snippet:", dec.get('reasoning', '')[:120])
