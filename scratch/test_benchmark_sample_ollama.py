import sys
import os
import json
import time
import base64
import asyncio
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

# 1. Initialize Kaggle Ollama Client
client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

# 2. Load Sample 1 from cache
with open("osworld_g_cache.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

sample = samples[0]
img_path = Path("osworld_g_images") / sample["image_path"]
instruction = sample["instruction"]
box_coords = sample["box_coordinates"]
x, y, w, h = box_coords
gt_box = (x, y, x + w, y + h)
gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0

print(f"[*] Testing Sample 1 on Kaggle Ollama: '{instruction}'")
print(f"[*] GT Box: {gt_box} | Center: ({gt_cx:.1f}, {gt_cy:.1f})")

img = cv2.imread(str(img_path))
assert img is not None, "Failed to load image"

async def main():
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    reasoner.use_ngrok = True
    reasoner.ngrok_client = client
    reasoner.ngrok_model = "qwen3.8:27b"

    # Spray droplets
    t0 = time.perf_counter()
    droplets = await sprayer.spray(img)
    print(f"[+] Sprayed {len(droplets)} droplets in {(time.perf_counter()-t0)*1000:.1f}ms")

    # Predict with Spatial Droplets via Kaggle Ollama
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    t1 = time.perf_counter()
    dec = await reasoner.decide(
        user_intent=instruction,
        droplets=droplets,
        window_ctx=win,
        memory=None,
        screenshot=img,
        mode="combined",
        requested_provider="ngrok"
    )
    dt = (time.perf_counter() - t1) * 1000
    tgt_id = dec.get("target_dot_id") or dec.get("target_droplet_id")
    px, py = dec.target_x, dec.target_y

    dx = max(0.0, gt_box[0] - px, px - gt_box[2]) if px else 999
    dy = max(0.0, gt_box[1] - py, py - gt_box[3]) if py else 999
    dist_box = (dx**2 + dy**2)**0.5
    dist_c = ((px - gt_cx)**2 + (py - gt_cy)**2)**0.5 if px else 999
    strict = (gt_box[0] <= px <= gt_box[2]) and (gt_box[1] <= py <= gt_box[3]) if px else False

    print("\n" + "=" * 60)
    print(" KAGGLE OLLAMA SPATIAL DROPLET BENCHMARK RESULT")
    print("=" * 60)
    print(f"Target Dot ID:    {tgt_id}")
    print(f"Predicted Coords: ({px}, {py})")
    print(f"GT Center:        ({gt_cx:.1f}, {gt_cy:.1f})")
    print(f"Distance to GT:   {dist_c:.1f}px (Box Gap: {dist_box:.1f}px)")
    print(f"Verdict:          {'[STRICT HIT ✔]' if strict else '[MISS ❌]'}")
    print(f"Latency:          {dt:.1f}ms")
    print(f"Reasoning:        {dec.reasoning}")
    print("=" * 60)

asyncio.run(main())
