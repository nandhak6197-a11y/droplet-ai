import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import asyncio
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext

sys.stdout.reconfigure(encoding='utf-8')

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")
sprayer = FullBleedLatticeSprayer()
win_ctx = WindowContext(hwnd=0, title="Save As", rect=(0, 0, img.shape[1], img.shape[0]))
droplets = asyncio.run(sprayer.spray(img, win_ctx=win_ctx, offline_frame=True))

print(f"Total droplets sprayed: {len(droplets)}")
for d in droplets:
    # Print droplets near the dialog (x: 396..634, y: 209..298)
    if 380 <= d.center_x <= 650 and 190 <= d.center_y <= 320:
        print(f"  Dialog Area Droplet [{d.id}] at ({d.center_x}, {d.center_y}): '{d.label}' (cat={d.category}, role={d.semantic_role})")

    # Also print any Yes / No / OK / Cancel droplets anywhere on screen
    lbl = d.label.lower()
    if any(k in lbl for k in ['yes', 'no', 'cancel', 'save', 'ok']):
        print(f"  Target Match Droplet [{d.id}] at ({d.center_x}, {d.center_y}): '{d.label}' (cat={d.category})")
