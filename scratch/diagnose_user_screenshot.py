import sys
import os
import cv2
import asyncio

sys.stdout.reconfigure(encoding='utf-8')

img_path = r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png"
if not os.path.exists(img_path):
    print("Error: Image does not exist at", img_path)
    sys.exit(1)

img = cv2.imread(img_path)
print(f"Image shape: {img.shape}")

# Test 1: Run WinOCR on full image
import winocr
op = winocr.recognize_cv2(img, lang='en')
res = op.get() if hasattr(op, 'get') else op
print(f"Total OCR lines: {len(res.lines)}")
for idx, line in enumerate(res.lines):
    t = line.text.strip()
    words_info = [(w.text, int(w.bounding_rect.x), int(w.bounding_rect.y), int(w.bounding_rect.width), int(w.bounding_rect.height)) for w in line.words]
    print(f"Line {idx}: '{t}' -> words: {words_info}")

# Test 2: Run FullBleedLatticeSprayer on the image
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext

sprayer = FullBleedLatticeSprayer()
win_ctx = WindowContext(hwnd=0, title="Save As", rect=(0, 0, img.shape[1], img.shape[0]))
droplets = asyncio.run(sprayer.spray(img, win_ctx=win_ctx, offline_frame=True))
print(f"\nTotal sprayed droplets: {len(droplets)}")
for d in droplets:
    lbl = d.label.lower()
    if any(k in lbl for k in ['yes', 'no', 'confirm', 'replace', 'login', 'cancel', 'save']):
        print(f"  Droplet [{d.id}] at ({d.center_x}, {d.center_y}): '{d.label}' (cat={d.category}, role={d.semantic_role})")
