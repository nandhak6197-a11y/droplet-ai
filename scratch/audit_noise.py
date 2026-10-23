import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import asyncio
from pathlib import Path
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext
from collections import Counter

with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

for idx in [0, 3]:
    s = samples[idx]
    img = cv2.imread(str(Path('osworld_g_images') / s['image_path']))
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title='OSWorld-G', rect=(0,0,sw,sh), width=sw, height=sh, is_valid=True)
    sprayer = FullBleedLatticeSprayer(48)
    droplets = asyncio.run(sprayer.spray(img, win_ctx=win, offline_frame=True))
    print(f"=== Sample {idx+1}: '{s['instruction']}' ===")
    print(f"Total droplets: {len(droplets)}")
    cats = Counter(d.category for d in droplets)
    print("Categories:", dict(cats))
    roles = Counter(getattr(d, 'semantic_role', '') for d in droplets)
    print("Roles:", dict(roles))
    archetypes = Counter(getattr(d, 'archetype', '') for d in droplets)
    print("Archetypes:", dict(archetypes))
    generic_buttons = sum(1 for d in droplets if d.label == "[Icon] Toolbar Tool / Visual Button")
    generic_near = sum(1 for d in droplets if d.label.startswith("[Icon] Toolbar Tool / Visual Button (Near"))
    print(f"Generic visual buttons (no context): {generic_buttons}")
    print(f"Generic visual buttons (with Near context): {generic_near}")
    probes = sum(1 for d in droplets if d.category == 'surface_probe')
    print(f"Surface probes (lattice stars): {probes}")
    ocr_texts = sum(1 for d in droplets if not d.label.startswith("[Icon]") and d.category != 'surface_probe')
    print(f"OCR text words/lines: {ocr_texts}")
    print()
