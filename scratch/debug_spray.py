import asyncio
import cv2
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.vision.icon_detector import EnhancedIconDetector

async def debug_droplets():
    img = cv2.imread(str(Path('osworld_g_images') / 'GDZVK92HHZ.png'))
    print(f"Img shape: {img.shape}")
    
    # 1. Test IconDetector directly
    icons = EnhancedIconDetector.detect_icons(img)
    print(f"IconDetector returned: {len(icons)} icons")
    
    # 2. Test Sprayer
    sprayer = FullBleedLatticeSprayer(48)
    droplets = await sprayer.spray(img)
    print(f"Sprayer returned: {len(droplets)} total droplets")
    active = [d for d in droplets if d.category != 'surface_probe']
    print(f"Active droplets: {len(active)}")
    
    # Check GT box for Sample 69: x: 724 to 771, y: 501 to 545
    for ic in icons:
        if 724 <= ic[0] <= 771 and 501 <= ic[1] <= 545:
            print(f"Icon in GT box: {ic}")
            
    for d in active:
        if 724 <= d.center_x <= 771 and 501 <= d.center_y <= 545:
            print(f"Droplet in GT box: {d.id} at ({d.center_x}, {d.center_y}) Label='{d.label}'")

asyncio.run(debug_droplets())
