import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose")

img = cv2.imread(str(Path('osworld_g_images') / '2ENZHM7E2X.png'))
# GT is at (233, 184)
# Let's see what contour was detected around (233, 184)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Let's crop 24x24 around 233, 184
patch = gray[184-12:184+12, 233-12:233+12]
edges = cv2.Canny(patch, 50, 150)
p24 = cv2.resize(edges, (24, 24), interpolation=cv2.INTER_NEAREST)

top_c = int(np.sum(p24[1:7, 8:16] > 0))
mid_l = int(np.sum(p24[7:13, 0:8] > 0))
mid_r = int(np.sum(p24[7:13, 16:24] > 0))
bot_l = int(np.sum(p24[16:23, 0:9] > 0))
bot_r = int(np.sum(p24[16:23, 15:24] > 0))
val_tl = int(np.sum(p24[:6, :7] > 0))
val_tr = int(np.sum(p24[:6, 17:] > 0))
bot_groove = int(np.sum(p24[21:24, 10:14] > 0))

print(f"top_c={top_c}, mid_l={mid_l}, mid_r={mid_r}, bot_l={bot_l}, bot_r={bot_r}")
print(f"val_tl={val_tl}, val_tr={val_tr}, bot_groove={bot_groove}")
