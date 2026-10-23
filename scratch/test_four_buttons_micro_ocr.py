import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import winocr

sys.stdout.reconfigure(encoding='utf-8')

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")

buttons = [
    ("Save", (897, 519, 954, 535)),
    ("Cancel", (965, 519, 1021, 535)),
    ("Yes", (528, 276, 576, 293)),
    ("No", (580, 276, 627, 293)),
]

def resolve_button_text(crop):
    if crop is None or crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    # Test scales: 3x and 4x
    for scale in [3.0, 4.0]:
        mean_col = [int(v) for v in crop.mean(axis=(0, 1))]
        pad = int(12 * scale)
        up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_col)
        
        op = winocr.recognize_cv2(padded, lang='en')
        res = op.get() if hasattr(op, 'get') else op
        lines = [l.text.strip() for l in res.lines if l.text.strip()]
        if lines:
            return f"raw@{scale}x: " + " ".join(lines)
        
        # Test inverted
        inv = 255 - padded
        op2 = winocr.recognize_cv2(inv, lang='en')
        res2 = op2.get() if hasattr(op2, 'get') else op2
        lines2 = [l.text.strip() for l in res2.lines if l.text.strip()]
        if lines2:
            return f"inv@{scale}x: " + " ".join(lines2)
            
        # Test contrast stretching / CLAHE on upscaled patch
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        enh_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        op3 = winocr.recognize_cv2(enh_bgr, lang='en')
        res3 = op3.get() if hasattr(op3, 'get') else op3
        lines3 = [l.text.strip() for l in res3.lines if l.text.strip()]
        if lines3:
            return f"clahe@{scale}x: " + " ".join(lines3)

    return ""

for expected, (x1, y1, x2, y2) in buttons:
    crop = img[y1:y2, x1:x2]
    text = resolve_button_text(crop)
    print(f"Target '{expected}' [{x1},{y1},{x2},{y2}] -> Micro-OCR: '{text}'")
