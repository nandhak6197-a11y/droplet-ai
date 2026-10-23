import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import winocr
import re
import numpy as np

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")
h, w = img.shape[:2]

from spatial_droplet.vision.icon_detector import EnhancedIconDetector

icons = EnhancedIconDetector.detect_icons(img)
print(f"Total detected icons: {len(icons)}")

def clean_dialog_token(text: str) -> str:
    t = text.strip()
    t_clean = re.sub(r'[^a-zA-Z0-9]', '', t).lower()
    if 'canc' in t_clean or t_clean in ('caneel', 'cance', 'cncl', 'can'):
        return "Cancel"
    if t_clean in ('sav', 'save', 'saved', 'sve'):
        return "Save"
    if t_clean in ('yes', 'ye', 'ys', 'yess'):
        return "Yes"
    if t_clean in ('no', 'n0', 'noo'):
        return "No"
    if t_clean in ('ok', '0k', 'okay', 'okk'):
        return "OK"
    if 'dont' in t_clean or 'dontsave' in t_clean:
        return "Don't Save"
    if 'retry' in t_clean:
        return "Retry"
    if 'ignore' in t_clean:
        return "Ignore"
    if 'abort' in t_clean:
        return "Abort"
    if 'apply' in t_clean:
        return "Apply"
    if 'open' in t_clean:
        return "Open"
    if 'close' in t_clean:
        return "Close"
    return t

def micro_ocr_button(crop) -> str:
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 12:
        return ""
    ch, cw = crop.shape[:2]
    mean_val = [int(v) for v in crop.mean(axis=(0, 1))]
    
    for scale in [3.5, 4.0]:
        pad = int(14 * scale)
        up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_val)
        
        # 1. Inverted pass (dark text on light button -> white on dark)
        inv = 255 - padded
        op = winocr.recognize_cv2(inv, lang='en')
        res = op.get() if hasattr(op, 'get') else op
        lines = [l.text.strip() for l in res.lines if l.text.strip()]
        if lines:
            txt = " ".join(lines)
            token = clean_dialog_token(txt)
            if len(token) >= 2:
                return token
            
        # 2. CLAHE contrast enhanced pass
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        enh_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        op2 = winocr.recognize_cv2(enh_bgr, lang='en')
        res2 = op2.get() if hasattr(op2, 'get') else op2
        lines2 = [l.text.strip() for l in res2.lines if l.text.strip()]
        if lines2:
            txt2 = " ".join(lines2)
            token2 = clean_dialog_token(txt2)
            if len(token2) >= 2:
                return token2

        # 3. Direct pass
        op3 = winocr.recognize_cv2(padded, lang='en')
        res3 = op3.get() if hasattr(op3, 'get') else op3
        lines3 = [l.text.strip() for l in res3.lines if l.text.strip()]
        if lines3:
            txt3 = " ".join(lines3)
            token3 = clean_dialog_token(txt3)
            if len(token3) >= 2:
                return token3
            
    return ""

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

def run_micro_ocr_worker(crop):
    return micro_ocr_button(crop)

resolved_buttons = []
futures = []
for ic in icons:
    cx, cy = ic[0], ic[1]
    lbl = ic[2]
    ibox = ic[5] if len(ic) > 5 else (cx - 12, cy - 12, cx + 12, cy + 12)
    bw = ibox[2] - ibox[0]
    bh = ibox[3] - ibox[1]
    if "Visual Button" in lbl or "Button" in lbl or "Graphic" in lbl:
        if 20 <= bw <= 320 and 12 <= bh <= 65:
            crop = img[max(0, ibox[1]):min(h, ibox[3]), max(0, ibox[0]):min(w, ibox[2])]
            fut = executor.submit(run_micro_ocr_worker, crop)
            futures.append((cx, cy, ibox, fut))

for cx, cy, ibox, fut in futures:
    text = fut.result()
    if text:
        resolved_buttons.append((cx, cy, text, ibox))
        print(f"Resolved Button at ({cx}, {cy}) [{ibox}]: '[Btn] {text}'")

executor.shutdown(wait=True)
print(f"\nTotal resolved buttons via micro-OCR: {len(resolved_buttons)}")

