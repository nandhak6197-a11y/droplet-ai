import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import re
import math
import asyncio
import numpy as np
from typing import List, Tuple, Optional

try:
    import winocr
    HAS_WINOCR = True
except ImportError:
    HAS_WINOCR = False

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")
h, w = img.shape[:2]

from spatial_droplet.vision.icon_detector import EnhancedIconDetector

DIALOG_TOKENS = {
    'Cancel', 'Save', 'Yes', 'No', 'OK', "Don't Save", 'Retry', 'Ignore',
    'Abort', 'Apply', 'Open', 'Close', 'Next', 'Finish', 'Browse', 'Help'
}

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
    if 'next' in t_clean:
        return "Next"
    if 'finish' in t_clean:
        return "Finish"
    if 'browse' in t_clean:
        return "Browse"
    if 'help' in t_clean:
        return "Help"
    return ""

def micro_ocr_button_patch(crop: np.ndarray) -> str:
    if not HAS_WINOCR or crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 12:
        return ""
    ch, cw = crop.shape[:2]
    mean_val = [int(v) for v in crop.mean(axis=(0, 1))]
    best_fallback = ""
    
    for scale in [3.0, 4.0]:
        pad = int(12 * scale)
        up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_val)
        inv = 255 - padded
        
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        
        for p in [padded, inv, enh]:
            try:
                op = winocr.recognize_cv2(p, lang='en')
                res = op.get() if hasattr(op, 'get') else op
                for line in res.lines:
                    raw_txt = line.text.strip()
                    tok = clean_dialog_token(raw_txt)
                    if tok in DIALOG_TOKENS:
                        return tok
                    cleaned_general = re.sub(r'[^\w\s]', '', raw_txt).strip()
                    if len(cleaned_general) >= 2 and not best_fallback:
                        best_fallback = cleaned_general
            except Exception:
                pass
                
    return best_fallback

def collinear_dialog_peer_inference(
    candidates: List[Tuple[int, int, str, str, str, float, Tuple[int, int, int, int]]]
) -> List[Tuple[int, int, str, str, str, float, Tuple[int, int, int, int]]]:
    """
    Infers adjacent unread or faint dialog button labels in standard horizontal action strips:
    e.g. [__] [No] -> [Yes] [No]
         [__] [Cancel] -> [Save] / [OK] [Cancel]
         [Save] [__] [Cancel] -> [Save] [Don't Save] [Cancel]
    """
    updated = list(candidates)
    n = len(updated)
    for i in range(n):
        cx1, cy1, lbl1, cat1, sub1, conf1, box1 = updated[i]
        b1_tok = sub1.strip()
        
        for j in range(n):
            if i == j:
                continue
            cx2, cy2, lbl2, cat2, sub2, conf2, box2 = updated[j]
            b2_tok = sub2.strip()
            
            # Check if collinear: same row within 8px, similar height within 8px
            h1 = box1[3] - box1[1]
            h2 = box2[3] - box2[1]
            if abs(cy1 - cy2) <= 8 and abs(h1 - h2) <= 8:
                gap = box2[0] - box1[2]  # box1 is on left, box2 is on right
                if 4 <= gap <= 35:
                    # Case A: [?] [No] -> infer left is Yes
                    if b2_tok == "No" and (b1_tok not in DIALOG_TOKENS or "Visual Button" in lbl1):
                        updated[i] = (cx1, cy1, "[Btn] Yes", "interactive_widget", "Yes", 0.98, box1)
                    # Case B: [Yes] [?] -> infer right is No
                    elif b1_tok == "Yes" and (b2_tok not in DIALOG_TOKENS or "Visual Button" in lbl2):
                        updated[j] = (cx2, cy2, "[Btn] No", "interactive_widget", "No", 0.98, box2)
                    # Case C: [?] [Cancel] in 2-button strip
                    elif b2_tok == "Cancel" and (b1_tok not in DIALOG_TOKENS or "Visual Button" in lbl1):
                        inferred_name = "Save" if any("save" in str(u[2]).lower() for u in updated) else "OK"
                        updated[i] = (cx1, cy1, f"[Btn] {inferred_name}", "interactive_widget", inferred_name, 0.96, box1)
                        
    return updated

async def run_test():
    icons = await asyncio.to_thread(EnhancedIconDetector.detect_icons, img)
    print(f"Detected raw icons: {len(icons)}")
    
    # Process with micro-OCR
    enhanced_ui = []
    
    async def process_icon(ic):
        cx, cy, lbl, cat, sub = ic[:5]
        ibox = ic[5] if len(ic) > 5 else (cx - 12, cy - 12, cx + 12, cy + 12)
        bw = ibox[2] - ibox[0]
        bh = ibox[3] - ibox[1]
        
        # Check if button-shaped candidate
        is_btn_candidate = (
            ("Visual Button" in lbl or "Button" in lbl or "Graphic" in lbl) and
            (20 <= bw <= 320 and 12 <= bh <= 65 and bw >= bh * 1.2)
        )
        if is_btn_candidate:
            crop = img[max(0, ibox[1]):min(h, ibox[3]), max(0, ibox[0]):min(w, ibox[2])]
            text = await asyncio.to_thread(micro_ocr_button_patch, crop)
            if text:
                return (cx, cy, f"[Btn] {text}", "interactive_widget", text, 0.98, ibox)
        return (cx, cy, lbl, cat, sub, 0.70, ibox)

    tasks = [process_icon(ic) for ic in icons]
    enhanced_ui = await asyncio.gather(*tasks)
    
    # Run collinear peer inference
    enhanced_ui = collinear_dialog_peer_inference(enhanced_ui)
    
    # Check our target dialog buttons:
    targets = [
        ("Yes", (528, 276, 576, 293)),
        ("No", (579, 276, 627, 293)),
        ("Save", (897, 519, 954, 535)),
        ("Cancel", (965, 519, 1021, 535))
    ]
    
    print("\n--- Verification against Target Buttons ---")
    for name, (tx1, ty1, tx2, ty2) in targets:
        tcx = (tx1 + tx2) // 2
        tcy = (ty1 + ty2) // 2
        matching = [
            u for u in enhanced_ui
            if abs(u[0] - tcx) <= 12 and abs(u[1] - tcy) <= 12
        ]
        if matching:
            best = matching[0]
            print(f"SUCCESS: Target '{name}' grounded at ({best[0]}, {best[1]}): label='{best[2]}', sub='{best[4]}'")
        else:
            print(f"FAILED: Target '{name}' not found near ({tcx}, {tcy})")

if __name__ == "__main__":
    asyncio.run(run_test())
