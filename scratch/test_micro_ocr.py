import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import winocr
import time

sys.stdout.reconfigure(encoding='utf-8')

def micro_ocr_patch(patch_bgr, pad=12, scale=3.5):
    """
    Super-Resolution Micro-OCR on candidate button patches.
    Upscales small button patches so WinOCR neural net easily resolves 8-12px glyphs.
    """
    if patch_bgr is None or patch_bgr.size == 0 or patch_bgr.shape[0] < 8 or patch_bgr.shape[1] < 12:
        return ""
    # Add border padding for neural network field-of-view margin
    # Use border replicate or mean color
    mean_val = [int(v) for v in patch_bgr.mean(axis=(0, 1))]
    padded = cv2.copyMakeBorder(patch_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_val)
    # Upscale 3.5x
    h, w = padded.shape[:2]
    up = cv2.resize(padded, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    op = winocr.recognize_cv2(up, lang='en')
    res = op.get() if hasattr(op, 'get') else op
    lines = [l.text.strip() for l in res.lines if l.text.strip()]
    if lines:
        return " ".join(lines)
    
    # Try inverted if dark background (mean < 120)
    if sum(mean_val) / 3 < 120:
        inv = 255 - up
        op_inv = winocr.recognize_cv2(inv, lang='en')
        res_inv = op_inv.get() if hasattr(op_inv, 'get') else op_inv
        inv_lines = [l.text.strip() for l in res_inv.lines if l.text.strip()]
        if inv_lines:
            return " ".join(inv_lines)
    return ""

# Test on bottom_right (Save / Cancel)
br = cv2.imread("scratch/bottom_right.png")
t0 = time.time()
txt_br = micro_ocr_patch(br)
t1 = time.time()
print(f"Bottom right micro-OCR result: '{txt_br}' (took {(t1-t0)*1000:.1f}ms)")

# Test on yes button
yes = cv2.imread("scratch/yes_exact_btn.png")
txt_yes = micro_ocr_patch(yes)
print(f"Yes button micro-OCR result: '{txt_yes}'")

# Test on dialog buttons
dialog = cv2.imread("scratch/dialog_exact.png")
strip = dialog[60:88, 110:235]
cv2.imwrite("scratch/dialog_btn_strip.png", strip)
txt_strip = micro_ocr_patch(strip)
print(f"Dialog Button Strip result: '{txt_strip}'")

btn1 = dialog[65:86, 130:182]
btn2 = dialog[65:86, 182:234]
print(f"Dialog Btn 1 (Yes area): '{micro_ocr_patch(btn1)}'")
print(f"Dialog Btn 2 (No area): '{micro_ocr_patch(btn2)}'")
