import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import winocr

sys.stdout.reconfigure(encoding='utf-8')

img_path = r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png"
img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Original image size: {w}x{h}")

def run_ocr(frame, desc):
    op = winocr.recognize_cv2(frame, lang='en')
    res = op.get() if hasattr(op, 'get') else op
    lines = [l.text.strip() for l in res.lines if l.text.strip()]
    print(f"\n[{desc}] Total lines detected: {len(lines)}")
    for l in res.lines:
        txt = l.text.strip()
        if any(k in txt.lower() for k in ['yes', 'no', 'confirm', 'replace', 'login', 'cancel', 'save', 'already']):
            print(f"  -> MATCH: '{txt}' at ({l.words[0].bounding_rect.x}, {l.words[0].bounding_rect.y})")
    return res

# 1. Raw
run_ocr(img, "Raw Image")

# 2. Upscaled 1.5x (INTER_CUBIC)
img_1_5x = cv2.resize(img, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
run_ocr(img_1_5x, "Upscaled 1.5x (INTER_CUBIC)")

# 3. Upscaled 2.0x (INTER_CUBIC)
img_2x = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
run_ocr(img_2x, "Upscaled 2.0x (INTER_CUBIC)")

# 4. Upscaled 2.0x + Unsharp Mask (Sharpened)
gaussian = cv2.GaussianBlur(img_2x, (0, 0), 2.0)
unsharp = cv2.addWeighted(img_2x, 1.5, gaussian, -0.5, 0)
run_ocr(unsharp, "Upscaled 2.0x + Unsharp Mask")

# 5. Crop around the modal dialog (approx 380..650 x, 190..320 y)
crop_dialog = img[int(h * 0.35):int(h * 0.62), int(w * 0.37):int(w * 0.65)]
cv2.imwrite("scratch/dialog_crop_raw.png", crop_dialog)
run_ocr(crop_dialog, "Dialog Crop (Raw)")

crop_dialog_2x = cv2.resize(crop_dialog, (crop_dialog.shape[1] * 2, crop_dialog.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("scratch/dialog_crop_2x.png", crop_dialog_2x)
run_ocr(crop_dialog_2x, "Dialog Crop 2x")
