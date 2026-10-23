import cv2
import winocr
import re
from concurrent.futures import ThreadPoolExecutor

img = cv2.imread(r"C:/Users/NANDHA K/.gemini/antigravity/brain/77172af1-65f6-41cc-bf5d-0a2f27cf94ff/.user_uploaded/media_1791998496786_2d52b8ac.png")

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
    return ""

DIALOG_TOKENS = {'Cancel', 'Save', 'Yes', 'No', 'OK', "Don't Save", 'Retry', 'Ignore', 'Abort', 'Apply', 'Open', 'Close'}

def resolve(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 12:
        return ""
    h, w = crop.shape[:2]
    mean_val = [int(v) for v in crop.mean(axis=(0, 1))]
    for scale in [3.0, 4.0]:
        up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        pad = int(12 * scale)
        padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_val)
        inv = 255 - padded
        
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        
        for p in [padded, inv, enh]:
            op = winocr.recognize_cv2(p, lang='en')
            r = op.get() if hasattr(op, 'get') else op
            for line in r.lines:
                tok = clean_dialog_token(line.text)
                if tok in DIALOG_TOKENS:
                    return tok
    return ""

def test():
    for name, (x1, y1, x2, y2) in [
        ('Yes', (528, 276, 576, 293)),
        ('No', (579, 276, 627, 293)),
        ('Save', (897, 519, 954, 535)),
        ('Cancel', (965, 519, 1021, 535))
    ]:
        crop = img[y1:y2, x1:x2]
        print(f"{name} -> {resolve(crop)}")

ThreadPoolExecutor().submit(test).result()
