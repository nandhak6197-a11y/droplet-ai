import time
import base64
import cv2
from openai import OpenAI

client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

# Load a small crop or sample image
img = cv2.imread("osworld_g_images/0FOB4CLBT2.png")
if img is not None:
    # Resize down to 800px width for fast network transfer
    h, w = img.shape[:2]
    new_w = 800
    new_h = int(h * (800 / w))
    small = cv2.resize(img, (new_w, new_h))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode("utf-8")
    
    print(f"[*] Testing multimodal image input to qwen3-vl:30b-a3b-thinking (image bytes: {len(b64)})...")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="qwen3-vl:30b-a3b-thinking",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this desktop UI screenshot in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            max_tokens=150,
            timeout=60.0
        )
        print(f"[SUCCESS] Multimodal call took {time.time() - t0:.2f}s!")
        print("Reply:", resp.choices[0].message.content)
    except Exception as e:
        print("[ERROR] Multimodal call failed:", e)
else:
    print("[ERROR] Could not load image")
