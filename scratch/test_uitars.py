import urllib.request
import json
import time

URL = "https://greedy-porcupine-precut.ngrok-free.dev"
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"
}

MODEL = "0000/ui-tars-1.5-7b:latest"

print(f"[*] Loading and warming up {MODEL} with keep_alive: -1 and num_ctx=8192...")
t0 = time.time()
payload = {
    "model": MODEL,
    "prompt": "You are a GUI agent. Confirm you are online and working.",
    "stream": False,
    "options": {
        "num_ctx": 8192,
        "temperature": 0.01
    },
    "keep_alive": -1
}

req = urllib.request.Request(
    f"{URL}/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers=HEADERS
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        print(f"\n[SUCCESS] Response received in {dt:.2f}s:")
        print("Response:", data.get("response"))
        print("Eval Count:", data.get("eval_count"))
        if data.get("eval_duration"):
            tok_per_sec = data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)
            print(f"Generation Speed: {tok_per_sec:.2f} tokens/sec")
except Exception as e:
    print(f"\n[ERROR] Loading failed: {e}")

# Check VRAM residency
try:
    req_ps = urllib.request.Request(f"{URL}/api/ps", headers=HEADERS)
    with urllib.request.urlopen(req_ps, timeout=15) as resp:
        ps_data = json.loads(resp.read().decode("utf-8"))
        print("\n[*] VRAM Status (/api/ps):")
        for m in ps_data.get("models", []):
            print(f" - {m.get('name')}: {m.get('size_vram', 0)/1e9:.2f} GB resident in VRAM. Expires: {m.get('expires_at')}")
except Exception as e:
    print(f"[ERROR] /api/ps check failed: {e}")
