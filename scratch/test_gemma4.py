import urllib.request
import json
import time

URL = "https://greedy-porcupine-precut.ngrok-free.dev"
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"
}

print("[*] Testing gemma4:31b with num_ctx=8192...")
t0 = time.time()
payload = {
    "model": "gemma4:31b",
    "prompt": "Hello Gemma! Confirm you are working and respond with: 'Gemma 4 31B is online.'",
    "stream": False,
    "options": {
        "num_ctx": 8192,
        "temperature": 0.1
    },
    "keep_alive": -1
}

req = urllib.request.Request(
    f"{URL}/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers=HEADERS
)

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        print(f"\n[SUCCESS] Response received in {dt:.2f}s:")
        print("Response:", data.get("response"))
        print("Eval Count:", data.get("eval_count"))
        if data.get("eval_duration"):
            tok_per_sec = data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)
            print(f"Speed: {tok_per_sec:.2f} tokens/sec")
except Exception as e:
    print(f"\n[ERROR] Generation failed: {e}")

# Check VRAM residency
try:
    req_ps = urllib.request.Request(f"{URL}/api/ps", headers=HEADERS)
    with urllib.request.urlopen(req_ps, timeout=15) as resp:
        ps_data = json.loads(resp.read().decode("utf-8"))
        print("\n[*] VRAM Status (/api/ps):")
        for m in ps_data.get("models", []):
            print(f" - {m.get('name')}: {m.get('size_vram')} bytes ({m.get('size_vram', 0)/1e9:.2f} GB) resident in VRAM.")
            print(f"   Context Length: {m.get('context_length', 'default')}, Expires: {m.get('expires_at')}")
except Exception as e:
    print(f"[ERROR] /api/ps check failed: {e}")
