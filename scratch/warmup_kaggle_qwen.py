import sys
import time
import json
import urllib.request
import ssl

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

base_url = "https://greedy-porcupine-precut.ngrok-free.dev"
headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "curl/7.68.0",
}

print(f"[*] Warming up and loading qwen2.5vl:7b on Kaggle...")
t0 = time.time()

# Use Ollama /api/generate
payload = {
    "model": "qwen2.5vl:7b",
    "prompt": "Hello! Confirm in 5 words that you are live.",
    "stream": False,
    "keep_alive": -1
}

req = urllib.request.Request(
    f"{base_url}/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        ans = res.get("response", "")
        total_duration = res.get("total_duration", 0) / 1e9
        load_duration = res.get("load_duration", 0) / 1e9
        eval_count = res.get("eval_count", 0)
        eval_duration = res.get("eval_duration", 0) / 1e9
        rate = eval_count / eval_duration if eval_duration > 0 else 0
        print(f"[+] Model Response:\n    \"{ans.strip()}\"")
        print(f"\n[Stats] Total: {total_duration:.2f}s | Load into VRAM: {load_duration:.2f}s | Eval: {eval_count} tokens @ {rate:.2f} t/s")
except Exception as e:
    print(f"\n[-] Error during generate: {e}")
