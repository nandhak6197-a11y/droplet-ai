import urllib.request
import json
import ssl
import time
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

base_url = 'https://greedy-porcupine-precut.ngrok-free.dev'
headers = {
    'ngrok-skip-browser-warning': 'true',
    'User-Agent': 'curl/7.68.0',
    'Content-Type': 'application/json'
}

print(f"Connecting to Kaggle endpoint: {base_url}")

try:
    req = urllib.request.Request(f"{base_url}/api/tags", headers=headers)
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        data = json.loads(resp.read().decode())
        models = data.get("models", [])
        print(f"[+] Total Models in Ollama: {len(models)}")
        
        target_model = None
        for m in models:
            name = m.get("name", "")
            size_gb = m.get("size", 0) / (1024**3)
            print(f"    - {name} ({size_gb:.2f} GB)")
            if "qwen" in name.lower() or "vl" in name.lower():
                target_model = name
        
        if not target_model and models:
            target_model = models[0].get("name")

        if target_model:
            print(f"\n[*] Testing inference with model: '{target_model}'...")
            t0 = time.time()
            payload = {
                "model": target_model,
                "prompt": "Hello! Confirm in 5 words that you are live and ready.",
                "stream": False
            }
            req_gen = urllib.request.Request(
                f"{base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req_gen, timeout=180, context=ctx) as gen_resp:
                elapsed = time.time() - t0
                res_data = json.loads(gen_resp.read().decode("utf-8"))
                response_text = res_data.get("response", "")
                eval_count = res_data.get("eval_count", 0)
                eval_duration = res_data.get("eval_duration", 0) / 1e9
                speed = eval_count / eval_duration if eval_duration > 0 else 0
                print(f"[+] INFERENCE SUCCESSFUL! ({elapsed:.2f}s)")
                print(f"[+] Output: {response_text.strip()}")
                print(f"[+] Speed: {eval_count} tokens @ {speed:.2f} tokens/sec")
        else:
            print("[-] No model listed in /api/tags yet. The pull might still be downloading or copying layers on Kaggle.")
except Exception as e:
    print(f"[-] Error: {e}")
