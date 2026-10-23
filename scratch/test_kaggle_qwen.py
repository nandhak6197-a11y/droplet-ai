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
    "Authorization": "Bearer ollama"
}

print(f"[*] Testing Kaggle Qwen endpoint: {base_url}")
print(f"[*] Model: qwen3.8:27b")

# Test 1: Simple hello / sanity test
t0 = time.time()
payload_simple = {
    "model": "qwen3.8:27b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in 5 words."}
    ],
    "temperature": 0.1,
    "max_tokens": 50
}

req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps(payload_simple).encode("utf-8"),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        elapsed = time.time() - t0
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"]
        print(f"[+] Test 1 Passed ({elapsed:.2f}s): {content.strip()}")
except Exception as e:
    print(f"[-] Test 1 Failed: {e}")
    sys.exit(1)

# Test 2: Spatial Droplet Trajectory Reasoning Decision
print("\n[*] Test 2: Testing Spatial Droplet deliberative cognitive decision...")

droplet_manifest = """
[139] (552, 284) [Btn] Yes
[140] (603, 284) [Btn] No
[199] (925, 527) [Btn] Save
[200] (993, 527) [Btn] Cancel
[258] (1765, 589) [WIDG] shopping-app
"""

system_prompt = """You are an autonomous GUI agent. Select the next action to achieve the user's goal based on the screen droplet manifest.
Output your decision strictly as JSON with keys:
"observation", "critique", "strategy", "action" (click, double_click, type, press, stop), "target_dot_id", "input_text"
"""

user_goal = "We are at the 'Confirm Save As' dialog. Click the 'Yes' button to confirm overwriting."

payload_reasoning = {
    "model": "qwen3.8:27b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Manifest:\n{droplet_manifest}\n\nGoal: {user_goal}"}
    ],
    "temperature": 0.1,
    "max_tokens": 512
}

t1 = time.time()
req2 = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=json.dumps(payload_reasoning).encode("utf-8"),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req2, timeout=60, context=ctx) as resp:
        elapsed2 = time.time() - t1
        res2 = json.loads(resp.read().decode("utf-8"))
        content2 = res2["choices"][0]["message"]["content"]
        print(f"[+] Test 2 Passed ({elapsed2:.2f}s):\n{content2.strip()}")
except Exception as e:
    print(f"[-] Test 2 Failed: {e}")
