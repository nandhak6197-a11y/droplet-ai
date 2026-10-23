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

def query_qwen(prompt, manifest, history=None):
    system_prompt = (
        "You are an autonomous desktop GUI agent. Analyze the screen manifest and previous history to select the next action.\n"
        "Output strict JSON with keys:\n"
        "- observation: string\n"
        "- critique: string (self-critical analysis of previous step or state)\n"
        "- strategy: string\n"
        "- action: 'click' | 'double_click' | 'type' | 'press' | 'stop'\n"
        "- target_dot_id: int or null\n"
        "- input_text: string or null\n"
    )
    user_content = f"Active Manifest:\n{manifest}\n"
    if history:
        user_content += f"\nPrevious Steps History:\n{history}\n"
    user_content += f"\nCurrent Goal: {prompt}"

    payload = {
        "model": "qwen3.8:27b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        return res["choices"][0]["message"]["content"], elapsed

# Scenario A: Folder open
manifest_a = """
[10] (45, 120) [Icon] Folder 'src'
[11] (45, 160) [Icon] Folder 'shopping-app'
[12] (45, 200) [Icon] File 'package.json'
[99] (500, 700) [Btn] Select Folder
"""
goal_a = "Open the 'shopping-app' project folder."
resp_a, t_a = query_qwen(goal_a, manifest_a)
print(f"--- Scenario A (Folder Open) [{t_a:.2f}s] ---")
print(resp_a.strip())

# Scenario B: Text input
manifest_b = """
[50] (200, 300) [Edit] File name: 'Untitled'
[51] (500, 500) [Btn] Save
[52] (580, 500) [Btn] Cancel
"""
goal_b = "Type 'shopping_app_v2' into the file name box."
resp_b, t_b = query_qwen(goal_b, manifest_b)
print(f"\n--- Scenario B (Text Typing) [{t_b:.2f}s] ---")
print(resp_b.strip())

# Scenario C: Stop when finished
manifest_c = """
[1] (100, 200) [Text] Workspace 'shopping-app' loaded successfully in VS Code
[2] (100, 300) [Icon] Explorer
"""
goal_c = "Open shopping-app in VS Code"
history_c = "Turn 1: CLICK dot 11 ('shopping-app') -> Confirmed\nTurn 2: PRESS 'enter' -> Confirmed"
resp_c, t_c = query_qwen(goal_c, manifest_c, history=history_c)
print(f"\n--- Scenario C (Completion / Stop) [{t_c:.2f}s] ---")
print(resp_c.strip())
