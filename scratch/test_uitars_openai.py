import time
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

MODEL = "0000/ui-tars-1.5-7b:latest"

print(f"[*] Testing OpenAI /v1/chat/completions with {MODEL}...")
t0 = time.time()

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a GUI visual grounding agent. Analyze the candidate manifest and identify the exact target Dot ID matching the user intent. Respond with valid JSON: {\"target_dot_id\": <int>, \"reasoning\": \"<rationale>\"}."},
            {"role": "user", "content": "Instruction: 'Click the filter settings button'. Candidates:\nDot [12]: [Button] Save File\nDot [42]: [Icon] Filter Settings Options\nDot [99]: [Button] Cancel"}
        ],
        temperature=0.01,
        max_tokens=300,
        timeout=60.0
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    print(f"\n[SUCCESS] Completed in {dt:.2f}s!")
    print("Content:", msg.content)
    if hasattr(msg, "reasoning") and msg.reasoning:
        print("Reasoning trace:", msg.reasoning)
except Exception as e:
    print(f"\n[ERROR] OpenAI call failed: {e}")
