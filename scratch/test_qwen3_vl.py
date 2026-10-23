import time
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

MODEL = "qwen3-vl:30b-a3b-thinking"

print(f"[*] Testing {MODEL} on Kaggle endpoint...")
t0 = time.time()

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a desktop GUI grounding agent. Analyze the candidate manifest and select the exact Dot ID for the user instruction. Return JSON: {\"target_dot_id\": <int>, \"reasoning\": \"<brief>\"}."
            },
            {
                "role": "user",
                "content": "Instruction: 'Open the slide transition panel'.\n\nCandidate Dots:\nDot [14]: [Button] File Menu\nDot [52]: [Tab] Transitions\nDot [93]: [Chevron] Auto Save Settings"
            }
        ],
        temperature=0.01,
        max_tokens=500,
        timeout=90.0
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    print(f"\n[SUCCESS] Response in {dt:.2f}s!")
    print("Content:\n", msg.content)
    if hasattr(msg, "reasoning") and msg.reasoning:
        print("\nReasoning Trace:\n", msg.reasoning[:300], "...")
except Exception as e:
    print(f"\n[ERROR] Call failed: {e}")
