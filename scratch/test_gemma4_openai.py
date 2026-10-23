import time
from openai import OpenAI

client = OpenAI(
    base_url="https://greedy-porcupine-precut.ngrok-free.dev/v1",
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

print("[*] Testing OpenAI /v1/chat/completions with gemma4:31b...")
t0 = time.time()

try:
    resp = client.chat.completions.create(
        model="gemma4:31b",
        messages=[
            {"role": "system", "content": "You are a JSON-only API. Respond with ONLY valid JSON."},
            {"role": "user", "content": "Analyze this request: 'Click search filter button'. Candidates: Dot 1: Filter, Dot 2: Cancel. Return JSON with target_dot_id."}
        ],
        temperature=0.01,
        max_tokens=500,
        timeout=60.0
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    print(f"\n[SUCCESS] Completed in {dt:.2f}s!")
    print("Content:", msg.content)
    if hasattr(msg, "reasoning") and msg.reasoning:
        print("Reasoning:", msg.reasoning)
except Exception as e:
    print(f"\n[ERROR] OpenAI compatibility call failed: {e}")
