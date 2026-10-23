from openai import OpenAI
import os

key = os.environ.get("GROQ_API_KEY", "")
if not key:
    # Try loading from .groq_keys
    base_dir = os.path.dirname(__file__)
    groq_keys_path = os.path.join(base_dir, ".groq_keys")
    if os.path.exists(groq_keys_path):
        with open(groq_keys_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            if lines:
                key = lines[0]

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key or "placeholder")

test_models = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-specdec",
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-3b-preview",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
]

for m in test_models:
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10
        )
        print(f"[SUCCESS] {m}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
