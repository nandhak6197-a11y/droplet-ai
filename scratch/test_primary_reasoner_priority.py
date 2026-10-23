import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("--- Testing Reasoner Priority & Kaggle Auto-Discovery ---")

# Clear environment variables to simulate default launch
os.environ.pop("USE_NGROK", None)
os.environ.pop("ENABLE_KAGGLE_QWEN", None)
os.environ.pop("KAGGLE_QWEN_MODEL", None)

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner

reasoner = DiagonalDropletReasoner()
print(f"use_ngrok:             {reasoner.use_ngrok}")
print(f"ngrok_model:           {reasoner.ngrok_model}")
print(f"Vision enabled:        {reasoner.ngrok_use_vision}")
print(f"Primary key provider:  {reasoner.key_pool[0]['provider']}")
print(f"Primary key masked:    {reasoner.key_pool[0]['masked']}")

assert reasoner.use_ngrok is True, "Expected use_ngrok to be True when Kaggle is live!"
assert "vl" in reasoner.ngrok_model.lower(), f"Expected vision model, got {reasoner.ngrok_model}"
print("\n✔ SUCCESS: Kaggle Qwen 2.5 VL automatically prioritized as primary reasoner!")
