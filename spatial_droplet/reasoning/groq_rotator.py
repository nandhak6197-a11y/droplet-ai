"""
Groq Multi-Key Rotator (25 API Keys).
Rotates requests across a pool of Groq API keys with automatic model failover
(openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b, qwen/qwen3.6-27b).
"""
import os
import threading
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _load_groq_keys() -> List[str]:
    """Loads Groq API keys from environment variables or gitignored local file."""
    # 1. Comma-separated list in GROQ_API_KEYS
    env_keys = os.environ.get("GROQ_API_KEYS", "").strip()
    if env_keys:
        return [k.strip() for k in env_keys.split(",") if k.strip()]
    # 2. Single key in GROQ_API_KEY
    single_key = os.environ.get("GROQ_API_KEY", "").strip()
    if single_key:
        return [single_key]
    # 3. Local gitignored keyfile (.groq_keys)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_keys_path = os.path.join(base_dir, ".groq_keys")
    if os.path.exists(local_keys_path):
        try:
            with open(local_keys_path, "r", encoding="utf-8") as f:
                keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                if keys:
                    return keys
        except Exception:
            pass
    return []


GROQ_API_KEYS = _load_groq_keys()

GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

POLLINATIONS_BASE_URL = "https://gen.pollinations.ai/v1"
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")
POLLINATIONS_MODELS = [
    os.environ.get(
        "POLLINATIONS_MODEL",
        "community/YoannDev90/poolside-laguna-s-2.1:free",
    )
]

class GroqRotator:
    """Manages rotation across the 25-key pool and model failover."""
    def __init__(self, keys: Optional[List[str]] = None, models: Optional[List[str]] = None):
        self.keys = keys or GROQ_API_KEYS
        self.models = models or GROQ_MODELS
        self.key_pool: List[Dict[str, Any]] = []
        if HAS_OPENAI:
            for key in self.keys:
                try:
                    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
                    masked = f"{key[:8]}...{key[-6:]}" if len(key) >= 14 else "***"
                    self.key_pool.append({"key": key, "client": client, "masked": masked, "provider": "groq"})
                except Exception:
                    pass
            # Kaggle Dual-T4 Qwen endpoint prioritized if ngrok is active
            if os.environ.get("USE_NGROK", "0").strip() in {"1", "true", "yes"} or os.environ.get("ENABLE_KAGGLE_QWEN", "0").strip() in {"1", "true", "yes"} or os.environ.get("KAGGLE_QWEN_URL"):
                kaggle_url = os.environ.get("KAGGLE_QWEN_URL", "https://greedy-porcupine-precut.ngrok-free.dev/v1")
                kaggle_model = os.environ.get("KAGGLE_QWEN_MODEL", "qwen2.5vl:7b")
                try:
                    client = OpenAI(
                        base_url=kaggle_url,
                        api_key="ollama",
                        default_headers={"ngrok-skip-browser-warning": "true"},
                        timeout=90.0
                    )
                    self.key_pool.insert(0, {
                        "key": "kaggle_qwen",
                        "client": client,
                        "masked": f"kaggle:{kaggle_model}",
                        "provider": "kaggle",
                        "models": [kaggle_model]
                    })
                except Exception:
                    pass
        self.current_idx = 0
        self.lock = threading.Lock()
        if self.key_pool:
            primary = self.key_pool[0].get("provider", "groq")
            print(f"[*] LLM rotator initialized with {len(self.key_pool)} active keys; primary={primary}")

    def get_client(self) -> Optional[Dict[str, Any]]:
        if not self.key_pool:
            return None
        with self.lock:
            entry = self.key_pool[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.key_pool)
            return entry
