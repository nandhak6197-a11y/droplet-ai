"""
Groq Multi-Key Rotator Grounding Provider (Phase 3).
Rotates across 25 keys with 3-tier fallback (Qwen 3.8 27B -> GPT-OSS 120B -> GPT-OSS 20B).
"""
import time
import os
import threading
from typing import Optional, List, Dict, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .base import BaseGroundingProvider, GroundingDecision
from ..groq_rotator import GROQ_API_KEYS
from ...core.models import DiagonalDroplet


class GroqGroundingProvider(BaseGroundingProvider):
    """Production Groq provider with multi-key rotation and 3-tier failover."""

    def __init__(self, keys: Optional[List[str]] = None, timeout: float = 45.0):
        self.keys = keys or GROQ_API_KEYS
        self.timeout = timeout
        self.key_pool: List[Dict[str, Any]] = []
        self._key_cooldowns: Dict[tuple, float] = {}
        self.current_idx = 0
        self.lock = threading.Lock()

        if HAS_OPENAI:
            for key in self.keys:
                try:
                    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
                    masked = f"{key[:8]}...{key[-6:]}"
                    self.key_pool.append({"key": key, "client": client, "masked": masked})
                except Exception:
                    pass

    async def ground(
        self,
        instruction: str,
        manifest_text: str,
        all_candidates: List[DiagonalDroplet],
        som_image_url: Optional[str] = None,
        win_info: str = "Desktop",
        **kwargs
    ) -> GroundingDecision:
        if not self.key_pool:
            return GroundingDecision(error="No Groq clients available in pool", confidence=0.0)

        num_keys = len(self.key_pool)
        with self.lock:
            start_idx = self.current_idx % num_keys
            self.current_idx = (self.current_idx + 1) % num_keys

        target_model = kwargs.get("target_model")
        if target_model:
            model_tiers = [target_model]
            if target_model == "qwen/qwen3.8-27b":
                model_tiers.append("openai/gpt-oss-120b")
                model_tiers.append("openai/gpt-oss-20b")
        else:
            model_tiers = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

        user_content = f"""<user_request>
{instruction}
</user_request>

<screen_droplet_manifest>
{manifest_text}
</screen_droplet_manifest>

<instruction>
Select the single best candidate Dot ID from the manifest that matches the user request. Respond with valid JSON.
</instruction>"""

        system_instruction = (
            "You are an expert GUI visual grounding agent. Choose the exact candidate Dot ID from the manifest that matches the user request. Respond with ONLY valid JSON."
        )

        now = time.time()
        last_err = ""

        for model_name in model_tiers:
            for offset in range(num_keys):
                active_idx = (start_idx + offset) % num_keys
                entry = self.key_pool[active_idx]
                client = entry["client"]
                masked_key = entry["masked"]

                cooldown_until = self._key_cooldowns.get((masked_key, model_name), 0.0)
                if now < cooldown_until:
                    continue

                try:
                    if "qwen" in model_name and som_image_url:
                        user_msg_content = [
                            {"type": "text", "text": user_content},
                            {"type": "image_url", "image_url": {"url": som_image_url}}
                        ]
                    else:
                        user_msg_content = user_content

                    call_kwargs = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_msg_content}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 200,
                        "timeout": self.timeout
                    }
                    if "gpt-oss" in model_name:
                        call_kwargs["reasoning_effort"] = "low"

                    response = client.chat.completions.create(**call_kwargs)
                    raw = response.choices[0].message.content or ""
                    decision = self.parse_raw_response(raw, all_candidates)
                    decision.model_used = model_name
                    decision.provider_used = "groq"
                    decision.key_used = f"Key #{active_idx + 1} ({masked_key})"

                    with self.lock:
                        self.current_idx = (active_idx + 1) % num_keys
                    return decision

                except Exception as e:
                    err_str = str(e)
                    last_err = f"Key #{active_idx + 1} ({masked_key}) [{model_name}]: {err_str}"
                    if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                        self._key_cooldowns[(masked_key, model_name)] = time.time() + 300.0
                    elif "429" in err_str or "rate limit" in err_str.lower():
                        self._key_cooldowns[(masked_key, model_name)] = time.time() + 10.0
                    continue

        return GroundingDecision(
            error=f"All Groq keys failed. Last: {last_err}",
            confidence=0.0,
            reasoning=f"Failures across Groq keys: {last_err}"
        )
