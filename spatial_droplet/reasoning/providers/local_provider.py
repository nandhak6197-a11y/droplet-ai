"""
Local & Offline Grounding Provider (Phase 3).
Supports Ollama, vLLM, LMDeploy, LocalAI, and any OpenAI-compatible local server.
100% Rate-Limit Free and Offline Capable.
"""
import os
from typing import Optional, List, Dict, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .base import BaseGroundingProvider, GroundingDecision
from ...core.models import DiagonalDroplet


class LocalGroundingProvider(BaseGroundingProvider):
    """
    Local grounding provider for offline execution without rate limits.
    Connects to Ollama (default: http://localhost:11434/v1) or
    vLLM / LMDeploy / LocalAI (default: http://localhost:8000/v1).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "ollama",
        timeout: float = 60.0
    ):
        self.base_url = base_url or os.environ.get("LOCAL_VLM_URL") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model_name = model_name or os.environ.get("LOCAL_VLM_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.timeout = timeout
        self.client = None

        if HAS_OPENAI:
            try:
                headers = {"ngrok-skip-browser-warning": "true"}
                self.client = OpenAI(base_url=self.base_url, api_key=api_key, default_headers=headers)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Checks if local server is reachable."""
        if not self.client:
            return False
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    async def ground(
        self,
        instruction: str,
        manifest_text: str,
        all_candidates: List[DiagonalDroplet],
        som_image_url: Optional[str] = None,
        win_info: str = "Desktop",
        **kwargs
    ) -> GroundingDecision:
        if not self.client:
            return GroundingDecision(error=f"Local client not initialized for {self.base_url}", confidence=0.0)

        model = kwargs.get("target_model") or self.model_name

        user_content = f"""<user_request>
{instruction}
</user_request>

<screen_droplet_manifest>
{manifest_text}
</screen_droplet_manifest>

<instruction>
Select the single best candidate Dot ID from the manifest that matches the user request. Respond with ONLY valid JSON:
{{"target_dot_id": <int>, "action": "click", "reasoning": "<1 sentence>"}}
</instruction>"""

        system_instruction = (
            "You are an expert GUI visual grounding agent. Choose the exact candidate Dot ID from the manifest that matches the user request. Respond with ONLY valid JSON."
        )

        try:
            if som_image_url and any(vm in model.lower() for vm in ("vl", "vision", "llava", "qwen2.5-vl")):
                user_msg_content = [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": som_image_url}}
                ]
            else:
                user_msg_content = user_content

            call_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_msg_content}
                ],
                "temperature": 0.1,
                "max_tokens": 512,
                "timeout": self.timeout
            }

            response = self.client.chat.completions.create(**call_kwargs)
            msg = response.choices[0].message
            raw = (msg.content or getattr(msg, "reasoning", "") or "").strip()
            decision = self.parse_raw_response(raw, all_candidates)
            decision.model_used = model
            decision.provider_used = "local"
            decision.key_used = self.base_url
            return decision

        except Exception as e:
            return GroundingDecision(
                error=f"Local provider error at {self.base_url}: {e}",
                confidence=0.0,
                reasoning=f"Local provider call failed: {e}"
            )
