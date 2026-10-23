"""
OpenAI Multimodal Grounding Provider (Phase 3).
Supports official OpenAI models (gpt-4o, gpt-4o-mini, o3-mini) and OpenAI-compatible gateways.
Full Vision & Text support with high-resolution image payloads.
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


class OpenAIGroundingProvider(BaseGroundingProvider):
    """
    Multimodal Grounding Provider using official OpenAI or compatible Vision-Language APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
        self.model_name = (model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")).strip()
        self.timeout = timeout
        self.client = None

        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Returns True if OpenAI client is initialized and an API key is present."""
        return self.client is not None and bool(self.api_key)

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
            return GroundingDecision(
                error="OpenAI client is not initialized. Please ensure the 'openai' package is installed and OPENAI_API_KEY is configured.",
                confidence=0.0
            )

        model = kwargs.get("target_model") or self.model_name

        user_content = f"""<user_request>
{instruction}
</user_request>

<active_window>
{win_info}
</active_window>

<screen_droplet_manifest>
{manifest_text}
</screen_droplet_manifest>

<instruction>
Select the single best candidate Dot ID from the manifest that matches the user request.
Respond with ONLY valid JSON containing:
{{
  "target_dot_id": <int>,
  "action": "click",
  "reasoning": "<1 sentence justifying the choice>"
}}
</instruction>"""

        system_instruction = (
            "You are an expert GUI visual grounding agent. "
            "Examine the screen manifest and optional image to choose the exact candidate Dot ID. "
            "Respond ONLY with valid JSON."
        )

        try:
            if som_image_url:
                user_msg_content = [
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": som_image_url,
                            "detail": "high"
                        }
                    }
                ]
            else:
                user_msg_content = user_content

            call_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_msg_content}
                ],
                "temperature": 0.1,
                "max_tokens": 300,
                "timeout": self.timeout
            }

            # If model supports response_format
            if any(m in model.lower() for m in ("gpt-4o", "gpt-4-turbo", "o3", "o1")):
                call_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**call_kwargs)
            raw = response.choices[0].message.content or ""
            decision = self.parse_raw_response(raw, all_candidates)
            decision.model_used = model
            decision.provider_used = "openai"
            decision.key_used = self.api_key[:8] + "..." if self.api_key else ""
            return decision

        except Exception as e:
            return GroundingDecision(
                error=f"OpenAI provider error with model {model}: {e}",
                confidence=0.0,
                reasoning=f"OpenAI API call failed: {e}"
            )
