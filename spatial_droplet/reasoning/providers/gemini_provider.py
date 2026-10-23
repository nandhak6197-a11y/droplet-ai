"""
Google Gemini Multimodal Grounding Provider (Milestone 2).
Uses official Google OpenAI-compatible endpoint:
https://generativelanguage.googleapis.com/v1beta/openai/
Supports gemini-1.5-pro, gemini-2.0-flash, gemini-1.5-flash with high-resolution Set-of-Marks visual grounding.
"""
import os
import re
from typing import Optional, List, Dict, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .base import BaseGroundingProvider, GroundingDecision
from ...core.models import DiagonalDroplet


class GeminiGroundingProvider(BaseGroundingProvider):
    """
    Multimodal Grounding Provider for Google Gemini via the official OpenAI-compatible endpoint.
    Supports high-resolution Set-of-Marks visual grounding, 5-stage CoT deliberation traces,
    and structured JSON decision generation.
    """
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_MODEL = "gemini-1.5-pro"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0
    ):
        # 4-Tier API Key Resolution:
        # 1. Explicit api_key parameter
        # 2. GEMINI_API_KEY environment variable
        # 3. GOOGLE_API_KEY environment variable
        # 4. OPENAI_API_KEY environment variable fallback
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()

        # Base URL resolution
        self.base_url = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).strip()

        # Model name resolution
        self.model_name = (
            model_name
            or os.environ.get("GEMINI_MODEL")
            or self.DEFAULT_MODEL
        ).strip()

        self.timeout = timeout
        self.client = None

        if HAS_OPENAI and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                pass

    def is_available(self) -> bool:
        """Returns True if Gemini client is initialized and an API key is present."""
        return self.client is not None and bool(self.api_key)

    def ground_sync(
        self,
        instruction: str,
        manifest_text: str,
        all_candidates: List[DiagonalDroplet],
        som_image_url: Optional[str] = None,
        win_info: str = "Desktop",
        system_instruction: Optional[str] = None,
        user_content: Optional[str] = None,
        **kwargs
    ) -> GroundingDecision:
        """Synchronous grounding execution for synchronous callers (e.g. decide_turn)."""
        if not self.client:
            return GroundingDecision(
                error=(
                    "Gemini client is not initialized. Please ensure the 'openai' package is installed "
                    "and GEMINI_API_KEY (or GOOGLE_API_KEY / OPENAI_API_KEY) is configured."
                ),
                confidence=0.0,
                provider_used="gemini",
                model_used=self.model_name
            )

        model = kwargs.get("target_model") or self.model_name

        if user_content is None:
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
Respond with ONLY valid JSON containing 5-stage deliberation:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "<query decomposition: target application & completion proof>",
    "screen_state_and_visual_hierarchy": "<active window, zones, modal dialog precedence>",
    "mistake_critique_and_loop_diagnosis": "<critique of previous step outcome; diagnose root causes>",
    "end_to_end_master_plan": ["Phase 1: <step> (DONE/ACTIVE/PENDING)"],
    "tactical_action_justification": "<step-by-step rationale for chosen action & Dot ID, distractor elimination>"
  }},
  "target_dot_id": <int>,
  "action": "click",
  "reasoning": "<1 sentence justifying the choice>"
}}
</instruction>"""

        if system_instruction is None:
            system_instruction = (
                "You are an expert GUI visual grounding agent with high-resolution multimodal perception. "
                "Examine the screen manifest, spatial layout, and high-resolution Set-of-Marks image to select the exact candidate Dot ID. "
                "Respond ONLY with valid JSON."
            )

        try:
            if som_image_url:
                clean_img_url = som_image_url.strip()
                if (
                    not clean_img_url.startswith("data:image/")
                    and not clean_img_url.startswith("http://")
                    and not clean_img_url.startswith("https://")
                ):
                    clean_img_url = f"data:image/jpeg;base64,{clean_img_url}"

                user_msg_content: Any = [
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": clean_img_url,
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
                "max_tokens": 1200,
                "timeout": self.timeout
            }

            if any(m in model.lower() for m in ("gemini", "gpt", "json")):
                call_kwargs["response_format"] = {"type": "json_object"}

            try:
                response = self.client.chat.completions.create(**call_kwargs)
            except Exception as e:
                # Resilient fallback if endpoint rejects response_format
                if "response_format" in str(e).lower() and "response_format" in call_kwargs:
                    call_kwargs.pop("response_format", None)
                    response = self.client.chat.completions.create(**call_kwargs)
                else:
                    raise

            raw = response.choices[0].message.content or ""
            decision = self.parse_raw_response(raw, all_candidates)
            decision.model_used = model
            decision.provider_used = "gemini"
            decision.key_used = self.api_key[:8] + "..." if self.api_key else ""
            return decision

        except Exception as e:
            return GroundingDecision(
                error=f"Gemini provider error with model {model}: {e}",
                confidence=0.0,
                reasoning=f"Gemini API call failed: {e}",
                provider_used="gemini",
                model_used=model
            )

    async def ground(
        self,
        instruction: str,
        manifest_text: str,
        all_candidates: List[DiagonalDroplet],
        som_image_url: Optional[str] = None,
        win_info: str = "Desktop",
        **kwargs
    ) -> GroundingDecision:
        """Asynchronous grounding execution for async callers (e.g. decide)."""
        return self.ground_sync(
            instruction=instruction,
            manifest_text=manifest_text,
            all_candidates=all_candidates,
            som_image_url=som_image_url,
            win_info=win_info,
            **kwargs
        )
