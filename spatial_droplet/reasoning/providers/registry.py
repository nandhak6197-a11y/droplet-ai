"""
Grounding Provider Registry & Factory (Phase 3).
Dynamically instantiates and returns the appropriate provider:
  - "groq": Multi-key 25-key rotator with 3 model tiers
  - "local" / "ollama" / "vllm": Offline local server (rate-limit free)
  - "openai": Official OpenAI / Azure API
  - "auto": Auto-detects available provider
"""
import os
from typing import Optional, Dict

from .base import BaseGroundingProvider
from .gemini_provider import GeminiGroundingProvider
from .groq_provider import GroqGroundingProvider
from .local_provider import LocalGroundingProvider
from .openai_provider import OpenAIGroundingProvider


_CACHED_PROVIDERS: Dict[str, BaseGroundingProvider] = {}


def get_grounding_provider(name: Optional[str] = None, **kwargs) -> BaseGroundingProvider:
    """Returns a grounding provider instance by name or auto-detects."""
    target_name = (name or os.environ.get("SPATIAL_PROVIDER") or "auto").strip().lower()

    if target_name == "auto":
        # Check if local Ollama/vLLM is running first for rate-limit free operation
        local_p = LocalGroundingProvider(**kwargs)
        if local_p.is_available():
            print(f"[ProviderRegistry] 🚀 Auto-detected active local VLM provider at {local_p.base_url}")
            return local_p

        # Check if Gemini key is set (high-capacity multimodal VLM)
        gemini_p = GeminiGroundingProvider(**kwargs)
        if gemini_p.is_available():
            print(f"[ProviderRegistry] 🚀 Auto-detected Gemini VLM provider ({gemini_p.model_name})")
            return gemini_p

        # Check if OpenAI key is set
        openai_p = OpenAIGroundingProvider(**kwargs)
        if openai_p.is_available():
            print(f"[ProviderRegistry] 🚀 Auto-detected OpenAI provider")
            return openai_p

        target_name = "groq"

    if target_name in _CACHED_PROVIDERS:
        return _CACHED_PROVIDERS[target_name]

    if target_name in ("local", "ollama", "vllm", "sglang"):
        provider = LocalGroundingProvider(**kwargs)
    elif target_name in ("gemini", "gemini-pro", "google", "gemini_pro", "gemini-flash", "gemini-1.5-pro", "gemini-2.0-flash"):
        provider = GeminiGroundingProvider(**kwargs)
    elif target_name in ("openai", "azure"):
        provider = OpenAIGroundingProvider(**kwargs)
    elif target_name in ("groq", "groq_rotator"):
        provider = GroqGroundingProvider(**kwargs)
    else:
        # Default to Groq
        provider = GroqGroundingProvider(**kwargs)

    _CACHED_PROVIDERS[target_name] = provider
    return provider
