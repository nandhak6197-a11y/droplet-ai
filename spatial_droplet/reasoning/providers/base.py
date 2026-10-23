"""
Base Grounding Provider Interface & Data Models (Phase 3).
Standardizes multimodal grounding across Groq, Ollama, vLLM, OpenAI, Anthropic, and local endpoints.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re
from typing import Optional, Dict, Any, List

from ...core.models import DiagonalDroplet


@dataclass
class GroundingDecision:
    """Standardized multimodal grounding result returned by any provider."""
    target_dot_id: Optional[int] = None
    action: str = "click"
    x: Optional[float] = None
    y: Optional[float] = None
    drag_destination_dot_id: Optional[int] = None
    drag_dest_x: Optional[float] = None
    drag_dest_y: Optional[float] = None
    confidence: float = 1.0
    reasoning: str = ""
    deliberation: Dict[str, Any] = field(default_factory=dict)
    model_used: str = ""
    provider_used: str = ""
    key_used: str = ""
    selection_source: str = "model"
    manifest_candidate_ids: List[int] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_dot_id": self.target_dot_id,
            "target_droplet_id": self.target_dot_id,
            "target_id": self.target_dot_id,
            "action": self.action,
            "action_type": self.action,
            "target_x": self.x,
            "target_y": self.y,
            "x": self.x,
            "y": self.y,
            "drag_destination_dot_id": self.drag_destination_dot_id,
            "drag_dest_x": self.drag_dest_x,
            "drag_dest_y": self.drag_dest_y,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "deliberation": self.deliberation,
            "model_used": self.model_used,
            "provider": self.provider_used,
            "provider_used": self.provider_used,
            "key_used": self.key_used,
            "selection_source": self.selection_source,
            "manifest_candidate_ids": self.manifest_candidate_ids,
            "error": self.error
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        return d.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        return d.get(key, default)


class BaseGroundingProvider(ABC):
    """Abstract interface for all Vision-Language and Language Grounding Providers."""

    @abstractmethod
    async def ground(
        self,
        instruction: str,
        manifest_text: str,
        all_candidates: List[DiagonalDroplet],
        som_image_url: Optional[str] = None,
        win_info: str = "Desktop",
        **kwargs
    ) -> GroundingDecision:
        """Dispatches multimodal reasoning to ground instruction into a specific Dot ID."""
        pass

    @staticmethod
    def parse_raw_response(
        raw_text: str,
        all_candidates: List[DiagonalDroplet]
    ) -> GroundingDecision:
        """
        Resilient cognitive parser: recovers structured decision from JSON,
        partially malformed JSON, and markdown-wrapped or raw text blocks.
        """
        clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        clean = clean.replace('```json', '').replace('```', '').strip()

        cand_ids = {d.id for d in all_candidates}
        fb = clean.find('{')
        lb = clean.rfind('}')
        data: Optional[Dict[str, Any]] = None

        if fb != -1 and lb != -1 and lb > fb:
            json_str = clean[fb:lb + 1]
            try:
                data = json.loads(json_str)
            except Exception:
                try:
                    # Clean trailing commas
                    data = json.loads(re.sub(r',\s*([}\]])', r'\1', json_str))
                except Exception:
                    pass

        if data is None:
            # Fallback regex extraction
            m_id = re.search(r'["\']?(?:target_dot_id|target_id|dot_id|id)["\']?\s*:\s*["\']?(?:dot[\s_\[#:]*)?(\d+)', clean, re.IGNORECASE)
            m_act = re.search(r'["\']?action["\']?\s*:\s*["\']([a-z_]+)["\']', clean)
            m_x = re.search(r'["\']?x["\']?\s*:\s*(\d+(?:\.\d+)?)', clean)
            m_y = re.search(r'["\']?y["\']?\s*:\s*(\d+(?:\.\d+)?)', clean)
            m_coord = re.search(r'["\']?(?:coordinate|point)["\']?\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', clean)
            target_num = int(m_id.group(1)) if m_id else None
            data = {
                "target_dot_id": target_num,
                "action": m_act.group(1) if m_act else "click",
                "x": float(m_x.group(1)) if m_x else (float(m_coord.group(1)) if m_coord else None),
                "y": float(m_y.group(1)) if m_y else (float(m_coord.group(2)) if m_coord else None),
                "reasoning": clean[:180].replace("\n", " ")
            }

        # Resolve target ID
        raw_id = data.get("target_dot_id") or data.get("target_id") or data.get("dot_id") or data.get("id")
        if isinstance(raw_id, list) and raw_id:
            raw_id = raw_id[0]
        elif isinstance(raw_id, str):
            m = re.search(r'\d+', raw_id)
            raw_id = int(m.group()) if m else None

        # Check deliberation if top-level ID is missing
        if raw_id is None:
            delib = data.get("deliberation", {})
            text_eval = str(delib.get("action_plan", "")) + " " + str(data.get("reasoning", "")) + " " + str(delib.get("candidate_evaluation", ""))
            mentions = [int(m) for m in re.findall(r'(?:\[|Dot\s*|#)(\d+)(?:\]|\b)', text_eval, re.IGNORECASE) if int(m) in cand_ids]
            if mentions:
                raw_id = mentions[0]

        target_node = next((d for d in all_candidates if d.id == raw_id), None) if raw_id is not None else None
        target_x = float(target_node.center_x) if target_node else (data.get("x"))
        target_y = float(target_node.center_y) if target_node else (data.get("y"))

        # Grounded fallback if ungrounded
        selection_source = "manifest_id" if target_node else "model_coord"
        if target_node is None and all_candidates:
            target_node = all_candidates[0]
            raw_id = target_node.id
            target_x = float(target_node.center_x)
            target_y = float(target_node.center_y)
            selection_source = "manifest_fallback"

        return GroundingDecision(
            target_dot_id=raw_id,
            action=str(data.get("action", "click")),
            x=target_x,
            y=target_y,
            confidence=1.0 if selection_source == "manifest_id" else 0.6,
            reasoning=str(data.get("reasoning", "")),
            deliberation=data.get("deliberation", {}) if isinstance(data.get("deliberation"), dict) else {},
            selection_source=selection_source,
            manifest_candidate_ids=[d.id for d in all_candidates]
        )
