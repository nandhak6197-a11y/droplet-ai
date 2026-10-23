"""
Production Blackbox Flight Recorder & Incident Telemetry (Pillar 5).
Captures end-to-end execution spans, stage latency breakdowns, and generates
forensic post-mortem incident bundles (images, manifests, deliberations) on failures.
"""
import os
import time
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import cv2
import numpy as np


@dataclass
class StageTiming:
    capture_ms: float = 0.0
    settle_ms: float = 0.0
    spray_ms: float = 0.0
    reason_ms: float = 0.0
    actuate_ms: float = 0.0
    verify_ms: float = 0.0


@dataclass
class ExecutionTrace:
    trace_id: str
    task_goal: str
    status: str                         # "SUCCESS", "VERIFICATION_FAILED", "ERROR"
    total_latency_ms: float
    timings: StageTiming = field(default_factory=StageTiming)
    target_dot_id: Optional[int] = None
    action_type: str = "click"
    target_coords: Optional[List[int]] = None
    provider: str = "groq"
    model: str = ""
    verification_outcome: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    timestamp: float = field(default_factory=time.time)


class FlightRecorder:
    """
    Production flight recorder and forensic incident logger.
    """
    _instance = None

    @classmethod
    def get_instance(cls, incident_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(incident_dir)
        return cls._instance

    def __init__(self, incident_dir: Optional[str] = None):
        self.incident_dir = Path(incident_dir or os.path.join(".", "diagnostics", "incidents"))
        self.traces: List[ExecutionTrace] = []
        self.max_in_memory_traces = 100
        # Option 3: Adaptive Long-Term Knowledge Base
        self.knowledge_base_path = Path(os.path.join(".", "diagnostics", "knowledge_base.json"))
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Any]:
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def update_knowledge(self, task_goal: str, successful_target: str, context: str):
        """Update adaptive KB with successful execution strategies."""
        self.knowledge_base[task_goal] = {
            "target": successful_target,
            "context": context,
            "timestamp": time.time()
        }
        try:
            self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.knowledge_base_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception:
            pass

    def query_knowledge(self, task_goal: str) -> Optional[Dict[str, Any]]:
        """Retrieve past successful strategies for similar tasks."""
        # Exact match or simple keyword match for now
        for known_goal, data in self.knowledge_base.items():
            if known_goal.lower() in task_goal.lower() or task_goal.lower() in known_goal.lower():
                return data
        return None

    def start_trace(self, task_goal: str) -> str:
        """Starts a new execution trace, returning its trace_id."""
        trace_id = f"trace_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        return trace_id

    def record_trace(self, trace: ExecutionTrace):
        """Saves trace to in-memory rolling buffer."""
        self.traces.append(trace)
        if len(self.traces) > self.max_in_memory_traces:
            self.traces.pop(0)

    def log_incident(
        self,
        trace: ExecutionTrace,
        pre_screenshot: Optional[np.ndarray] = None,
        post_screenshot: Optional[np.ndarray] = None,
        manifest_text: Optional[str] = None,
        deliberation_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Dumps a forensic incident bundle to disk for post-mortem analysis.
        Returns the directory path of the bundle.
        """
        try:
            folder_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{trace.trace_id}"
            bundle_dir = self.incident_dir / folder_name
            bundle_dir.mkdir(parents=True, exist_ok=True)

            # 1. Summary JSON
            summary = {
                "trace": asdict(trace),
                "deliberation": deliberation_data or {},
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(bundle_dir / "incident_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # 2. Screenshots
            if pre_screenshot is not None and isinstance(pre_screenshot, np.ndarray):
                cv2.imwrite(str(bundle_dir / "pre_action.png"), pre_screenshot)

            if post_screenshot is not None and isinstance(post_screenshot, np.ndarray):
                cv2.imwrite(str(bundle_dir / "post_action.png"), post_screenshot)

            # 3. Manifest text
            if manifest_text:
                with open(bundle_dir / "screen_droplet_manifest.txt", "w", encoding="utf-8") as f:
                    f.write(manifest_text)

            print(f"[FlightRecorder] [ARCHIVE] Forensic incident bundle archived to: {bundle_dir}")
            return str(bundle_dir)

        except Exception as e:
            print(f"[FlightRecorder] Failed to archive incident bundle: {e}")
            return None

    def get_latency_summary(self) -> Dict[str, float]:
        """Calculates rolling average latencies across recent traces."""
        if not self.traces:
            return {}
        n = len(self.traces)
        return {
            "avg_total_ms": sum(t.total_latency_ms for t in self.traces) / n,
            "avg_capture_ms": sum(t.timings.capture_ms for t in self.traces) / n,
            "avg_spray_ms": sum(t.timings.spray_ms for t in self.traces) / n,
            "avg_reason_ms": sum(t.timings.reason_ms for t in self.traces) / n,
            "avg_verify_ms": sum(t.timings.verify_ms for t in self.traces) / n,
            "trace_count": n
        }
