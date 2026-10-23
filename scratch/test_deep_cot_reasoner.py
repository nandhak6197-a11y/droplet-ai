"""
Test suite for Deep Chain-of-Thought (CoT) Cognitive Reasoner.
Tests 5-stage deliberation parsing, master plan milestone extraction,
bidirectional schema synchronization, backward compatibility, and episodic memory formatting.
"""

import sys
import os
import json
import pytest

# Ensure utf-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import (
    DiagonalDropletReasoner,
    DecisionResult,
    _clean_decision,
    _extract_json,
)
from spatial_droplet.core.models import TrajectoryStep
from spatial_droplet.core.memory import EpisodicTrajectoryMemory


def test_decision_result_5stage_cot_properties():
    """Verify DecisionResult correctly exposes 5-stage CoT fields and backward-compatible properties."""
    delib = {
        "user_intent_and_success_criteria": "User wants to open Edge and search for python docs.",
        "screen_state_and_visual_hierarchy": "Desktop visible, Edge pinned in taskbar [FOOTER].",
        "mistake_critique_and_loop_diagnosis": "No previous mistakes. Initial step.",
        "end_to_end_master_plan": [
            "[Phase 1] Launch Microsoft Edge (ACTIVE)",
            "[Phase 2] Enter query in search bar (PENDING)",
            "[Phase 3] Verify documentation loaded (PENDING)"
        ],
        "tactical_action_justification": "Clicking Dot [5] (Edge icon) cleanly launches browser."
    }

    res = DecisionResult(
        action="click",
        target_dot_id=5,
        deliberation=delib,
        raw_response="{...}",
        semantic_intention="Launch Edge browser",
        expected_transition="Edge window appears on desktop"
    )

    # Test new 5-stage accessors
    assert res.user_intent_and_success_criteria == delib["user_intent_and_success_criteria"]
    assert res.screen_state_and_visual_hierarchy == delib["screen_state_and_visual_hierarchy"]
    assert res.mistake_critique_and_loop_diagnosis == delib["mistake_critique_and_loop_diagnosis"]
    assert res.end_to_end_master_plan == delib["end_to_end_master_plan"]
    assert res.tactical_action_justification == delib["tactical_action_justification"]

    # Test backward-compatible legacy properties
    assert res.intent_analysis == delib["user_intent_and_success_criteria"]
    assert res.observation_analysis == delib["screen_state_and_visual_hierarchy"]
    assert res.critique_of_previous_step == delib["mistake_critique_and_loop_diagnosis"]
    assert "Launch Microsoft Edge" in res.strategy_plan
    assert res.candidate_evaluation == delib["tactical_action_justification"]
    assert res.action_plan == delib["tactical_action_justification"]
    assert "User wants to open Edge" in res.reasoning


def test_decision_result_legacy_fallback():
    """Verify DecisionResult fallback behavior when given legacy deliberation dictionary."""
    legacy_delib = {
        "intent_analysis": "User wants to type greeting in Notepad.",
        "observation_analysis": "Notepad is open and focused.",
        "critique_of_previous_step": "Step 1 succeeded in launching Notepad.",
        "strategy_plan": "Type greeting text into the text area.",
        "candidate_evaluation": "Dot [14] is the main text area.",
        "action_plan": "Click text area and type text."
    }

    res = DecisionResult(
        action="type",
        target_dot_id=14,
        text="Hello World",
        deliberation=legacy_delib,
        raw_response="{...}"
    )

    # New properties should cleanly map from legacy keys
    assert res.user_intent_and_success_criteria == legacy_delib["intent_analysis"]
    assert res.screen_state_and_visual_hierarchy == legacy_delib["observation_analysis"]
    assert res.mistake_critique_and_loop_diagnosis == legacy_delib["critique_of_previous_step"]
    assert legacy_delib["strategy_plan"] in res.end_to_end_master_plan
    assert res.tactical_action_justification == legacy_delib["candidate_evaluation"]


def test_clean_decision_5stage_normalization():
    """Verify _clean_decision normalizes 5-stage deliberation and synchronizes legacy fields."""
    raw = {
        "deliberation": {
            "user_intent_and_success_criteria": "Navigate to github.com and search repository.",
            "screen_state_and_visual_hierarchy": "Browser is open on home page.",
            "mistake_critique_and_loop_diagnosis": "Prior click missed address bar by 10px.",
            "end_to_end_master_plan": [
                "[Phase 1] Focus address bar (ACTIVE)",
                "[Phase 2] Navigate to github.com (PENDING)"
            ],
            "tactical_action_justification": "Use hotkey Alt+D to avoid coordinate clicking."
        },
        "action": "press_enter",
        "status": "in_progress",
        "semantic_intention": "Submit address bar",
        "expected_transition": "Page loads"
    }

    cleaned = _clean_decision(raw)

    assert cleaned["action"] in ("press", "press_enter")
    delib = cleaned["deliberation"]
    assert isinstance(delib, dict)
    assert delib["user_intent_and_success_criteria"] == raw["deliberation"]["user_intent_and_success_criteria"]
    assert delib["end_to_end_master_plan"] == raw["deliberation"]["end_to_end_master_plan"]

    # Legacy synchronization
    assert delib["intent_analysis"] == delib["user_intent_and_success_criteria"]
    assert delib["observation_analysis"] == delib["screen_state_and_visual_hierarchy"]
    assert delib["critique_of_previous_step"] == delib["mistake_critique_and_loop_diagnosis"]
    assert delib["strategy_plan"] == "[Phase 1] Focus address bar (ACTIVE) | [Phase 2] Navigate to github.com (PENDING)"
    assert delib["candidate_evaluation"] == delib["tactical_action_justification"]


def test_clean_decision_bidirectional_legacy_sync():
    """Verify _clean_decision maps legacy format into 5-stage schema."""
    raw = {
        "deliberation": {
            "intent_analysis": "Close modal popup.",
            "observation_analysis": "Confirmation modal visible in center.",
            "critique_of_previous_step": "Modal was triggered by save button.",
            "strategy_plan": "[1] Click Confirm. [2] Verify file saved.",
            "candidate_evaluation": "Dot [30] is the primary Confirm button."
        },
        "action": "click",
        "target_dot_id": 30
    }

    cleaned = _clean_decision(raw)
    delib = cleaned["deliberation"]

    assert delib["user_intent_and_success_criteria"] == "Close modal popup."
    assert delib["screen_state_and_visual_hierarchy"] == "Confirmation modal visible in center."
    assert delib["mistake_critique_and_loop_diagnosis"] == "Modal was triggered by save button."
    assert isinstance(delib["end_to_end_master_plan"], list)
    assert delib["end_to_end_master_plan"] == ["[1] Click Confirm. [2] Verify file saved."]
    assert delib["tactical_action_justification"] == "Dot [30] is the primary Confirm button."


def test_extract_json_with_deep_cot():
    """Verify _extract_json parses markdown fenced JSON containing 5-stage CoT."""
    text = """
Here is my deep deliberation and tactical decision:

```json
{
  "deliberation": {
    "user_intent_and_success_criteria": "Search for query 'weather' in Chrome.",
    "screen_state_and_visual_hierarchy": "Chrome is active, search bar [8] ready.",
    "mistake_critique_and_loop_diagnosis": "No mistakes detected.",
    "end_to_end_master_plan": [
      "[Phase 1] Click search input [8] (ACTIVE)",
      "[Phase 2] Type query (PENDING)"
    ],
    "tactical_action_justification": "Dot [8] is focused and ready for text input."
  },
  "status": "in_progress",
  "action": "click",
  "target_dot_id": 8,
  "semantic_intention": "Click search box",
  "expected_transition": "Cursor appears in box"
}
```
    """

    data = _extract_json(text)
    assert data is not None
    assert data["action"] == "click"
    assert data["target_dot_id"] == 8
    assert "deliberation" in data
    assert len(data["deliberation"]["end_to_end_master_plan"]) == 2
    assert "Search for query" in data["deliberation"]["user_intent_and_success_criteria"]


def test_extract_json_regex_fallback():
    """Verify _extract_json falls back to regex parser when JSON is slightly malformed."""
    malformed_text = """
{
  "deliberation": {
    "user_intent_and_success_criteria": "Open Settings and toggle Wi-Fi.",
    "screen_state_and_visual_hierarchy": "Taskbar visible at bottom.",
    "mistake_critique_and_loop_diagnosis": "None.",
    "end_to_end_master_plan": [
      "Phase 1: Open Settings",
      "Phase 2: Toggle Wi-Fi"
    ],
    "tactical_action_justification": "Hotkey Win+I opens Settings directly.",
  },
  "action": "hotkey",
  "key_combo": ["win", "i"],
  "status": "in_progress",
}
"""
    data = _extract_json(malformed_text)
    assert data is not None
    assert data["action"] == "hotkey"
    assert "deliberation" in data
    assert data["deliberation"]["user_intent_and_success_criteria"] == "Open Settings and toggle Wi-Fi."
    assert "win" in str(data.get("key_combo", []))


def test_episodic_memory_master_plan_formatting():
    """Verify EpisodicTrajectoryMemory formats master plan milestones (✔, ►, ○) and failure critique."""
    mem = EpisodicTrajectoryMemory(goal="Open calculator and calculate 12 * 8", max_steps=5)

    step1 = TrajectoryStep(
        step_num=1,
        goal="Open calculator and calculate 12 * 8",
        subgoal="Launch Calculator via Win+R",
        action="hotkey",
        target_id=None,
        target_label="None",
        input_text="calc",
        verified=True,
        verification_reason="Calculator window opened",
        active_window="Calculator",
        semantic_intention="Launch calc.exe",
        expected_transition="Calculator opens",
        verdict="SUCCESS",
        master_plan=[
            "[Phase 1] Launch Calculator (DONE)",
            "[Phase 2] Enter 12 * 8 (ACTIVE)",
            "[Phase 3] Read result (PENDING)"
        ],
        intent_analysis="Calculate 12 * 8 in Calculator.",
        mistake_critique="Initial step succeeded."
    )
    mem.add_step(step1)

    step2 = TrajectoryStep(
        step_num=2,
        goal="Open calculator and calculate 12 * 8",
        subgoal="Click number 1",
        action="click",
        target_id=10,
        target_label="Button 1",
        input_text=None,
        verified=False,
        verification_reason="Calculator input did not register key",
        active_window="Calculator",
        semantic_intention="Input digit 1",
        expected_transition="Display shows 1",
        verdict="NO_EFFECT",
        master_plan=[
            "[Phase 1] Launch Calculator (DONE)",
            "[Phase 2] Enter 12 * 8 (ACTIVE)",
            "[Phase 3] Read result (PENDING)"
        ],
        intent_analysis="Calculate 12 * 8 in Calculator.",
        mistake_critique="Click on Dot [10] had no effect; window focus may have been lost."
    )
    mem.add_step(step2)

    prompt_history = mem.format_history_for_prompt()

    # Verify master plan milestone symbols are rendered
    assert "Active Master Plan:" in prompt_history
    assert "✔ [Phase 1] Launch Calculator (DONE)" in prompt_history
    assert "► [Phase 2] Enter 12 * 8 (ACTIVE)" in prompt_history
    assert "○ [Phase 3] Read result (PENDING)" in prompt_history

    # Verify failure warning and critique are rendered
    assert "NOTE: Step 2 yielded no verified screen change" in prompt_history
    assert "Prev Critique: \"Click on Dot [10] had no effect" in prompt_history
    assert "DIAGNOSED MISTAKE: Click on Dot [10] had no effect" in prompt_history
    assert "MANDATE: Avoid repeating this diagnosed mistake" in prompt_history
