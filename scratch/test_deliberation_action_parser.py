"""
Opaque-Box Test Suite: Deliberation-to-Action Parser & 5-Stage CoT Schema (F3, F6).
Verifies:
- 11 canonical actions (click, double_click, type_and_enter, clear_and_type, launch_app,
  open_path, switch_window, minimize, maximize, hotkey, stop)
- 5-stage CoT deliberation parsing, master plan milestone extraction, and bidirectional schema sync
- Fault-tolerant JSON extraction: markdown fences, <think> tags, trailing commas, regex fallback
- Target dot-id normalization and hotkey/key auto-recovery from deliberation text
- Prevention of passive text termination
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


# ============================================================================
# TIER 1: FEATURE COVERAGE (Canonical Action Normalization & 5-Stage CoT Schema)
# ============================================================================

class TestTier1CanonicalActionCoverage:
    """Verifies that all 11 canonical actions are deterministically parsed and normalized."""

    @pytest.mark.parametrize("canonical_action", [
        "click",
        "double_click",
        "type_and_enter",
        "clear_and_type",
        "launch_app",
        "open_path",
        "switch_window",
        "minimize",
        "maximize",
        "hotkey",
        "stop",
    ])
    def test_all_11_canonical_actions_extracted(self, canonical_action):
        """Tier 1: Verify each of the 11 canonical actions is extracted from clean JSON."""
        raw_json = json.dumps({
            "action": canonical_action,
            "target_dot_id": 10 if canonical_action in ("click", "double_click", "clear_and_type") else None,
            "input_text": "sample text" if "type" in canonical_action else None,
            "key_combo": ["ctrl", "c"] if canonical_action == "hotkey" else None,
            "deliberation": {
                "user_intent_and_success_criteria": f"Execute canonical action {canonical_action}.",
                "screen_state_and_visual_hierarchy": "Active window focused.",
                "mistake_critique_and_loop_diagnosis": "No previous errors.",
                "end_to_end_master_plan": [f"[Phase 1] Perform {canonical_action} (ACTIVE)"],
                "tactical_action_justification": f"Executing {canonical_action} directly achieves the subgoal."
            }
        })

        parsed = _extract_json(raw_json)
        assert parsed is not None, f"Failed to parse clean JSON for action {canonical_action}"
        assert parsed["action"] == canonical_action
        assert parsed.get("action_type") == canonical_action

    def test_5stage_cot_deliberation_schema_complete(self):
        """Tier 1: Verify 5-stage CoT schema is completely extracted into deliberation dict."""
        raw_json = json.dumps({
            "action": "click",
            "target_dot_id": 42,
            "deliberation": {
                "user_intent_and_success_criteria": "Open Settings application and verify window loaded.",
                "screen_state_and_visual_hierarchy": "Desktop visible, Settings icon in [FOOTER] taskbar.",
                "mistake_critique_and_loop_diagnosis": "Previous click on blank space produced 0 motion delta.",
                "end_to_end_master_plan": [
                    "[Phase 1] Click Settings icon in taskbar (ACTIVE)",
                    "[Phase 2] Wait for Settings window (PENDING)",
                    "[Phase 3] Verify search input loaded (PENDING)"
                ],
                "tactical_action_justification": "Dot [42] is the high-confidence Windows Settings launcher icon."
            }
        })

        parsed = _extract_json(raw_json)
        assert parsed is not None
        delib = parsed["deliberation"]

        assert delib["user_intent_and_success_criteria"] == "Open Settings application and verify window loaded."
        assert delib["screen_state_and_visual_hierarchy"] == "Desktop visible, Settings icon in [FOOTER] taskbar."
        assert delib["mistake_critique_and_loop_diagnosis"] == "Previous click on blank space produced 0 motion delta."
        assert len(delib["end_to_end_master_plan"]) == 3
        assert delib["tactical_action_justification"] == "Dot [42] is the high-confidence Windows Settings launcher icon."

    def test_5stage_cot_bidirectional_legacy_synchronization(self):
        """Tier 1: Verify bidirectional sync between 5-stage CoT and legacy deliberation fields."""
        # 1. From 5-stage to legacy
        delib_5stage = {
            "user_intent_and_success_criteria": "Search for Python documentation.",
            "screen_state_and_visual_hierarchy": "Browser window in [WORKSPACE], search bar in [HEADER/NAV].",
            "mistake_critique_and_loop_diagnosis": "No previous mistakes.",
            "end_to_end_master_plan": ["Phase 1: Enter query", "Phase 2: Press Enter"],
            "tactical_action_justification": "Dot [7] is the main search input box."
        }
        cleaned = _clean_decision({"action": "click", "deliberation": delib_5stage})
        cdelib = cleaned["deliberation"]

        assert cdelib["intent_analysis"] == delib_5stage["user_intent_and_success_criteria"]
        assert cdelib["observation_analysis"] == delib_5stage["screen_state_and_visual_hierarchy"]
        assert cdelib["critique_of_previous_step"] == delib_5stage["mistake_critique_and_loop_diagnosis"]
        assert "Phase 1: Enter query" in cdelib["strategy_plan"]
        assert cdelib["candidate_evaluation"] == delib_5stage["tactical_action_justification"]

        # 2. From legacy to 5-stage
        legacy_delib = {
            "intent_analysis": "Close current active tab.",
            "observation_analysis": "Browser active with 3 tabs open.",
            "critique_of_previous_step": "Tab opened successfully.",
            "strategy_plan": "Click the close button on tab 3.",
            "candidate_evaluation": "Dot [19] is the X button on the active tab."
        }
        cleaned_legacy = _clean_decision({"action": "click", "deliberation": legacy_delib})
        cldelib = cleaned_legacy["deliberation"]

        assert cldelib["user_intent_and_success_criteria"] == legacy_delib["intent_analysis"]
        assert cldelib["screen_state_and_visual_hierarchy"] == legacy_delib["observation_analysis"]
        assert cldelib["mistake_critique_and_loop_diagnosis"] == legacy_delib["critique_of_previous_step"]
        assert legacy_delib["strategy_plan"] in cldelib["end_to_end_master_plan"]
        assert cldelib["tactical_action_justification"] == legacy_delib["candidate_evaluation"]

    def test_decision_result_object_properties(self):
        """Tier 1: Verify DecisionResult exposes all 5-stage properties and legacy accessors."""
        delib = {
            "user_intent_and_success_criteria": "Intent A",
            "screen_state_and_visual_hierarchy": "State B",
            "mistake_critique_and_loop_diagnosis": "Diagnosis C",
            "end_to_end_master_plan": ["[Phase 1] Plan D (ACTIVE)"],
            "tactical_action_justification": "Justification E"
        }
        res = DecisionResult(
            action="clear_and_type",
            target_dot_id=5,
            text="hello world",
            deliberation=delib,
            raw_response="{}",
            semantic_intention="Type greeting into text input",
            expected_transition="Text input shows 'hello world'"
        )

        assert res.action == "clear_and_type"
        assert res.target_dot_id == 5
        assert res.text == "hello world"
        assert res.user_intent_and_success_criteria == "Intent A"
        assert res.screen_state_and_visual_hierarchy == "State B"
        assert res.mistake_critique_and_loop_diagnosis == "Diagnosis C"
        assert res.end_to_end_master_plan == ["[Phase 1] Plan D (ACTIVE)"]
        assert res.tactical_action_justification == "Justification E"

        # Legacy fallbacks
        assert res.intent_analysis == "Intent A"
        assert res.observation_analysis == "State B"
        assert res.critique_of_previous_step == "Diagnosis C"
        assert "Plan D" in res.strategy_plan


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES (Fuzzy JSON, Bad Types & Recovery)
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """Verifies edge cases, malformed outputs, and parameter boundary normalizations."""

    def test_think_tag_removal_before_extraction(self):
        """Tier 2: Verify <think>...</think> tags from reasoning models are completely stripped."""
        raw = """
        <think>
        The user wants to launch calculator.
        Let's inspect the screen:
        Taskbar has Start button and Calc shortcut.
        I will select Dot [8] which is the Calc icon.
        Action: click.
        </think>
        ```json
        {
            "action": "click",
            "target_dot_id": 8,
            "deliberation": {
                "user_intent_and_success_criteria": "Launch Calculator",
                "screen_state_and_visual_hierarchy": "Taskbar visible",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": ["Launch Calc"],
                "tactical_action_justification": "Dot [8] is Calc icon"
            }
        }
        ```
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 8

    def test_trailing_commas_in_json_cleaned(self):
        """Tier 2: Verify parser cleans trailing commas in objects and arrays."""
        raw = """
        {
            "action": "type_and_enter",
            "target_dot_id": 12,
            "input_text": "pytest -v",
            "deliberation": {
                "user_intent_and_success_criteria": "Run tests",
                "screen_state_and_visual_hierarchy": "Terminal open",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": [
                    "Step 1: run tests",
                ],
                "tactical_action_justification": "Execute test command",
            },
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "type_and_enter"
        assert parsed["target_dot_id"] == 12
        assert parsed["input_text"] == "pytest -v"

    @pytest.mark.parametrize("input_val, expected_val", [
        (42, 42),
        (42.0, 42),
        ("42", 42),
        ("Dot [42]", 42),
        ("dot_42", 42),
        ("Target: 42", 42),
        ("null", None),
        ("None", None),
        ("", None),
        (None, None),
    ])
    def test_target_dot_id_normalization(self, input_val, expected_val):
        """Tier 2: Verify target_dot_id is converted to int or None across all formats."""
        decision = {"action": "click", "target_dot_id": input_val}
        cleaned = _clean_decision(decision)
        assert cleaned["target_dot_id"] == expected_val

    def test_window_action_aliases_normalization(self):
        """Tier 2: Verify minimize_window and maximize_window aliases map to canonical minimize/maximize."""
        d_min = _clean_decision({"action": "minimize_window"})
        assert d_min["action"] == "minimize"
        assert d_min["action_type"] == "minimize"

        d_max = _clean_decision({"action": "maximize_window"})
        assert d_max["action"] == "maximize"
        assert d_max["action_type"] == "maximize"

    def test_hotkey_auto_recovery_from_deliberation_text(self):
        """Tier 2: When key_combo is missing in hotkey action, auto-recovers from deliberation text."""
        raw = """
        {
            "action": "hotkey",
            "deliberation": {
                "user_intent_and_success_criteria": "Switch to next running application.",
                "screen_state_and_visual_hierarchy": "Multiple windows open.",
                "mistake_critique_and_loop_diagnosis": "Previous window click missed title bar.",
                "end_to_end_master_plan": ["Switch application using Alt+Tab shortcut."],
                "tactical_action_justification": "Pressing Alt+Tab cycles foreground focus immediately."
            }
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "hotkey"
        assert parsed.get("key_combo") == ["alt", "tab"]

    def test_press_key_auto_recovery_from_deliberation(self):
        """Tier 2: When key is missing in press action, auto-recovers from deliberation text."""
        raw = """
        {
            "action": "press",
            "deliberation": {
                "user_intent_and_success_criteria": "Submit login form.",
                "screen_state_and_visual_hierarchy": "Password field focused.",
                "mistake_critique_and_loop_diagnosis": "None.",
                "end_to_end_master_plan": ["Submit by pressing enter."],
                "tactical_action_justification": "Pressing enter key commits form credentials."
            }
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "press"
        assert parsed.get("key") == "enter"

    def test_empty_and_unparseable_inputs_return_none(self):
        """Tier 2: Verify parser returns None for empty, null, or completely unparseable input."""
        assert _extract_json("") is None
        assert _extract_json(None) is None
        assert _extract_json("Just conversational gibberish without any action markers or JSON.") is None


# ============================================================================
# TIER 3: PAIRWISE & CROSS-FEATURE COMBINATIONS
# ============================================================================

class TestTier3PairwiseCombinations:
    """Verifies interactions between JSON formatting, Deliberation, and Action dispatching."""

    def test_deliberation_with_nested_braces_and_code_snippets(self):
        """Tier 3: Verify parser succeeds when deliberation text contains brackets, braces, and quotes."""
        raw = """
        I analyzed the screen and determined the optimal next step:
        ```json
        {
            "action": "clear_and_type",
            "target_dot_id": 14,
            "input_text": "function test() { return [1, 2, 3]; }",
            "deliberation": {
                "user_intent_and_success_criteria": "Insert snippet {code: true} into editor.",
                "screen_state_and_visual_hierarchy": "VS Code editor in [WORKSPACE] zone with rect (100, 100, 1200, 800).",
                "mistake_critique_and_loop_diagnosis": "No mistakes detected. Previous typing delta Δ = 14.5px.",
                "end_to_end_master_plan": [
                    "[Phase 1] Focus editor (DONE)",
                    "[Phase 2] Type function snippet (ACTIVE)",
                    "[Phase 3] Save file via Ctrl+S (PENDING)"
                ],
                "tactical_action_justification": "Dot [14] represents line 4 in the active document buffer."
            }
        }
        ```
        Hope this action accomplishes your goal!
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "clear_and_type"
        assert parsed["target_dot_id"] == 14
        assert parsed["input_text"] == "function test() { return [1, 2, 3]; }"
        assert "VS Code" in parsed["deliberation"]["screen_state_and_visual_hierarchy"]

    def test_regex_fallback_when_json_is_severely_malformed(self):
        """Tier 3: Verify regex fallback extracts action, target, and deliberation when JSON syntax is broken."""
        # Intentionally broken syntax: missing quotes around keys, unclosed braces
        raw = """
        Here is what I will do:
        "action": "launch_app",
        "input_text": "notepad.exe",
        "target_dot_id": null,
        "user_intent_and_success_criteria": "Launch Notepad to start text editing.",
        "screen_state_and_visual_hierarchy": "Desktop visible.",
        "tactical_action_justification": "Direct process launch is faster than icon hunting."
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "launch_app"
        assert parsed["input_text"] == "notepad.exe"
        assert parsed["target_dot_id"] is None
        assert parsed["deliberation"]["user_intent_and_success_criteria"] == "Launch Notepad to start text editing."

    def test_action_and_deliberation_mirrored_to_toplevel_decision(self):
        """Tier 3: Verify deliberation fields are mirrored to top-level dict for direct access."""
        raw = json.dumps({
            "action": "switch_window",
            "input_text": "Chrome",
            "deliberation": {
                "user_intent_and_success_criteria": "Switch to Chrome",
                "screen_state_and_visual_hierarchy": "Chrome behind Notepad",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": ["Switch window"],
                "tactical_action_justification": "Bring Chrome to front"
            }
        })
        parsed = _extract_json(raw)
        assert parsed is not None
        # Verify top-level mirror
        assert parsed["user_intent_and_success_criteria"] == "Switch to Chrome"
        assert parsed["screen_state_and_visual_hierarchy"] == "Chrome behind Notepad"
        assert parsed["tactical_action_justification"] == "Bring Chrome to front"


# ============================================================================
# TIER 4: REAL-WORLD DESKTOP APPLICATION SCENARIOS
# ============================================================================

class TestTier4RealWorldScenarios:
    """Verifies realistic multi-paragraph outputs from leading VLMs (Gemini Pro, Qwen, DeepSeek)."""

    def test_gemini_pro_verbose_conversational_response(self):
        """Tier 4: Realistic Gemini Pro output with extensive explanation preceding the JSON block."""
        gemini_response = """
        Based on visual inspection of the Set-of-Marks annotated screenshot:
        1. We observe a confirmation dialog titled 'Confirm File Replace' centered at (450, 300, 1050, 650).
        2. Dot [21] is the primary commit button labeled '[btn] Replace'.
        3. Dot [22] is '[btn] Cancel'.
        4. In accordance with the user's intent to overwrite the file, Dot [21] must be activated.

        ```json
        {
            "deliberation": {
                "user_intent_and_success_criteria": "Confirm replacing existing file with downloaded version.",
                "screen_state_and_visual_hierarchy": "Modal dialog [MODAL/POPUP] elevated over Explorer viewport.",
                "mistake_critique_and_loop_diagnosis": "No previous mistakes. Turn 1 modal appearance detected.",
                "end_to_end_master_plan": [
                    "[Phase 1] Confirm file replace dialog (ACTIVE)",
                    "[Phase 2] Verify Explorer returns to file list (PENDING)",
                    "[Phase 3] Confirm operation complete (PENDING)"
                ],
                "tactical_action_justification": "Dot [21] is the focused primary commit button in the modal."
            },
            "action": "click",
            "target_dot_id": 21,
            "semantic_intention": "Click Replace in modal confirmation dialog",
            "expected_transition": "Modal dialog closes, showing updated file in Explorer"
        }
        ```
        The click will be dispatched via hardware SendInput.
        """
        parsed = _extract_json(gemini_response)
        assert parsed is not None
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 21
        assert parsed["semantic_intention"] == "Click Replace in modal confirmation dialog"
        assert "[MODAL/POPUP]" in parsed["deliberation"]["screen_state_and_visual_hierarchy"]
        assert len(parsed["deliberation"]["end_to_end_master_plan"]) == 3

    def test_zero_passive_deliberation_invariant(self):
        """Tier 4: Zero turns terminate in passive text deliberation without an action."""
        # Passive response without any action tag
        passive_response = """
        I am currently thinking about what to do next. The application has opened,
        and there are many buttons on screen. I should probably click one of them soon,
        perhaps the submit button. Let's wait and see.
        """
        parsed = _extract_json(passive_response)
        # Parser refuses to return a valid action from passive text alone
        assert parsed is None or "action" not in parsed or not parsed["action"]

    def test_multistep_action_sequence_cleanliness(self):
        """Tier 4: Verify parsing across consecutive steps in a multi-turn desktop workflow."""
        steps = [
            # Step 1: Launch application
            """```json
            {"action": "launch_app", "input_text": "calc", "target_dot_id": null, "deliberation": {"user_intent_and_success_criteria": "Open Calc", "screen_state_and_visual_hierarchy": "Desktop", "mistake_critique_and_loop_diagnosis": "None", "end_to_end_master_plan": ["Open Calc"], "tactical_action_justification": "Launch executable"}}
            ```""",
            # Step 2: Clear and type
            """```json
            {"action": "clear_and_type", "input_text": "128", "target_dot_id": 105, "deliberation": {"user_intent_and_success_criteria": "Enter 128", "screen_state_and_visual_hierarchy": "Calc open", "mistake_critique_and_loop_diagnosis": "None", "end_to_end_master_plan": ["Enter 128"], "tactical_action_justification": "Click calc display"}}
            ```""",
            # Step 3: Stop upon task completion
            """```json
            {"action": "stop", "target_dot_id": null, "status": "completed", "deliberation": {"user_intent_and_success_criteria": "Calculation verified", "screen_state_and_visual_hierarchy": "Result shows 128", "mistake_critique_and_loop_diagnosis": "None", "end_to_end_master_plan": ["[Phase 1] Completed"], "tactical_action_justification": "Task goal satisfied"}}
            ```"""
        ]

        actions_extracted = []
        for s in steps:
            p = _extract_json(s)
            assert p is not None
            actions_extracted.append(p["action"])

        assert actions_extracted == ["launch_app", "clear_and_type", "stop"]
