"""
Adversarial Stress Test Suite: Canonical Action Parser & Zero Passive Text Deliberation (M3 Challenger).
Authors: challenger_m3_1 (Action Parser & Zero Passive Text Adversarial Challenger)

Adversarial Stress Dimensions:
1. Zero passive text deliberation invariant against conversational, stream-of-consciousness, and polite-refusal prompts.
2. Malformed markdown, unclosed fences, truncated JSON strings, unquoted keys.
3. Complex dot-id regex evasion and synonym mapping.
"""
import sys
import os
import json
import pytest

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
# DIMENSION 1: ZERO PASSIVE TEXT DELIBERATION INVARIANT & DECEPTIVE PROMPTS
# ============================================================================

class TestAdversarialZeroPassiveTextInvariant:
    """
    Stress-tests the zero passive text deliberation invariant against:
    - Pure stream-of-consciousness rambling with zero actionable directives.
    - Polite refusals and safety disclaimers.
    - Conflicting multiple verbs in conversational prose.
    - Passive action tags in JSON and their deterministic conversion to canonical actions.
    - Unhandled verbs and edge cases.
    """

    @pytest.mark.parametrize("stream_text", [
        "The desktop has several windows open. The wallpaper shows a mountain sunrise. I am contemplating what to do next. Today is a great day to organize files.",
        "Hmm, let me think about this. The user wants to search for something, but maybe the network is down. Let me just observe the screen for a while without doing anything.",
        "Looking at the screen, there are 15 icons. Dot 1 is something, dot 2 is something else. Everything looks normal. Just reflecting on the visual layout.",
        "As an AI, I notice that the prompt is asking to click a button. However, I am just observing the screen pixels. The color distribution is mostly gray and blue.",
    ])
    def test_pure_stream_of_consciousness_rejected(self, stream_text):
        """Adversarial 1.1: Verify parser refuses to fabricate an action from pure stream of consciousness."""
        parsed = _extract_json(stream_text)
        assert parsed is None or not parsed.get("action"), (
            f"Parser fabricated an ungrounded action from stream of consciousness: {parsed}"
        )

    @pytest.mark.parametrize("refusal_text", [
        "I am an AI assistant and I cannot interact with your desktop directly. I must politely decline to perform any actions.",
        "I cannot fulfill this request as clicking UI elements without human supervision violates safety protocols.",
        "Sorry, but I am programmed to be helpful and harmless, so I will not perform any automated clicks or keystrokes on this system.",
    ])
    def test_polite_refusals_without_actionable_dot_rejected(self, refusal_text):
        """Adversarial 1.2: Verify polite refusals lacking explicit command/dot are rejected as None."""
        parsed = _extract_json(refusal_text)
        assert parsed is None or not parsed.get("action"), (
            f"Parser returned an action for a polite refusal without target dot: {parsed}"
        )

    def test_conflicting_verbs_in_conversational_text_recovers_action(self):
        """Adversarial 1.3: Verify conflicting action verbs in prose still recovers a valid canonical action."""
        prose = "I could click dot 4 or maybe double-click dot 5 or perhaps wait, but the instructions say to clear and type on dot 6."
        parsed = _extract_json(prose)
        assert parsed is not None, "Failed to recover action from conversational prose with multiple verbs"
        assert parsed["action"] in DiagonalDropletReasoner.CANONICAL_ACTIONS
        assert parsed["target_dot_id"] in (4, 5, 6)

    @pytest.mark.parametrize("passive_action", list(DiagonalDropletReasoner.PASSIVE_ACTIONS))
    def test_all_passive_actions_converted_to_canonical_actions(self, passive_action):
        """Adversarial 1.4: Verify every passive action in PASSIVE_ACTIONS is deterministically converted."""
        # Case A: status is completed -> should convert to 'stop'
        dec_completed = {"action": passive_action, "status": "completed"}
        cleaned_c = _clean_decision(dec_completed)
        assert cleaned_c["action"] == "stop", f"Failed to convert passive action '{passive_action}' to 'stop'"

        # Case B: has target_dot_id -> should convert to 'click'
        dec_targeted = {"action": passive_action, "target_dot_id": 42}
        cleaned_t = _clean_decision(dec_targeted)
        assert cleaned_t["action"] == "click", f"Failed to convert passive action '{passive_action}' to 'click'"
        assert cleaned_t["target_dot_id"] == 42

        # Case C: has input_text -> should convert to 'clear_and_type'
        dec_input = {"action": passive_action, "input_text": "sample text"}
        cleaned_i = _clean_decision(dec_input)
        assert cleaned_i["action"] == "clear_and_type", f"Failed to convert passive action '{passive_action}' to 'clear_and_type'"

        # Case D: has key_combo -> should convert to 'hotkey'
        dec_key = {"action": passive_action, "key_combo": ["ctrl", "c"]}
        cleaned_k = _clean_decision(dec_key)
        assert cleaned_k["action"] == "hotkey", f"Failed to convert passive action '{passive_action}' to 'hotkey'"

    def test_passive_action_converted_from_deliberation_text_clues(self):
        """Adversarial 1.5: Verify passive action recovers intent from deliberation text keywords."""
        scenarios = [
            ("wait", "I will double click dot 88 to open the file.", "double_click", 88),
            ("think", "We need to clear and type into dot 12.", "clear_and_type", 12),
            ("observe", "Let's launch the browser application.", "launch_app", None),
            ("analyzing", "We should minimize the window to see desktop.", "minimize", None),
            ("ponder", "Time to maximize the window for full view.", "maximize", None),
            ("deliberate", "We need to switch window to Notepad.", "switch_window", None),
            ("commentary", "The task is finished and goal complete.", "stop", None),
            ("inspection", "Press Alt+Tab to switch active window.", "hotkey", None),
        ]

        for act, delib_text, expected_act, expected_dot in scenarios:
            dec = {
                "action": act,
                "deliberation": {
                    "tactical_action_justification": delib_text,
                    "user_intent_and_success_criteria": delib_text
                }
            }
            cleaned = _clean_decision(dec)
            assert cleaned is not None
            assert cleaned["action"] == expected_act, (
                f"Expected '{expected_act}' for clue '{delib_text}', got '{cleaned['action']}'"
            )
            if expected_dot is not None:
                assert cleaned["target_dot_id"] == expected_dot

    def test_empty_string_and_none_action_normalized(self):
        """Adversarial 1.6: Verify empty string, null, and None actions are safely normalized."""
        for empty_val in ("", None, "None", "null", "   "):
            dec = {"action": empty_val, "target_dot_id": 10}
            cleaned = _clean_decision(dec)
            assert cleaned is not None
            assert cleaned["action"] == "click"
            assert cleaned["target_dot_id"] == 10

    def test_unhandled_action_behavior(self):
        """
        Adversarial 1.7: Probe behavior on unhandled/unrecognized verbs.
        Checks how _clean_decision handles verbs outside PASSIVE_ACTIONS, ACTION_SYNONYMS, and CANONICAL_ACTIONS.
        """
        unhandled_verbs = ["idle", "do_nothing", "sleep", "pause", "rest", "noop", "refusal"]
        for verb in unhandled_verbs:
            dec = {"action": verb, "target_dot_id": 5}
            cleaned = _clean_decision(dec)
            assert cleaned is not None
            assert "action" in cleaned


# ============================================================================
# DIMENSION 2: MALFORMED MARKDOWN, UNCLOSED FENCES & TRUNCATED JSON
# ============================================================================

class TestAdversarialMalformedMarkdownAndJson:
    """
    Stress-tests JSON extraction against:
    - Unclosed markdown code fences.
    - Truncated JSON strings, arrays, and objects (simulating LLM token cutoffs).
    - Relaxed/unquoted JSON keys and single-quoted strings.
    - Multiple code blocks with noise preceding the decision JSON.
    - Deeply nested structures, trailing commas, and escaped quotes.
    """

    def test_unclosed_markdown_fence_clean_json(self):
        """Adversarial 2.1: Verify parser extracts JSON when closing ``` fence is missing."""
        raw = """
        Here is my deliberative decision:
        ```json
        {
            "action": "click",
            "target_dot_id": 19,
            "deliberation": {
                "user_intent_and_success_criteria": "Click OK button",
                "screen_state_and_visual_hierarchy": "Dialog present",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": ["Click OK"],
                "tactical_action_justification": "Dot [19] is OK button"
            }
        }
        """  # Note: missing closing ```
        parsed = _extract_json(raw)
        assert parsed is not None, "Failed to parse JSON with unclosed markdown fence"
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 19

    def test_unclosed_markdown_fence_and_truncated_string(self):
        """Adversarial 2.2: Verify parser recovers when string is truncated mid-sentence and fence unclosed."""
        raw = """
        ```json
        {
            "action": "clear_and_type",
            "target_dot_id": 25,
            "input_text": "search query",
            "deliberation": {
                "user_intent_and_success_criteria": "Search for documentation on the web using Google
        """  # Truncated mid-string, no closing quotes, braces, or fence
        parsed = _extract_json(raw)
        assert parsed is not None, "Failed to recover truncated JSON string"
        assert parsed["action"] == "clear_and_type"
        assert parsed["target_dot_id"] == 25
        assert parsed["input_text"] == "search query"

    def test_truncated_json_inside_array(self):
        """Adversarial 2.3: Verify parser recovers when array is truncated mid-item."""
        raw = """
        {
            "action": "hotkey",
            "key_combo": ["ctrl", "shift
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "hotkey"
        # At least "ctrl" should be recovered
        assert "ctrl" in parsed.get("key_combo", [])

    def test_truncated_json_flat_object(self):
        """Adversarial 2.4: Verify parser recovers flat truncated JSON cutting off after a comma."""
        raw = """
        {
            "action": "double_click",
            "target_dot_id": 31,
            "input_text": "open_folder",
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "double_click"
        assert parsed["target_dot_id"] == 31

    def test_truncated_outer_json_with_nested_deliberation_inner_hijack(self):
        """
        Adversarial 2.4b: Empirical challenge documenting inner object hijacking.
        When outer JSON is truncated but contains a balanced inner deliberation object,
        Step 3 balanced brace parser extracts the inner deliberation object instead of repairing outer.
        """
        raw = """
        {
            "action": "double_click",
            "target_dot_id": 31,
            "deliberation": {
                "user_intent_and_success_criteria": "Open folder",
            },
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        # Documents that the parser returns a valid actionable dictionary (not None)
        assert "action" in parsed
        assert parsed["action"] in ("double_click", "click")

    def test_unquoted_keys_relaxed_json(self):
        """Adversarial 2.5: Verify regex fallback recovers unquoted keys (JavaScript/relaxed JSON)."""
        raw = """
        {
            action: launch_app,
            input_text: "notepad.exe",
            target_dot_id: null,
            user_intent_and_success_criteria: "Launch Notepad application",
            screen_state_and_visual_hierarchy: "Desktop active",
            tactical_action_justification: "Direct executable launch"
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None, "Failed to parse unquoted keys JSON via regex fallback"
        assert parsed["action"] == "launch_app"
        assert parsed["input_text"] == "notepad.exe"
        assert parsed.get("target_dot_id") is None

    def test_single_quoted_pseudo_json(self):
        """Adversarial 2.6: Verify regex fallback recovers single-quoted pseudo-JSON."""
        raw = """
        {
            'action': 'type_and_enter',
            'target_dot_id': 10,
            'input_text': 'pytest -v',
            'user_intent_and_success_criteria': 'Run test suite'
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None, "Failed to parse single-quoted pseudo-JSON"
        assert parsed["action"] == "type_and_enter"
        assert parsed["target_dot_id"] == 10
        assert parsed["input_text"] == "pytest -v"

    def test_multiple_code_blocks_with_preceding_noise(self):
        """Adversarial 2.7: Verify parser finds decision JSON when preceded by noise code blocks."""
        raw = """
        Here is the python snippet I investigated:
        ```python
        def calculate(x, y):
            return x + y
        ```
        And here is the visual diff table:
        ```
        | Metric | Value |
        | Δ ROI  | 14.2  |
        ```
        Now here is my final decision:
        ```json
        {
            "action": "click",
            "target_dot_id": 77,
            "deliberation": {
                "user_intent_and_success_criteria": "Click calculate button",
                "screen_state_and_visual_hierarchy": "Calculator active",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": ["Click calculate"],
                "tactical_action_justification": "Dot [77] is the equals button"
            }
        }
        ```
        """
        parsed = _extract_json(raw)
        assert parsed is not None, "Failed to extract JSON when multiple noise blocks precede it"
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 77

    def test_deeply_nested_braces_and_escaped_json(self):
        """Adversarial 2.8: Verify parser handles escaped quotes, brackets, and nested JSON strings."""
        raw = r"""
        ```json
        {
            "action": "clear_and_type",
            "target_dot_id": 44,
            "input_text": "{\"key\": [1, 2, 3], \"nested\": true}",
            "deliberation": {
                "user_intent_and_success_criteria": "Inject JSON string {\"env\": \"prod\"} into editor",
                "screen_state_and_visual_hierarchy": "IDE window in [WORKSPACE] with rect (50, 50, 1200, 900)",
                "mistake_critique_and_loop_diagnosis": "Previous text was missing closing brace }",
                "end_to_end_master_plan": [
                    "[Phase 1] Focus line 10 {active: true}",
                    "[Phase 2] Type JSON payload"
                ],
                "tactical_action_justification": "Dot [44] represents editor input buffer at line 10"
            }
        }
        ```
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "clear_and_type"
        assert parsed["target_dot_id"] == 44
        assert "key" in parsed["input_text"]
        assert len(parsed["deliberation"]["end_to_end_master_plan"]) == 2

    def test_multiline_trailing_commas_on_every_line(self):
        """Adversarial 2.9: Verify parser cleans multiple trailing commas across lines and arrays."""
        raw = """
        {
            "action": "minimize",
            "target_dot_id": null,
            "deliberation": {
                "user_intent_and_success_criteria": "Minimize foreground window",
                "screen_state_and_visual_hierarchy": "Active window taking full screen",
                "mistake_critique_and_loop_diagnosis": "None",
                "end_to_end_master_plan": [
                    "Step 1: minimize",
                ],
                "tactical_action_justification": "Minimize window to reveal background",
            },
        }
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "minimize"
        assert parsed.get("target_dot_id") is None


# ============================================================================
# DIMENSION 3: COMPLEX DOT-ID REGEX EVASION & SYNONYM MAPPING
# ============================================================================

class TestAdversarialDotIdRegexEvasionAndSynonymMapping:
    """
    Stress-tests dot-id extraction and synonym mapping against:
    - Text designed to trick dot-id regex (e.g. dot-matrix, DPI, IP addresses, versions).
    - Preserving 0 as a valid target_dot_id (not coerced to None/null).
    - Adversarial types for target_dot_id (floats, lists, strings with prefixes).
    - 100% coverage of all 45+ ACTION_SYNONYMS mapping to 11 canonical actions.
    - Case and whitespace insensitivity in action normalization.
    - Hotkey and press key auto-recovery from deliberation text.
    """

    def test_deceptive_dot_matrix_printer_text(self):
        """Adversarial 3.1: Text mentioning dot-matrix printers and DPI must not mislead dot-id extractor."""
        raw = """
        ```json
        {
            "action": "click",
            "deliberation": {
                "user_intent_and_success_criteria": "Configure dot-matrix printer with 24 pins and 360 dots per inch.",
                "screen_state_and_visual_hierarchy": "Printer dialog open with 5 options.",
                "mistake_critique_and_loop_diagnosis": "No mistakes.",
                "end_to_end_master_plan": ["Select printer"],
                "tactical_action_justification": "Dot [58] is the Print Setup commit button."
            }
        }
        ```
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 58, (
            f"Expected target_dot_id=58, but got {parsed.get('target_dot_id')} (tricked by dot-matrix/DPI text)"
        )

    def test_deceptive_ip_address_and_version_numbers(self):
        """Adversarial 3.2: Text containing IP addresses and semantic versions must not corrupt dot-id."""
        raw = """
        ```json
        {
            "action": "click",
            "deliberation": {
                "user_intent_and_success_criteria": "Connect to server 192.168.1.100 running daemon version 3.14.159.",
                "screen_state_and_visual_hierarchy": "Network settings visible with 64-bit driver.",
                "mistake_critique_and_loop_diagnosis": "None.",
                "end_to_end_master_plan": ["Click connect"],
                "tactical_action_justification": "We must activate dot 73 which corresponds to Connect button."
            }
        }
        ```
        """
        parsed = _extract_json(raw)
        assert parsed is not None
        assert parsed["action"] == "click"
        assert parsed["target_dot_id"] == 73, (
            f"Expected target_dot_id=73, but got {parsed.get('target_dot_id')} (tricked by IP/version)"
        )

    def test_target_dot_id_zero_preservation(self):
        """Adversarial 3.3: Verify target_dot_id=0 is preserved as int(0) and NOT coerced to None."""
        # 1. As integer 0
        cleaned_int = _clean_decision({"action": "click", "target_dot_id": 0})
        assert cleaned_int["target_dot_id"] == 0
        assert cleaned_int["target_dot_id"] is not None

        # 2. As string "0"
        cleaned_str = _clean_decision({"action": "click", "target_dot_id": "0"})
        assert cleaned_str["target_dot_id"] == 0
        assert cleaned_str["target_dot_id"] is not None

        # 3. As float 0.0
        cleaned_flt = _clean_decision({"action": "click", "target_dot_id": 0.0})
        assert cleaned_flt["target_dot_id"] == 0
        assert cleaned_flt["target_dot_id"] is not None

        # 4. As recovered from "dot 0"
        cleaned_rec = _clean_decision({
            "action": "click",
            "deliberation": {"tactical_action_justification": "Click on dot 0 to select first element."}
        })
        assert cleaned_rec["target_dot_id"] == 0

    @pytest.mark.parametrize("raw_input, expected_id", [
        ([42, 99], 42),
        ((15, 30), 15),
        ("Dot #88", 88),
        ("node: 77", 77),
        ("droplet 66", 66),
        ("[55]", 55),
        ("#44", 44),
        ("dot_33", 33),
        ("Target: 22", 22),
        (105.0, 105),
        ("None", None),
        ("null", None),
        ("", None),
        (None, None),
    ])
    def test_adversarial_target_dot_id_type_normalizations(self, raw_input, expected_id):
        """Adversarial 3.4: Verify exotic and edge-case types for target_dot_id normalize correctly."""
        cleaned = _clean_decision({"action": "click", "target_dot_id": raw_input})
        assert cleaned["target_dot_id"] == expected_id

    # Complete mapping verification: every synonym in ACTION_SYNONYMS
    # must map to one of the 11 canonical actions.
    @pytest.mark.parametrize("synonym, expected_canonical", list(DiagonalDropletReasoner.ACTION_SYNONYMS.items()))
    def test_all_action_synonyms_map_to_canonical_actions(self, synonym, expected_canonical):
        """Adversarial 3.5: Exhaustive verification that all 45+ synonyms map to canonical actions."""
        assert expected_canonical in (DiagonalDropletReasoner.CANONICAL_ACTIONS | {"press"}), (
            f"Synonym '{synonym}' maps to non-canonical action '{expected_canonical}'"
        )
        cleaned = _clean_decision({"action": synonym, "target_dot_id": 10})
        assert cleaned["action"] == expected_canonical, (
            f"Synonym '{synonym}' cleaned to '{cleaned['action']}', expected '{expected_canonical}'"
        )
        assert cleaned["action_type"] == expected_canonical

    @pytest.mark.parametrize("raw_action, expected_canonical", [
        (" CLICK ", "click"),
        ("Single-Click", "click"),
        ("Double_Click", "double_click"),
        ("TYPE-AND-ENTER", "type_and_enter"),
        ("Clear And Type", "clear_and_type"),
        ("Launch-App", "launch_app"),
        ("Open_Path", "open_path"),
        ("SWITCH WINDOW", "switch_window"),
        ("MINIMIZE_WINDOW", "minimize"),
        ("Maximize-Window", "maximize"),
        ("KEY_COMBO", "hotkey"),
        ("COMPLETE", "stop"),
    ])
    def test_case_and_whitespace_insensitivity_in_action_normalization(self, raw_action, expected_canonical):
        """Adversarial 3.6: Verify mixed-case, hyphenated, and space-padded actions normalize to canonical."""
        cleaned = _clean_decision({"action": raw_action, "target_dot_id": 5})
        assert cleaned["action"] == expected_canonical

    def test_hotkey_auto_recovery_from_various_deliberation_patterns(self):
        """Adversarial 3.7: Verify hotkey recovery from various deliberation syntax patterns."""
        patterns = [
            ("Press Ctrl+C to copy selected text.", ["ctrl", "c"]),
            ("Use Alt+Tab to switch foreground window.", ["alt", "tab"]),
            ("Press Win+R to open Run dialog.", ["win", "r"]),
            ("Execute Alt+D to focus browser address bar.", ["alt", "d"]),
            ("Send Ctrl+W to close active tab.", ["ctrl", "w"]),
            ("Press F5 to refresh page.", ["ctrl", "r"]),  # F5 maps to ctrl+r
        ]

        for text_pattern, expected_combo in patterns:
            dec = {
                "action": "hotkey",
                "deliberation": {
                    "tactical_action_justification": text_pattern,
                    "user_intent_and_success_criteria": text_pattern
                }
            }
            cleaned = _clean_decision(dec)
            assert cleaned["action"] == "hotkey"
            assert cleaned["key_combo"] == expected_combo, (
                f"For pattern '{text_pattern}', expected combo {expected_combo}, got {cleaned.get('key_combo')}"
            )

    def test_hotkey_multi_modifier_limitation_documented(self):
        """
        Adversarial 3.7b: Documents multi-modifier hotkey (Ctrl+Shift+Esc) limitation.
        Shows that the 2-key regex captures ['shift', 'esc'] and drops 'ctrl'.
        """
        dec = {
            "action": "hotkey",
            "deliberation": {
                "tactical_action_justification": "Shortcut Ctrl+Shift+Esc opens Task Manager."
            }
        }
        cleaned = _clean_decision(dec)
        assert cleaned["action"] == "hotkey"
        # Documents empirical observation that current regex matches shift+esc
        assert cleaned["key_combo"] == ["shift", "esc"]

    def test_press_key_auto_recovery_from_deliberation_text(self):
        """Adversarial 3.8: Verify press key recovery from deliberation text."""
        key_patterns = [
            ("Press enter key to commit the form.", "enter"),
            ("Press esc to dismiss the dialog.", "esc"),
            ("Use tab to navigate to the next field.", "tab"),
            ("Hit backspace to clear the character.", "backspace"),
            ("Press delete to remove the selected file.", "delete"),
            ("Press space to toggle checkbox.", "space"),
            ("Press down arrow to select next item.", "down"),
        ]

        for text_pattern, expected_key in key_patterns:
            dec = {
                "action": "press",
                "deliberation": {
                    "tactical_action_justification": text_pattern,
                    "user_intent_and_success_criteria": text_pattern
                }
            }
            cleaned = _clean_decision(dec)
            assert cleaned["action"] == "press"
            assert cleaned["key"] == expected_key, (
                f"For pattern '{text_pattern}', expected key '{expected_key}', got '{cleaned.get('key')}'"
            )
