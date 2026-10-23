"""
Adversarial Test Suite for Milestone 5 (Tier 5 Adversarial Coverage Hardening):
Perception & Reasoning Track.

Stress-tests:
a) Extreme token spike inputs (>5,000 to 20,000 tokens) & 3-tier token budget guard compaction.
b) Degenerate visual hierarchy, inverted bounding boxes, off-screen nodes, multi-modal dialog distractors.
c) Severely malformed LLM responses: mixed markdown fences, unicode null bytes, raw JSON wrapped in conversational chat, missing fields, passive thinking text loops.
d) 5-stage CoT parser robustness against deceptive prefixes, dot ID extractions, and provider resilience.
"""
import sys
import os
import re
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Ensure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.vision.screen_parser import ScreenStateParser
from spatial_droplet.reasoning.reasoner import (
    DiagonalDropletReasoner,
    DecisionResult,
    _clean_decision,
    _extract_json,
    classify_screen_zone,
    _build_spatial_layout_graph,
)
from spatial_droplet.reasoning.token_guard import TokenBudgetGuard
from spatial_droplet.reasoning.providers.base import GroundingDecision
from spatial_droplet.reasoning.providers.gemini_provider import GeminiGroundingProvider
from spatial_droplet.reasoning.providers.openai_provider import OpenAIGroundingProvider
from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    ModalDialogContext,
    ScreenAwarenessContext,
    TrajectoryStep,
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory


# ============================================================================
# TIER 5 - TRACK A: Extreme Token Spike Inputs (>5k-20k Tokens) & Compaction
# ============================================================================

class TestAdversarialTokenBudgetCompaction:
    """Stress-tests TokenBudgetGuard under massive token floods and extreme input spikes."""

    def test_extreme_token_spike_flood_over_20k_tokens(self):
        """
        Adversarial scenario: An extreme input flood of >20,000 tokens consisting of:
        - 120 candidate droplets with verbose labels
        - 40 historical trajectory steps
        - Massive multi-paragraph system instruction with exemplars
        - Repetitive bloated user prompt
        TokenBudgetGuard must strictly compact the prompt to < 4,800 tokens (< 5,000 ceiling).
        """
        # 1. Generate 120 candidate droplets
        candidates = [
            DiagonalDroplet(
                id=i,
                center_x=(i * 15) % 1920,
                center_y=(i * 12) % 1080,
                category="interactive_widget",
                label=f"SuperVerboseElementDescriptor_{i}_DetailedSubsystemComponent_WithExtraneousContext",
                diag_wave=i * 27,
                is_foreground=True,
                archetype="button" if i % 2 == 0 else "text",
            )
            for i in range(1, 121)
        ]

        # 2. Generate episodic memory with 40 steps
        memory = EpisodicTrajectoryMemory(goal="Autonomous end-to-end task")
        for s in range(1, 41):
            step = TrajectoryStep(
                step_num=s,
                goal="Autonomous end-to-end task",
                subgoal=f"Execute comprehensive milestone stage {s} to advance autonomous workflow",
                action="click" if s % 2 == 0 else "clear_and_type",
                target_id=s,
                target_label=f"VerboseCandidateTargetLabel_{s}_InSubsystem",
                input_text="Detailed payload input string" if s % 2 != 0 else "",
                verified=True if s < 38 else False,
                verification_reason=f"Detailed verification telemetry diagnostics reason for step {s}",
                active_window="Desktop",
                verdict="TRANSITION_VERIFIED" if s < 38 else "NO_MOTION_DETECTED",
                end_to_end_master_plan=[
                    f"Phase 1: Milestone setup and pre-checks (DONE)",
                    f"Phase 2: Active operation execution stage {s} (ACTIVE)",
                    f"Phase 3: Final state verification and commit (PENDING)",
                ]
            )
            memory.add_step(step)

        # 3. Massive system instruction (>3,000 tokens)
        system_instruction = (
            "You are an expert GUI visual grounding agent operating under strict latency constraints.\n"
            "<few_shot_exemplars>\n"
            "--- EXEMPLAR 1 ---\n" + ("Exemplar 1 demonstration rationale and step instructions.\n" * 100) +
            "--- EXEMPLAR 2 ---\n" + ("Exemplar 2 extended historical edge case documentation.\n" * 200) +
            "</few_shot_exemplars>\n" +
            ("Core architectural instructions and guidance rules.\n" * 150)
        )

        # 4. Massive user prompt (>15,000 tokens)
        user_content = (
            "<overall_user_goal>\n"
            + ("Autonomous end-to-end task requiring deep multi-step desktop coordination. " * 200)
            + "\n</overall_user_goal>\n\n"
            + "<system_grounding>\n"
            + ("Active drive paths, user configuration details, and operating environment constants.\n" * 300)
            + "</system_grounding>\n\n"
            + "<trajectory_history>\n"
            + TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=40)
            + "\n</trajectory_history>\n\n"
            + "<spatial_layout_graph>\n"
            + _build_spatial_layout_graph(candidates, 1920, 1080)
            + "\n</spatial_layout_graph>\n\n"
            + ("Extraneous multi-turn dialogue context.\n" * 400)
        )

        initial_tokens = TokenBudgetGuard.count_tokens(system_instruction) + TokenBudgetGuard.count_tokens(user_content)
        assert initial_tokens > 10000, f"Setup verification: initial prompt must exceed 10,000 tokens, got {initial_tokens}"

        # Execute 3-tier dynamic compaction
        sys_out, usr_out, active_nodes = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=candidates,
            max_tokens=4800,
            goal="Autonomous end-to-end task requiring deep multi-step desktop coordination.",
            memory=memory,
            sw=1920,
            sh=1080,
        )

        final_tokens = TokenBudgetGuard.count_tokens(sys_out) + TokenBudgetGuard.count_tokens(usr_out)

        # Invariant 1: Total prompt tokens must strictly respect the hard budget (< 4800)
        assert final_tokens < 4800, f"Token budget violated! Expected < 4800 tokens, got {final_tokens}"

        # Invariant 2: Candidate droplets pruned to tier 1 limits (<= 36 or <= 24)
        assert len(active_nodes) <= 36, f"Candidate count not pruned! Got {len(active_nodes)}"

        # Invariant 3: Output strings are non-empty and well-formed
        assert len(sys_out.strip()) > 0
        assert len(usr_out.strip()) > 0

    def test_system_exemplar_stripping_under_pressure(self):
        """
        Verifies Tier 4 headroom compaction: When prompt remains over budget after
        candidate pruning and spatial graph condensation, TokenBudgetGuard excises
        secondary exemplars (--- EXEMPLAR 2) from system_instruction.
        """
        sys_with_exemplars = (
            "System instruction header.\n"
            "<few_shot_exemplars>\n"
            "--- EXEMPLAR 1: Baseline Action ---\n"
            "Action details for exemplar 1.\n"
            "--- EXEMPLAR 2: Complex Action ---\n"
            "Action details for exemplar 2.\n"
            "</few_shot_exemplars>\n"
            "System instruction footer."
        )

        # Create user content large enough to trigger exemplar pruning (> 6,000 tokens)
        user_bloat = "Adversarial word token pressure testing content block.\n" * 1200

        sys_out, usr_out, _ = TokenBudgetGuard.enforce(
            system_instruction=sys_with_exemplars,
            user_content=user_bloat,
            nodes=[],
            max_tokens=4800,
        )

        # Verify exemplar 2 was stripped to recover headroom
        assert "--- EXEMPLAR 2" not in sys_out
        assert "--- EXEMPLAR 1" in sys_out
        assert TokenBudgetGuard.count_tokens(sys_out) + TokenBudgetGuard.count_tokens(usr_out) < 4800

    def test_trajectory_history_compaction_and_loop_warning_preservation(self):
        """
        Verifies Tier 3 compaction: Truncates history to recent 3 steps while
        strictly preserving active master plan symbols and critical loop warnings.
        """
        memory = EpisodicTrajectoryMemory(goal="Test loop warning")

        # Step 1: Completed
        s1 = TrajectoryStep(
            step_num=1,
            goal="Test loop warning",
            subgoal="Open File Menu",
            action="click",
            target_id=5,
            target_label="FileMenu",
            input_text=None,
            verified=True,
            verification_reason="Menu opened",
            active_window="Notepad",
            verdict="TRANSITION_VERIFIED",
            end_to_end_master_plan=[
                "✔ Setup environment (DONE)",
                "► Open target file (ACTIVE)",
                "○ Save processed file (PENDING)",
            ]
        )
        memory.add_step(s1)

        # Step 2: Failed attempt 1 on same control (ping-pong setup)
        s2 = TrajectoryStep(
            step_num=2,
            goal="Test loop warning",
            subgoal="Save file",
            action="click",
            target_id=9,
            target_label="SaveButton",
            input_text=None,
            verified=False,
            verification_reason="Zero visual change",
            active_window="Notepad",
            verdict="NO_MOTION_DETECTED",
        )
        memory.add_step(s2)

        # Step 3: Failed attempt 2 on same control
        s3 = TrajectoryStep(
            step_num=3,
            goal="Test loop warning",
            subgoal="Save file",
            action="click",
            target_id=9,
            target_label="SaveButton",
            input_text=None,
            verified=False,
            verification_reason="Zero visual change",
            active_window="Notepad",
            verdict="NO_MOTION_DETECTED",
        )
        memory.add_step(s3)

        compacted = TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=3)

        # Invariants:
        assert "Active Master Plan:" in compacted
        assert "✔" in compacted
        assert "►" in compacted
        assert "○" in compacted
        assert "Chronological Trajectory History (Recent 3 Steps):" in compacted
        assert "Step 1:" in compacted
        assert "Step 2:" in compacted
        assert "Step 3:" in compacted
        assert "CRITICAL LOOP WARNING" in compacted or "⚠️" in compacted

    def test_token_counting_resilience_and_encoding_fallback(self):
        """
        Verifies TokenBudgetGuard.count_tokens handles empty strings, unicode nulls,
        and cleanly falls back to (len + 3) // 4 when tiktoken is unavailable.
        """
        # Empty string
        assert TokenBudgetGuard.count_tokens("") == 0
        assert TokenBudgetGuard.count_tokens(None) == 0

        # Normal text with tiktoken
        sample = "Screen-aware deep self-reasoning with spatial layout graph."
        count_normal = TokenBudgetGuard.count_tokens(sample)
        assert count_normal > 0

        # Unicode null bytes and emojis
        messy_sample = "Test\x00with\x00nulls and emojis 🎯🚀 " + ("abc " * 50)
        count_messy = TokenBudgetGuard.count_tokens(messy_sample)
        assert count_messy > 0

        # Fallback when tiktoken is disabled / fails
        with patch.object(TokenBudgetGuard, "get_encoding", return_value=None):
            count_fallback = TokenBudgetGuard.count_tokens(sample)
            expected_fallback = max(1, (len(sample) + 3) // 4)
            assert count_fallback == expected_fallback

    def test_count_messages_tokens_multimodal_complex_payload(self):
        """
        Verifies TokenBudgetGuard.count_messages_tokens correctly calculates
        text tokens, low-detail image tokens (85), high-detail image tokens (1105),
        and per-message framing overhead.
        """
        messages = [
            {"role": "system", "content": "You are a grounding assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Locate the save icon in this screenshot."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
                ],
            },
        ]

        total = TokenBudgetGuard.count_messages_tokens(messages)

        # Expected:
        # Message 1 overhead: 4
        # System text: count_tokens("You are a grounding assistant.")
        # Message 2 overhead: 4
        # User text: count_tokens("Locate the save icon in this screenshot.")
        # Image 1 (high): 1105
        # Image 2 (low): 85
        # Assistant priming: 3
        sys_tokens = TokenBudgetGuard.count_tokens("You are a grounding assistant.")
        user_tokens = TokenBudgetGuard.count_tokens("Locate the save icon in this screenshot.")
        expected = 4 + sys_tokens + 4 + user_tokens + 1105 + 85 + 3

        assert total == expected

    def test_modal_droplets_immortality_under_heavy_candidate_pruning(self):
        """
        Adversarial test: 100 candidate droplets where only 2 droplets are inside an active
        modal dialog rect, and 98 are background distractors with high goal-keyword matches.
        TokenBudgetGuard MUST reserve and retain the modal droplets during Tier 1 compaction.
        """
        modal_rect = (400, 300, 800, 600)

        # Droplet 1 & 2: inside modal dialog
        modal_drop1 = DiagonalDroplet(
            id=1,
            center_x=500,
            center_y=450,
            category="interactive_widget",
            label="Save Confirmation Confirm",
            diag_wave=950,
            archetype="button",
        )
        modal_drop2 = DiagonalDroplet(
            id=2,
            center_x=600,
            center_y=450,
            category="interactive_widget",
            label="Discard Cancel",
            diag_wave=1050,
            archetype="button",
        )

        # 98 background distractors with heavy goal-matching words
        background_distractors = [
            DiagonalDroplet(
                id=i,
                center_x=(i * 18) % 1920,
                center_y=150 if i % 2 == 0 else 900,
                category="interactive_widget",
                label="Save Target Folder Document File Confirm",
                diag_wave=i * 20,
                is_foreground=False,
                archetype="button",
            )
            for i in range(3, 101)
        ]

        all_nodes = [modal_drop1, modal_drop2] + background_distractors

        pruned = TokenBudgetGuard.compact_candidates_tier1(
            nodes=all_nodes,
            goal="Confirm save target document",
            modal_rect=modal_rect,
            max_nodes=24,
        )

        assert len(pruned) == 24
        pruned_ids = {d.id for d in pruned}
        assert 1 in pruned_ids, "Modal Droplet 1 was evicted by background distractors!"
        assert 2 in pruned_ids, "Modal Droplet 2 was evicted by background distractors!"


# ============================================================================
# TIER 5 - TRACK B: Degenerate Visual Hierarchy & Distractor Stress
# ============================================================================

class TestAdversarialDegenerateVisualHierarchyAndDistractors:
    """Stress-tests screen state parsing, inverted bboxes, and degenerate geometry."""

    def test_inverted_bounding_boxes_and_negative_coordinates(self):
        """
        Adversarial scenario: Inverted window bounding boxes where x1 > x2 or y1 > y2,
        negative coordinates, and zero-area geometry.
        classify_screen_zone and _build_spatial_layout_graph must not crash with ZeroDivisionError.
        """
        # Inverted window rect: x1=1200, y1=800, x2=400, y2=200
        inverted_win = (1200, 800, 400, 200)

        # Inverted modal rect: x1=900, y1=700, x2=500, y2=300
        inverted_modal = (900, 700, 500, 300)

        # Negative window rect
        negative_win = (-500, -300, -100, -50)

        # Zero-area rect
        zero_rect = (400, 400, 400, 400)

        # 1. classify_screen_zone must survive and return a valid canonical zone
        zone1 = classify_screen_zone(x=500, y=400, sw=1920, sh=1080, modal_rect=inverted_modal, win_rect=inverted_win)
        assert zone1 in ("[MODAL/POPUP]", "[HEADER/NAV]", "[SIDEBAR]", "[WORKSPACE]", "[FOOTER]")

        zone2 = classify_screen_zone(x=100, y=100, sw=1920, sh=1080, modal_rect=None, win_rect=negative_win)
        assert zone2 in ("[HEADER/NAV]", "[SIDEBAR]", "[WORKSPACE]", "[FOOTER]")

        zone3 = classify_screen_zone(x=400, y=400, sw=1920, sh=1080, modal_rect=None, win_rect=zero_rect)
        assert zone3 in ("[HEADER/NAV]", "[SIDEBAR]", "[WORKSPACE]", "[FOOTER]")

        # 2. _build_spatial_layout_graph must survive with inverted rects
        droplets = [
            DiagonalDroplet(id=1, center_x=300, center_y=300, category="button", label="TestBtn1", diag_wave=600),
            DiagonalDroplet(id=2, center_x=-50, center_y=-50, category="button", label="NegativeCoordBtn", diag_wave=-100),
        ]
        graph = _build_spatial_layout_graph(droplets, sw=1920, sh=1080, modal_rect=inverted_modal, win_rect=inverted_win)
        assert "=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH" in graph
        assert "• [1]" in graph

    def test_degenerate_screen_dimensions(self):
        """
        Adversarial scenario: Screen dimensions are 0x0 or negative.
        Both classify_screen_zone and _build_spatial_layout_graph must clamp dimensions
        using max(1, sw) and max(1, sh) without raising exceptions.
        """
        # Zero dimensions
        z1 = classify_screen_zone(x=10, y=10, sw=0, sh=0)
        assert isinstance(z1, str)

        # Negative dimensions
        z2 = classify_screen_zone(x=10, y=10, sw=-1920, sh=-1080)
        assert isinstance(z2, str)

        # Layout graph with 0x0 screen
        droplet = DiagonalDroplet(id=1, center_x=5, center_y=5, category="button", label="OriginBtn", diag_wave=10)
        graph = _build_spatial_layout_graph([droplet], sw=0, sh=0)
        assert "=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH" in graph

    def test_extreme_offscreen_and_nan_droplet_coordinates(self):
        """
        Adversarial scenario: Droplets located far beyond standard monitors (e.g. x=500,000)
        or negative coordinates. Relative neighbor calculations must survive cleanly.
        """
        droplets = [
            DiagonalDroplet(id=1, center_x=-99999, center_y=-50000, category="button", label="FarNegative", diag_wave=-149999),
            DiagonalDroplet(id=2, center_x=500000, center_y=800000, category="button", label="FarPositive", diag_wave=1300000),
            DiagonalDroplet(id=3, center_x=960, center_y=540, category="button", label="CenterScreen", diag_wave=1500),
        ]
        graph = _build_spatial_layout_graph(droplets, sw=1920, sh=1080)
        assert "• [1]" in graph
        assert "• [2]" in graph
        assert "• [3]" in graph

    def test_screen_state_parser_degenerate_window_context(self):
        """
        Adversarial scenario: WindowContext is completely empty, None, or has invalid handles.
        ScreenStateParser must safely return None for modals and valid fallback ScreenAwarenessContext.
        """
        # None window context
        modal_none = ScreenStateParser.detect_modal_dialog(None)
        assert modal_none is None

        # Invalid window context
        invalid_win = WindowContext(hwnd=0, title="", rect=None, is_valid=False)
        modal_inv = ScreenStateParser.detect_modal_dialog(invalid_win)
        assert modal_inv is None

        # Desktop window context
        desktop_win = WindowContext(hwnd=1001, title="Desktop", is_valid=True)
        modal_desk = ScreenStateParser.detect_modal_dialog(desktop_win)
        assert modal_desk is None

        # Screen awareness extraction with invalid context
        awareness = ScreenStateParser.parse_screen_awareness(win_ctx=None, all_ocr_lines=[], droplets=[])
        assert awareness.scene_type == "DESKTOP"
        assert awareness.app_name == "Desktop"
        assert len(awareness.summary) <= 280

    def test_in_window_modal_sheet_heuristics_with_contradictory_ocr(self):
        """
        Adversarial scenario: Win32 API doesn't detect a separate HWND, but OCR text
        contains explicit modal sheet triggers ("Are you sure you want to exit? Unsaved changes will be lost").
        ScreenStateParser must elevate the window into an InWindowModalSheet.
        """
        win = WindowContext(hwnd=5432, title="TextEditor - Untitled.txt", rect=(100, 100, 1100, 700), is_valid=True)
        ocr_lines = [
            "File Edit View Format Help",
            "Do you want to save changes to Untitled.txt?",
            "Save", "Don't Save", "Cancel",
            "Random editor text line 1",
            "Random editor text line 2",
        ]

        awareness = ScreenStateParser.parse_screen_awareness(win_ctx=win, all_ocr_lines=ocr_lines, droplets=[])
        assert awareness.active_modal is not None
        assert awareness.active_modal.is_modal is True
        assert awareness.active_modal.dialog_type == "InWindowSheet"
        assert awareness.scene_type == "MODAL_DIALOG"

    def test_multimodal_dialog_distractor_prioritization(self):
        """
        Adversarial scenario: Three buttons on screen all share the exact label 'Cancel':
        - Dot 10: In active modal dialog (500, 450)
        - Dot 20: In parent window header (200, 50)
        - Dot 30: In global taskbar / footer (800, 1000)
        TokenBudgetGuard.score_droplet_relevance must prioritize the modal control
        by an overwhelming score margin (> 10,000 points).
        """
        modal_rect = (400, 300, 800, 600)
        win_rect = (100, 100, 1200, 800)

        d_modal = DiagonalDroplet(id=10, center_x=500, center_y=450, category="button", label="Cancel", diag_wave=950)
        d_parent = DiagonalDroplet(id=20, center_x=200, center_y=150, category="button", label="Cancel", diag_wave=350)
        d_footer = DiagonalDroplet(id=30, center_x=800, center_y=1000, category="button", label="Cancel", diag_wave=1800)

        s_modal = TokenBudgetGuard.score_droplet_relevance(d_modal, goal="Cancel operation", modal_rect=modal_rect, win_rect=win_rect)
        s_parent = TokenBudgetGuard.score_droplet_relevance(d_parent, goal="Cancel operation", modal_rect=modal_rect, win_rect=win_rect)
        s_footer = TokenBudgetGuard.score_droplet_relevance(d_footer, goal="Cancel operation", modal_rect=modal_rect, win_rect=win_rect)

        assert s_modal > 10000.0, f"Modal droplet did not receive +10,000 boost: {s_modal}"
        assert s_modal > s_parent + 9000.0
        assert s_modal > s_footer + 9000.0


# ============================================================================
# TIER 5 - TRACK C: Severely Malformed LLM Responses & Thinking Loops
# ============================================================================

class TestAdversarialMalformedLLMResponses:
    """Stress-tests JSON extraction against corrupt markdown, null bytes, and passive prose."""

    def test_unicode_null_bytes_and_control_chars_sanitization(self):
        """
        Adversarial scenario: Model output contains injected unicode null bytes (\x00),
        vertical tabs (\x0b), backspaces (\x08), and escape sequences surrounding JSON.
        _extract_json must cleanly extract and parse the decision.
        """
        corrupted_raw = (
            "Thinking text with control chars:\x00\x08\x0b\r\n"
            "```json\n"
            "{\n"
            '  "action": "type_and_enter",\n'
            '  "target_dot_id": 42,\n'
            '  "input_text": "sanitized_input\\x00_test"\n'
            "}\n"
            "```\n\x00\x00Finished response."
        )

        decision = _extract_json(corrupted_raw)
        assert decision is not None
        assert decision["action"] == "type_and_enter"
        assert decision["target_dot_id"] == 42
        assert "sanitized_input" in decision["input_text"]

    def test_mixed_nested_markdown_and_unclosed_fences(self):
        """
        Adversarial scenario: Complex nested markdown fences (```markdown followed by ```json)
        or unclosed markdown blocks at the end of stream.
        """
        # Nested fences
        nested_raw = (
            "Here is the plan:\n"
            "````markdown\n"
            "```json\n"
            "{\n"
            '  "action": "click",\n'
            '  "target_dot_id": 19,\n'
            '  "reasoning": "Nested markdown block test"\n'
            "}\n"
            "```\n"
            "````"
        )
        dec_nested = _extract_json(nested_raw)
        assert dec_nested is not None
        assert dec_nested["action"] == "click"
        assert dec_nested["target_dot_id"] == 19

        # Unclosed markdown block
        unclosed_raw = (
            "Assistant dispatch:\n"
            "```json\n"
            "{\n"
            '  "action": "double_click",\n'
            '  "target_dot_id": 88,\n'
            '  "reasoning": "Unclosed fence at token cutoff"'
        )
        dec_unclosed = _extract_json(unclosed_raw)
        assert dec_unclosed is not None
        assert dec_unclosed["action"] == "double_click"
        assert dec_unclosed["target_dot_id"] == 88

    def test_severely_truncated_json_repair(self):
        """
        Adversarial scenario: Output cut off mid-string or mid-structure due to max_token limits.
        _extract_json must repair unclosed quotes and braces and recover available fields.
        """
        truncated_raw = (
            '{"action": "clear_and_type", "target_dot_id": 73, "input_text": "hello adversarial world", '
            '"deliberation": {"user_intent_and_success_criteria": "Cutoff mid sentence...'
        )
        decision = _extract_json(truncated_raw)
        assert decision is not None
        assert decision["action"] == "clear_and_type"
        assert decision["target_dot_id"] == 73
        assert decision["input_text"] == "hello adversarial world"

    def test_zero_passive_text_deliberation_invariant(self):
        """
        Adversarial scenario: Model terminates in pure conversational introspection
        without producing any actionable command or targeting a dot.
        _extract_json MUST reject pure passive prose (returning None),
        while recovering action when a grounded dot directive is provided.
        """
        # 1. Pure passive contemplation without action -> REJECT (return None)
        passive_text = (
            "<think>\n"
            "I should check the screen layout and examine the icons.\n"
            "Perhaps I will wait for the screen to settle.\n"
            "</think>\n"
            "I am observing the application. It looks like the window is currently loading."
        )
        res_passive = _extract_json(passive_text)
        assert res_passive is None, "Passive contemplation without grounded action was improperly accepted!"

        # 2. Conversational response containing explicit dot action directive -> ACCEPT & RECOVER
        actionable_prose = (
            "<think>Analyzing candidates...</think>\n"
            "To save the document, we should click on Dot [31] which corresponds to the Save button."
        )
        res_actionable = _extract_json(actionable_prose)
        assert res_actionable is not None
        assert res_actionable["action"] == "click"
        assert res_actionable["target_dot_id"] == 31

    def test_deeply_nested_conversational_chat_wrapper(self):
        """
        Adversarial scenario: JSON block buried within a multi-turn chat persona
        with greetings, quotes, multiple code snippets, and trailing banter.
        """
        chat_wrapped = (
            "Hello! I am your AI desktop automation copilot.\n\n"
            "Here is an example of what I usually do:\n"
            "```python\n"
            "# Example script\n"
            "print('Hello world')\n"
            "```\n\n"
            "For this turn, here is the exact decision payload you must execute:\n"
            "```json\n"
            "{\n"
            '  "action": "switch_window",\n'
            '  "target_dot_id": null,\n'
            '  "input_text": "Visual Studio Code",\n'
            '  "deliberation": {\n'
            '    "user_intent_and_success_criteria": "Switch to IDE window.",\n'
            '    "screen_state_and_visual_hierarchy": "IDE is in background.",\n'
            '    "mistake_critique_and_loop_diagnosis": "No errors.",\n'
            '    "end_to_end_master_plan": ["Phase 1: Switch (ACTIVE)"],\n'
            '    "tactical_action_justification": "Bring target app to focus."\n'
            "  }\n"
            "}\n"
            "```\n\n"
            "Please execute this and report back the visual settle result!"
        )

        decision = _extract_json(chat_wrapped)
        assert decision is not None
        assert decision["action"] == "switch_window"
        assert decision["input_text"] == "Visual Studio Code"
        assert decision["deliberation"]["user_intent_and_success_criteria"] == "Switch to IDE window."


# ============================================================================
# TIER 5 - TRACK D: 5-Stage CoT Parser Robustness & Provider Resilience
# ============================================================================

class TestAdversarialCoTParserRobustnessAndDotExtractions:
    """Stress-tests 5-stage CoT schema, dot ID exotic formats, and provider exception handling."""

    def test_deceptive_action_prefixes_and_negations(self):
        """
        Adversarial scenario: Action string contains deceptive prefixes or negations:
        'do_not_click', 'avoid_closing', 'minimize_window', 'maximize_app'.
        _clean_decision must deterministically normalize to one of the 11 canonical actions.
        """
        # Alias: minimize_window -> minimize
        d1 = _clean_decision({"action": "minimize_window"})
        assert d1["action"] == "minimize"

        # Alias: maximize_window -> maximize
        d2 = _clean_decision({"action": "maximize_window"})
        assert d2["action"] == "maximize"

        # Non-canonical / deceptive action with target dot -> converts to 'click'
        d3 = _clean_decision({
            "action": "do_not_click",
            "target_dot_id": 14,
            "deliberation": {"tactical_action_justification": "Clicking button 14"}
        })
        assert d3["action"] == "click"
        assert d3["target_dot_id"] == 14

    @pytest.mark.parametrize("dot_input,expected_output", [
        (42, 42),
        (42.0, 42),
        ("42", 42),
        ("[Dot 42]", 42),
        ("Dot [42]", 42),
        ("dot_42", 42),
        ("#42", 42),
        ("Target Dot: 42", 42),
        ("node: 42", 42),
        ([42], 42),
        ("null", None),
        ("None", None),
        ("", None),
        (None, None),
        ([], None),
    ])
    def test_dot_id_exotic_string_and_numeric_variants(self, dot_input, expected_output):
        """
        Verifies _clean_decision handles exotic representations of target_dot_id
        (brackets, prefixes, floats, lists, strings) and normalizes to int or None.
        """
        decision = {"action": "click", "target_dot_id": dot_input}
        cleaned = _clean_decision(decision)
        assert cleaned["target_dot_id"] == expected_output
        assert cleaned.get("target_droplet_id") == expected_output

    def test_5stage_cot_bidirectional_cross_contamination(self):
        """
        Verifies bidirectional schema sync:
        1. When legacy keys are provided, modern 5-stage CoT keys are generated.
        2. When modern 5-stage CoT keys are provided, legacy keys are populated.
        3. DecisionResult provides property accessors for all schema variants.
        """
        # 1. Legacy input
        legacy_input = {
            "action": "click",
            "target_dot_id": 7,
            "deliberation": {
                "intent_analysis": "Decompose user query for file save.",
                "observation_analysis": "Notepad active on screen.",
                "critique_of_previous_step": "Step 1 executed successfully.",
                "strategy_plan": "Click File -> Click Save As.",
                "candidate_evaluation": "Dot 7 matches File menu.",
            }
        }
        c1 = _clean_decision(legacy_input)
        res1 = DecisionResult(c1)

        assert res1.user_intent_and_success_criteria == "Decompose user query for file save."
        assert res1.screen_state_and_visual_hierarchy == "Notepad active on screen."
        assert res1.mistake_critique_and_loop_diagnosis == "Step 1 executed successfully."
        assert res1.end_to_end_master_plan == ["Click File -> Click Save As."]
        assert res1.tactical_action_justification == "Dot 7 matches File menu."

        # 2. Modern input
        modern_input = {
            "action": "click",
            "target_dot_id": 8,
            "deliberation": {
                "user_intent_and_success_criteria": "Confirm dialog exit.",
                "screen_state_and_visual_hierarchy": "Modal sheet elevated.",
                "mistake_critique_and_loop_diagnosis": "No previous failures.",
                "end_to_end_master_plan": ["Phase 1: Confirm (ACTIVE)"],
                "tactical_action_justification": "Select OK commit button.",
            }
        }
        c2 = _clean_decision(modern_input)
        res2 = DecisionResult(c2)

        assert res2.intent_analysis == "Confirm dialog exit."
        assert res2.spatial_anchoring == "Modal sheet elevated."
        assert res2.critique_of_previous_step == "No previous failures."
        assert "Phase 1: Confirm (ACTIVE)" in res2.strategy_plan
        assert res2.candidate_evaluation == "Select OK commit button."

    def test_master_plan_arbitrary_formatting_and_dict_accessors(self):
        """
        Verifies master plan parsing handles multi-line strings, bullet points,
        and clean list formatting across DecisionResult attributes.
        """
        plan_str = (
            "1. ✔ Setup workspace environment (DONE)\n"
            "2. ► Click save button (ACTIVE)\n"
            "3. ○ Verify file saved to disk (PENDING)"
        )
        decision = {
            "action": "click",
            "target_dot_id": 11,
            "deliberation": {"end_to_end_master_plan": plan_str}
        }
        cleaned = _clean_decision(decision)
        res = DecisionResult(cleaned)

        assert isinstance(res.end_to_end_master_plan, list)
        assert len(res.end_to_end_master_plan) == 3
        assert "✔" in res.end_to_end_master_plan[0]
        assert "►" in res.end_to_end_master_plan[1]
        assert "○" in res.end_to_end_master_plan[2]

    def test_hotkey_and_press_recovery_from_deliberation_text(self):
        """
        Verifies that when the model outputs action='hotkey' or action='press'
        without explicit key/key_combo parameters, _clean_decision scans the
        deliberation text to recover the keys.
        """
        # Hotkey auto-recovery
        d_hotkey = {
            "action": "hotkey",
            "deliberation": {
                "tactical_action_justification": "We should use Alt+Tab to switch to the active editor."
            }
        }
        c_hk = _clean_decision(d_hotkey)
        assert c_hk["key_combo"] == ["alt", "tab"]

        # Press key auto-recovery
        d_press = {
            "action": "press",
            "deliberation": {
                "tactical_action_justification": "Press Enter to commit the filename in the dialog."
            }
        }
        c_pr = _clean_decision(d_press)
        assert c_pr["key"] == "enter"

    def test_gemini_and_openai_providers_adversarial_error_recovery(self):
        """
        Adversarial scenario: External APIs throw connection resets, rate limit errors,
        or return empty choice payloads. Providers must catch all exceptions and return
        a structured GroundingDecision with confidence=0.0 and error details without crashing.
        """
        # 1. Gemini Grounding Provider error handling
        gemini = GeminiGroundingProvider(api_key="test-api-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("503 Service Unavailable: Rate quota exhausted")
        gemini.client = mock_client

        cand = DiagonalDroplet(id=1, center_x=100, center_y=100, category="button", label="OK", diag_wave=200)
        res_gemini = gemini.ground_sync(
            instruction="Click OK",
            manifest_text="• [1] OK",
            all_candidates=[cand],
        )
        assert res_gemini.confidence == 0.0
        assert "503 Service Unavailable" in res_gemini.error
        assert res_gemini.provider_used == "gemini"

        # 2. OpenAI Grounding Provider error handling
        openai = OpenAIGroundingProvider(api_key="test-api-key")
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.side_effect = ConnectionResetError("Connection aborted by peer")
        openai.client = mock_openai_client

        # Call async ground in test
        import asyncio
        res_openai = asyncio.run(openai.ground(
            instruction="Click OK",
            manifest_text="• [1] OK",
            all_candidates=[cand],
        ))
        assert res_openai.confidence == 0.0
        assert "Connection aborted by peer" in res_openai.error

    def test_som_unoccluded_overlay_memory_and_rendering_robustness(self):
        """
        Verifies Set-of-Marks visual overlay rendering with 1px non-occluding boundaries,
        2px hollow micro-anchors, border-pinned badges, and memory buffer deallocation.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        droplets = [
            DiagonalDroplet(id=1, center_x=10, center_y=10, category="button", label="TopLeftCorner", diag_wave=20),
            DiagonalDroplet(id=2, center_x=1910, center_y=1070, category="button", label="BottomRightCorner", diag_wave=2980),
            DiagonalDroplet(id=3, center_x=960, center_y=540, category="text", label="CenterTextInput", diag_wave=1500),
        ]

        img_url, (img_w, img_h), scale = DiagonalDropletReasoner._render_unoccluded_som(frame, droplets, max_image_side=1280)

        assert img_url.startswith("data:image/jpeg;base64,")
        assert len(img_url) > 1000
        assert img_w == 1280 or img_w == 1920
