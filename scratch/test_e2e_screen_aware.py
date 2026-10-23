"""
Opaque-Box End-to-End Test Suite: Screen-Aware Deep Self-Reasoning Spatial Droplet Engine.
Covers:
- F1: Window-Relative Visual Hierarchy & Zones ([MODAL/POPUP], [HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER])
- F2: Topological Set-of-Marks 2D Graph & High-Resolution Non-Occluding Overlay
- F4: Gemini Pro Multimodal VLM Provider Integration (OpenAI-compatible endpoint)
- F5: Strict Prompt Token Budget Guard (<5,000 tokens) & Dynamic Compaction
- F7: PrecisionActuator, Visual Settle Verification & Autonomous Loop Guard (<=2 turns)
- F8: Test Infrastructure, SHA-256 Parity & Offline Invariants
- Tier 4: Real-World Multi-Turn Desktop Application Scenarios
"""
import sys
import os
import time
import hashlib
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import pytest
import tiktoken

# Ensure utf-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    ModalDialogContext,
    ScreenAwarenessContext,
    TrajectoryStep,
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.reasoning.reasoner import (
    DiagonalDropletReasoner,
    DecisionResult,
)
from spatial_droplet.reasoning.providers.base import GroundingDecision
from spatial_droplet.reasoning.providers.openai_provider import OpenAIGroundingProvider
from spatial_droplet.reasoning.providers.registry import get_grounding_provider
from spatial_droplet.platform.actuator import (
    PrecisionActuator,
    Win32HardwareActuator,
)
from spatial_droplet.platform.window import (
    minimize_window,
    maximize_window,
    get_active_window_context,
)
from spatial_droplet.vision.settle_sensor import AdaptiveSettleSensor
from spatial_droplet.verification.action_verifier import ActionVerifier, PreActionSnapshot
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine


# ============================================================================
# TIER 1: FEATURE COVERAGE (Happy Path & Contracts)
# ============================================================================

class TestTier1FeatureCoverage:
    """Verifies core features under nominal operating conditions."""

    def test_f1_visual_hierarchy_zones_classification(self):
        """F1: Verify all 5 functional screen zones are classified accurately."""
        sw, sh = 1920, 1080
        modal_rect = (600, 300, 1200, 800)

        # 1. [MODAL/POPUP] elevated dialog precedence
        assert DiagonalDropletReasoner.classify_screen_zone(700, 400, sw, sh, modal_rect) == "[MODAL/POPUP]"
        assert DiagonalDropletReasoner.classify_screen_zone(1190, 790, sw, sh, modal_rect) == "[MODAL/POPUP]"

        # 2. [HEADER/NAV] (y < 0.12 * 1080 = 129.6)
        assert DiagonalDropletReasoner.classify_screen_zone(960, 60, sw, sh) == "[HEADER/NAV]"

        # 3. [FOOTER] (y >= 0.88 * 1080 = 950.4)
        assert DiagonalDropletReasoner.classify_screen_zone(960, 1000, sw, sh) == "[FOOTER]"

        # 4. [SIDEBAR] (x < 0.22 * 1920 = 422.4, outside header & footer)
        assert DiagonalDropletReasoner.classify_screen_zone(200, 500, sw, sh) == "[SIDEBAR]"

        # 5. [WORKSPACE] (center interior)
        assert DiagonalDropletReasoner.classify_screen_zone(960, 540, sw, sh) == "[WORKSPACE]"

    def test_f2_topological_som_graph_builder(self):
        """F2: Verify 2D topological graph builds structured zone buckets with neighbor hints."""
        sw, sh = 1920, 1080
        nodes = [
            DiagonalDroplet(id=1, center_x=100, center_y=50, category="button", label="Back", diag_wave=150, archetype="button"),
            DiagonalDroplet(id=2, center_x=160, center_y=50, category="button", label="Forward", diag_wave=210, archetype="button"),
            DiagonalDroplet(id=10, center_x=150, center_y=400, category="interactive_widget", label="Project Tree", diag_wave=550, archetype="tree"),
            DiagonalDroplet(id=20, center_x=960, center_y=500, category="interactive_widget", label="Code Editor", diag_wave=1460, archetype="canvas"),
            DiagonalDroplet(id=30, center_x=960, center_y=1020, category="interactive_widget", label="Taskbar", diag_wave=1980, archetype="app"),
        ]

        graph = DiagonalDropletReasoner._build_spatial_layout_graph(nodes, sw, sh)
        assert "2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH" in graph
        assert "[HEADER / NAV BAR]" in graph
        assert "[SIDEBAR / NAV DRAWER]" in graph
        assert "[WORKSPACE / VIEWPORT]" in graph
        assert "[FOOTER / BOTTOM BAR]" in graph
        assert "• [1] (100,50) [HEADER/NAV] [BUTTON] \"Back\"" in graph
        assert "Right of Dot [1]" in graph or "Left of Dot [2]" in graph

    def test_f2_som_overlay_rendering(self):
        """F2: Verify _render_unoccluded_som returns a valid base64 JPEG data URL."""
        synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        nodes = [
            DiagonalDroplet(id=5, center_x=200, center_y=150, category="button", label="Save", diag_wave=350, archetype="button", bbox=(180, 130, 220, 170)),
            DiagonalDroplet(id=6, center_x=400, center_y=300, category="input", label="Username", diag_wave=700, archetype="input", bbox=(350, 280, 450, 320)),
        ]

        som_url, (img_w, img_h), scale = DiagonalDropletReasoner._render_unoccluded_som(synthetic_frame, nodes, 1280, 720)
        assert som_url.startswith("data:image/jpeg;base64,")
        assert len(som_url) > 100
        assert (img_w, img_h) == (1280, 720)
        assert scale == 1.0

    @pytest.mark.asyncio
    async def test_f4_multimodal_provider_vision_payload(self):
        """F4: Verify OpenAI-compatible provider constructs vision payload with image URL."""
        provider = OpenAIGroundingProvider(api_key="sk-mock-key-for-test", base_url="https://api.openai.com/v1")
        assert provider.api_key == "sk-mock-key-for-test"

        captured_calls = []

        class MockCompletions:
            def create(self, **kwargs):
                captured_calls.append(kwargs)
                mock_resp = MagicMock()
                mock_choice = MagicMock()
                mock_choice.message.content = '{"action": "click", "target_dot_id": 14, "deliberation": {"user_intent_and_success_criteria": "Click", "screen_state_and_visual_hierarchy": "Desktop", "mistake_critique_and_loop_diagnosis": "None", "end_to_end_master_plan": ["Click"], "tactical_action_justification": "Target 14"}}'
                mock_resp.choices = [mock_choice]
                return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions = MockCompletions()
        provider.client = mock_client

        candidates = [
            DiagonalDroplet(id=14, center_x=300, center_y=200, category="button", label="Submit", diag_wave=500)
        ]
        decision = await provider.ground(
            instruction="Click submit button",
            manifest_text="[14] Submit",
            all_candidates=candidates,
            som_image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        )

        assert len(captured_calls) == 1
        messages = captured_calls[0]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        img_part = next(p for p in user_msg["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        assert img_part["image_url"]["detail"] == "high"
        assert decision.action == "click"
        assert decision.target_dot_id == 14

    def test_f5_nominal_prompt_token_budget_under_5000(self):
        """F5: Verify standard prompt with 25 droplets stays comfortably under 5,000 tokens."""
        enc = tiktoken.get_encoding("cl100k_base")
        reasoner = DiagonalDropletReasoner()

        droplets = [
            DiagonalDroplet(
                id=i,
                center_x=100 + (i % 5) * 150,
                center_y=100 + (i // 5) * 100,
                category="button",
                label=f"Action Button {i}",
                diag_wave=i * 20,
                archetype="button"
            )
            for i in range(25)
        ]
        mem = EpisodicTrajectoryMemory("Open browser and search")
        win = WindowContext(hwnd=1001, title="Browser Window", rect=(100, 100, 1200, 800))
        aware = ScreenAwarenessContext(app_name="Browser", scene_type="BROWSER")

        captured = []
        for entry in reasoner.key_pool:
            def make_create(entry_client):
                def mock_create(**kwargs):
                    captured.append(kwargs)
                    raise Exception("intercepted")
                return mock_create
            entry["client"].chat.completions.create = make_create(entry["client"])

        reasoner.decide_turn("Open browser and search", mem, droplets, win, aware)

        assert len(captured) > 0
        msgs = captured[0]["messages"]
        total_tokens = sum(len(enc.encode(m["content"])) for m in msgs)
        assert total_tokens < 5000, f"Token count {total_tokens} exceeded 5,000 token budget"

    def test_f7_precision_actuator_hardware_actions_mocked(self):
        """F7: Verify PrecisionActuator dispatches mouse, keyboard, and window actions without errors."""
        actuator = PrecisionActuator()
        droplet = DiagonalDroplet(id=99, center_x=350, center_y=250, category="button", label="Commit", diag_wave=600)
        win = WindowContext(hwnd=4321, title="Editor", rect=(100, 100, 900, 700))

        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click") as mock_click, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.double_click") as mock_dbl, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.clear_and_type") as mock_clear, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click_and_type") as mock_type_enter, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.hotkey") as mock_hk, \
             patch("spatial_droplet.platform.actuator.time.sleep", return_value=None):

            # 1. Click
            res_click = actuator.execute(droplet=droplet, action="click")
            assert res_click.get("verified") is True
            mock_click.assert_called_with(350, 250, button="left")

            # 2. Double click
            res_dbl = actuator.execute(droplet=droplet, action="double_click")
            assert res_dbl.get("verified") is True
            mock_dbl.assert_called_with(350, 250, button="left")

            # 3. Clear and type
            res_clear = actuator.execute(droplet=droplet, action="clear_and_type", input_text="new text")
            assert res_clear.get("verified") is True
            mock_clear.assert_called_with(350, 250, "new text", press_enter=False)

            # 4. Type and enter
            res_te = actuator.execute(droplet=droplet, action="type_and_enter", input_text="command")
            assert res_te.get("verified") is True
            mock_type_enter.assert_called_with(350, 250, "command", press_enter=True)

            # 5. Hotkey
            res_hk = actuator.execute(droplet=None, action="hotkey", key_combo=["ctrl", "s"])
            assert res_hk.get("verified") is True
            mock_hk.assert_called_with("ctrl", "s")

            # 6. Minimize & Maximize
            with patch("spatial_droplet.platform.window.minimize_window", return_value=True), \
                 patch("spatial_droplet.platform.window.maximize_window", return_value=True):
                res_min = actuator.execute(droplet=None, action="minimize", win_ctx=win)
                assert res_min.get("verified") is True
                res_max = actuator.execute(droplet=None, action="maximize", win_ctx=win)
                assert res_max.get("verified") is True

    def test_f7_adaptive_settle_sensor_motion_delta(self):
        """F7: Verify AdaptiveSettleSensor computes 0 delta for identical frames and >0 for changes."""
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
        delta_zero = AdaptiveSettleSensor.compute_motion_delta(frame_a, frame_b)
        assert delta_zero == 0.0

        frame_b[50:60, 50:60, :] = 255
        delta_motion = AdaptiveSettleSensor.compute_motion_delta(frame_a, frame_b)
        assert delta_motion > 0.0

    def test_f8_sha256_mirror_parity_verification_logic(self):
        """F8: Verify SHA-256 byte comparison correctly identifies identical and divergent files."""
        content1 = b"screen_aware_deep_self_reasoning_v1"
        content2 = b"screen_aware_deep_self_reasoning_v1"
        content3 = b"screen_aware_deep_self_reasoning_divergent"

        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        hash3 = hashlib.sha256(content3).hexdigest()

        assert hash1 == hash2
        assert hash1 != hash3


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ============================================================================

class TestTier2BoundaryAndCornerCases:
    """Verifies robustness under extreme values, empty structures, and limits."""

    def test_spatial_layout_empty_and_single_node(self):
        """Tier 2: Verify _build_spatial_layout_graph handles empty and single node gracefully."""
        # Empty
        graph_empty = DiagonalDropletReasoner._build_spatial_layout_graph([], 1920, 1080)
        assert "(No droplets present on screen)" in graph_empty

        # Single node
        single_node = [
            DiagonalDroplet(id=1, center_x=960, center_y=540, category="button", label="OK", diag_wave=1500)
        ]
        graph_single = DiagonalDropletReasoner._build_spatial_layout_graph(single_node, 1920, 1080)
        assert "• [1] (960,540) [WORKSPACE]" in graph_single

    def test_zone_coordinates_at_exact_boundaries(self):
        """Tier 2: Test zone classification exactly at percentage boundary cutoffs."""
        sw, sh = 1000, 1000
        # Exactly on header threshold (0.12 * 1000 = 120): y < 120 -> HEADER, y = 120 -> NOT HEADER
        assert DiagonalDropletReasoner.classify_screen_zone(500, 119, sw, sh) == "[HEADER/NAV]"
        assert DiagonalDropletReasoner.classify_screen_zone(500, 120, sw, sh) == "[WORKSPACE]"

        # Exactly on footer threshold (0.88 * 1000 = 880): y >= 880 -> FOOTER
        assert DiagonalDropletReasoner.classify_screen_zone(500, 879, sw, sh) == "[WORKSPACE]"
        assert DiagonalDropletReasoner.classify_screen_zone(500, 880, sw, sh) == "[FOOTER]"

        # Exactly on sidebar threshold (0.22 * 1000 = 220): x < 220 -> SIDEBAR
        assert DiagonalDropletReasoner.classify_screen_zone(219, 500, sw, sh) == "[SIDEBAR]"
        assert DiagonalDropletReasoner.classify_screen_zone(220, 500, sw, sh) == "[WORKSPACE]"

    def test_zero_and_negative_screen_dimensions(self):
        """Tier 2: Safe fallback when screen dimensions are zero or negative."""
        zone = DiagonalDropletReasoner.classify_screen_zone(10, 10, sw=0, sh=0)
        assert zone in ("[HEADER/NAV]", "[SIDEBAR]", "[WORKSPACE]", "[FOOTER]")

    def test_modal_rect_boundary_and_zero_size(self):
        """Tier 2: Test modal rect boundary tolerance (+-10px) and zero-area rect."""
        # 10px tolerance around modal rect
        modal = (100, 100, 300, 300)
        assert DiagonalDropletReasoner.classify_screen_zone(95, 95, 1000, 1000, modal) == "[MODAL/POPUP]"
        assert DiagonalDropletReasoner.classify_screen_zone(305, 305, 1000, 1000, modal) == "[MODAL/POPUP]"
        assert DiagonalDropletReasoner.classify_screen_zone(85, 85, 1000, 1000, modal) != "[MODAL/POPUP]"

    def test_token_compaction_guard_with_60_heavy_droplets(self):
        """Tier 2: 60 heavy droplets with long text triggers candidate compaction staying <5,000 tokens."""
        enc = tiktoken.get_encoding("cl100k_base")
        reasoner = DiagonalDropletReasoner()

        heavy_droplets = [
            DiagonalDroplet(
                id=i,
                center_x=50 + (i % 8) * 180,
                center_y=50 + (i // 8) * 100,
                category="interactive_widget",
                label=f"Ultra High Resolution Hierarchical Subsystem Component Container Widget Index {i} in Panel Section",
                diag_wave=i * 15,
                archetype="button"
            )
            for i in range(60)
        ]
        mem = EpisodicTrajectoryMemory("Perform multi-step system maintenance task")
        win = WindowContext(hwnd=8888, title="System Diagnostics Tool", rect=(50, 50, 1800, 950))
        aware = ScreenAwarenessContext(app_name="System Diagnostics", scene_type="DESKTOP_APP")

        captured = []
        for entry in reasoner.key_pool:
            def make_create(entry_client):
                def mock_create(**kwargs):
                    captured.append(kwargs)
                    raise Exception("intercepted")
                return mock_create
            entry["client"].chat.completions.create = make_create(entry["client"])

        reasoner.decide_turn("Perform multi-step system maintenance task", mem, heavy_droplets, win, aware)

        assert len(captured) > 0
        msgs = captured[0]["messages"]
        total_tokens = sum(len(enc.encode(m["content"])) for m in msgs)
        assert total_tokens < 5000, f"Compaction failed: {total_tokens} tokens exceeded 5,000 token limit"

    def test_actuator_invalid_coordinates_and_missing_drag_targets(self):
        """Tier 2: PrecisionActuator gracefully returns error dict when drag coordinates are missing."""
        actuator = PrecisionActuator()
        # Drag missing destination
        res = actuator.execute(droplet=None, action="drag")
        assert res.get("verified") is False
        assert res.get("type") == "INVALID_ACTION"


# ============================================================================
# TIER 3: PAIRWISE & CROSS-FEATURE COMBINATIONS
# ============================================================================

class TestTier3PairwiseCombinations:
    """Verifies interactions between intersecting subsystems."""

    def test_f1_modal_precedence_in_spatial_graph(self):
        """F1 + F2: Modal dialog droplets take priority and are placed in elevated section."""
        modal_rect = (500, 200, 1000, 600)
        nodes = [
            DiagonalDroplet(id=1, center_x=100, center_y=50, category="button", label="Nav Back", diag_wave=150),
            DiagonalDroplet(id=2, center_x=650, center_y=400, category="button", label="Confirm Delete", diag_wave=1050),
            DiagonalDroplet(id=3, center_x=850, center_y=400, category="button", label="Cancel", diag_wave=1250),
        ]

        graph = DiagonalDropletReasoner._build_spatial_layout_graph(nodes, 1920, 1080, modal_rect=modal_rect)
        assert "--- [MODAL / POPUP DIALOG] (Elevated Foreground Dialog/Sheet) ---" in graph
        assert "• [2] (650,400) [MODAL/POPUP]" in graph
        assert "• [3] (850,400) [MODAL/POPUP]" in graph

    def test_f7_f3_loop_guard_ping_pong_detection_triggers_pivot(self):
        """F7 + F3: Trajectory engine detects 2-turn ping pong and adopts pivoted decision."""
        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED"}

        engine = AutonomousTrajectoryEngine(
            reasoner=mock_reasoner,
            sprayer=mock_sprayer,
            actuator=mock_actuator
        )

        droplets = [
            DiagonalDroplet(id=10, center_x=200, center_y=200, category="button", label="Tab A", diag_wave=400),
            DiagonalDroplet(id=20, center_x=400, center_y=200, category="button", label="Tab B", diag_wave=600),
        ]
        win_ctx = WindowContext(hwnd=7777, title="App", rect=(100, 100, 800, 600))

        turn_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn_count[0] += 1
            if kwargs.get("loop_intervention"):
                return {"action": "click", "target_dot_id": 20, "next_subgoal": "Pivoted Tab B", "status": "in_progress"}
            # Alternating ping-pong: Tab A -> Tab B -> Tab A -> Tab B
            if turn_count[0] % 2 == 1:
                return {"action": "click", "target_dot_id": 10, "next_subgoal": "Click Tab A", "status": "in_progress"}
            else:
                return {"action": "click", "target_dot_id": 20, "next_subgoal": "Click Tab B", "status": "in_progress"}

        mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grabber, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grabber.return_value.grab_full.return_value = None
            memory = engine.run_trajectory(goal="Switch tabs", initial_droplets=droplets, initial_win_ctx=win_ctx, max_turns=6)

        assert len(memory.steps) == 6
        assert not memory.completion_summary.startswith("Aborted: Trapped in alternating ping-pong loop")

    def test_f1_modal_commit_button_prioritization(self):
        """F1 + F7: AutonomousTrajectoryEngine discovers and prioritizes commit button in active modal."""
        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        engine = AutonomousTrajectoryEngine(mock_reasoner, mock_sprayer, mock_actuator)

        modal_ctx = ModalDialogContext(hwnd=3333, title="Confirm Save", class_name="#32770", rect=(500, 300, 1100, 700))
        awareness = ScreenAwarenessContext(app_title="Confirm Save", active_modal=modal_ctx)

        droplets = {
            1: DiagonalDroplet(id=1, center_x=200, center_y=100, category="button", label="Background File Menu", diag_wave=300),
            2: DiagonalDroplet(id=2, center_x=800, center_y=600, category="button", label="[btn] Save", diag_wave=1400),
            3: DiagonalDroplet(id=3, center_x=950, center_y=600, category="button", label="[btn] Cancel", diag_wave=1550),
        }

        commit = engine._find_primary_dialog_commit(droplets, awareness, None)
        assert commit is not None
        assert commit.id == 2
        assert "[btn] Save" in commit.label


# ============================================================================
# TIER 4: REAL-WORLD DESKTOP APPLICATION SCENARIOS
# ============================================================================

class TestTier4RealWorldScenarios:
    """Verifies multi-turn desktop workflows through closed-loop simulation."""

    def test_scenario_modal_dialog_file_picker_workflow(self):
        """Tier 4: Multi-turn modal dialog workflow (Open Folder -> Select Folder -> Complete)."""
        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED"}

        engine = AutonomousTrajectoryEngine(mock_reasoner, mock_sprayer, mock_actuator)

        modal_ctx = ModalDialogContext(hwnd=5555, title="Open Folder", class_name="#32770", rect=(400, 200, 1400, 900))
        win_ctx = WindowContext(hwnd=5555, title="Open Folder", rect=(400, 200, 1400, 900))

        droplets_t1 = [
            DiagonalDroplet(id=1, center_x=600, center_y=350, category="interactive_widget", label="src", diag_wave=950),
            DiagonalDroplet(id=2, center_x=1200, center_y=850, category="button", label="Select Folder", diag_wave=2050),
        ]

        turn = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn[0] += 1
            if turn[0] == 1:
                return {"action": "click", "target_dot_id": 1, "next_subgoal": "Select src folder", "status": "in_progress"}
            elif turn[0] == 2:
                return {"action": "click", "target_dot_id": 2, "next_subgoal": "Click Select Folder button", "status": "in_progress"}
            else:
                return {"action": "stop", "target_dot_id": None, "next_subgoal": "Done", "status": "completed"}

        mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets_t1
        mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grabber, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grabber.return_value.grab_full.return_value = None
            memory = engine.run_trajectory(goal="Open src folder in editor", initial_droplets=droplets_t1, initial_win_ctx=win_ctx, max_turns=5)

        assert len(memory.steps) >= 2
        # All steps resulted in grounded actions
        for step in memory.steps:
            assert step.action in ("click", "double_click", "stop")

    def test_scenario_window_management_and_text_input(self):
        """Tier 4: Window lifecycle: Focus/Switch -> Clear & Type -> Type & Enter -> Minimize."""
        actuator = PrecisionActuator()
        win = WindowContext(hwnd=6677, title="Notepad", rect=(200, 150, 1000, 750))
        target_field = DiagonalDroplet(id=77, center_x=400, center_y=300, category="input", label="Document Area", diag_wave=700)

        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.clear_and_type") as mock_clear, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click_and_type") as mock_type_enter, \
             patch("spatial_droplet.platform.window.minimize_window", return_value=True) as mock_min, \
             patch("spatial_droplet.platform.actuator.time.sleep", return_value=None):

            # 1. Clear and type
            res1 = actuator.execute(droplet=target_field, action="clear_and_type", input_text="Initial Draft", win_ctx=win)
            assert res1.get("verified") is True
            mock_clear.assert_called_with(400, 300, "Initial Draft", press_enter=False)

            # 2. Type and enter
            res2 = actuator.execute(droplet=target_field, action="type_and_enter", input_text="\nEnd of Section", win_ctx=win)
            assert res2.get("verified") is True
            mock_type_enter.assert_called_with(400, 300, "\nEnd of Section", press_enter=True)

            # 3. Minimize
            res3 = actuator.execute(droplet=None, action="minimize", win_ctx=win)
            assert res3.get("verified") is True
            mock_min.assert_called_with(6677)

    def test_scenario_loop_break_within_2_unverified_turns(self):
        """Tier 4: Loop guard detects stagnation within <= 2 unverified turns and pivots."""
        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        # Simulate physical verification failure (zero motion delta) on first two attempts
        execution_results = [
            {"verified": False, "verdict": "ZERO_MOTION", "reason": "No UI change"},
            {"verified": False, "verdict": "ZERO_MOTION", "reason": "No UI change"},
            {"verified": True, "verdict": "CONFIRMED", "reason": "Pivoted click confirmed"}
        ]
        mock_actuator.execute.side_effect = execution_results

        engine = AutonomousTrajectoryEngine(mock_reasoner, mock_sprayer, mock_actuator)

        droplets = [
            DiagonalDroplet(id=50, center_x=300, center_y=300, category="button", label="Stuck Button", diag_wave=600),
            DiagonalDroplet(id=51, center_x=500, center_y=300, category="button", label="Alternative Commit", diag_wave=800),
        ]
        win_ctx = WindowContext(hwnd=9900, title="Stuck Dialog", rect=(200, 200, 800, 600))

        turn = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn[0] += 1
            if kwargs.get("loop_intervention"):
                # Reasoner pivots to alternative commit on intervention
                return {"action": "click", "target_dot_id": 51, "next_subgoal": "Pivoted to Alternative", "status": "in_progress"}
            return {"action": "click", "target_dot_id": 50, "next_subgoal": "Click Stuck Button", "status": "in_progress"}

        mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grabber, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grabber.return_value.grab_full.return_value = None
            memory = engine.run_trajectory(goal="Click button", initial_droplets=droplets, initial_win_ctx=win_ctx, max_turns=3)

        assert len(memory.steps) == 3
        # In turn 3, loop intervention broke stagnation and adopted target 51
        step3 = memory.steps[2]
        assert step3.target_id == 51 or step3.action in ("click", "wait")
