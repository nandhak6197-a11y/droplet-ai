"""
Adversarial Stress Test Suite: Milestone 3 Closed-Loop Settle Sensor & Loop Guard.
Role: challenger_m3_2 (Settle & Loop Guard Adversarial Challenger)

Adversarially tests:
1. Strict <= 2 turn loop guard:
   - Ping-pong alternating patterns (A -> B -> A) triggering at Turn 3
   - Direct repetition of unverified action (A [fail] -> A) triggering at Turn 2
   - Target stagnation (dot 4 [fail] -> dot 4) triggering within <= 2 turns
   - Consecutive unverified turns triggering guard
   - Repeated ungrounded actions (wait, switch_window) triggering guard
2. Smart Tactical Escape behavior & prevention of infinite loops:
   - Primary dialog commit selection ([btn] Select Folder)
   - Commit priorities & spatial score ranking in dialogs
   - Focus cycling via Tab key outside dialogs
   - Safety cap guarantee: No infinite loops when all tactics fail
   - Immediate cancellation via abort_event
3. Degenerate frames & strobe animations in cv2.absdiff settle sensor:
   - None frames, mismatched dimensions in compute_motion_delta
   - 0x0 degenerate frame behavior and NaN/empty slice resilience
   - None frame handling in wait_for_settle (initial and mid-stream)
   - Rapid strobe animation (black/white oscillation) timeout detection
   - Strobe stabilization: detection of stability as soon as strobe halts
4. Desktop window transitions & None droplet handling:
   - Transition from App window to Desktop (pre.title != Desktop -> post_title == Desktop)
   - Desktop to Desktop (clean rejection of false title shift)
   - Desktop to App window transition
   - Async polling minimize-to-desktop detection
   - None droplet handling in snapshot_pre_action, verify_action, suggest_pivot, is_blacklisted
"""
import sys
import os
import time
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spatial_droplet.core.models import (
    DiagonalDroplet, WindowContext, TrajectoryStep,
    ModalDialogContext, ScreenAwarenessContext
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.vision.settle_sensor import AdaptiveSettleSensor
from spatial_droplet.verification.action_verifier import ActionVerifier, PreActionSnapshot
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine


# ============================================================================
# 1. STRICT <= 2 TURN LOOP GUARD ADVERSARIAL CHALLENGES
# ============================================================================

class TestAdversarialLoopGuard(unittest.TestCase):
    """Adversarial challenge for Strict <= 2 turn Loop Guard invariants."""

    def setUp(self):
        self.mock_reasoner = MagicMock()
        self.mock_sprayer = MagicMock()
        self.mock_actuator = MagicMock()
        self.mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED"}
        self.engine = AutonomousTrajectoryEngine(
            reasoner=self.mock_reasoner,
            sprayer=self.mock_sprayer,
            actuator=self.mock_actuator
        )

    def test_ping_pong_triggers_turn_3_unverified(self):
        """Adversarial: Sequence [A (failed), B (failed)] followed by A triggers loop guard on Turn 3."""
        droplets = [
            DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Search", diag_wave=200),
            DiagonalDroplet(id=2, center_x=200, center_y=200, category="interactive_widget", label="Filter", diag_wave=400)
        ]
        win_ctx = WindowContext(hwnd=100, title="Data Explorer", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "No delta"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append((call_count[0], kwargs["loop_intervention"]))
                return {"action": "press", "key": "tab", "status": "in_progress"}

            if call_count[0] == 1:
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}
            elif call_count[0] == 2:
                return {"action": "click", "target_dot_id": 2, "status": "in_progress"}
            else:
                # Turn 3: oscillates back to A
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Ping pong unverified test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertEqual(len(interventions), 1)
        self.assertIn("Alternating ping-pong oscillation detected", interventions[0][1])

    def test_ping_pong_triggers_turn_3_verified(self):
        """Adversarial: Even when actions were verified, ping-pong oscillation [A, B] -> A triggers loop guard on Turn 3."""
        droplets = [
            DiagonalDroplet(id=10, center_x=100, center_y=100, category="interactive_widget", label="Tab 1", diag_wave=200),
            DiagonalDroplet(id=20, center_x=200, center_y=200, category="interactive_widget", label="Tab 2", diag_wave=400)
        ]
        win_ctx = WindowContext(hwnd=100, title="Multi-Tab App", rect=(0, 0, 800, 600))
        # Actuator returns verified = True for all steps
        self.mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append((call_count[0], kwargs["loop_intervention"]))
                return {"action": "press", "key": "tab", "status": "in_progress"}

            if call_count[0] == 1:
                return {"action": "click", "target_dot_id": 10, "status": "in_progress"}
            elif call_count[0] == 2:
                return {"action": "click", "target_dot_id": 20, "status": "in_progress"}
            else:
                return {"action": "click", "target_dot_id": 10, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Ping pong verified test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertEqual(len(interventions), 1)
        self.assertIn("Alternating ping-pong oscillation detected", interventions[0][1])

    def test_repeated_unverified_action_triggers_turn_2(self):
        """Adversarial: Repeating an unverified action triggers loop guard immediately on Turn 2."""
        droplets = [DiagonalDroplet(id=7, center_x=300, center_y=300, category="interactive_widget", label="Run Query", diag_wave=600)]
        win_ctx = WindowContext(hwnd=100, title="Query Tool", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "No response"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append((call_count[0], kwargs["loop_intervention"]))
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "click", "target_dot_id": 7, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Direct repetition test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertEqual(len(interventions), 1)
        self.assertIn("was repeated after failing verification", interventions[0][1])

    def test_repeated_unverified_action_with_different_dot_id_same_label(self):
        """Adversarial: When droplet IDs re-index between sprays (id 1 -> id 99) but label is identical, guard still triggers on Turn 2."""
        turn_droplets = {
            1: [DiagonalDroplet(id=1, center_x=300, center_y=300, category="interactive_widget", label="Save All", diag_wave=600)],
            2: [DiagonalDroplet(id=99, center_x=300, center_y=300, category="interactive_widget", label="Save All", diag_wave=600)]
        }
        win_ctx = WindowContext(hwnd=100, title="Code Editor", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append((call_count[0], kwargs["loop_intervention"]))
                return {"action": "press", "key": "esc", "status": "in_progress"}

            if call_count[0] == 1:
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}
            else:
                return {"action": "click", "target_dot_id": 99, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return turn_droplets.get(call_count[0] + 1, turn_droplets[1])
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Reindexed dot id repetition test",
                initial_droplets=turn_droplets[1],
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertEqual(len(interventions), 1)
        self.assertIn("was repeated after failing verification", interventions[0][1])

    def test_target_stagnation_triggers_turn_2(self):
        """Adversarial: Target stagnation on Dot 4 (failed on Turn 1, targeted again on Turn 2) triggers loop guard on Turn 2."""
        droplets = [DiagonalDroplet(id=4, center_x=400, center_y=400, category="interactive_widget", label="Submit Order", diag_wave=800)]
        win_ctx = WindowContext(hwnd=100, title="Shopping Cart", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "click", "target_dot_id": 4, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Dot 4 stagnation test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertEqual(len(interventions), 1)
        self.assertTrue(any("Submit Order" in i or "was repeated" in i for i in interventions))

    def test_target_stagnation_across_different_actions_triggers_turn_3(self):
        """Adversarial: When different actions target the same failing Dot 4, after 2 consecutive failures it triggers guard on Turn 3."""
        droplets = [DiagonalDroplet(id=4, center_x=400, center_y=400, category="interactive_widget", label="Unresponsive Element", diag_wave=800)]
        win_ctx = WindowContext(hwnd=100, title="Form", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT"}

        interventions = []
        call_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}

            # Turn 1: click, Turn 2: double_click, Turn 3: right_click
            if call_count[0] == 1:
                return {"action": "click", "target_dot_id": 4, "status": "in_progress"}
            elif call_count[0] == 2:
                return {"action": "double_click", "target_dot_id": 4, "status": "in_progress"}
            else:
                return {"action": "right_click", "target_dot_id": 4, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Dot 4 different actions stagnation test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertEqual(len(interventions), 1)
        self.assertTrue(
            "has failed verification" in interventions[0] or
            "2 consecutive turns failed" in interventions[0]
        )

    def test_repeated_ungrounded_wait_triggers_guard_on_turn_2(self):
        """Adversarial: Two consecutive ungrounded 'wait' actions trigger loop guard on Turn 2."""
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        interventions = []
        call_count = [0]

        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "wait", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return []
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Repeated wait test",
                initial_droplets=[],
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertEqual(len(interventions), 1)
        self.assertTrue(
            "Consecutive or ungrounded 'wait'" in interventions[0] or
            "was repeated after failing verification" in interventions[0]
        )

    def test_repeated_ungrounded_switch_window_triggers_guard(self):
        """Adversarial: Repeated ungrounded switch_window actions trigger loop guard on Turn 2."""
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        interventions = []
        call_count = [0]

        def mock_decide(goal, memory, *args, **kwargs):
            call_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "switch_window", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return []
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Repeated switch window test",
                initial_droplets=[],
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertEqual(len(interventions), 1)
        self.assertTrue(
            "Repeated ungrounded 'switch_window'" in interventions[0] or
            "was repeated after failing verification" in interventions[0]
        )


# ============================================================================
# 2. SMART TACTICAL ESCAPE & INFINITE LOOP PREVENTION
# ============================================================================

class TestAdversarialSmartTacticalEscape(unittest.TestCase):
    """Adversarial challenge for Smart Tactical Escape and finite execution guarantees."""

    def setUp(self):
        self.mock_reasoner = MagicMock()
        self.mock_sprayer = MagicMock()
        self.mock_actuator = MagicMock()
        self.engine = AutonomousTrajectoryEngine(
            reasoner=self.mock_reasoner,
            sprayer=self.mock_sprayer,
            actuator=self.mock_actuator
        )

    def test_smart_escape_commit_priorities_in_dialog(self):
        """Adversarial: In a file dialog with multiple candidates, Smart Escape prefers 'Select Folder' over 'Cancel'."""
        droplets = {
            1: DiagonalDroplet(id=1, center_x=300, center_y=300, category="interactive_widget", label="docs folder item", diag_wave=600),
            2: DiagonalDroplet(id=2, center_x=1050, center_y=750, category="interactive_widget", label="Cancel", diag_wave=1800),
            3: DiagonalDroplet(id=3, center_x=1150, center_y=750, category="interactive_widget", label="[btn] Select Folder", diag_wave=1900)
        }
        win_ctx = WindowContext(hwnd=555, title="Select Directory Dialog", rect=(200, 100, 1300, 800))
        modal_ctx = ModalDialogContext(hwnd=555, title="Select Directory Dialog", class_name="#32770", rect=(200, 100, 1300, 800))
        screen_awareness = ScreenAwarenessContext(
            scene_type="desktop_app",
            app_title="Select Directory Dialog",
            active_modal=modal_ctx,
            modal_context=modal_ctx
        )

        commit = AutonomousTrajectoryEngine._find_primary_dialog_commit(droplets, screen_awareness, win_ctx)
        self.assertIsNotNone(commit)
        self.assertEqual(commit.id, 3)
        self.assertEqual(commit.label, "[btn] Select Folder")

    def test_infinite_loop_prevention_at_max_turns(self):
        """Adversarial: When both model and Smart Tactical Escape fail persistently, trajectory terminates cleanly at max_turns."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Rigid Btn", diag_wave=200)]
        win_ctx = WindowContext(hwnd=100, title="Rigid App", rect=(0, 0, 800, 600))

        # Actuator fails every single turn
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "Hardware locked"}

        # Reasoner stubbornly repeats identical failing action
        self.mock_reasoner.decide_turn.return_value = {"action": "click", "target_dot_id": 1, "status": "in_progress"}

        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        start_t = time.time()
        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            # Set small max_turns to verify finite loop termination
            memory = self.engine.run_trajectory(
                goal="Infinite loop prevention test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=5
            )

        elapsed = time.time() - start_t
        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(memory.steps), 5)
        self.assertFalse(memory.is_completed)
        self.assertIn("Reached max safety turn cap of 5 turns", memory.completion_summary)

    def test_abort_event_immediate_interruption(self):
        """Adversarial: Setting abort_event immediately halts trajectory execution."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Btn", diag_wave=200)]
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        abort_ev = threading.Event()
        abort_ev.set()  # Immediately pre-aborted

        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        memory = self.engine.run_trajectory(
            goal="Abort event test",
            initial_droplets=droplets,
            initial_win_ctx=win_ctx,
            max_turns=10,
            abort_event=abort_ev
        )

        self.assertEqual(len(memory.steps), 0)
        self.assertIn("Cancelled by user via ESC", memory.completion_summary)


# ============================================================================
# 3. CV2.ABSDIFF SETTLE SENSOR DEGENERATE FRAMES & STROBE ANIMATIONS
# ============================================================================

class TestAdversarialSettleSensorDegenerateAndStrobe(unittest.TestCase):
    """Adversarial challenge for cv2.absdiff settle sensor edge cases and strobe handling."""

    def test_compute_motion_delta_none_and_mismatched(self):
        """Adversarial: compute_motion_delta safely handles None and dimension mismatches by returning 999.0."""
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        other_frame = np.zeros((300, 300, 3), dtype=np.uint8)

        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(None, None), 999.0)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(valid_frame, None), 999.0)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(None, valid_frame), 999.0)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(valid_frame, other_frame), 999.0)

    def test_compute_motion_delta_zero_dimensions_does_not_crash(self):
        """Adversarial: compute_motion_delta on 0x0 empty numpy arrays executes without unhandled exception."""
        empty_frame_a = np.zeros((0, 0, 3), dtype=np.uint8)
        empty_frame_b = np.zeros((0, 0, 3), dtype=np.uint8)

        # Must not raise an unhandled exception
        delta = AdaptiveSettleSensor.compute_motion_delta(empty_frame_a, empty_frame_b)
        self.assertIsNotNone(delta)

    def test_wait_for_settle_grabber_returns_none_initially(self):
        """Adversarial: wait_for_settle handles grabber returning None on first call."""
        grabber = MagicMock(return_value=None)
        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=grabber,
            timeout_sec=0.10
        )
        self.assertFalse(is_settled)
        self.assertIsNone(final_f)

    def test_wait_for_settle_grabber_returns_none_midway(self):
        """Adversarial: wait_for_settle handles grabber returning None after initial valid frame."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        call_count = [0]
        def grabber():
            call_count[0] += 1
            if call_count[0] == 1:
                return frame
            return None

        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=grabber,
            timeout_sec=0.08,
            check_interval=0.01
        )
        self.assertFalse(is_settled)
        self.assertIsNotNone(final_f)

    def test_wait_for_settle_rapid_strobe_animation_times_out(self):
        """Adversarial: Rapidly strobing alternating frames (black / white) prevents false settlement and times out cleanly."""
        frame_black = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_white = np.ones((200, 200, 3), dtype=np.uint8) * 255
        toggle = [False]

        def strobe_grabber():
            toggle[0] = not toggle[0]
            return frame_white if toggle[0] else frame_black

        start_t = time.time()
        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=strobe_grabber,
            timeout_sec=0.10,
            check_interval=0.01,
            motion_threshold=0.75
        )
        real_elapsed = time.time() - start_t

        self.assertFalse(is_settled, "Strobing screen must NOT be falsely marked as settled!")
        self.assertGreaterEqual(real_elapsed, 0.08)
        self.assertIsNotNone(final_f)

    def test_wait_for_settle_strobe_settles_after_animation_halts(self):
        """Adversarial: Screen strobes for 3 frames, then becomes completely static -> sensor detects stabilization."""
        frame_black = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_white = np.ones((200, 200, 3), dtype=np.uint8) * 255
        frame_stable = np.ones((200, 200, 3), dtype=np.uint8) * 128
        call_idx = [0]

        def dynamic_grabber():
            call_idx[0] += 1
            if call_idx[0] == 1:
                return frame_black
            elif call_idx[0] == 2:
                return frame_white
            elif call_idx[0] == 3:
                return frame_black
            else:
                # Stabilizes from frame 4 onwards
                return frame_stable

        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=dynamic_grabber,
            timeout_sec=0.30,
            check_interval=0.01,
            motion_threshold=0.75
        )

        self.assertTrue(is_settled, "Sensor must detect stabilization when animation ceases!")
        self.assertIsNotNone(final_f)
        self.assertTrue(np.array_equal(final_f, frame_stable))


# ============================================================================
# 4. DESKTOP WINDOW TRANSITIONS & NONE DROPLET HANDLING
# ============================================================================

class TestAdversarialWindowTransitionsAndNoneDroplet(unittest.TestCase):
    """Adversarial challenge for Desktop transitions and None droplet robustness."""

    def setUp(self):
        self.verifier = ActionVerifier(sw=1920, sh=1080)
        self.roi = np.zeros((120, 120, 3), dtype=np.uint8)

    def test_window_transition_app_to_desktop_verified(self):
        """Adversarial: Transition from App to Desktop (pre.title != Desktop -> post_title == Desktop) is confirmed."""
        pre = PreActionSnapshot(hwnd=100, title="Notepad - Untitled", cursor_shape="ARROW", roi=self.roi, cx=200, cy=200)
        post_ctx = WindowContext(hwnd=100, title="Desktop", rect=(0, 0, 1920, 1080))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, droplet=None, action="click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "WINDOW_CHANGE")
            self.assertIn("Desktop", res["reason"])

    def test_window_transition_desktop_to_desktop_not_verified(self):
        """Adversarial: Desktop to Desktop is NOT falsely verified as a window shift."""
        pre = PreActionSnapshot(hwnd=10, title="Desktop", cursor_shape="ARROW", roi=self.roi, cx=200, cy=200)
        post_ctx = WindowContext(hwnd=10, title="Desktop", rect=(0, 0, 1920, 1080))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch.object(self.verifier, "capture_full_frame", return_value=None), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, droplet=None, action="click")
            self.assertFalse(res["verified"])
            self.assertEqual(res["type"], "NO_CHANGE")

    def test_window_transition_desktop_to_app_verified(self):
        """Adversarial: Desktop to App window launch (pre.title == Desktop -> post_title == Calculator) is confirmed."""
        pre = PreActionSnapshot(hwnd=10, title="Desktop", cursor_shape="ARROW", roi=self.roi, cx=200, cy=200)
        post_ctx = WindowContext(hwnd=200, title="Calculator", rect=(100, 100, 600, 700))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, droplet=None, action="double_click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "WINDOW_CHANGE")

    def test_none_droplet_pre_action_snapshot_without_target_coords(self):
        """Adversarial: snapshot_pre_action with droplet=None and no target coords defaults safely to screen center."""
        snap = self.verifier.snapshot_pre_action(droplet=None, win_ctx=None, target_x=None, target_y=None)
        self.assertEqual(snap.cx, 1920 // 2)
        self.assertEqual(snap.cy, 1080 // 2)
        self.assertEqual(snap.title, "Desktop")

    def test_none_droplet_verify_action_zero_delta_no_attribute_error(self):
        """Adversarial: verify_action with droplet=None on zero delta fails cleanly without raising AttributeError."""
        pre = PreActionSnapshot(hwnd=100, title="Terminal", cursor_shape="ARROW", roi=self.roi, cx=500, cy=500)
        post_ctx = WindowContext(hwnd=100, title="Terminal", rect=(0, 0, 800, 600))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch.object(self.verifier, "capture_full_frame", return_value=None), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, droplet=None, action="type_and_enter")
            self.assertFalse(res["verified"])
            self.assertEqual(res["type"], "NO_CHANGE")
            self.assertIsNone(res["blacklisted_id"])

    def test_none_droplet_suggest_pivot_with_candidates(self):
        """Adversarial: suggest_pivot with failed_droplet=None ranks candidates from screen center safely."""
        cands = [
            DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="TopLeft", diag_wave=200),
            DiagonalDroplet(id=2, center_x=960, center_y=540, category="interactive_widget", label="Center", diag_wave=1500)
        ]
        pivot = self.verifier.suggest_pivot(failed_droplet=None, candidates=cands, action="click")
        self.assertEqual(pivot["pivot_type"], "ALTERNATIVE_CANDIDATE")
        # Center candidate (id 2) should be closer to screen center (960, 540)
        self.assertEqual(pivot["target_dot_id"], 2)

    def test_none_droplet_suggest_pivot_without_candidates(self):
        """Adversarial: suggest_pivot with failed_droplet=None and empty candidates escalates action safely."""
        pivot = self.verifier.suggest_pivot(failed_droplet=None, candidates=[], action="click")
        self.assertEqual(pivot["pivot_type"], "ACTION_ESCALATION")
        self.assertEqual(pivot["suggested_action"], "double_click")
        self.assertIsNone(pivot["target_dot_id"])

    def test_is_blacklisted_none_returns_false(self):
        """Adversarial: is_blacklisted(None) returns False without error."""
        self.assertFalse(self.verifier.is_blacklisted(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
