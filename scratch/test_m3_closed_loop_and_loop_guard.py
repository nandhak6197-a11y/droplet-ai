"""
Comprehensive Milestone 3 Verification Suite: Closed-Loop Settle & Loop Guard.
Verifies:
1. Closed-loop visual settle verification via cv2.absdiff motion delta (<0.5ms)
2. Window title and HWND shift verification (including Desktop transitions)
3. Cursor transformation validation for hover vs rejection for click
4. Loop Guard strict <= 2 turn detection (ping-pong at Turn 3, repetition at Turn 2, target failure >= 2)
5. Automated in-turn cognitive auto-pivot adoption
6. Generalized Smart Tactical Escape (dialog primary commit click, tab focus cycling)
"""
import sys
import os
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
# TIER 1: CLOSED-LOOP VISUAL SETTLE & CV2.ABSDIFF MOTION DELTA
# ============================================================================

class TestTier1VisualSettleAndDelta(unittest.TestCase):
    """Verifies cv2.absdiff motion delta and AdaptiveSettleSensor convergence."""

    def test_cv2_absdiff_motion_delta_identical_frames(self):
        """Tier 1: Identical frames produce zero motion delta."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 120
        delta = AdaptiveSettleSensor.compute_motion_delta(frame, frame.copy())
        self.assertEqual(delta, 0.0)

    def test_cv2_absdiff_motion_delta_different_frames(self):
        """Tier 1: Frames with motion produce accurate positive delta."""
        frame_a = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_b = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_b[100:200, 100:200] = 255  # Motion box
        delta = AdaptiveSettleSensor.compute_motion_delta(frame_a, frame_b)
        self.assertGreater(delta, 0.5)

    def test_adaptive_settle_sensor_wait_for_settle_converges(self):
        """Tier 1: wait_for_settle stabilizes and returns is_settled = True."""
        static_frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        grabber = MagicMock(return_value=static_frame)
        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=grabber,
            timeout_sec=0.20,
            motion_threshold=0.75
        )
        self.assertTrue(is_settled)
        self.assertIsNotNone(final_f)

    def test_action_verifier_pixel_delta_pass(self):
        """Tier 1: Local ROI delta > 15 pixels produces verified = True (PIXEL_DELTA)."""
        verifier = ActionVerifier(sw=1920, sh=1080)
        droplet = DiagonalDroplet(id=1, center_x=300, center_y=300, category="interactive_widget", label="Submit", diag_wave=600)
        pre_roi = np.zeros((120, 120, 3), dtype=np.uint8)
        post_roi = np.zeros((120, 120, 3), dtype=np.uint8)
        post_roi[20:60, 20:60] = 200  # 1600 pixels changed > 20

        pre = PreActionSnapshot(hwnd=100, title="App", cursor_shape="ARROW", roi=pre_roi, cx=300, cy=300)

        with patch.object(verifier, "capture_roi", return_value=post_roi), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=WindowContext(hwnd=100, title="App")), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = verifier.verify_action(pre, droplet, "click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "PIXEL_DELTA")
            self.assertGreater(res["delta"], 15)

    def test_action_verifier_zero_delta_fails_and_blacklists(self):
        """Tier 1: Identical pre/post ROI produces verified = False (NO_CHANGE) and blacklists target."""
        verifier = ActionVerifier(sw=1920, sh=1080)
        droplet = DiagonalDroplet(id=42, center_x=300, center_y=300, category="interactive_widget", label="Stuck Btn", diag_wave=600)
        roi = np.zeros((120, 120, 3), dtype=np.uint8)
        pre = PreActionSnapshot(hwnd=100, title="App", cursor_shape="ARROW", roi=roi, cx=300, cy=300)

        with patch.object(verifier, "capture_roi", return_value=roi), \
             patch.object(verifier, "capture_full_frame", return_value=None), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=WindowContext(hwnd=100, title="App")), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = verifier.verify_action(pre, droplet, "click")
            self.assertFalse(res["verified"])
            self.assertEqual(res["type"], "NO_CHANGE")
            self.assertTrue(verifier.is_blacklisted(42))
            self.assertIsNotNone(res.get("suggested_pivot"))

    def test_action_verifier_none_droplet_safe_handling(self):
        """Tier 1: verify_action and snapshot_pre_action handle droplet = None without raising exceptions."""
        verifier = ActionVerifier(sw=1920, sh=1080)
        win_ctx = WindowContext(hwnd=50, title="Terminal", rect=(0, 0, 800, 600))

        with patch.object(verifier, "capture_roi", return_value=np.zeros((120, 120, 3), dtype=np.uint8)), \
             patch.object(verifier, "capture_full_frame", return_value=None), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            snap = verifier.snapshot_pre_action(droplet=None, win_ctx=win_ctx, target_x=400, target_y=300)
            self.assertEqual(snap.cx, 400)
            self.assertEqual(snap.cy, 300)
            res = verifier.verify_action(snap, droplet=None, action="type_and_enter")
            self.assertFalse(res["verified"])
            self.assertEqual(res["type"], "NO_CHANGE")


# ============================================================================
# TIER 2: SEMANTIC WINDOW TITLE & STATE TRANSITIONS
# ============================================================================

class TestTier2SemanticWindowVerification(unittest.TestCase):
    """Verifies window title and HWND shift tracking across desktop and app states."""

    def setUp(self):
        self.verifier = ActionVerifier(sw=1920, sh=1080)
        self.droplet = DiagonalDroplet(id=10, center_x=100, center_y=100, category="interactive_widget", label="Save", diag_wave=200)
        self.roi = np.zeros((120, 120, 3), dtype=np.uint8)

    def test_window_title_change_verified(self):
        """Tier 2: Window title transition (e.g. Document ➜ Document (Saved)) is verified."""
        pre = PreActionSnapshot(hwnd=200, title="Document1 - Editor", cursor_shape="ARROW", roi=self.roi, cx=100, cy=100)
        post_ctx = WindowContext(hwnd=200, title="Document1 (Saved) - Editor", rect=(0, 0, 1000, 700))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, self.droplet, "click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "WINDOW_CHANGE")
            self.assertIn("Document1 (Saved)", res["reason"])

    def test_window_close_to_desktop_verified(self):
        """Tier 2: Closing an application window returning to Desktop is confirmed as a valid transition."""
        pre = PreActionSnapshot(hwnd=300, title="Calculator", cursor_shape="ARROW", roi=self.roi, cx=100, cy=100)
        post_ctx = WindowContext(hwnd=10, title="Desktop", rect=(0, 0, 1920, 1080))

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=post_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"):
            res = self.verifier.verify_action(pre, self.droplet, "click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "WINDOW_CHANGE")

    def test_cursor_change_hover_pass_click_fail(self):
        """Tier 2: Cursor shape change verifies hover, but strictly rejects zero-delta click."""
        pre = PreActionSnapshot(hwnd=400, title="Browser", cursor_shape="ARROW", roi=self.roi, cx=100, cy=100)

        with patch.object(self.verifier, "capture_roi", return_value=self.roi), \
             patch.object(self.verifier, "capture_full_frame", return_value=None), \
             patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=WindowContext(hwnd=400, title="Browser")), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="HAND"):
            # Hover action with cursor change -> PASS
            hover_res = self.verifier.verify_action(pre, self.droplet, "hover")
            self.assertTrue(hover_res["verified"])
            self.assertEqual(hover_res["type"], "CURSOR_CHANGE")

            # Click action with cursor change but zero visual change -> FAIL
            click_res = self.verifier.verify_action(pre, self.droplet, "click")
            self.assertFalse(click_res["verified"])
            self.assertEqual(click_res["type"], "NO_CHANGE")


# ============================================================================
# TIER 3: LOOP GUARD STRICT <= 2 TURN DETECTION
# ============================================================================

class TestTier3LoopGuardStrictDetection(unittest.TestCase):
    """Verifies that loop guard detects stagnation and ping-pong patterns within <= 2 consecutive turns."""

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

    def test_ping_pong_detected_at_turn_3(self):
        """Tier 3: Sequence [A, B] followed by A triggers loop guard on Turn 3 (<= 2 cycles)."""
        droplets = [
            DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Item A", diag_wave=200),
            DiagonalDroplet(id=2, center_x=200, center_y=200, category="interactive_widget", label="Item B", diag_wave=400)
        ]
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        intervention_received = []

        call_idx = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_idx[0] += 1
            if kwargs.get("loop_intervention"):
                intervention_received.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}

            # Ping-pong sequence: Turn 1: A, Turn 2: B, Turn 3: A
            if call_idx[0] == 1:
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}
            elif call_idx[0] == 2:
                return {"action": "click", "target_dot_id": 2, "status": "in_progress"}
            else:
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
                goal="Ping pong test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertTrue(len(intervention_received) >= 1)
        self.assertIn("Alternating ping-pong oscillation detected", intervention_received[0])

    def test_direct_repetition_detected_at_turn_2(self):
        """Tier 3: Repeating an unverified action triggers loop guard immediately on Turn 2."""
        droplets = [DiagonalDroplet(id=5, center_x=100, center_y=100, category="interactive_widget", label="Broken Btn", diag_wave=200)]
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        intervention_received = []

        # Make turn 1 fail verification
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "No delta"}

        def mock_decide(goal, memory, *args, **kwargs):
            if kwargs.get("loop_intervention"):
                intervention_received.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "click", "target_dot_id": 5, "status": "in_progress"}

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
                goal="Repetition test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertTrue(len(intervention_received) >= 1)
        self.assertIn("was repeated after failing verification", intervention_received[0])

    def test_consecutive_unverified_turns_triggers_guard(self):
        """Tier 3: 2 consecutive unverified turns trigger loop guard on Turn 3."""
        droplets = [
            DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Btn 1", diag_wave=200),
            DiagonalDroplet(id=2, center_x=200, center_y=200, category="interactive_widget", label="Btn 2", diag_wave=400),
            DiagonalDroplet(id=3, center_x=300, center_y=300, category="interactive_widget", label="Btn 3", diag_wave=600),
        ]
        win_ctx = WindowContext(hwnd=100, title="App", rect=(0, 0, 800, 600))
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT"}

        interventions = []
        call_idx = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            call_idx[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "click", "target_dot_id": call_idx[0], "status": "in_progress"}

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
                goal="Consecutive failure test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertTrue(len(interventions) >= 1)
        self.assertIn("2 consecutive turns failed to produce verified UI transitions", interventions[0])


# ============================================================================
# TIER 4: AUTOMATED PIVOT & SMART TACTICAL ESCAPE
# ============================================================================

class TestTier4PivotAndTacticalEscape(unittest.TestCase):
    """Verifies heuristic smart tactical escape when model deliberation fails."""

    def setUp(self):
        self.mock_reasoner = MagicMock()
        self.mock_sprayer = MagicMock()
        self.mock_actuator = MagicMock()
        self.engine = AutonomousTrajectoryEngine(
            reasoner=self.mock_reasoner,
            sprayer=self.mock_sprayer,
            actuator=self.mock_actuator
        )

    def test_smart_tactical_escape_auto_clicks_dialog_commit(self):
        """Tier 4: When reasoner fails to break loop in modal dialog, engine auto-clicks commit button."""
        droplets = [
            DiagonalDroplet(id=1, center_x=300, center_y=300, category="interactive_widget", label="stuck item", diag_wave=600),
            DiagonalDroplet(id=99, center_x=1150, center_y=750, category="interactive_widget", label="[btn] Select Folder", diag_wave=1900)
        ]
        modal_ctx = ModalDialogContext(hwnd=888, title="Open Folder", class_name="#32770", rect=(200, 100, 1300, 800))
        win_ctx = WindowContext(hwnd=888, title="Open Folder", rect=(200, 100, 1300, 800))
        executed = []

        def mock_exec(*args, **kwargs):
            executed.append((kwargs.get("action"), kwargs.get("droplet")))
            return {"verified": True, "verdict": "CONFIRMED"}
        self.mock_actuator.execute.side_effect = mock_exec

        # Model stubbornly returns identical failing action
        self.mock_reasoner.decide_turn.return_value = {"action": "double_click", "target_dot_id": 1, "status": "in_progress"}
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Open folder test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        # Smart escape clicked Select Folder (id 99)
        clicked_droplet_ids = [d.id for act, d in executed if d is not None]
        self.assertIn(99, clicked_droplet_ids)

    def test_smart_tactical_escape_tab_cycles_focus_outside_dialog(self):
        """Tier 4: When reasoner fails to break loop outside dialog, engine dispatches 'tab' key."""
        droplets = [DiagonalDroplet(id=1, center_x=500, center_y=500, category="interactive_widget", label="Trapped Field", diag_wave=1000)]
        win_ctx = WindowContext(hwnd=999, title="Web Form", rect=(0, 0, 1920, 1080))
        executed_keys = []

        def mock_exec(*args, **kwargs):
            if kwargs.get("key"):
                executed_keys.append(kwargs.get("key"))
            return {"verified": True, "verdict": "CONFIRMED"}
        self.mock_actuator.execute.side_effect = mock_exec

        # Model stubbornly returns same failing action
        self.mock_reasoner.decide_turn.return_value = {"action": "click", "target_dot_id": 1, "status": "in_progress"}
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Form test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertIn("tab", executed_keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
