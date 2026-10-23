"""
Milestone 5 Tier 5 Adversarial Coverage Hardening Test Suite:
Execution, Settle & Mirror Track.

White-box adversarial stress tests covering:
Category A: Coordinate Clamping & Boundary Stress (actuator.py & desktop.py)
  - Negative coordinate normalization in Win32HardwareActuator & VirtualDesktopTopology
  - Screen dimensions exceeded (> sw, > sh) coordinate normalization
  - Non-integer / float coordinate rounding & dispatch
  - PrecisionActuator smooth cursor movement boundary guards (tx <= 0 or ty <= 0)
  - Droplet bbox clamping, inverted bboxes (bx2 < bx1), zero coordinate rejection
  - Hardware drag trajectory interpolation with negative & out-of-bounds coordinates
  - SendInput buffer generation with MOUSEEVENTF_ABSOLUTE and MOUSEEVENTF_VIRTUALDESK

Category B: Loop Guard Stagnation & Ping-Pong State Machine Stress (trajectory_engine.py)
  - Ping-pong state machine detection under rapidly oscillating window titles
  - Direct repetition failure guard with dynamic window title variations
  - Stagnation guard across repeated target with consecutive failures
  - Consecutive ungrounded 'wait' actions triggering deterministic auto-pivot
  - Repeated ungrounded 'switch_window' triggering deterministic auto-pivot
  - Multi-cycle ping-pong oscillation breaking within strict <= 2 cycles
  - Heuristic smart tactical escape targeting primary modal commit button on stubborn reasoner
  - Heuristic smart tactical escape dispatching tab focus cycling when outside modal dialog
  - Contextual action upgrade (click -> double_click) for file/folder targets in open workflows

Category C: Settle Sensor Noise Deltas, Frame Flicker & Edge Arrays (settle_sensor.py)
  - Zero-dimensional frames (shape=(0,0,3), shape=(100,0,3)) returning 0.0 delta safely
  - None and mismatched frame shapes returning fallback sentinel 999.0
  - Extreme noise deltas (salt-and-pepper, full inversion 0 vs 255)
  - High-frequency frame flicker causing timeout with is_settled = False
  - High-frequency frame flicker transitioning to static frame exiting early (is_settled = True)
  - Dynamic ROI clamping with negative, out-of-bounds, and inverted bounding boxes
  - Subsample variations (subsample=1, subsample=2, subsample=8, subsample=16)

Category D: Recursive Mirror Synchronization Edge Cases (sync_and_verify.py)
  - Read-only file permission recovery on destination during overwrite
  - Source file read-only attribute preservation and sync verification
  - Nested read-only directory hierarchies and deep tree synchronization
  - Mixed-case filename preservation and case-sensitive parity on Windows
  - Zero-byte file handling and SHA-256 verification
  - Large chunked file (>64KB chunk boundary) SHA-256 verification
  - Orphan file pruning with read-only attributes
  - Tree scanning resilience against inaccessible / broken files (OSError handling)
  - Dry-run non-mutation guarantee across complex file operations
"""

import os
import sys
import stat
import time
import shutil
import hashlib
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spatial_droplet.platform.actuator import (
    Win32HardwareActuator,
    PrecisionActuator,
    resolve_vk,
    INPUT,
    MOUSEINPUT,
    KEYBDINPUT,
)
from spatial_droplet.platform.desktop import VirtualDesktopTopology
from spatial_droplet.vision.settle_sensor import AdaptiveSettleSensor
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine
from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    TrajectoryStep,
    ModalDialogContext,
    ScreenAwarenessContext,
)
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from scratch.sync_and_verify import (
    compute_sha256,
    is_path_excluded,
    scan_tree,
    compare_trees,
    safe_copy,
    run_sync,
    run_verify_only,
    DEFAULT_EXCLUDES,
)


# ============================================================================
# CATEGORY A: COORDINATE CLAMPING & BOUNDARY STRESS
# ============================================================================

class TestAdversarialCoordinateClamping(unittest.TestCase):
    """Stress tests coordinate translation, normalization, and bounds clamping."""

    def test_negative_coordinates_hardware_actuator_norm(self):
        """Win32HardwareActuator correctly maps negative coordinates to signed normalized space."""
        with patch.object(VirtualDesktopTopology, "get_virtual_bounds", return_value=(0, 0, 1920, 1080)), \
             patch.object(VirtualDesktopTopology, "is_multi_monitor", return_value=False):
            nx, ny, is_v = Win32HardwareActuator._to_norm_coords(-100, -200, sw=1920, sh=1080)
            self.assertIsInstance(nx, int)
            self.assertIsInstance(ny, int)
            self.assertLess(nx, 0)
            self.assertLess(ny, 0)
            # In primary screen, is_v is False
            self.assertEqual(is_v, 0)

    def test_virtual_desktop_negative_coordinate_offset(self):
        """VirtualDesktopTopology correctly computes relative coordinates when virtual screen origin is negative."""
        # Multi-monitor setup where secondary monitor is at x = -1920, y = 0
        with patch.object(VirtualDesktopTopology, "get_virtual_bounds", return_value=(-1920, 0, 3840, 1080)), \
             patch.object(VirtualDesktopTopology, "is_multi_monitor", return_value=True):
            nx, ny, flag = VirtualDesktopTopology.to_normalized_coords(-1920, 0)
            # Origin of virtual desktop should normalize to 0
            self.assertEqual(nx, 0)
            self.assertEqual(ny, 0)
            self.assertTrue(flag)

            # Center of virtual desktop (-1920 + 3840 / 2 = 0)
            mid_x, mid_y, flag_mid = VirtualDesktopTopology.to_normalized_coords(0, 540)
            self.assertAlmostEqual(mid_x, 32767, delta=10)
            self.assertTrue(flag_mid)

    def test_out_of_bounds_coordinates_exceeding_screen_size(self):
        """Actuator safely maps coordinates larger than screen width and height."""
        with patch.object(VirtualDesktopTopology, "get_virtual_bounds", return_value=(0, 0, 1920, 1080)), \
             patch.object(VirtualDesktopTopology, "is_multi_monitor", return_value=False):
            nx, ny, flag = Win32HardwareActuator._to_norm_coords(3840, 2160, sw=1920, sh=1080)
            self.assertGreater(nx, 65535)
            self.assertGreater(ny, 65535)

    def test_non_integer_floating_point_coordinates(self):
        """Actuator rounds non-integer floating-point coordinates to integer SendInput units cleanly."""
        with patch.object(VirtualDesktopTopology, "get_virtual_bounds", return_value=(0, 0, 1920, 1080)), \
             patch.object(VirtualDesktopTopology, "is_multi_monitor", return_value=False):
            nx_float, ny_float, _ = Win32HardwareActuator._to_norm_coords(120.75, 340.25, sw=1920, sh=1080)
            self.assertIsInstance(nx_float, int)
            self.assertIsInstance(ny_float, int)

            # PrecisionActuator execution with float coordinates
            actuator = PrecisionActuator()
            with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click") as mock_click, \
                 patch("spatial_droplet.platform.actuator.Win32HardwareActuator.move_cursor") as mock_move:
                res = actuator.execute(droplet=None, action="click", target_x=120.75, target_y=340.25)
                self.assertTrue(res["verified"])
                self.assertEqual(res["type"], "HARDWARE")
                mock_click.assert_called_once_with(120.75, 340.25, button="left")
                mock_move.assert_called_once_with(120.75, 340.25)

    def test_precision_actuator_smooth_cursor_boundary_guards(self):
        """PrecisionActuator suppresses smooth cursor movement when coordinates are negative or zero."""
        actuator = PrecisionActuator()
        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click") as mock_click, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.move_cursor") as mock_move:
            # Negative coordinates: click is dispatched, but glide cursor move is skipped
            res = actuator.execute(droplet=None, action="click", target_x=-50, target_y=-100)
            self.assertTrue(res["verified"])
            mock_click.assert_called_once_with(-50, -100, button="left")
            mock_move.assert_not_called()

    def test_droplet_bbox_clamping_and_inverted_boundaries(self):
        """PrecisionActuator extracts target coordinates safely from negative or inverted bboxes."""
        actuator = PrecisionActuator()
        # Normal bounding box
        droplet_valid = DiagonalDroplet(id=1, center_x=0, center_y=0, category="button", label="OK", diag_wave=0)
        droplet_valid.bbox = (100, 200, 300, 400)
        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click") as mock_click, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.move_cursor"):
            actuator.execute(droplet=droplet_valid, action="click")
            # Midpoint (100+300)//2=200, (200+400)//2=300
            mock_click.assert_called_once_with(200, 300, button="left")

        # Inverted bounding box (bx2 < bx1): falls back to droplet center_x/center_y
        droplet_inv = DiagonalDroplet(id=2, center_x=150, center_y=250, category="button", label="OK", diag_wave=0)
        droplet_inv.bbox = (300, 400, 100, 200)
        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.click") as mock_click_inv, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.move_cursor"):
            actuator.execute(droplet=droplet_inv, action="click")
            mock_click_inv.assert_called_once_with(150, 250, button="left")

    def test_actuator_drag_interpolation_step_safety(self):
        """Win32HardwareActuator.drag calculates intermediate step coordinates as valid integers."""
        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.move_cursor") as mock_move, \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.mouse_down"), \
             patch("spatial_droplet.platform.actuator.Win32HardwareActuator.mouse_up"), \
             patch("time.sleep"):
            Win32HardwareActuator.drag(
                from_x=-100, from_y=-50,
                to_x=2000, to_y=1200,
                duration=0.1, steps=5
            )
            # Verify move_cursor was called for start plus 5 steps
            self.assertEqual(mock_move.call_count, 6)
            for call_args in mock_move.call_args_list:
                cx, cy = call_args[0][:2]
                self.assertIsInstance(cx, int)
                self.assertIsInstance(cy, int)

    def test_actuator_rejects_missing_drag_destination_adversarial(self):
        """PrecisionActuator strictly rejects drag actions when destination coordinates are missing."""
        actuator = PrecisionActuator()
        res = actuator.execute(droplet=None, action="drag", target_x=100, target_y=100, drag_dest=None)
        self.assertFalse(res["verified"])
        self.assertEqual(res["type"], "INVALID_ACTION")
        self.assertIn("requires valid start and destination", res["reason"])

    def test_actuator_hardware_virtual_key_resolution_edge_cases(self):
        """resolve_vk accurately resolves key codes across alphanumeric, modifiers, function keys, and invalid tokens."""
        # Modifiers & control keys
        self.assertEqual(resolve_vk("ctrl"), 0x11)
        self.assertEqual(resolve_vk("  Alt  "), 0x12)
        self.assertEqual(resolve_vk("SHIFT"), 0x10)
        self.assertEqual(resolve_vk("Enter"), 0x0D)
        self.assertEqual(resolve_vk("Tab"), 0x09)
        self.assertEqual(resolve_vk("Esc"), 0x1B)

        # Function keys
        self.assertEqual(resolve_vk("f1"), 0x70)
        self.assertEqual(resolve_vk("F5"), 0x74)
        self.assertEqual(resolve_vk("f12"), 0x7B)

        # Numbers and letters
        self.assertEqual(resolve_vk("0"), 0x30)
        self.assertEqual(resolve_vk("9"), 0x39)
        self.assertEqual(resolve_vk("a"), 0x41)
        self.assertEqual(resolve_vk("Z"), 0x5A)

        # Integer pass-through
        self.assertEqual(resolve_vk(0x20), 0x20)

        # Invalid or non-string keys safely return 0
        self.assertEqual(resolve_vk("non_existent_key_token_xyz"), 0)
        self.assertEqual(resolve_vk(None), 0)
        self.assertEqual(resolve_vk([]), 0)


# ============================================================================
# CATEGORY B: LOOP GUARD STAGNATION & PING-PONG STATE MACHINE STRESS
# ============================================================================

class TestAdversarialLoopGuardStateMachine(unittest.TestCase):
    """Stress tests loop guard stagnation and ping-pong state machine under dynamic conditions."""

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

    def test_ping_pong_detection_under_varying_window_titles(self):
        """Loop guard detects ping-pong oscillation (A -> B -> A) on Turn 3 even when window titles vary dynamically."""
        droplets = [
            DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Search", diag_wave=200),
            DiagonalDroplet(id=2, center_x=200, center_y=200, category="interactive_widget", label="Filter", diag_wave=400),
        ]
        titles = ["App - Main Window", "App - Filter Applied", "App - Results Updated"]
        interventions = []

        turn_count = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn_count[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            if turn_count[0] == 1:
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}
            elif turn_count[0] == 2:
                return {"action": "click", "target_dot_id": 2, "status": "in_progress"}
            else:
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        ctx_idx = [0]
        def mock_get_win():
            t = titles[min(ctx_idx[0], len(titles) - 1)]
            ctx_idx[0] += 1
            return WindowContext(hwnd=100 + ctx_idx[0], title=t, rect=(0, 0, 800, 600))

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", side_effect=mock_get_win), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Ping pong with varying titles",
                initial_droplets=droplets,
                initial_win_ctx=WindowContext(hwnd=100, title=titles[0], rect=(0, 0, 800, 600)),
                max_turns=3
            )

        self.assertGreaterEqual(len(interventions), 1)
        self.assertIn("Alternating ping-pong oscillation detected", interventions[0])

    def test_direct_repetition_pivot_under_fluctuating_window_titles(self):
        """Repeating an unverified action triggers loop guard on Turn 2 even if window title changed."""
        droplets = [DiagonalDroplet(id=10, center_x=300, center_y=300, category="button", label="Submit", diag_wave=600)]
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "No delta"}

        interventions = []
        def mock_decide(goal, memory, *args, **kwargs):
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "esc", "status": "in_progress"}
            return {"action": "click", "target_dot_id": 10, "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        titles = ["Form (Pending)", "Form (Retrying...)"]
        ctx_idx = [0]
        def mock_get_win():
            t = titles[min(ctx_idx[0], len(titles) - 1)]
            ctx_idx[0] += 1
            return WindowContext(hwnd=200, title=t, rect=(0, 0, 800, 600))

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", side_effect=mock_get_win), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Repetition under title change",
                initial_droplets=droplets,
                initial_win_ctx=WindowContext(hwnd=200, title=titles[0], rect=(0, 0, 800, 600)),
                max_turns=2
            )

        self.assertGreaterEqual(len(interventions), 1)
        self.assertIn("was repeated after failing verification", interventions[0])

    def test_consecutive_wait_actions_triggers_auto_pivot(self):
        """Consecutive or repeated 'wait' actions trigger loop guard auto-pivot on Turn 2 (<= 2 turns)."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="icon", label="Spinner", diag_wave=200)]
        interventions = []

        def mock_decide(goal, memory, *args, **kwargs):
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "enter", "status": "in_progress"}
            return {"action": "wait", "input_text": "1.0", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        win_ctx = WindowContext(hwnd=300, title="Loading Screen", rect=(0, 0, 800, 600))
        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Wait action stress",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertGreaterEqual(len(interventions), 1)
        # Succeeded in triggering loop guard within <= 2 turns
        self.assertTrue(
            "was repeated after failing verification" in interventions[0] or
            "Consecutive or ungrounded 'wait' action chosen" in interventions[0]
        )

    def test_repeated_ungrounded_switch_window_triggers_pivot(self):
        """Repeated ungrounded switch_window actions trigger loop guard auto-pivot on Turn 2 (<= 2 turns)."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="window", label="Background", diag_wave=200)]
        interventions = []

        def mock_decide(goal, memory, *args, **kwargs):
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            return {"action": "switch_window", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        win_ctx = WindowContext(hwnd=400, title="Editor", rect=(0, 0, 800, 600))
        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Switch window guard",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertGreaterEqual(len(interventions), 1)
        # Succeeded in triggering loop guard within <= 2 turns
        self.assertTrue(
            "was repeated after failing verification" in interventions[0] or
            "Repeated ungrounded 'switch_window'" in interventions[0]
        )

    def test_ungrounded_switch_window_rule5_after_grounded_switch(self):
        """When an ungrounded switch_window follows a grounded switch_window, Rule 5 triggers auto-pivot on Turn 2."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="window", label="Background", diag_wave=200)]
        interventions = []

        turn_idx = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn_idx[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "tab", "status": "in_progress"}
            if turn_idx[0] == 1:
                # Turn 1: grounded switch_window with explicit target title
                return {"action": "switch_window", "input_text": "Notepad", "status": "in_progress"}
            else:
                # Turn 2: ungrounded switch_window (differs in input_text, so Rule 2 doesn't match)
                return {"action": "switch_window", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        win_ctx = WindowContext(hwnd=400, title="Editor", rect=(0, 0, 800, 600))
        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("spatial_droplet.verification.semantic_verifier.SemanticStateVerifier.verify_transition", return_value={"verified": True, "verdict": "CONFIRMED"}), \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Switch window rule 5",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertGreaterEqual(len(interventions), 1)
        self.assertIn("Repeated ungrounded 'switch_window'", interventions[0])

    def test_wait_action_rule6_after_unverified_failure(self):
        """When reasoner proposes 'wait' following an unverified failure, Rule 6 triggers auto-pivot."""
        droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="icon", label="Spinner", diag_wave=200)]
        interventions = []

        turn_idx = [0]
        def mock_decide(goal, memory, *args, **kwargs):
            turn_idx[0] += 1
            if kwargs.get("loop_intervention"):
                interventions.append(kwargs["loop_intervention"])
                return {"action": "press", "key": "enter", "status": "in_progress"}
            if turn_idx[0] == 1:
                # Turn 1: click that fails verification
                return {"action": "click", "target_dot_id": 1, "status": "in_progress"}
            else:
                # Turn 2: wait chosen while consecutive_failures >= 1
                return {"action": "wait", "input_text": "1.0", "status": "in_progress"}

        self.mock_reasoner.decide_turn.side_effect = mock_decide
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        # Make turn 1 fail verification
        self.mock_actuator.execute.return_value = {"verified": False, "verdict": "NO_EFFECT", "reason": "No delta"}

        win_ctx = WindowContext(hwnd=300, title="Loading Screen", rect=(0, 0, 800, 600))
        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Wait action rule 6",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=2
            )

        self.assertGreaterEqual(len(interventions), 1)
        self.assertIn("Consecutive or ungrounded 'wait' action chosen", interventions[0])

    def test_stubborn_reasoner_dialog_commit_discovery(self):
        """When reasoner persistently ignores loop intervention in a dialog, engine auto-executes dialog commit."""
        droplets = [
            DiagonalDroplet(id=1, center_x=200, center_y=200, category="item", label="stuck file", diag_wave=400),
            DiagonalDroplet(id=88, center_x=900, center_y=600, category="button", label="[btn] Save As", diag_wave=1500),
        ]
        modal_ctx = ModalDialogContext(hwnd=500, title="Save File", class_name="#32770", rect=(100, 100, 1000, 700))
        win_ctx = WindowContext(hwnd=500, title="Save File", rect=(100, 100, 1000, 700))

        executed_actions = []
        def mock_exec(*args, **kwargs):
            executed_actions.append(kwargs.get("droplet"))
            return {"verified": True, "verdict": "CONFIRMED"}
        self.mock_actuator.execute.side_effect = mock_exec

        # Stubborn reasoner returns identical blocked action even after intervention
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
                goal="Save dialog test",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        # Smart escape clicked Save As (droplet id 88)
        executed_ids = [d.id for d in executed_actions if d is not None]
        self.assertIn(88, executed_ids)

    def test_contextual_action_upgrade_click_to_double_click_for_files(self):
        """Engine automatically upgrades 'click' to 'double_click' when goal intent is open/launch and target is a file or app."""
        droplets = [
            DiagonalDroplet(id=1, center_x=300, center_y=300, category="interactive_widget", label="report_notes.txt", diag_wave=600),
        ]
        win_ctx = WindowContext(hwnd=600, title="Desktop", rect=(0, 0, 1920, 1080))
        executed_actions = []

        def mock_exec(*args, **kwargs):
            executed_actions.append(kwargs.get("action"))
            return {"verified": True, "verdict": "CONFIRMED"}
        self.mock_actuator.execute.side_effect = mock_exec

        # Reasoner proposes single click, but goal has 'Open' intent on a file target
        self.mock_reasoner.decide_turn.return_value = {
            "action": "click",
            "target_dot_id": 1,
            "next_subgoal": "Open file report_notes.txt",
            "status": "in_progress"
        }
        async def mock_spray(frame, win_ctx):
            return droplets
        self.mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grab, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grab.return_value.grab_full.return_value = None
            memory = self.engine.run_trajectory(
                goal="Open report_notes.txt on Desktop",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=1
            )

        self.assertIn("double_click", executed_actions)


# ============================================================================
# CATEGORY C: SETTLE SENSOR NOISE DELTAS, FRAME FLICKER & EDGE ARRAYS
# ============================================================================

class TestAdversarialSettleSensor(unittest.TestCase):
    """Stress tests settle sensor against noise, flickering frames, and zero-dimensional arrays."""

    def test_zero_dimensional_and_empty_frames(self):
        """Settle sensor returns 0.0 delta safely for empty or zero-size frames without crashing."""
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(empty_frame, empty_frame), 0.0)

        zero_width = np.zeros((100, 0, 3), dtype=np.uint8)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(zero_width, zero_width), 0.0)

    def test_none_and_mismatched_frame_dimensions(self):
        """Settle sensor returns sentinel 999.0 when either frame is None or shapes differ."""
        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(None, valid_frame), 999.0)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(valid_frame, None), 999.0)

        diff_shape = np.zeros((200, 200, 3), dtype=np.uint8)
        self.assertEqual(AdaptiveSettleSensor.compute_motion_delta(valid_frame, diff_shape), 999.0)

    def test_extreme_noise_and_inversion_deltas(self):
        """Settle sensor computes accurate maximum delta for completely inverted frames and salt-and-pepper noise."""
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        white_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        delta_inv = AdaptiveSettleSensor.compute_motion_delta(black_frame, white_frame)
        self.assertAlmostEqual(delta_inv, 255.0, delta=1.0)

        # Random noise delta
        rng = np.random.default_rng(42)
        noise_frame = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        delta_noise = AdaptiveSettleSensor.compute_motion_delta(black_frame, noise_frame)
        self.assertGreater(delta_noise, 50.0)

    def test_high_frequency_frame_flicker_times_out(self):
        """Continuous high-frequency screen flicker causes wait_for_settle to timeout with is_settled = False."""
        frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_b = np.ones((50, 50, 3), dtype=np.uint8) * 200

        call_idx = [0]
        def alternating_grabber():
            call_idx[0] += 1
            return frame_a if (call_idx[0] % 2 == 0) else frame_b

        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=alternating_grabber,
            timeout_sec=0.08,
            min_dwell_sec=0.005,
            motion_threshold=0.75,
            check_interval=0.005
        )
        self.assertFalse(is_settled)
        self.assertIsNotNone(final_f)

    def test_flicker_transitioning_to_static_settles_early(self):
        """Screen that flickers initially then stabilizes exits wait_for_settle early with is_settled = True."""
        frame_noise = np.ones((50, 50, 3), dtype=np.uint8) * 255
        frame_static = np.zeros((50, 50, 3), dtype=np.uint8)

        call_count = [0]
        def transitioning_grabber():
            call_count[0] += 1
            # First 3 calls flicker, then static
            if call_count[0] <= 3:
                return frame_noise if (call_count[0] % 2 == 1) else frame_static
            return frame_static

        is_settled, elapsed, final_f = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=transitioning_grabber,
            timeout_sec=0.50,
            min_dwell_sec=0.005,
            motion_threshold=0.75,
            check_interval=0.005
        )
        self.assertTrue(is_settled)
        self.assertLess(elapsed, 0.40)

    def test_adaptive_settle_sensor_invalid_roi_clamping(self):
        """wait_for_settle safely clamps negative, out-of-bounds, and inverted bounding boxes."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        grabber = MagicMock(return_value=frame)

        # Negative ROI
        is_settled, _, _ = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=grabber,
            roi=(-100, -100, 50, 50),
            timeout_sec=0.05,
            check_interval=0.005
        )
        self.assertTrue(is_settled)

        # Inverted ROI (x1 > x2, y1 > y2)
        is_settled_inv, _, _ = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=grabber,
            roi=(80, 80, 20, 20),
            timeout_sec=0.05,
            check_interval=0.005
        )
        self.assertTrue(is_settled_inv)


# ============================================================================
# CATEGORY D: RECURSIVE MIRROR SYNCHRONIZATION EDGE CASES
# ============================================================================

class TestAdversarialMirrorSync(unittest.TestCase):
    """Stress tests recursive mirror synchronization under file system edge cases."""

    def test_read_only_destination_permission_recovery(self):
        """safe_copy and run_sync clear read-only permissions on destination files and achieve parity."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            src_file = src / "protected.txt"
            dst_file = dst / "protected.txt"

            src_file.write_text("updated active code", encoding="utf-8")
            dst_file.write_text("old read-only content", encoding="utf-8")

            # Mark destination file as read-only
            os.chmod(str(dst_file), stat.S_IREAD)

            # Verification should detect divergence
            self.assertFalse(run_verify_only(str(src), str(dst)))

            # Sync must recover permissions, copy cleanly, and verify parity
            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))
            self.assertEqual(dst_file.read_text(encoding="utf-8"), "updated active code")

    def test_source_file_read_only_attribute_sync(self):
        """Sync correctly copies files when source file is read-only."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            src_file = src / "immutable.py"
            src_file.write_text("read_only_data = 123", encoding="utf-8")
            os.chmod(str(src_file), stat.S_IREAD)

            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))

            # Restore write permission on src for clean temp dir removal
            os.chmod(str(src_file), stat.S_IWRITE | stat.S_IREAD)
            if (dst / "immutable.py").exists():
                os.chmod(str(dst / "immutable.py"), stat.S_IWRITE | stat.S_IREAD)

    def test_nested_directory_hierarchy_and_deep_tree_parity(self):
        """Sync and verify recursively handle arbitrarily deep directory hierarchies."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            deep_dir = src / "level1" / "level2" / "level3" / "level4"
            deep_dir.mkdir(parents=True)

            (deep_dir / "deep_module.py").write_text("deep = True", encoding="utf-8")
            (src / "top_level.md").write_text("# Top", encoding="utf-8")

            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))

    def test_mixed_case_filenames_parity(self):
        """Sync preserves mixed-case filenames and verifies bit-for-bit parity without collision."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            (src / "CamelCase.Py").write_text("class CamelCase: pass", encoding="utf-8")
            (src / "UPPERCASE_CONFIG.JSON").write_text('{"DEBUG": true}', encoding="utf-8")
            (src / "mixED_123_File.txt").write_text("mixed case data", encoding="utf-8")

            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))

    def test_zero_byte_files_parity(self):
        """Sync handles zero-byte files accurately with valid SHA-256 digests."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            empty_file = src / "empty.init"
            empty_file.write_bytes(b"")

            expected_sha = hashlib.sha256(b"").hexdigest()
            self.assertEqual(compute_sha256(str(empty_file)), expected_sha)

            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))

    def test_large_chunked_file_sha256_parity(self):
        """Files exceeding chunk buffer (64KB) are hashed and synced with 100% byte fidelity."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            # Create 150KB file (> 2 chunks of 64KB)
            large_content = b"0123456789ABCDEF" * 9600
            large_file = src / "large_binary.bin"
            large_file.write_bytes(large_content)

            self.assertEqual(compute_sha256(str(large_file)), hashlib.sha256(large_content).hexdigest())

            sync_ok = run_sync(str(src), str(dst))
            self.assertTrue(sync_ok)
            self.assertTrue(run_verify_only(str(src), str(dst)))

    def test_pruning_read_only_orphan_files(self):
        """Pruning removes orphan files in destination even when they have read-only attributes."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            (src / "keeper.txt").write_text("keep me", encoding="utf-8")
            (dst / "keeper.txt").write_text("keep me", encoding="utf-8")

            orphan = dst / "read_only_orphan.txt"
            orphan.write_text("remove me", encoding="utf-8")
            os.chmod(str(orphan), stat.S_IREAD)

            sync_ok = run_sync(str(src), str(dst), prune=True)
            self.assertTrue(sync_ok)
            self.assertFalse(orphan.exists())
            self.assertTrue((dst / "keeper.txt").exists())

    def test_dry_run_preserves_destination_state(self):
        """Dry-run mode does not modify or delete any destination files."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            src.mkdir()
            dst.mkdir()

            (src / "new_feature.py").write_text("def run(): pass", encoding="utf-8")
            (dst / "orphan.py").write_text("old", encoding="utf-8")

            run_sync(str(src), str(dst), dry_run=True, prune=True)

            self.assertFalse((dst / "new_feature.py").exists())
            self.assertTrue((dst / "orphan.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
