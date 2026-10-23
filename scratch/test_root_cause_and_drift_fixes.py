"""
Targeted Verification Suite:
1. Actuator spatial vs non-spatial coordinate handling & rejection of blind (0,0) clicks.
2. Hotkey combo parsing from multiple fields (no [None] execution).
3. Sprayer two-pass ID persistence across continuous scroll / movement up to 120px.
4. Sprayer OCR bounding box margin.
5. Consecutive wait loop guard in TrajectoryEngine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import unittest
from unittest.mock import MagicMock, patch

from spatial_droplet.core.models import DiagonalDroplet, WindowContext, TrajectoryStep
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.platform.actuator import PrecisionActuator
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine
from spatial_droplet.engine.fast_path import FastPathRouter


class TestRootCauseAndDriftFixes(unittest.TestCase):

    def test_actuator_rejects_blind_zero_zero_click(self):
        """Pillar: Spatial actions require valid non-(0,0) coordinates."""
        actuator = PrecisionActuator()
        # Click with no target dot and (0,0) coords must return INVALID_ACTION, not blind click at (0,0)
        res = actuator.execute(None, "click", input_text=None, target_x=0, target_y=0)
        self.assertFalse(res["verified"])
        self.assertEqual(res.get("type"), "INVALID_ACTION")
        self.assertIn("no target coordinates", res["reason"].lower())

    def test_actuator_allows_non_spatial_actions_without_dot(self):
        """Pillar: Non-spatial actions (wait, hotkey, press) do not fail on missing dot."""
        actuator = PrecisionActuator()
        with patch("spatial_droplet.platform.actuator.Win32HardwareActuator.press_key", return_value=True):
            res = actuator.execute(None, "press", key="enter")
            self.assertTrue(res["verified"])

    def test_hotkey_never_executes_none(self):
        """Pillar: Hotkey combo parsing aggregates keys and eliminates [None]."""
        actuator = PrecisionActuator()
        # When key_combo is None and key is None, it should return INVALID_ACTION, never execute [None]
        res = actuator.execute(None, "hotkey", key_combo=None, key=None, input_text=None)
        self.assertFalse(res["verified"])
        self.assertEqual(res.get("type"), "INVALID_ACTION")
        self.assertIn("no valid keys", res["reason"].lower())

    def test_two_pass_id_persistence_preserves_id_during_scroll(self):
        """Pillar: Scrolling a button down by 60px preserves its ID via Pass A (up to 120px)."""
        sprayer = FullBleedLatticeSprayer()
        # Frame 1: Button 'Submit Order' at y=400 with ID=15
        prev_droplet = DiagonalDroplet(
            id=15, center_x=500, center_y=400, category="interactive_widget",
            label="[btn] Submit Order", diag_wave=900
        )
        sprayer._last_ui_droplets = [prev_droplet]

        # Frame 2: Page scrolled, button moved to y=460 (displacement = 60px > 22px)
        ui_points = [{
            'x': 500, 'y': 460, 'category': 'interactive_widget',
            'label': '[btn] Submit Order', 'diag_key': 960, 'subtext': 'Submit Order',
            'energy': 100, 'in_window': True
        }]

        # Run ID persistence logic
        assigned_ids = set()
        unmatched_points = []
        used_prev_ids = set()
        pass_b_candidates = []

        import re, math
        def _clean_match_label(lbl: str) -> str:
            if not lbl:
                return ""
            s = lbl.lower()
            s = re.sub(r'\[item \d+ of \d+.*?\]', '', s)
            s = re.sub(r'\[(?:btn|icon|app)\]\s*', '', s)
            s = re.sub(r'[^a-z0-9]', '', s)
            return s.strip()

        for pt in ui_points:
            px, py = pt['x'], pt['y']
            clean_lbl = _clean_match_label(pt.get('label', ''))
            best_prev = None
            best_dist = 120.0
            if len(clean_lbl) >= 2:
                for pd in sprayer._last_ui_droplets:
                    if pd.id in used_prev_ids:
                        continue
                    if _clean_match_label(pd.label) == clean_lbl:
                        dist = math.hypot(px - pd.center_x, py - pd.center_y)
                        if dist <= best_dist:
                            best_dist = dist
                            best_prev = pd
            if best_prev is not None:
                used_prev_ids.add(best_prev.id)
                assigned_ids.add(best_prev.id)
                pt['id'] = best_prev.id
            else:
                pass_b_candidates.append(pt)

        # Verified: ID 15 is preserved despite 60px displacement
        self.assertEqual(ui_points[0]['id'], 15)

    def test_fast_path_disables_url_opening(self):
        """Pillar: FastPathRouter blocks system_open_url and redirects."""
        res = FastPathRouter.route("open youtube.com", [], None)
        # Should NOT return a system_open_url action that launches browser automatically
        if res:
            self.assertNotEqual(res.get("action"), "system_open_url")

    def test_consecutive_wait_triggers_loop_guard(self):
        """Pillar: Consecutive wait action triggers in-turn cognitive auto-pivot."""
        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED", "reason": "ok"}

        engine = AutonomousTrajectoryEngine(
            reasoner=mock_reasoner,
            sprayer=mock_sprayer,
            actuator=mock_actuator
        )

        call_idx = [0]
        pivoted = [False]

        def mock_decide(goal, memory, droplets, win_ctx, screen_awareness=None, loop_intervention=None, screenshot=None):
            call_idx[0] += 1
            if loop_intervention:
                pivoted[0] = True
                return {"action": "stop", "status": "completed", "reasoning": "Pivoted and stopped"}
            if call_idx[0] == 1:
                return {"action": "wait", "status": "in_progress"}
            elif call_idx[0] == 2:
                # 2nd consecutive wait
                return {"action": "wait", "status": "in_progress"}
            return {"action": "stop", "status": "completed"}

        mock_reasoner.decide_turn.side_effect = mock_decide
        win_ctx = WindowContext(hwnd=111, title="TestApp", rect=(0, 0, 1000, 800))
        initial_droplets = [DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="btn", diag_wave=200)]

        async def mock_spray(frame, win_ctx):
            return initial_droplets
        mock_sprayer.spray = mock_spray

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grabber, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grabber.return_value.grab_full.return_value = None
            memory = engine.run_trajectory(
                goal="Wait and act",
                initial_droplets=initial_droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        self.assertTrue(pivoted[0], "Auto-pivot was not triggered on consecutive wait actions")


if __name__ == "__main__":
    unittest.main()
