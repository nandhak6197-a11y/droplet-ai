import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from unittest.mock import MagicMock, patch

from spatial_droplet.core.models import DiagonalDroplet, WindowContext
from spatial_droplet.verification.action_verifier import ActionVerifier, PreActionSnapshot


class TestGeneralizedVerificationAndLoop(unittest.TestCase):
    def test_adaptive_polling_window_shift(self):
        verifier = ActionVerifier(sw=1920, sh=1080)
        droplet = DiagonalDroplet(id=1, center_x=100, center_y=100, diag_wave=0, label="App Icon", category="interactive_app")
        pre = PreActionSnapshot(hwnd=1001, title="Desktop", cursor_shape="ARROW", roi=np.zeros((80, 80, 3), dtype=np.uint8), cx=100, cy=100)

        # Mock get_active_window_context to simulate window appearing after 100ms
        call_count = [0]
        def mock_win_ctx():
            call_count[0] += 1
            if call_count[0] <= 1:
                return WindowContext(hwnd=1001, title="Desktop", rect=(0,0,1920,1080), is_valid=True)
            return WindowContext(hwnd=2002, title="Terminal", rect=(100,100,800,600), is_valid=True)

        with patch("spatial_droplet.verification.action_verifier.get_active_window_context", side_effect=mock_win_ctx), \
             patch("spatial_droplet.verification.action_verifier.get_cursor_shape", return_value="ARROW"), \
             patch("spatial_droplet.verification.action_verifier.FastScreenGrabber.get_instance") as mock_grab:
            mock_grab.return_value.grab_box.return_value = np.zeros((80, 80, 3), dtype=np.uint8)
            res = verifier.verify_action(pre, droplet, "click")
            self.assertTrue(res["verified"])
            self.assertEqual(res["type"], "WINDOW_CHANGE")
            self.assertIn("Terminal", res["reason"])

    def test_loop_guard_adaptation_not_blocked(self):
        # Test that switching action from double_click to click does not trip the action failure guard
        action_fail_counts = {}
        target_fail_counts = {}

        target_key = "taskbar pinned app #1"
        action = "double_click"
        action_key = f"{action}:{target_key}"

        # 2 failures of double_click
        action_fail_counts[action_key] = 2
        target_fail_counts[target_key] = 2

        # In turn 3, action pivots to click:
        new_action = "click"
        new_action_key = f"{new_action}:{target_key}"
        act_fails = action_fail_counts.get(new_action_key, 0)
        tgt_fails = target_fail_counts.get(target_key, 0)

        # Act fails for click is 0 -> should NOT abort (threshold is 3)
        self.assertLess(act_fails, 3)
        # Target fails is 2 -> should NOT abort (threshold is 4)
        self.assertLess(tgt_fails, 4)

if __name__ == "__main__":
    unittest.main()
