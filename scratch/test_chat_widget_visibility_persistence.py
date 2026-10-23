import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tkinter as tk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_droplet.ui.chat_widget import HoverChatWidget


class TestChatWidgetVisibilityPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.widget = HoverChatWidget(
            self.root,
            on_submit_callback=lambda x: None,
            on_toggle_callback=lambda: None,
            on_refresh_callback=lambda: None
        )

    def tearDown(self):
        try:
            if self.widget.win.winfo_exists():
                self.widget.win.destroy()
        except Exception:
            pass

    def test_ensure_visible_calls_topmost_and_lift(self):
        """Pillar: ensure_visible re-asserts HWND_TOPMOST and lifts window."""
        with patch("ctypes.windll.user32.SetWindowPos") as mock_set_pos:
            self.widget.ensure_visible()
            self.assertTrue(mock_set_pos.called)
            # Args: hwnd, hWndInsertAfter (-1 = HWND_TOPMOST), x, y, cx, cy, uFlags
            args = mock_set_pos.call_args[0]
            self.assertEqual(args[1], -1)  # HWND_TOPMOST

    def test_escape_does_not_collapse_ui(self):
        """Pillar: Pressing Escape in the chat widget does not collapse to orb."""
        self.assertTrue(self.widget.is_expanded)
        # Type some text into entry
        if hasattr(self.widget, "entry") and self.widget.entry:
            self.widget.entry.insert(0, "some search query")
            self.assertEqual(self.widget.entry.get(), "some search query")

            # Press Escape
            self.widget._handle_escape()
            # Entry text should be cleared, but UI remains expanded
            self.assertEqual(self.widget.entry.get(), "")
            self.assertTrue(self.widget.is_expanded)

    def test_drag_clamps_to_screen_and_ensures_visible(self):
        """Pillar: Dragging cannot move the chat widget off-screen."""
        with patch.object(self.widget, "ensure_visible") as mock_vis:
            mock_event = MagicMock()
            mock_event.x = 99999  # Attempt to drag off right edge
            mock_event.y = 99999  # Attempt to drag off bottom
            self.widget._start_drag(MagicMock(x=0, y=0))
            self.widget._do_drag(mock_event)

            # Check that window rect was clamped
            rx, ry, rr, rb = self.widget._cached_win_rect
            sw = self.widget.win.winfo_screenwidth()
            sh = self.widget.win.winfo_screenheight()
            self.assertLessEqual(rr, sw)
            self.assertLessEqual(rb, sh)
            self.assertTrue(mock_vis.called)

    def test_anim_tick_reschedules_and_maintains_heartbeat(self):
        """Pillar: _anim_tick reschedules itself and periodically triggers ensure_visible."""
        self.widget._topmost_tick_counter = 24
        with patch.object(self.widget, "ensure_visible") as mock_vis:
            self.widget._anim_tick()
            self.assertEqual(self.widget._topmost_tick_counter, 0)
            self.assertTrue(mock_vis.called)


if __name__ == "__main__":
    unittest.main()
