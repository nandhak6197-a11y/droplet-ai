"""
Verification test suite for Advanced Generalized Capabilities:
1. Option 1: Autonomous Scroll & Seek Vision Engine (Visual Pagination)
2. Option 2: Smart Clipboard Injection & Anti-Autocomplete Typing
3. Option 3: Adaptive Long-Term Knowledge Base (Self-Learning)
"""
import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from spatial_droplet.telemetry.flight_recorder import FlightRecorder
from spatial_droplet.platform.actuator import Win32HardwareActuator, PrecisionActuator


class TestAdvancedFeatures(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recorder = FlightRecorder(incident_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_option_3_knowledge_base_update_and_query(self):
        """Verify that Adaptive Knowledge Base writes and retrieves learned rules."""
        self.recorder.update_knowledge(
            task_goal="select ellipse tool in paint",
            successful_target="23",
            context="double click required on toolbar"
        )
        # Query by exact match
        data = self.recorder.query_knowledge("select ellipse tool in paint")
        self.assertIsNotNone(data)
        self.assertEqual(data["target"], "23")
        self.assertIn("double click", data["context"])

        # Query by partial substring match
        sub_data = self.recorder.query_knowledge("ellipse tool")
        self.assertIsNotNone(sub_data)
        self.assertEqual(sub_data["target"], "23")

    def test_option_2_smart_clipboard_typing_logic(self):
        """Verify that Smart Clipboard injection triggers for strings longer than 3 characters."""
        actuator = PrecisionActuator()
        with patch.object(Win32HardwareActuator, "click") as mock_click, \
             patch.object(Win32HardwareActuator, "hotkey") as mock_hotkey, \
             patch.object(Win32HardwareActuator, "press_key") as mock_press, \
             patch.object(Win32HardwareActuator, "type_unicode") as mock_unicode:
            
            # Short string <= 3 chars should use direct unicode typing
            actuator.execute(droplet=None, action="type", input_text="hi", target_x=100, target_y=100)
            mock_unicode.assert_called_with("hi")
            mock_hotkey.assert_not_called()

            # Long string > 3 chars should use clipboard injection (Ctrl+V + Esc)
            mock_unicode.reset_mock()
            actuator.execute(droplet=None, action="type", input_text="Hello World!", target_x=100, target_y=100)
            mock_hotkey.assert_called_with("ctrl", "v")
            mock_press.assert_called_with("esc")
            mock_unicode.assert_not_called()

    def test_option_1_planner_scroll_and_seek_wiring(self):
        """Verify that planner imports and references Visual Pagination logic."""
        import inspect
        from spatial_droplet.engine.planner import HierarchicalGoalPlanner
        source = inspect.getsource(HierarchicalGoalPlanner.execute_plan)
        self.assertIn("Autonomous \"Scroll & Seek\" Vision Engine", source)
        self.assertIn("requires_scroll", source)
        self.assertIn("scroll_amount=-300", source)


if __name__ == "__main__":
    unittest.main()
