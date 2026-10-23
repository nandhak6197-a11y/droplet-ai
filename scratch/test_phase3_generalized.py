"""
Comprehensive Unit Test Suite for Phase 3 Generalized Improvements:
1. Cross-Platform Actuator Guarding & PyAutoGUI Fallback
2. OpenAI Grounding Provider & Registry Integration
3. Closed-Loop ActionVerifier Auto-Pivot & Blacklist
"""
import os
import sys
import unittest
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.core.models import DiagonalDroplet, WindowContext
from spatial_droplet.platform import actuator
from spatial_droplet.platform.actuator import (
    Win32HardwareActuator,
    PrecisionActuator,
    get_cursor_shape,
    resolve_vk
)
from spatial_droplet.reasoning.providers.base import GroundingDecision
from spatial_droplet.reasoning.providers.openai_provider import OpenAIGroundingProvider
from spatial_droplet.reasoning.providers.registry import get_grounding_provider
from spatial_droplet.verification.action_verifier import ActionVerifier, PreActionSnapshot


class TestCrossPlatformActuator(unittest.TestCase):
    def test_cursor_and_vk_without_user32(self):
        # Save original
        orig_user32 = actuator._user32
        try:
            # Simulate non-Windows / headless
            actuator._user32 = None
            shape = get_cursor_shape()
            self.assertEqual(shape, "ARROW/DEFAULT")

            vk = resolve_vk('a')
            self.assertEqual(vk, 0x41)

            vk_enter = resolve_vk('enter')
            self.assertEqual(vk_enter, 0x0D)

            # Test hardware driver without user32 (should fall back safely without raising)
            Win32HardwareActuator.move_cursor(100, 100)
            Win32HardwareActuator.click(100, 100)
            Win32HardwareActuator.double_click(100, 100)
            Win32HardwareActuator.triple_click(100, 100)
            Win32HardwareActuator.right_click(100, 100)
            Win32HardwareActuator.scroll(-5)
            Win32HardwareActuator.type_unicode("test")
            Win32HardwareActuator.hotkey('ctrl', 'c')
        finally:
            actuator._user32 = orig_user32

    def test_precision_actuator_execute_mock(self):
        act = PrecisionActuator()
        droplet = DiagonalDroplet(id=42, center_x=150, center_y=120, category="interactive_widget", label="Test Button", diag_wave=270, bbox=(100, 100, 200, 140))
        # Execute click
        res = act.execute(droplet=droplet, action="click")
        self.assertTrue(res.get("verified", False))

        # Execute wait
        res_wait = act.execute(droplet=None, action="wait", input_text="0.01")
        self.assertTrue(res_wait.get("verified", False))


class TestOpenAIGroundingProvider(unittest.TestCase):
    def test_provider_initialization_and_registry(self):
        p = OpenAIGroundingProvider(api_key="sk-mock-key-for-test", model_name="gpt-4o")
        self.assertTrue(p.is_available())
        self.assertEqual(p.model_name, "gpt-4o")

        # Test Registry resolution
        reg_p = get_grounding_provider("openai", api_key="sk-test", model_name="gpt-4o-mini")
        self.assertIsInstance(reg_p, OpenAIGroundingProvider)

    def test_response_parsing(self):
        p = OpenAIGroundingProvider(api_key="sk-test")
        cands = [
            DiagonalDroplet(id=10, center_x=30, center_y=20, category="interactive_widget", label="File Menu", diag_wave=50, bbox=(10, 10, 50, 30)),
            DiagonalDroplet(id=20, center_x=80, center_y=20, category="interactive_widget", label="Save Button", diag_wave=100, bbox=(60, 10, 100, 30)),
        ]

        # Valid JSON
        raw_json = '{"target_dot_id": 20, "action": "click", "reasoning": "Save button selected"}'
        dec = p.parse_raw_response(raw_json, cands)
        self.assertEqual(dec.target_dot_id, 20)
        self.assertEqual(dec.action, "click")
        self.assertEqual(dec.x, 80.0)

        # Markdown wrapped JSON
        raw_md = '```json\n{"target_dot_id": 10, "action": "click", "reasoning": "Open File"}\n```'
        dec_md = p.parse_raw_response(raw_md, cands)
        self.assertEqual(dec_md.target_dot_id, 10)
        self.assertEqual(dec_md.x, 30.0)


class TestActionVerifierAutoPivot(unittest.TestCase):
    def setUp(self):
        self.verifier = ActionVerifier(sw=1920, sh=1080)
        self.d1 = DiagonalDroplet(id=5, center_x=130, center_y=115, category="interactive_widget", label="Cancel", diag_wave=245, bbox=(100, 100, 160, 130))
        self.d2 = DiagonalDroplet(id=6, center_x=210, center_y=115, category="interactive_widget", label="Confirm", diag_wave=325, bbox=(180, 100, 240, 130))
        self.d3 = DiagonalDroplet(id=7, center_x=525, center_y=515, category="interactive_widget", label="Options", diag_wave=1040, bbox=(500, 500, 550, 530))

    def test_blacklist_tracking(self):
        self.assertFalse(self.verifier.is_blacklisted(5))
        self.verifier.record_failure(5, "click", "No visual change", 0.0)
        self.assertTrue(self.verifier.is_blacklisted(5))
        self.verifier.reset_blacklist()
        self.assertFalse(self.verifier.is_blacklisted(5))

    def test_auto_pivot_to_alternative_candidate(self):
        # When action on d1 fails, suggest pivot to nearby candidate d2
        pivot = self.verifier.suggest_pivot(
            failed_droplet=self.d1,
            candidates=[self.d1, self.d2, self.d3],
            action="click",
            query="confirm"
        )
        self.assertEqual(pivot["pivot_type"], "ALTERNATIVE_CANDIDATE")
        self.assertEqual(pivot["target_dot_id"], 6)
        self.assertIn("runner-up Dot [6]", pivot["reason"])

    def test_auto_pivot_action_escalation(self):
        # When no alternative candidates are provided, escalate action
        pivot_click = self.verifier.suggest_pivot(self.d1, candidates=[], action="click")
        self.assertEqual(pivot_click["pivot_type"], "ACTION_ESCALATION")
        self.assertEqual(pivot_click["suggested_action"], "double_click")

        pivot_double = self.verifier.suggest_pivot(self.d1, candidates=[], action="double_click")
        self.assertEqual(pivot_double["pivot_type"], "ACTION_ESCALATION")
        self.assertEqual(pivot_double["suggested_action"], "press_enter")

    def test_verify_action_zero_delta(self):
        from spatial_droplet.verification import action_verifier
        orig_win = action_verifier.get_active_window_context
        orig_cur = action_verifier.get_cursor_shape
        from spatial_droplet.vision.grabber import FastScreenGrabber
        orig_grab = FastScreenGrabber.get_instance().grab_box

        mock_ctx = WindowContext(hwnd=123, title="App")
        action_verifier.get_active_window_context = lambda: mock_ctx
        action_verifier.get_cursor_shape = lambda: "ARROW"
        FastScreenGrabber.get_instance().grab_box = lambda x, y, w, h: np.zeros((h, w, 3), dtype=np.uint8)

        try:
            roi_pre = np.zeros((80, 80, 3), dtype=np.uint8)
            pre = PreActionSnapshot(hwnd=123, title="App", cursor_shape="ARROW", roi=roi_pre, cx=130, cy=115)
            ver_res = self.verifier.verify_action(
                pre=pre,
                droplet=self.d1,
                action="click",
                candidates=[self.d1, self.d2],
                query="confirm"
            )
            self.assertFalse(ver_res["verified"])
            self.assertEqual(ver_res["type"], "NO_CHANGE")
            self.assertTrue(self.verifier.is_blacklisted(self.d1.id))
            self.assertEqual(ver_res["suggested_alternative_id"], self.d2.id)
        finally:
            action_verifier.get_active_window_context = orig_win
            action_verifier.get_cursor_shape = orig_cur
            FastScreenGrabber.get_instance().grab_box = orig_grab


if __name__ == "__main__":
    unittest.main()
