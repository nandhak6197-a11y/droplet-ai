"""
Adversarial Stress Test Suite for Milestone 3 (Canonical Action Parser, PrecisionActuator, Win32 Window, ActionVerifier).
Constructs extreme boundary cases, malformed outputs, and adversarial conditions.
"""
import pytest
import json
import numpy as np
from unittest.mock import MagicMock, patch

from spatial_droplet.reasoning.reasoner import (
    DiagonalDropletReasoner,
    _clean_decision,
    _extract_json,
)
from spatial_droplet.platform.window import (
    minimize_window,
    maximize_window,
    switch_to_window,
    bring_to_foreground,
)
from spatial_droplet.platform.actuator import (
    PrecisionActuator,
    Win32HardwareActuator,
    resolve_vk,
)
from spatial_droplet.verification.action_verifier import (
    ActionVerifier,
    PreActionSnapshot,
)
from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
)


class TestAdversarialActionParser:
    """Stress tests for the Canonical Action Parser."""

    def test_truncated_json_string_literal(self):
        """Tests recovery from truncated JSON inside an unclosed string literal."""
        raw = '{"action": "click", "deliberation": {"user_intent_and_success_criteria": "Searching for files'
        res = _extract_json(raw)
        assert res is not None
        assert res.get("action") == "click"

    def test_four_backtick_markdown(self):
        """Tests markdown block enclosed in 4 backticks."""
        raw = "````json\n{\"action\": \"double_click\", \"target_dot_id\": 7}\n````"
        res = _extract_json(raw)
        assert res is not None
        assert res.get("action") == "double_click"
        assert res.get("target_dot_id") == 7

    def test_single_quotes_and_dot_prefix(self):
        """Tests Python dict style with single quotes and 'dot_99' string format."""
        raw = "{'action': 'type_and_enter', 'input_text': 'hello', 'target_dot_id': 'dot_99'}"
        res = _extract_json(raw)
        assert res is not None
        assert res.get("action") == "type_and_enter"
        assert res.get("target_dot_id") == 99
        assert res.get("input_text") == "hello"

    def test_passive_action_to_grounded_click(self):
        """Tests passive action conversion to grounded action enforcing zero passive turns."""
        raw = '{"action": "thinking", "deliberation": {"tactical_action_justification": "Observing screen"}}'
        res = _extract_json(raw)
        assert res is not None
        assert res.get("action") in ("click", "stop")

    def test_malformed_unquoted_regex_fallback(self):
        """Tests completely broken unquoted key-value output."""
        raw = "action: minimize, status: in_progress"
        res = _extract_json(raw)
        assert res is not None
        assert res.get("action") == "minimize"

    def test_hotkey_case_and_whitespace(self):
        """Tests hotkey strings with weird casing and spacing (e.g. ' Ctrl + Alt + Delete ')."""
        decision = {"action": "hotkey", "key_combo": " Ctrl + Alt + Delete "}
        cleaned = _clean_decision(decision)
        assert cleaned["key_combo"] == ["ctrl", "alt", "delete"]

    def test_excessively_deep_or_strange_keys(self):
        """Tests decision dictionary with unexpected extra types."""
        decision = {
            "action": "click",
            "target_dot_id": ["12", "34"],
            "extra_field": {"nested": [1, 2, 3]}
        }
        cleaned = _clean_decision(decision)
        assert cleaned["target_dot_id"] == 12
        assert cleaned["action"] == "click"


class TestAdversarialWindowManagement:
    """Stress tests for native Win32 window management."""

    def test_negative_hwnd_handling(self):
        """Defensive handling against negative / invalid HWND."""
        assert minimize_window(-99999) is False
        assert maximize_window(-99999) is False
        assert bring_to_foreground(-99999) is False

    def test_empty_or_none_target_window(self):
        """Defensive handling against empty or None search targets."""
        assert switch_to_window("") is None
        assert switch_to_window("   ") is None
        assert switch_to_window(None) is None

    def test_nonexistent_hwnd(self):
        """Defensive handling for non-existent window handle."""
        assert switch_to_window(9999999) is None


class TestAdversarialActuator:
    """Stress tests for PrecisionActuator."""

    def test_actuator_rejects_blind_click(self):
        """Actuator must reject blind clicks with no droplet and no target coordinates, or with (0,0) droplet."""
        actuator = PrecisionActuator()
        res = actuator.execute(droplet=None, action="click", target_x=None, target_y=None)
        assert res["verified"] is False
        assert res["type"] == "INVALID_ACTION"

        # Reject click when droplet has uninitialized (0, 0) coordinates
        droplet_zero = DiagonalDroplet(id=0, center_x=0, center_y=0, category="unknown", label="", diag_wave=0)
        res_zero = actuator.execute(droplet=droplet_zero, action="click")
        assert res_zero["verified"] is False
        assert res_zero["type"] == "INVALID_ACTION"

    def test_actuator_rejects_missing_drag_destination(self):
        """Actuator must reject drag actions when destination coordinates are missing."""
        actuator = PrecisionActuator()
        res = actuator.execute(droplet=None, action="drag", target_x=100, target_y=100, drag_dest=None)
        assert res["verified"] is False
        assert res["type"] == "INVALID_ACTION"

    def test_actuator_handles_stop_action(self):
        """Actuator handles stop action cleanly."""
        actuator = PrecisionActuator()
        res = actuator.execute(droplet=None, action="stop")
        assert res["verified"] is True
        assert res["type"] == "TERMINATION"

    def test_resolve_vk_keys(self):
        """Tests virtual key resolver for varied cases."""
        assert resolve_vk("enter") == 0x0D
        assert resolve_vk("esc") == 0x1B
        assert resolve_vk("ctrl") == 0x11
        assert resolve_vk("a") == 0x41
        assert resolve_vk("unknown_key_xyz") == 0


class TestAdversarialActionVerifier:
    """Stress tests for ActionVerifier."""

    def test_mismatched_frame_shapes(self):
        """ActionVerifier safely handles pre and post frames of different dimensions without crashing."""
        verifier = ActionVerifier(sw=1920, sh=1080)
        snap = PreActionSnapshot(
            hwnd=0,
            title="Desk",
            cursor_shape="ARROW",
            roi=None,
            cx=0,
            cy=0,
            full_frame=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        post_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch("spatial_droplet.verification.action_verifier.get_active_window_context", return_value=WindowContext(hwnd=0, title="Desk")):
            res = verifier.verify_action(snap, droplet=None, action="click", post_frame=post_frame)
            assert res["verified"] is False
            assert res["type"] == "NO_CHANGE"

    def test_compute_roi_delta_none_shapes(self):
        """Tests compute_roi_delta returns (0, 0.0) safely when either input is None."""
        d_pix, m_delta = ActionVerifier.compute_roi_delta(None, np.zeros((10, 10, 3)))
        assert d_pix == 0
        assert m_delta == 0.0

        d_pix2, m_delta2 = ActionVerifier.compute_roi_delta(np.zeros((10, 10, 3)), None)
        assert d_pix2 == 0
        assert m_delta2 == 0.0
