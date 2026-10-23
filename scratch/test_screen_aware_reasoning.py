"""
Test Suite: Screen-Aware Spatial Perception & 5-Stage CoT Deliberation Trace (Milestone 1).
Validates:
1. Window-relative functional zone classification for restored/floating non-maximized windows.
2. Retention of OS taskbar as [FOOTER] across arbitrary window states.
3. Modal dialog precedence over active window boundaries.
4. 2D relative cardinal neighbor hints (Right/Left/Above/Below) in spatial layout graph.
5. Modal element prioritization in spatial layout graph and candidate relevance scoring.
6. ScreenAwarenessContext active_window_rect propagation from ScreenStateParser.
7. TrajectoryStep 5-stage CoT attribute accessors and bidirectional backward compatibility.
8. Deterministic rejection of passive text deliberation with canonical action recovery.
9. Generalized dialog commit discovery (_find_primary_dialog_commit) prioritizing modal bounds.
"""

import sys
import os
import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import (
    DiagonalDropletReasoner,
    DecisionResult,
)
_clean_decision = DiagonalDropletReasoner._clean_decision

from spatial_droplet.core.models import (
    DiagonalDroplet,
    WindowContext,
    ModalDialogContext,
    ScreenAwarenessContext,
    TrajectoryStep,
)
from spatial_droplet.vision.screen_parser import ScreenStateParser
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine


def test_01_window_relative_functional_zones():
    """Verify classify_screen_zone computes zones relative to floating window frame."""
    sw, sh = 1920, 1080
    # Restored floating window: x1=300, y1=200, x2=1300, y2=900 (W=1000, H=700)
    win_rect = (300, 200, 1300, 900)

    # Window Header (y in [200, 200 + 0.14*700 = 298])
    # Coordinate y=240 is inside window header ribbon, but outside global 0.12*1080 (129.6px)
    assert DiagonalDropletReasoner.classify_screen_zone(500, 240, sw, sh, win_rect=win_rect) == "[HEADER/NAV]"

    # Window Sidebar (x in [300, 300 + 0.22*1000 = 520], y in [350, 750])
    assert DiagonalDropletReasoner.classify_screen_zone(400, 450, sw, sh, win_rect=win_rect) == "[SIDEBAR]"

    # Window Workspace (center of window)
    assert DiagonalDropletReasoner.classify_screen_zone(800, 500, sw, sh, win_rect=win_rect) == "[WORKSPACE]"

    # Window Footer / App Status Bar (y in [900 - 0.12*700 = 816, 900])
    assert DiagonalDropletReasoner.classify_screen_zone(600, 850, sw, sh, win_rect=win_rect) == "[FOOTER]"


def test_02_os_taskbar_retention_with_active_window():
    """Verify global OS taskbar (y >= 0.88 * sh) is preserved as [FOOTER] even with non-maximized window."""
    sw, sh = 1920, 1080
    win_rect = (200, 100, 1200, 700)  # Window terminates at y=700

    # Taskbar at y=1020 (>= 0.88 * 1080 = 950.4)
    assert DiagonalDropletReasoner.classify_screen_zone(500, 1020, sw, sh, win_rect=win_rect) == "[FOOTER]"
    assert DiagonalDropletReasoner.classify_screen_zone(1500, 980, sw, sh, win_rect=win_rect) == "[FOOTER]"


def test_03_modal_dialog_precedence_over_window():
    """Verify modal dialog bounds override both window zones and workspace."""
    sw, sh = 1920, 1080
    win_rect = (200, 100, 1600, 900)
    modal_rect = (600, 350, 1200, 750)

    # Point inside modal dialog
    zone = DiagonalDropletReasoner.classify_screen_zone(700, 450, sw, sh, modal_rect=modal_rect, win_rect=win_rect)
    assert zone == "[MODAL/POPUP]"

    # Point outside modal but inside window
    zone_outside = DiagonalDropletReasoner.classify_screen_zone(300, 450, sw, sh, modal_rect=modal_rect, win_rect=win_rect)
    assert zone_outside == "[SIDEBAR]"


def test_04_spatial_layout_graph_2d_cardinal_hints():
    """Verify _build_spatial_layout_graph generates Right/Left and Above/Below cardinal hints."""
    sw, sh = 1920, 1080

    # Horizontal pair
    d_left = DiagonalDroplet(id=1, center_x=100, center_y=100, category="interactive_widget", label="Search Box", diag_wave=200)
    d_right = DiagonalDroplet(id=2, center_x=220, center_y=100, category="interactive_widget", label="Submit Btn", diag_wave=320)

    # Vertical pair
    d_top = DiagonalDroplet(id=3, center_x=500, center_y=300, category="interactive_widget", label="Username", diag_wave=800)
    d_bottom = DiagonalDroplet(id=4, center_x=500, center_y=380, category="interactive_widget", label="Password", diag_wave=880)

    nodes = [d_left, d_right, d_top, d_bottom]
    graph = DiagonalDropletReasoner._build_spatial_layout_graph(nodes, sw, sh)

    # Verify horizontal hints
    assert "Right of Dot [1]" in graph or "Left of Dot [2]" in graph
    # Verify vertical cardinal hints
    assert "Below Dot [3]" in graph or "Above Dot [4]" in graph


def test_05_modal_elements_top_precedence_in_graph():
    """Verify modal dialog elements appear in the top [MODAL / POPUP DIALOG] section of the graph."""
    sw, sh = 1920, 1080
    modal_rect = (600, 400, 1000, 700)

    d_modal = DiagonalDroplet(id=10, center_x=700, center_y=550, category="interactive_widget", label="Save Changes", diag_wave=1250)
    d_window = DiagonalDroplet(id=20, center_x=500, center_y=500, category="interactive_widget", label="Document Text", diag_wave=1000)

    graph = DiagonalDropletReasoner._build_spatial_layout_graph([d_modal, d_window], sw, sh, modal_rect=modal_rect)

    lines = graph.splitlines()
    modal_section_idx = -1
    workspace_section_idx = -1
    for idx, l in enumerate(lines):
        if "[MODAL / POPUP DIALOG]" in l:
            modal_section_idx = idx
        elif "[WORKSPACE / VIEWPORT]" in l:
            workspace_section_idx = idx

    assert modal_section_idx != -1
    assert modal_section_idx < workspace_section_idx
    assert "• [10]" in graph and "Save Changes" in graph


def test_06_screen_awareness_context_window_rect_propagation():
    """Verify ScreenStateParser.parse_screen_awareness propagates active_window_rect."""
    win = WindowContext(
        hwnd=1234,
        title="Settings",
        rect=(100, 100, 900, 700),
        width=800,
        height=600,
        is_valid=True
    )
    awareness = ScreenStateParser.parse_screen_awareness(win, ["Settings", "Network & Internet"], [])

    assert awareness.active_window_rect == (100, 100, 900, 700)
    assert awareness.window_rect == (100, 100, 900, 700)
    assert awareness.app_name == "Settings"


def test_07_trajectory_step_5stage_cot_attributes():
    """Verify TrajectoryStep initializes and exposes 5-stage CoT attributes with bidirectional sync."""
    step = TrajectoryStep(
        step_num=1,
        goal="Open file",
        subgoal="Click File",
        action="click",
        target_id=1,
        target_label="File",
        input_text=None,
        verified=True,
        verification_reason="Menu opened",
        active_window="Editor",
        user_intent_and_success_criteria="Open the file menu.",
        screen_state_and_visual_hierarchy="Editor window focused.",
        mistake_critique_and_loop_diagnosis="Initial action.",
        end_to_end_master_plan=["[1] Open file (ACTIVE)", "[2] Read content (PENDING)"],
        tactical_action_justification="Clicking Dot [1] triggers dropdown."
    )

    # 5-stage CoT accessors
    assert step.user_intent_and_success_criteria == "Open the file menu."
    assert step.screen_state_and_visual_hierarchy == "Editor window focused."
    assert step.mistake_critique_and_loop_diagnosis == "Initial action."
    assert len(step.end_to_end_master_plan) == 2
    assert step.tactical_action_justification == "Clicking Dot [1] triggers dropdown."

    # Legacy fields bidirectional synchronization
    assert step.intent_analysis == "Open the file menu."
    assert step.observation_analysis == "Editor window focused."
    assert step.mistake_critique == "Initial action."
    assert step.master_plan == step.end_to_end_master_plan


def test_08_zero_passive_text_action_recovery():
    """Verify _clean_decision rejects turns with passive text and extracts canonical action or fallback."""
    raw = {
        "deliberation": {
            "user_intent_and_success_criteria": "Click the Save button.",
            "screen_state_and_visual_hierarchy": "Save button Dot [12] is visible.",
            "mistake_critique_and_loop_diagnosis": "None.",
            "end_to_end_master_plan": ["Click Save"],
            "tactical_action_justification": "Dot [12] is the Save button."
        },
        # Model produced conversational text without explicit canonical action
        "action": "",
        "reasoning": "I should probably click on Dot [12] to save.",
        "target_dot_id": 12
    }

    cleaned = _clean_decision(raw)
    assert cleaned is not None
    # Action must be non-empty and canonical
    assert cleaned["action"] in ("click", "wait", "press")
    assert cleaned["target_dot_id"] == 12


def test_09_dialog_commit_discovery_prefers_modal():
    """Verify AutonomousTrajectoryEngine._find_primary_dialog_commit selects button inside modal bounds."""
    modal_rect = (400, 300, 1000, 700)
    modal_ctx = ModalDialogContext(
        hwnd=5555,
        title="Confirm Exit",
        class_name="#32770",
        rect=modal_rect
    )
    awareness = ScreenAwarenessContext(
        app_title="Confirm Exit",
        active_modal=modal_ctx
    )

    droplets = {
        1: DiagonalDroplet(id=1, center_x=200, center_y=100, category="interactive_widget", label="[btn] Save Outside", diag_wave=300),
        2: DiagonalDroplet(id=2, center_x=850, center_y=650, category="interactive_widget", label="[btn] Save", diag_wave=1500),
        3: DiagonalDroplet(id=3, center_x=950, center_y=650, category="interactive_widget", label="[btn] Cancel", diag_wave=1600),
    }

    commit = AutonomousTrajectoryEngine._find_primary_dialog_commit(droplets, awareness, None)
    assert commit is not None
    assert commit.id == 2
    assert "save" in commit.label.lower()
