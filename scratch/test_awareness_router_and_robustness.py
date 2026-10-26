"""
Unit & Integration Tests for Awareness Router, Deliberation Token Capping,
Active Window Spatial Pruning, and Cognitive Verifier False-Positive Elimination.
"""
import pytest
import re
from unittest.mock import MagicMock, patch

from spatial_droplet.core.models import DiagonalDroplet, WindowContext, TrajectoryStep
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.reasoning.awareness_router import AwarenessRouter, Checkpoint
from spatial_droplet.reasoning.token_guard import TokenBudgetGuard
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner, _clean_decision
from spatial_droplet.verification.semantic_verifier import SemanticStateVerifier
from spatial_droplet.platform.actuator import PrecisionActuator


# ==============================================================================
# 1. Awareness Router: Sequential Checkpoint Decomposition & Awareness
# ==============================================================================

class TestAwarenessRouter:
    """Verifies sequential checkpoint decomposition, step awareness, and cleanup."""

    def test_decompose_chained_calculator_workflow(self):
        """
        Decomposes 'open calculator, add 2+2, verify, close' into 4 sequential checkpoints:
        1. Open Calculator (launch, ACTIVE)
        2. Perform Input / Calculation (interaction, PENDING)
        3. Verify Result (verify, PENDING)
        4. Close Calculator (cleanup, PENDING)
        """
        goal = "open calculator, add 2+2, verify, close"
        cps = AwarenessRouter.decompose_goal(goal)

        assert len(cps) == 4, f"Expected 4 checkpoints, got {len(cps)}"
        assert cps[0].id == 1
        assert cps[0].action_type == "launch"
        assert cps[0].status == "ACTIVE"
        assert "calculator" in cps[0].expected_window.lower()

        assert cps[1].id == 2
        assert cps[1].action_type == "interaction"
        assert cps[1].status == "PENDING"
        assert "2+2" in cps[1].description

        assert cps[2].id == 3
        assert cps[2].action_type == "verify"
        assert cps[2].status == "PENDING"

        assert cps[3].id == 4
        assert cps[3].action_type == "cleanup"
        assert cps[3].cleanup is True
        assert cps[3].status == "PENDING"

    def test_decompose_multitask_workflow(self):
        """Verifies multi-task chained workflows e.g. 'open this, do that, play, forward, backward, skip'."""
        goal = "open browser, search music, play, forward, backward, skip"
        cps = AwarenessRouter.decompose_goal(goal)

        assert len(cps) >= 5
        assert cps[0].action_type == "launch"
        assert any("play" in cp.description.lower() for cp in cps)
        assert any("forward" in cp.description.lower() for cp in cps)
        assert any("backward" in cp.description.lower() for cp in cps)
        assert any("skip" in cp.description.lower() for cp in cps)

    def test_step_by_step_awareness_and_advancement(self):
        """
        Verifies that Step 1 MUST be verified on screen before advancing to Step 2.
        - While window is 'Desktop', Step 1 remains ACTIVE.
        - When active window becomes 'Calculator', Step 1 becomes DONE, Step 2 becomes ACTIVE.
        - Step 4 (cleanup) remains PENDING until earlier steps succeed.
        """
        goal = "open calculator, add 2+2, verify, close"
        cps = AwarenessRouter.decompose_goal(goal)

        # Initial turn: Desktop active
        desk_win = WindowContext(hwnd=10, title="Desktop", is_valid=True)
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, desk_win)
        assert cps[0].status == "ACTIVE"
        assert cps[1].status == "PENDING"

        # Turn 1: Calculator launches and is active
        calc_win = WindowContext(hwnd=100, title="Calculator", rect=(100, 100, 600, 800), is_valid=True)
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win)
        assert cps[0].status == "DONE"
        assert cps[1].status == "ACTIVE"
        assert cps[2].status == "PENDING"
        assert cps[3].status == "PENDING"

        # Turn 2: Calculation performed and verified
        step_calc = TrajectoryStep(
            step_num=2, goal=goal, subgoal="Add 2+2", action="type",
            target_id=None, target_label="", input_text="2+2=",
            verified=True, verification_reason="Calculation typed",
            active_window="Calculator", verdict="CONFIRMED"
        )
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win, last_step=step_calc)
        assert cps[1].status == "DONE"
        assert cps[2].status == "ACTIVE"
        assert cps[3].status == "PENDING"

        # Turn 3: Result verified on screen (screen tokens contain '4')
        step_ver = TrajectoryStep(
            step_num=3, goal=goal, subgoal="Verify 4", action="wait",
            target_id=None, target_label="", input_text=None,
            verified=True, verification_reason="Output displayed",
            active_window="Calculator", verdict="CONFIRMED"
        )
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win, screen_tokens={"4", "display", "result"}, last_step=step_ver)
        assert cps[2].status == "DONE"
        assert cps[3].status == "ACTIVE"  # Cleanup now active!

        # Turn 4: Close performed and window returns to Desktop
        step_close = TrajectoryStep(
            step_num=4, goal=goal, subgoal="Close", action="close",
            target_id=None, target_label="", input_text=None,
            verified=True, verification_reason="Window closed",
            active_window="Desktop", verdict="CONFIRMED"
        )
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, desk_win, last_step=step_close)
        assert cps[3].status == "DONE"


    def test_prompt_checkpoint_formatting(self):
        """Verifies checkpoint block is formatted with clear milestone tags and turn mandate."""
        goal = "open notepad, type hello, close"
        cps = AwarenessRouter.decompose_goal(goal)
        txt = AwarenessRouter.format_checkpoints_for_prompt(cps)

        assert "=== AWARENESS ROUTER SEQUENTIAL CHECKPOINTS ===" in txt
        assert "► [ACTIVE] Step 1" in txt
        assert "○ [PENDING] Step 2" in txt
        assert "MANDATE FOR THIS TURN:" in txt


# ==============================================================================
# 2. Deliberation Token Capping (~400 Tokens Budget)
# ==============================================================================

class TestDeliberationTokenCapping:
    """Verifies that deliberation and thinking phase are strictly bounded to ~400 tokens."""

    def test_cap_deliberation_truncates_rambling_prose(self):
        """A rambling deliberation with > 500 tokens is compacted under 400 tokens."""
        long_paragraph = "Analyzing screen hierarchy and active UI nodes. " * 80  # ~800 tokens
        verbose_delib = {
            "user_intent_and_success_criteria": "Perform simple calculator addition.",
            "screen_state_and_visual_hierarchy": long_paragraph,
            "mistake_critique_and_loop_diagnosis": "No mistakes yet. " * 20,
            "end_to_end_master_plan": ["[Phase 1] Launch", "[Phase 2] Calculate"],
            "tactical_action_justification": long_paragraph,
        }

        capped = TokenBudgetGuard.cap_deliberation(verbose_delib, max_tokens=400)
        total_tokens = sum(TokenBudgetGuard.count_tokens(str(v)) for v in capped.values())
        assert total_tokens <= 400, f"Deliberation was not capped to <= 400 tokens: {total_tokens}"

    def test_concise_deliberation_preserved_intact(self):
        """Concise deliberation (< 300 tokens) is preserved completely without modification."""
        concise = {
            "user_intent_and_success_criteria": "Open Calculator.",
            "screen_state_and_visual_hierarchy": "Desktop visible.",
            "mistake_critique_and_loop_diagnosis": "Initial step.",
            "end_to_end_master_plan": ["[Phase 1] Launch Calculator"],
            "tactical_action_justification": "Click launcher icon.",
        }
        capped = TokenBudgetGuard.cap_deliberation(concise, max_tokens=400)
        assert capped == concise


# ==============================================================================
# 3. Active Window Spatial Pruning (Eliminating Background Misclicks)
# ==============================================================================

class TestActiveWindowSpatialPruning:
    """Verifies droplets outside the active window rect (e.g. PowerShell at (1388, 31)) are pruned."""

    def test_filter_spatial_droplets_prunes_background_windows(self):
        """
        When Calculator window is at (200, 200, 700, 800):
        - Droplet 1 inside Calculator at (350, 450) is PRESERVED.
        - Droplet 2 in background PowerShell at (1388, 31) is PRUNED.
        - Droplet 3 on desktop wallpaper at (1800, 900) is PRUNED.
        """
        win_ctx = WindowContext(
            hwnd=500, title="Calculator", rect=(200, 200, 700, 800), is_valid=True
        )

        d1_inside = DiagonalDroplet(id=1, center_x=350, center_y=450, category="interactive_widget", label="+ button", diag_wave=800, is_foreground=True)
        d2_powershell = DiagonalDroplet(id=2, center_x=1388, center_y=31, category="interactive_app", label="PowerShell Console", diag_wave=1419, is_foreground=False)
        d3_wallpaper = DiagonalDroplet(id=3, center_x=1800, center_y=900, category="interactive_app", label="Desktop Wallpaper", diag_wave=2700, is_foreground=False)

        all_droplets = [d1_inside, d2_powershell, d3_wallpaper]
        pruned = AwarenessRouter.filter_spatial_droplets(all_droplets, win_ctx)

        assert len(pruned) == 1
        assert pruned[0].id == 1
        assert d2_powershell not in pruned
        assert d3_wallpaper not in pruned

    def test_relevance_score_penalizes_outside_droplets(self):
        """TokenBudgetGuard scores background distractors outside active window with heavy penalty (-5000)."""
        win_rect = (200, 200, 700, 800)
        d_inside = DiagonalDroplet(id=1, center_x=350, center_y=450, category="button", label="Plus", diag_wave=800, is_foreground=True)
        d_outside = DiagonalDroplet(id=2, center_x=1388, center_y=31, category="button", label="Plus", diag_wave=1419, is_foreground=False)

        s_in = TokenBudgetGuard.score_droplet_relevance(d_inside, goal="Click plus", win_rect=win_rect)
        s_out = TokenBudgetGuard.score_droplet_relevance(d_outside, goal="Click plus", win_rect=win_rect)

        assert s_in > s_out + 4000.0, f"Outside droplet was not severely penalized: in={s_in}, out={s_out}"


# ==============================================================================
# 4. LAUNCH_APP Elimination & Keyboard Input Normalization
# ==============================================================================

class TestLaunchAppNormalizationAndGuard:
    """Verifies LAUNCH_APP is restricted to genuine launches and never used for keyboard input."""

    def test_clean_decision_normalizes_numeric_launch_app_to_type(self):
        """If model hallucinates action 'launch_app' with '2', normalized to action 'type' with input_text '2'."""
        bad_decision = {
            "action": "launch_app",
            "input_text": "2",
            "reasoning": "Launch 2 on the calculator"
        }
        cleaned = _clean_decision(bad_decision)
        assert cleaned["action"] == "type"
        assert cleaned["input_text"] == "2"

    def test_clean_decision_normalizes_formula_launch_app_to_type(self):
        """If model hallucinates action 'launch_app' with '2+2', normalized to action 'type'."""
        bad_decision = {
            "action": "launch_app",
            "input_text": "2+2=",
            "reasoning": "Compute 2+2"
        }
        cleaned = _clean_decision(bad_decision)
        assert cleaned["action"] == "type"
        assert cleaned["input_text"] == "2+2="

    def test_clean_decision_prevents_launch_when_already_in_foreground(self):
        """When Calculator is already in the foreground, launch_app 'calc' is converted to click or wait."""
        calc_win = WindowContext(hwnd=500, title="Calculator", is_valid=True)
        decision = {
            "action": "launch_app",
            "input_text": "calc",
            "target_dot_id": 10,
            "active_window": "Calculator"
        }
        cleaned = _clean_decision(decision, win_ctx=calc_win)
        assert cleaned["action"] != "launch_app"
        assert cleaned["action"] in ("click", "wait")

    def test_actuator_rejects_numeric_launch_app(self):
        """PrecisionActuator explicitly rejects launching numeric targets like '2' with INVALID_ACTION."""
        actuator = PrecisionActuator()
        res = actuator.execute(droplet=None, action="launch_app", input_text="2")
        assert res["verified"] is False
        assert res["type"] == "INVALID_ACTION"
        assert "Cannot launch" in res["reason"]

    def test_actuator_rejects_launch_when_already_active(self):
        """PrecisionActuator rejects launch_app when application is already in foreground window."""
        actuator = PrecisionActuator()
        win_ctx = WindowContext(hwnd=100, title="Calculator", is_valid=True)
        res = actuator.execute(droplet=None, action="launch_app", input_text="calc", win_ctx=win_ctx)
        assert res["verified"] is False
        assert res["type"] == "ALREADY_ACTIVE"


# ==============================================================================
# 5. Cognitive Verifier False-Positive Elimination
# ==============================================================================

class TestCognitiveVerifierFalsePositiveElimination:
    """Verifies SemanticStateVerifier never reports CONFIRMED_EXPECTED_TRANSITION for invalid launches."""

    def test_verifier_rejects_numeric_launch_app(self):
        """
        Calling verify_transition with action='launch_app' and input_text='2'
        MUST return verdict='INVALID_ACTION' and verified=False, even if tokens changed on screen.
        """
        pre_snap = {"tokens": {"desktop", "clock"}, "window_title": "Desktop", "hwnd": 10}
        # Simulate noisy screen where background tokens appeared
        post_snap = {"tokens": {"desktop", "clock", "news", "weather"}, "window_title": "Desktop", "hwnd": 10}

        ver = SemanticStateVerifier.verify_transition(
            pre_snap=pre_snap,
            post_snap=post_snap,
            action="launch_app",
            input_text="2",
            semantic_intention="Launch 2",
            expected_transition="2 opens"
        )

        assert ver["verified"] is False
        assert ver["verdict"] == "INVALID_ACTION"
        assert "CONFIRMED" not in ver["verdict"]

    def test_verifier_rejects_failed_launch_without_window_open(self):
        """
        Calling verify_transition with action='launch_app' and input_text='calc'
        when no window switch occurred MUST return verdict='NO_EFFECT'.
        """
        pre_snap = {"tokens": {"desktop"}, "window_title": "Desktop", "hwnd": 10}
        post_snap = {"tokens": {"desktop"}, "window_title": "Desktop", "hwnd": 10}

        ver = SemanticStateVerifier.verify_transition(
            pre_snap=pre_snap,
            post_snap=post_snap,
            action="launch_app",
            input_text="calc",
            semantic_intention="Launch Calculator",
            expected_transition="Calculator opens"
        )

        assert ver["verified"] is False
        assert ver["verdict"] == "NO_EFFECT"

    def test_verifier_confirms_genuine_launch(self):
        """When Calculator window actually opens (hwnd switch and title presence), launch is confirmed."""
        pre_snap = {"tokens": {"desktop"}, "window_title": "Desktop", "hwnd": 10}
        post_snap = {"tokens": {"calculator", "standard"}, "window_title": "Calculator", "hwnd": 500}

        ver = SemanticStateVerifier.verify_transition(
            pre_snap=pre_snap,
            post_snap=post_snap,
            action="launch_app",
            input_text="calc",
            semantic_intention="Launch Calculator",
            expected_transition="Calculator window opens"
        )

        assert ver["verified"] is True
        assert ver["verdict"] == "CONFIRMED_EXPECTED_TRANSITION"
        assert "Calculator" in ver["reason"]


# ==============================================================================
# 6. Label Citation Verification
# ==============================================================================

class TestLabelCitationVerification:
    """Verifies reasoner matches target_label against candidate droplet manifest."""

    def test_label_citation_resolved_from_candidates(self):
        """When candidate droplets are passed to _clean_decision, target_label is resolved from dot ID."""
        d_plus = DiagonalDroplet(id=15, center_x=300, center_y=400, category="button", label="Plus Button", diag_wave=700)
        decision = {
            "action": "click",
            "target_dot_id": 15,
            "deliberation": {
                "tactical_action_justification": "Click Dot [15] 'Plus Button' to add numbers."
            }
        }
        cleaned = _clean_decision(decision, candidates=[d_plus])
        assert cleaned["target_label"] == "Plus Button"

    def test_label_citation_mismatch_auto_corrects_to_matching_candidate(self):
        """
        When model hallucinates target_dot_id=2 ('PowerShell Console') but cited target_label='+' or 'Plus Button',
        label citation verification detects mismatch and auto-corrects to Dot [15] ('+ Button').
        """
        d_powershell = DiagonalDroplet(id=2, center_x=1388, center_y=31, category="interactive_app", label="PowerShell Console", diag_wave=1419)
        d_plus = DiagonalDroplet(id=15, center_x=450, center_y=550, category="button", label="+ Button", diag_wave=1000)

        decision = {
            "action": "click",
            "target_dot_id": 2,
            "target_label": "+",
            "deliberation": {
                "tactical_action_justification": "Click Dot [2] '+' to add numbers."
            }
        }
        cleaned = _clean_decision(decision, candidates=[d_powershell, d_plus])
        assert cleaned["target_dot_id"] == 15
        assert cleaned["target_label"] == "+ Button"


# ==============================================================================
# 7. Navigation Drawer Loop Guard & Turn 1 Awareness
# ==============================================================================

class TestNavigationDrawerLoopGuardAndTurn1Awareness:
    """Verifies prevention of navigation drawer toggle loops and Turn 1 awareness advancement."""

    def test_navigation_drawer_click_does_not_advance_calculation(self):
        """Clicking navigation drawer toggle does NOT mark calculation checkpoint as satisfied."""
        goal = "open calculator, add 2+2, verify, close"
        cps = AwarenessRouter.decompose_goal(goal)
        calc_win = WindowContext(hwnd=100, title="Calculator", rect=(100, 100, 600, 800), is_valid=True)

        # Advance past Step 1
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win)
        assert cps[1].status == "ACTIVE"

        # Model clicked navigation drawer
        step_drawer = TrajectoryStep(
            step_num=2, goal=goal, subgoal="Toggle nav", action="click",
            target_id=1, target_label="Open Navigation", input_text=None,
            verified=True, verification_reason="Screen text changed (+Standard)",
            active_window="Calculator", verdict="CONFIRMED_EXPECTED_TRANSITION"
        )
        cps = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win, last_step=step_drawer)
        # Calculation checkpoint MUST NOT be declared DONE!
        assert cps[1].status == "ACTIVE"
        assert cps[1].status != "DONE"

    def test_immediate_turn1_screen_awareness(self):
        """When Calculator is already in foreground on Turn 1, Step 1 is immediately DONE."""
        goal = "open calculator, add 2+2, verify, close"
        cps = AwarenessRouter.decompose_goal(goal)
        calc_win = WindowContext(hwnd=100, title="Calculator", rect=(100, 100, 600, 800), is_valid=True)

        # Immediate evaluation at launch
        evaluated = AwarenessRouter.evaluate_screen_and_advance(cps, calc_win)
        assert evaluated[0].status == "DONE"
        assert evaluated[1].status == "ACTIVE"

    def test_navigation_drawer_toggle_loop_detection_and_escape(self):
        """Verifies engine detects nav drawer toggle loop and tactical escape presses 'esc'."""
        from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine

        mock_reasoner = MagicMock()
        mock_sprayer = MagicMock()
        mock_actuator = MagicMock()
        mock_actuator.execute.return_value = {"verified": True, "verdict": "CONFIRMED"}

        engine = AutonomousTrajectoryEngine(
            reasoner=mock_reasoner,
            sprayer=mock_sprayer,
            actuator=mock_actuator
        )

        d_nav = DiagonalDroplet(id=1, center_x=30, center_y=45, category="button", label="Open Navigation", diag_wave=75)
        droplets = [d_nav]
        win_ctx = WindowContext(hwnd=100, title="Calculator", rect=(100, 100, 600, 800), is_valid=True)

        async def mock_spray(frame, win_ctx):
            return droplets
        mock_sprayer.spray = mock_spray

        mock_reasoner.decide_turn.side_effect = [
            {"action": "click", "target_dot_id": 1, "target_label": "Open Navigation", "status": "in_progress"},
            {"action": "click", "target_dot_id": 1, "target_label": "Close Navigation", "status": "in_progress"},
            {"action": "click", "target_dot_id": 1, "target_label": "Open Navigation", "status": "in_progress"},
            {"action": "stop", "status": "completed"}
        ]

        with patch("time.sleep", return_value=None), \
             patch("spatial_droplet.engine.trajectory_engine.get_active_window_context", return_value=win_ctx), \
             patch("spatial_droplet.engine.trajectory_engine.FastScreenGrabber.get_instance") as mock_grabber, \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            mock_grabber.return_value.grab_full.return_value = None
            memory = engine.run_trajectory(
                goal="add 2+2 in calculator",
                initial_droplets=droplets,
                initial_win_ctx=win_ctx,
                max_turns=3
            )

        # Confirm actuator received 'press' with 'esc' to dismiss the drawer
        called_actions = [call.kwargs.get("action") or (call.args[1] if len(call.args) > 1 else None) for call in mock_actuator.execute.call_args_list]
        called_keys = [call.kwargs.get("key") for call in mock_actuator.execute.call_args_list]
        assert "press" in called_actions
        assert "esc" in called_keys


# ==============================================================================
# 8. Command Prefix Parsing (/goal, /boost, /turns)
# ==============================================================================

class TestGoalBoostPrefixHandling:
    """Verifies robust stripping and extraction of /goal, /boost, /turns in any order."""

    def test_decompose_goal_strips_slash_prefixes(self):
        """AwarenessRouter.decompose_goal strips /goal, /boost, and /turns prefixes."""
        g1 = "/goal open calculator, add 2+2, verify, close"
        cps1 = AwarenessRouter.decompose_goal(g1)
        assert cps1[0].action_type == "launch"
        assert "calculator" in cps1[0].expected_window.lower()

        g2 = "/boost /goal /turns 20 open notepad, write notes, close"
        cps2 = AwarenessRouter.decompose_goal(g2)
        assert cps2[0].action_type == "launch"
        assert "notepad" in cps2[0].expected_window.lower()

    def test_cognitive_verifier_rejects_already_active_window(self):
        """SemanticStateVerifier rejects launch_app when app is already in foreground window."""
        pre_snap = {"tokens": {"calculator", "standard"}, "window_title": "Calculator", "hwnd": 100}
        post_snap = {"tokens": {"calculator", "standard"}, "window_title": "Calculator", "hwnd": 100}

        ver = SemanticStateVerifier.verify_transition(
            pre_snap=pre_snap,
            post_snap=post_snap,
            action="launch_app",
            input_text="calc",
            semantic_intention="Launch Calculator",
            expected_transition="Calculator opens"
        )
        assert ver["verified"] is False
        assert ver["verdict"] == "ALREADY_ACTIVE"
        assert "already active" in ver["reason"]

    def test_cognitive_verifier_rejects_formula_launch(self):
        """SemanticStateVerifier rejects formula strings like '2+2' with INVALID_ACTION."""
        pre_snap = {"tokens": {"calculator"}, "window_title": "Calculator", "hwnd": 100}
        post_snap = {"tokens": {"calculator"}, "window_title": "Calculator", "hwnd": 100}

        ver = SemanticStateVerifier.verify_transition(
            pre_snap=pre_snap,
            post_snap=post_snap,
            action="launch_app",
            input_text="2+2=",
            semantic_intention="Launch 2+2=",
            expected_transition="2+2= opens"
        )
        assert ver["verified"] is False
        assert ver["verdict"] == "INVALID_ACTION"
