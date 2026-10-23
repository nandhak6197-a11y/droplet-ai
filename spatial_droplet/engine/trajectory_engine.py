"""
Autonomous Multi-Turn Trajectory Engine (Phase 5, 6, 7 & 8).
Executes closed-loop goal trajectories with Dynamic Re-Spraying,
Cognitive Screen Awareness, and Semantic State Verification.
"""

import time
import re
import ctypes
import inspect
import asyncio
import threading
from typing import Optional, List, Callable, Any, Tuple, Dict

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

from ..core.models import DiagonalDroplet, WindowContext, TrajectoryStep, ScreenAwarenessContext
from ..core.memory import EpisodicTrajectoryMemory
from ..platform.window import get_active_window_context
from ..platform.actuator import PrecisionActuator, Win32HardwareActuator
from ..vision.grabber import FastScreenGrabber
from ..vision.sprayer import FullBleedLatticeSprayer
from ..vision.screen_parser import ScreenStateParser
from ..verification.semantic_verifier import SemanticStateVerifier
from ..reasoning.reasoner import DiagonalDropletReasoner


class AutonomousTrajectoryEngine:
    """
    Autonomous Multi-Turn Trajectory Engine.
    Executes closed-loop goal trajectories with Dynamic Re-Spraying:
      Turn K: Settle/Observe fresh frame -> Spray fresh droplets -> Reason turn with Groq ->
              Act via Precision/Win32 Actuator -> Verify visual delta -> Settle & Repeat.
    Eliminates stale coordinates by dynamically re-capturing and re-spraying every turn.
    Maintains chronological action & verification histories within EpisodicTrajectoryMemory.
    """

    def __init__(
        self,
        reasoner: DiagonalDropletReasoner,
        sprayer: FullBleedLatticeSprayer,
        actuator: PrecisionActuator,
        hud: Optional[Any] = None
    ):
        self.reasoner = reasoner
        self.sprayer = sprayer
        self.actuator = actuator
        self.hud = hud

    def _call_decide_turn(
        self,
        goal: str,
        memory: EpisodicTrajectoryMemory,
        droplets: List[DiagonalDroplet],
        win_ctx: Optional[WindowContext],
        screen_awareness: Optional[ScreenAwarenessContext] = None,
        loop_intervention: Optional[str] = None,
        frame: Optional[Any] = None
    ) -> Dict:
        kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(self.reasoner.decide_turn)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if screen_awareness is not None and ("screen_awareness" in params or has_var_kw):
                kwargs["screen_awareness"] = screen_awareness
            if loop_intervention is not None and ("loop_intervention" in params or has_var_kw):
                kwargs["loop_intervention"] = loop_intervention
            if frame is not None and ("screenshot" in params or has_var_kw):
                kwargs["screenshot"] = frame
        except Exception:
            if screen_awareness is not None:
                kwargs["screen_awareness"] = screen_awareness
            if loop_intervention is not None:
                kwargs["loop_intervention"] = loop_intervention
            if frame is not None:
                kwargs["screenshot"] = frame
        return self.reasoner.decide_turn(goal, memory, droplets, win_ctx, **kwargs)

    @staticmethod
    def _find_primary_dialog_commit(
        droplet_map: Dict[int, DiagonalDroplet],
        screen_awareness: Optional[ScreenAwarenessContext],
        win_ctx: Optional[WindowContext]
    ) -> Optional[DiagonalDroplet]:
        """
        Generalized heuristic discovery of primary commit/confirmation control
        in active dialogs, file/folder pickers, or confirmation sheets.
        """
        if not droplet_map:
            return None

        # Determine reference bounding box for dialog
        dialog_rect = None
        if screen_awareness and screen_awareness.active_modal and screen_awareness.active_modal.rect:
            dialog_rect = screen_awareness.active_modal.rect
        elif screen_awareness and screen_awareness.modal_context and screen_awareness.modal_context.rect:
            dialog_rect = screen_awareness.modal_context.rect
        elif win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop":
            # Check if current window looks like a dialog
            w = win_ctx.rect[2] - win_ctx.rect[0]
            h = win_ctx.rect[3] - win_ctx.rect[1]
            if w < 1200 and h < 900:  # Typical dialog dimensions
                dialog_rect = win_ctx.rect

        COMMIT_PRIORITIES = [
            # Highest priority: explicit multi-word commit phrases
            ("select folder", 1000),
            ("save as", 950),
            ("open folder", 950),
            # Standard commit verbs
            ("select", 900),
            ("open", 900),
            ("save", 900),
            ("choose", 850),
            ("confirm", 850),
            ("commit", 850),
            ("ok", 850),
            ("yes", 800),
            ("done", 800),
            ("finish", 800),
            ("proceed", 800),
            ("continue", 800),
            ("next", 750),
            ("install", 750),
            ("apply", 750),
            ("submit", 750),
            ("create", 700),
            ("folder", 600),
        ]

        best_droplet = None
        best_score = -1

        for d in droplet_map.values():
            if getattr(d, 'category', '') == 'surface_probe':
                continue
            raw_lbl = getattr(d, 'label', '') or ""
            lbl = raw_lbl.strip().lower()
            if not lbl:
                continue

            # Strip archetype prefixes like [btn] or [icon]
            clean_btn = re.sub(r'\[(?:btn|icon|app)\]\s*', '', lbl).strip()

            # Skip clear dismissal / cancel tokens
            if any(clean_btn == d_word or clean_btn.startswith(d_word + " ") for d_word in ["cancel", "no", "abort", "close", "dont save", "don't save"]):
                continue

            # Evaluate commit keyword matches
            match_score = 0
            for token, score in COMMIT_PRIORITIES:
                if token == clean_btn:
                    match_score = max(match_score, score + 200)  # Exact match bonus
                elif token in clean_btn:
                    match_score = max(match_score, score)

            if match_score <= 0:
                continue

            # Positional score: dialogs place primary commit buttons in bottom and bottom-right
            spatial_bonus = 0
            if dialog_rect:
                x1, y1, x2, y2 = dialog_rect
                dw = max(1, x2 - x1)
                dh = max(1, y2 - y1)
                # Inside dialog bounds?
                if (x1 - 20 <= d.center_x <= x2 + 20) and (y1 - 20 <= d.center_y <= y2 + 20):
                    spatial_bonus += 300
                    # Bottom half bonus
                    if d.center_y >= y1 + dh * 0.5:
                        spatial_bonus += 200
                    # Bottom-right quadrant bonus
                    if d.center_x >= x1 + dw * 0.5 and d.center_y >= y1 + dh * 0.6:
                        spatial_bonus += 250
                else:
                    # Outside dialog bounds - heavy penalty
                    spatial_bonus -= 500

            total_score = match_score + spatial_bonus
            if total_score > best_score:
                best_score = total_score
                best_droplet = d

        return best_droplet

    def run_trajectory(
        self,
        goal: str,
        initial_droplets: Optional[List[DiagonalDroplet]] = None,
        initial_win_ctx: Optional[WindowContext] = None,
        max_turns: int = 50,
        status_callback: Optional[Callable[[str, str], None]] = None,
        render_callback: Optional[Callable[[List[DiagonalDroplet], Optional[WindowContext]], None]] = None,
        abort_event: Optional[threading.Event] = None
    ) -> EpisodicTrajectoryMemory:
        memory = EpisodicTrajectoryMemory(goal=goal, max_steps=max_turns)
        current_droplets = initial_droplets or []
        current_win_ctx = initial_win_ctx
        frame = None
        target_fail_counts: Dict[str, int] = {}
        action_fail_counts: Dict[str, int] = {}
        recent_action_sigs: List[Tuple[str, str, str, str, str, str]] = []
        failed_status_count = 0
        last_loop_turn = -10

        def _notify_status(text: str, color: str = "#00e5ff", details: Optional[str] = None):
            if status_callback:
                try:
                    import inspect
                    sig = inspect.signature(status_callback)
                    if len(sig.parameters) >= 3 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        status_callback(text, color, details=details)
                    else:
                        status_callback(text, color)
                except Exception:
                    try:
                        status_callback(text, color)
                    except Exception:
                        pass

        print(f"\n" + "=" * 76)
        print(f"[Trajectory Engine] 🚀 Initiating Closed-Loop Trajectory for Goal: \"{goal}\"")
        print(f"                     Max Safety Cap: {max_turns} turns | Dynamic Re-Spraying Active")
        print("=" * 76)

        if hasattr(self.sprayer, 'reset_tracking'):
            self.sprayer.reset_tracking()
        if self.hud:
            try:
                self.hud.clear_target_reticle()
            except Exception:
                pass

        for turn in range(1, max_turns + 1):
            if abort_event and abort_event.is_set():
                print(f"[Trajectory Engine] 🛑 Trajectory manually cancelled by user via abort event.")
                _notify_status("🛑 Trajectory Cancelled", "#ff1744")
                memory.completion_summary = "Cancelled by user via ESC"
                break

            print(f"\n--- [Turn {turn}/{max_turns}] ----------------------------------------------------")
            print(f"🧠 [Trajectory Engine] Consulting Cognitive AI Reasoner for Turn {turn}...")
            _notify_status(f"🧠 Turn {turn}/{max_turns}: Reasoning...", "#00e5ff")

            # Turn > 1 or missing frame: Settle screen & Dynamic Re-Spraying (fresh frame + fresh droplets)
            if turn > 1 or not current_droplets or frame is None:
                if turn > 1:
                    time.sleep(0.35)  # Allow UI animations, page loads, and menu transitions to settle
                if self.hud:
                    try:
                        self.hud.clear_target_reticle()
                        self.hud.clear()
                    except Exception:
                        pass
                user32 = ctypes.windll.user32
                if current_win_ctx and current_win_ctx.hwnd and current_win_ctx.title != "Desktop":
                    if not user32.IsWindow(current_win_ctx.hwnd):
                        print(f"[Trajectory Engine] ⚠️ Active window {current_win_ctx.hwnd} closed during trajectory ➜ Re-grounding to new focus...")
                current_win_ctx = get_active_window_context()
                frame = FastScreenGrabber.get_instance().grab_full()
                if turn > 1 or not current_droplets:
                    print(f"[Trajectory Engine] 📸 Re-capturing fresh screen & re-spraying droplets (Turn {turn})...")
                    current_droplets = asyncio.run(self.sprayer.spray(frame, current_win_ctx))
                    if render_callback:
                        try:
                            render_callback(current_droplets, current_win_ctx)
                        except Exception:
                            pass

            droplet_map = {d.id: d for d in current_droplets}

            # Cognitive Screen Awareness Extraction (Phase 6)
            ocr_lines = getattr(self.sprayer, 'last_ocr_lines', [])
            screen_awareness = ScreenStateParser.parse_screen_awareness(current_win_ctx, ocr_lines, current_droplets)
            pre_snap = SemanticStateVerifier.capture_snapshot(current_win_ctx)

            if screen_awareness.active_modal:
                print(f"[Trajectory Engine] ⛶ Active Modal Detected: '{screen_awareness.active_modal.title}'")
                _notify_status(f"⛶ Modal: {screen_awareness.active_modal.title[:16]}", "#ffb300")

            # Reason next step using live screen awareness & trajectory memory
            decision = self._call_decide_turn(
                goal=goal,
                memory=memory,
                droplets=current_droplets,
                win_ctx=current_win_ctx,
                screen_awareness=screen_awareness,
                frame=frame
            )

            if "error" in decision:
                err_msg = decision["error"]
                print(f"[Trajectory Engine] ⚠️ Groq/LLM Reasoner Error on Turn {turn}: {err_msg}")
                _notify_status(f"⚠️ AI Error: {err_msg[:24]}", "#ffb300")
                time.sleep(1.0)
                win_title = current_win_ctx.title if (current_win_ctx and current_win_ctx.is_valid) else "Desktop"
                step = TrajectoryStep(
                    step_num=turn,
                    goal=goal,
                    subgoal=f"Turn {turn}",
                    action="wait",
                    target_id=None,
                    target_label="None",
                    input_text=None,
                    verified=False,
                    verification_reason=f"Transient reasoner error: {err_msg}. Retrying next turn.",
                    active_window=win_title,
                    semantic_intention="Retry turn after reasoner transient error",
                    expected_transition="None",
                    verdict="NO_EFFECT"
                )
                memory.add_step(step)
                continue

            status = decision.get("status", "in_progress")
            # The reasoner may expose the action under either schema name; never
            # silently default a malformed response to a click.
            raw_action = decision.get("action") or decision.get("action_type")
            action = str(raw_action or "").strip().lower()

            # Canonical normalization of human action names & key aliases
            if action in ("enter", "press_enter", "return"):
                action = "press"
                if not decision.get("key"):
                    decision["key"] = "enter"
            elif action in ("esc", "escape", "press_esc"):
                action = "press"
                if not decision.get("key"):
                    decision["key"] = "esc"
            elif action in ("tab", "press_tab"):
                action = "press"
                if not decision.get("key"):
                    decision["key"] = "tab"
            elif action in ("backspace", "press_backspace"):
                action = "press"
                if not decision.get("key"):
                    decision["key"] = "backspace"
            elif action in ("type_text", "write_text"):
                action = "type"
            elif action in ("minimize", "minimize_window"):
                action = "minimize"
            elif action in ("maximize", "maximize_window"):
                action = "maximize"

            allowed_actions = {
                "click", "single_click", "double_click", "triple_click",
                "right_click", "context_click", "middle_click", "hover",
                "move_to", "drag", "drag_to", "drag_and_drop", "free_drag", "drag_area",
                "scroll", "system_scroll", "wheel", "type", "fill", "press", "hotkey",
                "chord", "key_down", "key_up", "clear", "check", "uncheck",
                "select_option", "wait", "open_path", "system_open",
                "system_open_path", "launch_app", "stop", "finish",
                "clear_and_type", "replace_text", "clear_type",
                "type_and_enter", "submit", "click_and_type",
                "ctrl_click", "control_click", "shift_click", "alt_click", "modifier_click",
                "hover_and_wait", "hover_wait",
                "copy", "paste", "select_all", "undo", "redo", "save", "find",
                "close_tab", "new_tab", "switch_window", "win_run",
                "minimize", "minimize_window", "maximize", "maximize_window"
            }
            if action not in allowed_actions:
                print(f"[Trajectory Engine] ⚠️ Rejecting unrecognized action: {raw_action!r}; re-prompting on next turn.")
                _notify_status(f"⚠️ Invalid action: {raw_action!r}", "#ffb300")
                win_title = current_win_ctx.title if (current_win_ctx and current_win_ctx.is_valid) else "Desktop"
                step = TrajectoryStep(
                    step_num=turn,
                    goal=goal,
                    subgoal=f"Turn {turn}",
                    action="wait",
                    target_id=None,
                    target_label="None",
                    input_text=None,
                    verified=False,
                    verification_reason=f"Action {raw_action!r} is not an allowed action.",
                    active_window=win_title,
                    semantic_intention="Recover from invalid action",
                    expected_transition="None",
                    verdict="EXECUTION_FAILED"
                )
                memory.add_step(step)
                continue
            subgoal = decision.get("next_subgoal", f"Step {turn}")
            target_id = decision.get("target_dot_id")
            if target_id is None:
                target_id = decision.get("target_droplet_id")
            if target_id is not None:
                if isinstance(target_id, str):
                    m_tid = re.search(r'\d+', target_id)
                    target_id = int(m_tid.group(0)) if m_tid else None
                else:
                    try:
                        target_id = int(target_id)
                    except (ValueError, TypeError):
                        target_id = None
            input_text = decision.get("input_text")
            key_combo = decision.get("key_combo")
            semantic_intention = decision.get("semantic_intention") or decision.get("intention") or subgoal
            expected_transition = decision.get("expected_transition") or "Screen updates accordingly"
            reasoning = decision.get("reasoning", "")

            # Unpack 5-stage deep CoT deliberation with fallback to legacy keys
            delib = decision.get("deliberation") if isinstance(decision.get("deliberation"), dict) else {}
            user_intent = decision.get("user_intent_and_success_criteria") or delib.get("user_intent_and_success_criteria") or decision.get("intent_analysis") or delib.get("intent_analysis") or ""
            screen_hierarchy = decision.get("screen_state_and_visual_hierarchy") or delib.get("screen_state_and_visual_hierarchy") or decision.get("observation_analysis") or delib.get("observation_analysis") or ""
            mistake_critique = decision.get("mistake_critique_and_loop_diagnosis") or delib.get("mistake_critique_and_loop_diagnosis") or decision.get("critique_of_previous_step") or delib.get("critique_of_previous_step") or ""
            master_plan = decision.get("end_to_end_master_plan") or delib.get("end_to_end_master_plan") or []
            if isinstance(master_plan, str):
                master_plan = [p.strip() for p in master_plan.splitlines() if p.strip()]
            tactical_just = decision.get("tactical_action_justification") or delib.get("tactical_action_justification") or decision.get("candidate_evaluation") or delib.get("candidate_evaluation") or ""

            observation_analysis = screen_hierarchy or decision.get("observation_analysis", "") or delib.get("observation_analysis", "")
            critique = mistake_critique or decision.get("critique_of_previous_step", "") or delib.get("critique_of_previous_step", "")
            cand_eval = tactical_just or decision.get("candidate_evaluation", "") or delib.get("candidate_evaluation", "")
            strategy = (" | ".join(master_plan) if isinstance(master_plan, list) else str(master_plan)) or decision.get("strategy_plan", "") or delib.get("strategy_plan", "")
            model_used = decision.get("model_used", "Groq")

            print(f"[Trajectory Turn {turn}] Deliberative Cognitive Decision ({model_used}):")
            if user_intent:
                print(f"  • User Intent: {user_intent}")
            if screen_hierarchy:
                print(f"  • Screen State:{screen_hierarchy}")
            if mistake_critique:
                print(f"  • Critique:    {mistake_critique}")
            if master_plan:
                p_disp = master_plan if isinstance(master_plan, list) else [str(master_plan)]
                print(f"  • Master Plan: {' -> '.join(p_disp)}")
            if tactical_just:
                print(f"  • Justification:{tactical_just}")
            print(f"  • Scene:       {screen_awareness.scene_type} in '{screen_awareness.app_title}'")
            print(f"  • Status:      {status}")
            print(f"  • Sub-goal:    {subgoal}")
            print(f"  • Intention:   {semantic_intention}")
            print(f"  • Expected:    {expected_transition}")
            print(f"  • Action:      {action.upper()} (Target: Dot [{target_id}], Input: {repr(input_text)})")

            # Check termination
            if action in ("stop", "finish") or status == "completed":
                print(f"\n[Trajectory Engine] ✔ Goal Completed autonomously at Turn {turn}!")
                print(f"                    Summary: {reasoning}")
                memory.is_completed = True
                memory.completion_summary = reasoning or "Goal successfully completed."
                _notify_status(f"✔ Completed: {subgoal[:20]}", "#3fb950")
                break

            if status == "failed":
                if failed_status_count == 0 and turn < max_turns - 5:
                    failed_status_count += 1
                    print(f"[Trajectory Engine] ⚠️ Reasoner flagged status='failed' at Turn {turn}/{max_turns}; requesting re-deliberation on fallback paths before accepting defeat.")
                    _notify_status(f"⚠️ Re-evaluating fallback paths...", "#ffb300")
                    win_title = current_win_ctx.title if (current_win_ctx and current_win_ctx.is_valid) else "Desktop"
                    step = TrajectoryStep(
                        step_num=turn,
                        goal=goal,
                        subgoal=subgoal,
                        action="wait",
                        target_id=None,
                        target_label="None",
                        input_text=None,
                        verified=False,
                        verification_reason=f"Model reported failure: {reasoning}. Attempting recovery and alternate strategies.",
                        active_window=win_title,
                        semantic_intention="Re-evaluate fallback tactics",
                        expected_transition="None",
                        verdict="NO_EFFECT",
                        observation_analysis=observation_analysis,
                        critique=critique,
                        strategy_plan=strategy,
                        master_plan=master_plan,
                        intent_analysis=user_intent,
                        mistake_critique=mistake_critique
                    )
                    memory.add_step(step)
                    continue
                else:
                    print(f"\n[Trajectory Engine] ❌ Reasoner confirmed goal as unachievable/failed: {reasoning}")
                    memory.completion_summary = f"Goal failed: {reasoning}"
                    _notify_status(f"❌ Failed: {reasoning[:20]}", "#ff1744")
                    break

            # Execute the turn action
            target_node = droplet_map.get(target_id)

            target_label = target_node.label if target_node else (f"Dot [{target_id}]" if target_id is not None else action)
            win_title = current_win_ctx.title if (current_win_ctx and current_win_ctx.is_valid) else "Desktop"

            # Determine active plan phase for UI status banner
            active_phase = ""
            if master_plan and isinstance(master_plan, list):
                for p in master_plan:
                    p_str = str(p).strip()
                    p_up = p_str.upper()
                    if any(k in p_up for k in ("ACTIVE", "IN PROGRESS", "►", "CURRENT")):
                        active_phase = p_str
                        break
                if not active_phase and master_plan:
                    active_phase = str(master_plan[0]).strip()

            details_str = ""
            if master_plan:
                p_disp = master_plan if isinstance(master_plan, list) else [str(master_plan)]
                details_str = "Master Plan:\n" + "\n".join(f"  • {p}" for p in p_disp)
                if tactical_just:
                    details_str += f"\n\nJustification: {tactical_just}"
            elif tactical_just:
                details_str = f"Justification: {tactical_just}"

            if active_phase:
                clean_phase = re.sub(r'[\(\[]?(?:ACTIVE|IN PROGRESS|CURRENT|PENDING|DONE|COMPLETED)[\)\]]?', '', active_phase, flags=re.IGNORECASE).strip()
                _notify_status(f"Turn {turn}: {clean_phase[:36]}...", "#00e5ff", details=details_str or active_phase)
            else:
                _notify_status(f"Turn {turn}: {semantic_intention[:24]}...", "#00e5ff", details=details_str or semantic_intention)

            drag_dest = None
            dest_id = decision.get("drag_destination_dot_id")
            if dest_id is not None:
                if isinstance(dest_id, str):
                    m_did = re.search(r'\d+', dest_id)
                    dest_id = int(m_did.group(0)) if m_did else None
                else:
                    try:
                        dest_id = int(dest_id)
                    except (ValueError, TypeError):
                        dest_id = None
            if dest_id is not None:
                dest_node = droplet_map.get(dest_id)
                if dest_node:
                    drag_dest = (dest_node.center_x, dest_node.center_y)
            elif decision.get("drag_dest_x") is not None and decision.get("drag_dest_y") is not None:
                drag_dest = (int(decision.get("drag_dest_x")), int(decision.get("drag_dest_y")))
            elif decision.get("drag_dest"):
                drag_dest = tuple(decision.get("drag_dest"))
                
            target_x = None
            target_y = None
            if target_node:
                target_x = target_node.center_x
                target_y = target_node.center_y
            elif decision.get("x") is not None and decision.get("y") is not None:
                target_x = int(decision.get("x"))
                target_y = int(decision.get("y"))

            key_single = decision.get("key")
            scroll_dir = decision.get("scroll_direction", "vertical")
            scroll_amt = decision.get("scroll_amount")

            clean_tgt = ""
            if target_node:
                clean_tgt = re.sub(r'\[item \d+ of \d+.*?\]', '', target_node.label.lower()).strip()
                clean_tgt = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean_tgt).strip()
            elif target_label:
                clean_tgt = re.sub(r'\[item \d+ of \d+.*?\]', '', target_label.lower()).strip()
                clean_tgt = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean_tgt).strip()

            target_semantic = clean_tgt or (f"dot_{target_id}" if target_id is not None else "")
            target_key = clean_tgt or (str(input_text).strip().lower() if action in ('open_path', 'system_open_path', 'type') else "") or (f"dot_{target_id}" if target_id is not None else (key_single or str(key_combo) or action))

            current_sig = (
                action.lower().strip(),
                target_semantic,
                clean_tgt,
                str(input_text).strip().lower() if input_text else "",
                str(key_single).strip().lower() if key_single else "",
                str(key_combo).lower() if key_combo else ""
            )

            # ---------------------------------------------------------------
            # Dynamic Loop Guard: Self-Healing In-Turn Cognitive Auto-Pivot (<= 2 Turns)
            # ---------------------------------------------------------------
            commit_droplet = None
            loop_detected = False
            loop_reason = ""

            # 1. Ping-Pong Loop Detection (Strict <= 2 turns / 1 cycle):
            # In sequence A -> B -> A, Turn 3 proposes A again after B.
            # Engages immediately on Turn 3 when alternating oscillation is attempted.
            if len(recent_action_sigs) >= 2 and recent_action_sigs[-2] == current_sig and recent_action_sigs[-1] != current_sig:
                loop_detected = True
                loop_reason = (
                    f"Alternating ping-pong oscillation detected: '{current_sig[0]}' ➜ "
                    f"'{recent_action_sigs[-1][0]}' ➜ '{current_sig[0]}' within <= 2 turns without forward progress."
                )

            # 2. Direct Repetition Failure Guard (Strict <= 2 consecutive turns):
            # If immediately preceding turn chose identical action and failed verification:
            elif len(recent_action_sigs) >= 1 and recent_action_sigs[-1] == current_sig:
                prev_step = memory.steps[-1] if memory.steps else None
                prev_unverified = prev_step and (not prev_step.verified or prev_step.verdict in ("NO_EFFECT", "EXECUTION_FAILED"))
                if prev_unverified:
                    loop_detected = True
                    loop_reason = (
                        f"Action '{action}' on '{target_semantic or action}' was repeated after failing verification "
                        f"in the immediately preceding turn (<= 2 consecutive unverified turns)."
                    )
                elif len(recent_action_sigs) >= 2 and recent_action_sigs[-2] == current_sig:
                    loop_detected = True
                    loop_reason = (
                        f"Action '{action}' on '{target_semantic or action}' was chosen 2 consecutive times "
                        f"without verified forward state change."
                    )

            # 3. Target / Action Stagnation Guard (Strict <= 2 consecutive failures):
            elif not loop_detected and target_key and (
                action_fail_counts.get(f"{action}:{target_key}", 0) >= 2 or target_fail_counts.get(target_key, 0) >= 2
            ):
                act_fails = action_fail_counts.get(f"{action}:{target_key}", 0)
                tgt_fails = target_fail_counts.get(target_key, 0)
                loop_detected = True
                loop_reason = (
                    f"Target '{target_key}' has failed verification {max(act_fails, tgt_fails)} consecutive times (>= 2). "
                    f"The control is unresponsive, occluded, or refusing input."
                )

            # 4. Overall Consecutive Failure Guard (Strict <= 2 consecutive turns):
            elif not loop_detected and memory.get_consecutive_failures() >= 2 and (turn - last_loop_turn > 1):
                loop_detected = True
                loop_reason = (
                    f"2 consecutive turns failed to produce verified UI transitions. "
                    f"The interaction tactic is stalled."
                )

            # 5. Guard against repeated ungrounded switch_window / Alt+Tab:
            elif not loop_detected and action == "switch_window":
                sw_count = sum(1 for sig in recent_action_sigs[-2:] if sig[0] == "switch_window")
                if sw_count >= 1 and not target_node and not input_text:
                    loop_detected = True
                    loop_reason = "Repeated ungrounded 'switch_window' without target droplet or application title."

            # 6. Guard against consecutive or ungrounded idle wait actions:
            elif not loop_detected and action == "wait":
                prev_wait = bool(recent_action_sigs and recent_action_sigs[-1][0] == "wait")
                if not prev_wait and memory.steps and memory.steps[-1].action == "wait":
                    prev_wait = True
                if prev_wait or memory.get_consecutive_failures() >= 1:
                    loop_detected = True
                    loop_reason = "Consecutive or ungrounded 'wait' action chosen without active UI transition."

            if loop_detected:
                last_loop_turn = turn
                print(f"[Trajectory Loop Guard] ⚠️ Cognitive Loop Detected on Turn {turn}: {loop_reason}")
                print(f"[Trajectory Loop Guard] 🔄 Initiating in-turn cognitive re-deliberation & self-healing auto-pivot...")
                _notify_status("🔄 Auto-pivot re-deliberation", "#ffb300")

                critique_note = f"\nDIAGNOSED PREVIOUS MISTAKE: {mistake_critique}\n" if mistake_critique else ""
                intervention_msg = (
                    f"CRITICAL COGNITIVE RE-DELIBERATION REQUIRED (Turn {turn}/{max_turns}):\n"
                    f"LOOP / STAGNATION DIAGNOSIS: {loop_reason}\n"
                    f"{critique_note}\n"
                    f"MANDATES FOR THIS PIVOTED DECISION:\n"
                    f"1. You are CONSTITUTIONALLY FORBIDDEN from choosing action '{action}' on '{target_semantic or action}'.\n"
                    f"2. You MUST NOT repeat any action from the recent oscillation: '{current_sig[0]}' or '{recent_action_sigs[-1][0] if recent_action_sigs else 'None'}'.\n"
                    f"3. DIALOG / MODAL / PICKER PROTOCOL:\n"
                    f"   - If an active modal dialog, confirmation sheet, or file/folder picker is open (e.g. 'Open Folder', 'Select Folder', 'Save', 'Open', 'Save As', 'OK', 'Confirm'):\n"
                    f"     DO NOT touch the file/folder list item or repeat keyboard Enter!\n"
                    f"     You MUST click the dialog's primary commit button ('Select Folder', 'Open', 'Save', 'OK', 'Done', 'Apply') at the bottom right of the dialog, OR focus the address bar / path input field.\n"
                    f"4. GENERAL APPLICATION PROTOCOL:\n"
                    f"   - If not in a dialog, switch interaction modality entirely: cycle focus via Tab, press Escape to dismiss stuck popups, or use keyboard navigation/hotkeys.\n"
                    f"5. Formulate a revised end_to_end_master_plan and explain in tactical_action_justification why your new target and action break the deadlock."
                )

                pivoted_decision = self._call_decide_turn(
                    goal=goal,
                    memory=memory,
                    droplets=current_droplets,
                    win_ctx=current_win_ctx,
                    screen_awareness=screen_awareness,
                    loop_intervention=intervention_msg,
                    frame=frame
                )

                # Evaluate pivoted decision
                pivoted_action = str(pivoted_decision.get("action") or pivoted_decision.get("action_type") or "").strip().lower()
                piv_target_id = pivoted_decision.get("target_dot_id")
                if piv_target_id is None:
                    piv_target_id = pivoted_decision.get("target_droplet_id")
                piv_node = droplet_map.get(piv_target_id)
                piv_clean_tgt = ""
                if piv_node:
                    piv_clean_tgt = re.sub(r'\[item \d+ of \d+.*?\]', '', getattr(piv_node, 'label', '').lower()).strip()
                    piv_clean_tgt = re.sub(r'\[(?:btn|icon|app)\]\s*', '', piv_clean_tgt).strip()
                piv_semantic = piv_clean_tgt or (f"dot_{piv_target_id}" if piv_target_id is not None else "")
                piv_sig = (
                    pivoted_action,
                    piv_semantic,
                    piv_clean_tgt,
                    str(pivoted_decision.get("input_text") or "").strip().lower(),
                    str(pivoted_decision.get("key") or "").strip().lower(),
                    str(pivoted_decision.get("key_combo") or "").lower()
                )

                # Check if model successfully pivoted to a new, non-looping signature
                if (
                    pivoted_action in allowed_actions
                    and piv_sig != current_sig
                    and (not recent_action_sigs or piv_sig != recent_action_sigs[-1])
                    and "error" not in pivoted_decision
                ):
                    print(f"[Trajectory Loop Guard] 🔀 Auto-Pivot SUCCESS: Model shifted to '{pivoted_action.upper()}' on '{piv_semantic or 'None'}'")
                    decision = pivoted_decision
                    status = decision.get("status", "in_progress")
                    action = pivoted_action
                    subgoal = decision.get("next_subgoal", subgoal)
                    target_id = piv_target_id
                    target_node = piv_node
                    target_label = target_node.label if target_node else (f"Dot [{target_id}]" if target_id is not None else action)
                    input_text = decision.get("input_text")
                    key_combo = decision.get("key_combo")
                    key_single = decision.get("key")
                    semantic_intention = decision.get("semantic_intention") or decision.get("intention") or subgoal
                    expected_transition = decision.get("expected_transition") or "Screen updates accordingly"
                    reasoning = decision.get("reasoning", "")
                    piv_delib = decision.get("deliberation") if isinstance(decision.get("deliberation"), dict) else {}
                    user_intent = decision.get("user_intent_and_success_criteria") or piv_delib.get("user_intent_and_success_criteria") or decision.get("intent_analysis") or piv_delib.get("intent_analysis") or user_intent
                    screen_hierarchy = decision.get("screen_state_and_visual_hierarchy") or piv_delib.get("screen_state_and_visual_hierarchy") or decision.get("observation_analysis") or piv_delib.get("observation_analysis") or screen_hierarchy
                    mistake_critique = decision.get("mistake_critique_and_loop_diagnosis") or piv_delib.get("mistake_critique_and_loop_diagnosis") or decision.get("critique_of_previous_step") or piv_delib.get("critique_of_previous_step") or mistake_critique
                    piv_plan = decision.get("end_to_end_master_plan") or piv_delib.get("end_to_end_master_plan")
                    if piv_plan:
                        master_plan = [p.strip() for p in piv_plan.splitlines() if p.strip()] if isinstance(piv_plan, str) else list(piv_plan)
                    tactical_just = decision.get("tactical_action_justification") or piv_delib.get("tactical_action_justification") or decision.get("candidate_evaluation") or piv_delib.get("candidate_evaluation") or tactical_just
                    observation_analysis = screen_hierarchy or decision.get("observation_analysis", "") or piv_delib.get("observation_analysis", "")
                    critique = mistake_critique or decision.get("critique_of_previous_step", "") or piv_delib.get("critique_of_previous_step", "")
                    cand_eval = tactical_just or decision.get("candidate_evaluation", "") or piv_delib.get("candidate_evaluation", "")
                    strategy = (" | ".join(master_plan) if isinstance(master_plan, list) else str(master_plan)) or decision.get("strategy_plan", "") or piv_delib.get("strategy_plan", "")
                    clean_tgt = piv_clean_tgt
                    target_semantic = piv_semantic
                    target_key = clean_tgt or (str(input_text).strip().lower() if action in ('open_path', 'system_open_path', 'type') else "") or (f"dot_{target_id}" if target_id is not None else (key_single or str(key_combo) or action))
                    current_sig = piv_sig
                    if target_node:
                        target_x = target_node.center_x
                        target_y = target_node.center_y
                    elif decision.get("x") is not None and decision.get("y") is not None:
                        target_x = int(decision.get("x"))
                        target_y = int(decision.get("y"))
                    else:
                        target_x = None
                        target_y = None
                else:
                    # Heuristic Smart Tactical Escape: When reasoner still returns the same blocked action or errors
                    print(f"[Trajectory Loop Guard] 🛡️ Model deliberation did not break deadlock; engaging generalized smart tactical escape...")
                    commit_droplet = self._find_primary_dialog_commit(droplet_map, screen_awareness, current_win_ctx)
                    if commit_droplet:
                        print(f"[Trajectory Loop Guard] 🎯 Auto-targeting primary dialog commit control: '{commit_droplet.label}' (Dot [{commit_droplet.id}])")
                        action = "click"
                        target_node = commit_droplet
                        target_id = commit_droplet.id
                        target_label = commit_droplet.label
                        target_x = commit_droplet.center_x
                        target_y = commit_droplet.center_y
                        input_text = None
                        key_single = None
                        key_combo = None
                        semantic_intention = f"Smart Escape: Clicking primary commit control '{commit_droplet.label}'"
                        expected_transition = "Dialog commits and dismisses"
                        subgoal = f"Confirm dialog via {commit_droplet.label}"
                        clean_tgt = re.sub(r'\[(?:btn|icon|app)\]\s*', '', commit_droplet.label.lower()).strip()
                        target_semantic = clean_tgt
                        target_key = clean_tgt
                        current_sig = (action, target_semantic, clean_tgt, "", "", "")
                    else:
                        # Non-dialog or no commit droplet found: cycle focus to escape keyboard trap
                        print(f"[Trajectory Loop Guard] 🎯 Cycling focus via 'tab' key to escape focus trap.")
                        action = "press"
                        target_node = None
                        target_id = None
                        target_label = "Keyboard [tab]"
                        target_x = None
                        target_y = None
                        input_text = None
                        key_single = "tab"
                        key_combo = None
                        semantic_intention = "Smart Escape: Cycle focus with Tab to escape UI deadlock"
                        expected_transition = "Focus shifts to next actionable UI control"
                        subgoal = "Cycle focus to recover interaction"
                        target_semantic = "key_tab"
                        target_key = "tab"
                        current_sig = (action, target_semantic, "", "", "tab", "")

                # Dampen failure counts to allow the pivoted action to execute and verify
                if target_key in target_fail_counts:
                    target_fail_counts[target_key] = max(0, target_fail_counts[target_key] - 1)
                action_fail_counts.clear()

            act_res = None
            if target_node or (target_x is not None and target_y is not None):
                if self.hud and target_node:
                    try:
                        self.hud.highlight_target(target_node)
                    except Exception:
                        pass

                # Contextual double-click vs single-click upgrade:
                # Upgrade to double_click if open/launch intent is present for desktop apps or file/folder items
                is_open_intent = any(k in (goal.lower() + " " + subgoal.lower()) for k in ['open', 'launch', 'start', 'run', 'double'])
                is_desktop_ctx = (not current_win_ctx) or (not current_win_ctx.is_valid) or (current_win_ctx.title == "Desktop")
                
                is_file_or_folder = False
                if target_node and hasattr(target_node, 'label'):
                    is_file_or_folder = any(k in target_node.label.lower() for k in ['folder', 'file', '.exe', '.txt', '.pdf', '.docx', '.png', '.jpg', 'directory'])
                
                if action == 'click' and is_open_intent and target_node and not (commit_droplet and target_node == commit_droplet):
                    if (is_desktop_ctx and getattr(target_node, 'category', '') == 'interactive_app') or is_file_or_folder:
                        action = 'double_click'

                # Adaptive escalation if the exact same action was repeated after failing
                act_prev_fails = action_fail_counts.get(f"{action}:{target_key}", 0)
                if act_prev_fails == 1:
                    if action == 'double_click':
                        print(f"[Trajectory Loop Guard] Escalating repeated 'double_click' ➜ single 'click' + Enter on '{clean_tgt}'")
                        action = 'click'
                        key_single = 'enter'
                    elif action == 'click':
                        print(f"[Trajectory Loop Guard] Escalating repeated 'click' ➜ 'double_click' on '{clean_tgt}'")
                        action = 'double_click'

                act_res = self.actuator.execute(
                    droplet=target_node,
                    action=action,
                    input_text=input_text,
                    win_ctx=current_win_ctx,
                    target_x=target_x,
                    target_y=target_y,
                    drag_dest=drag_dest,
                    key_combo=key_combo,
                    key=key_single,
                    scroll_direction=scroll_dir,
                    scroll_amount=scroll_amt,
                    press_enter=decision.get("press_enter", False)
                )
            elif action in (
                "type", "write", "fill", "input", "hotkey", "chord", "press", "key_down", "key_up",
                "scroll", "system_scroll", "wheel", "wait", "open_path", "system_open", "system_open_path", "launch_app",
                "clear_and_type", "replace_text", "clear_type",
                "type_and_enter", "submit", "click_and_type",
                "ctrl_click", "control_click", "shift_click", "alt_click", "modifier_click",
                "hover_and_wait", "hover_wait",
                "copy", "paste", "select_all", "undo", "redo", "save", "find",
                "close_tab", "new_tab", "switch_window", "win_run",
                "drag", "drag_to", "drag_and_drop", "free_drag", "drag_area",
                "minimize", "minimize_window", "maximize", "maximize_window", "stop"
            ):
                act_res = self.actuator.execute(
                    droplet=None,
                    action=action,
                    input_text=input_text,
                    win_ctx=current_win_ctx,
                    target_x=target_x,
                    target_y=target_y,
                    drag_dest=drag_dest,
                    key_combo=key_combo,
                    key=key_single,
                    scroll_direction=scroll_dir,
                    scroll_amount=scroll_amt,
                    press_enter=decision.get("press_enter", False)
                )
            else:
                print(f"[Trajectory Engine] ⚠️ Target node [{target_id}] not found in live droplet spray; recording ungrounded step and continuing.")
                _notify_status(f"⚠️ Target [{target_id}] not visible", "#ffb300")
                step = TrajectoryStep(
                    step_num=turn,
                    goal=goal,
                    subgoal=subgoal,
                    action=action,
                    target_id=target_id,
                    target_label=f"Dot [{target_id}] (Not Found)",
                    input_text=input_text,
                    verified=False,
                    verification_reason=f"Target dot [{target_id}] was not found in live droplet spray. Choose an existing visible dot.",
                    active_window=win_title,
                    semantic_intention=semantic_intention,
                    expected_transition=expected_transition,
                    verdict="NO_EFFECT",
                    observation_analysis=observation_analysis,
                    critique=critique,
                    strategy_plan=strategy
                )
                memory.add_step(step)
                continue

            recent_action_sigs.append(current_sig)

            # Cognitive Semantic State Verification (Phase 6)
            is_fatal_hardware_error = act_res and (not act_res.get("verified", True)) and (act_res.get("type") in ("INVALID_ACTION", "EXECUTION_ERROR", "ERROR"))

            if is_fatal_hardware_error:
                # Hardware actuator explicitly failed (e.g. file not found, OS error, missing target)
                verified = False
                verdict = "EXECUTION_FAILED"
                ver_reason = act_res.get("reason") or act_res.get("error") or "Actuator execution failed"
            else:
                time.sleep(0.20)  # Brief settling for OS visual transition
                post_snap = SemanticStateVerifier.capture_snapshot(current_win_ctx)
                sem_ver = SemanticStateVerifier.verify_transition(
                    pre_snap, post_snap, action, input_text, semantic_intention, expected_transition
                )

                # Fuse physical verifier and semantic verifier:
                phys_verified = act_res.get("verified", False) if act_res else False
                sem_verified = sem_ver.get("verified", False)

                if phys_verified or sem_verified:
                    verified = True
                    verdict = sem_ver.get("verdict", "CONFIRMED")
                    if sem_verified and sem_ver.get("reason"):
                        ver_reason = sem_ver.get("reason")
                    else:
                        ver_reason = act_res.get("reason", "Physical transition confirmed")
                else:
                    verified = False
                    verdict = sem_ver.get("verdict", "NO_EFFECT")
                    ver_reason = sem_ver.get("reason") or (act_res.get("reason") if act_res else "No visual or semantic state change detected")

            # Track failure counts for target loop prevention
            action_key = f"{action}:{target_key}" if target_key else ""
            if target_key:
                if not verified or verdict in ("NO_EFFECT", "EXECUTION_FAILED"):
                    target_fail_counts[target_key] = target_fail_counts.get(target_key, 0) + 1
                    if action_key:
                        action_fail_counts[action_key] = action_fail_counts.get(action_key, 0) + 1
                else:
                    target_fail_counts[target_key] = 0
                    if action_key:
                        action_fail_counts[action_key] = 0

            print(f"[Cognitive Verifier] Verdict: {verdict} ➜ {ver_reason}")
            if verified:
                _notify_status(f"✔ Turn {turn}: {verdict[:14]}", "#3fb950")
            else:
                _notify_status(f"⚠️ Turn {turn}: {verdict[:14]}", "#ffb300")

            # Record step in episodic memory with deliberation fields
            step = TrajectoryStep(
                step_num=turn,
                goal=goal,
                subgoal=subgoal,
                action=action,
                target_id=target_id,
                target_label=target_label,
                input_text=input_text,
                verified=verified,
                verification_reason=ver_reason,
                active_window=win_title,
                semantic_intention=semantic_intention,
                expected_transition=expected_transition,
                verdict=verdict,
                observation_analysis=observation_analysis,
                critique=critique,
                strategy_plan=strategy,
                master_plan=master_plan,
                intent_analysis=user_intent,
                mistake_critique=mistake_critique
            )
            memory.add_step(step)

        if not memory.is_completed and len(memory.steps) >= max_turns:
            print(f"\n[Trajectory Engine] ⚠️ Reached maximum turn cap ({max_turns}) without explicit stop signal.")
            memory.completion_summary = f"Reached max safety turn cap of {max_turns} turns."
            _notify_status(f"✦ Max turns ({max_turns}) reached", "#ffb300")

        return memory
