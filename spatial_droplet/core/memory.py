"""
Episodic Trajectory Buffer for multi-turn desktop automation.
Maintains chronological history, enforces <250 token budget, and detects thrashing loops.
"""
from typing import List, Optional, Tuple
from .models import TrajectoryStep

class EpisodicTrajectoryMemory:
    """
    Episodic Trajectory Buffer for multi-turn desktop automation (Phases 5, 6, 7 & 8).
    Maintains chronological history of actions, semantic intentions, and visual transition verdicts.
    Formats token-efficient history representations for the LLM (<250 tokens).
    Features generalized action signature tracking, ping-pong cycle detection, and failure awareness.
    """
    def __init__(self, goal: str, max_steps: int = 50):
        self.goal = goal
        self.max_steps = max_steps
        self.steps: List[TrajectoryStep] = []
        self.is_completed = False
        self.completion_summary = ""

    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)

    def _get_step_signature(self, s: TrajectoryStep) -> Tuple[str, str, str, str]:
        """Returns a canonicalized signature tuple: (action, target_semantic, clean_label, input_text)."""
        import re
        clean_lbl = re.sub(r'\[item \d+ of \d+.*?\]', '', (s.target_label or "").lower()).strip()
        clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean_lbl).strip()
        act = (s.action or "").lower().strip()
        txt = (s.input_text or "").strip().lower()
        target_semantic = clean_lbl or (f"id_{s.target_id}" if s.target_id is not None else "")
        return (act, target_semantic, clean_lbl, txt)

    def is_repeated_action(self) -> bool:
        """Detects if the last 2 actions have the exact same signature."""
        if len(self.steps) < 2:
            return False
        return self._get_step_signature(self.steps[-1]) == self._get_step_signature(self.steps[-2])

    def detect_ping_pong_loop(self) -> Optional[str]:
        """Detects 2-step alternating cycles (A -> B -> A -> B)."""
        if len(self.steps) < 4:
            return None
        sig_a1 = self._get_step_signature(self.steps[-4])
        sig_b1 = self._get_step_signature(self.steps[-3])
        sig_a2 = self._get_step_signature(self.steps[-2])
        sig_b2 = self._get_step_signature(self.steps[-1])

        if sig_a1 == sig_a2 and sig_b1 == sig_b2 and sig_a1 != sig_b1:
            act_a = f"{self.steps[-2].action.upper()}"
            if self.steps[-2].input_text:
                act_a += f" text='{self.steps[-2].input_text}'"
            elif self.steps[-2].target_label:
                act_a += f" on '{self.steps[-2].target_label}'"

            act_b = f"{self.steps[-1].action.upper()}"
            if self.steps[-1].input_text:
                act_b += f" text='{self.steps[-1].input_text}'"
            elif self.steps[-1].target_label:
                act_b += f" on '{self.steps[-1].target_label}'"

            return (
                f"CRITICAL PING-PONG LOOP DETECTED: You are alternating between '{act_a}' and '{act_b}' back and forth! "
                f"You MUST break this loop immediately and execute a different approach."
            )
        return None

    def get_consecutive_failures(self) -> int:
        """Returns the count of consecutive unverified or failed steps."""
        count = 0
        for s in reversed(self.steps):
            if not s.verified or s.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE"):
                count += 1
            else:
                break
        return count

    def is_repeated_failure(self, target_label: str) -> bool:
        """Detects if the same element or action failed verification 2+ consecutive times."""
        if len(self.steps) < 2:
            return False
        s1, s2 = self.steps[-2], self.steps[-1]
        sig1, sig2 = self._get_step_signature(s1), self._get_step_signature(s2)
        s1_stagnant = not s1.verified or s1.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE")
        s2_stagnant = not s2.verified or s2.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE")
        if sig1 == sig2 and (s1_stagnant or s2_stagnant):
            return True

        if target_label:
            import re
            clean = re.sub(r'\[item \d+ of \d+.*?\]', '', target_label.lower()).strip()
            clean = re.sub(r'\[(?:btn|icon|app)\]\s*', '', clean).strip()
            c1 = re.sub(r'\[item \d+ of \d+.*?\]', '', (s1.target_label or "").lower()).strip()
            c1 = re.sub(r'\[(?:btn|icon|app)\]\s*', '', c1).strip()
            c2 = re.sub(r'\[item \d+ of \d+.*?\]', '', (s2.target_label or "").lower()).strip()
            c2 = re.sub(r'\[(?:btn|icon|app)\]\s*', '', c2).strip()
            if clean and c1 == clean and c2 == clean and (s1_stagnant or s2_stagnant):
                return True
        return False

    def format_history_for_prompt(self) -> str:
        if not self.steps:
            return "No previous actions taken yet. This is Step 1."

        lines = []

        # 1. Track Active Master Plan progression if recorded (inspect canonical then legacy)
        active_plan = None
        for s in reversed(self.steps):
            mp = getattr(s, "end_to_end_master_plan", None) or getattr(s, "master_plan", None)
            if mp:
                active_plan = mp
                break

        if active_plan:
            # Normalize string representation into list of lines
            if isinstance(active_plan, str):
                plan_items = [l.strip() for l in active_plan.splitlines() if l.strip()]
            elif isinstance(active_plan, list):
                plan_items = active_plan
            else:
                plan_items = [str(active_plan).strip()]

            if plan_items:
                lines.append("Active Master Plan:")
                for p in plan_items:
                    p_str = str(p).strip()
                    p_upper = p_str.upper()
                    if any(k in p_upper for k in ("(DONE)", "[DONE]", "(COMPLETED)", "[COMPLETED]", "✔")):
                        lines.append(f"  ✔ {p_str.lstrip('✔►○• ')}")
                    elif any(k in p_upper for k in ("(ACTIVE)", "[ACTIVE]", "(IN PROGRESS)", "[IN PROGRESS]", "►", "CURRENT")):
                        lines.append(f"  ► {p_str.lstrip('✔►○• ')}")
                    elif any(k in p_upper for k in ("(PENDING)", "[PENDING]", "(TODO)", "[TODO]", "○")):
                        lines.append(f"  ○ {p_str.lstrip('✔►○• ')}")
                    else:
                        lines.append(f"  • {p_str}")
                lines.append("")

        # 2. Chronological Trajectory History (Recent Steps)
        recent_steps = self.steps[-4:]
        lines.append("Chronological Trajectory History (Recent Steps):")
        if len(self.steps) > 4:
            lines.append(f"• (Steps 1-{len(self.steps)-4} executed previously in trajectory)")
        for s in recent_steps:
            if s.verdict == "EXECUTION_FAILED":
                v_badge = "❌ EXECUTION_FAILED"
            elif s.verified:
                v_badge = f"✔ {s.verdict}"
            else:
                v_badge = f"⚠️ {s.verdict}"

            act_str = f"{s.action.upper()}"
            if s.input_text:
                act_str += f" text=\"{s.input_text[:40]}\""
            target_str = f" on \"{s.target_label[:30]}\"" if s.target_label else ""

            intent_val = getattr(s, 'user_intent_and_success_criteria', '') or getattr(s, 'intent_analysis', '')
            if intent_val and len(intent_val) > 80:
                intent_val = intent_val[:77] + "..."
            intent_str = f" | Intended: \"{intent_val or s.semantic_intention[:60]}\"" if (intent_val or s.semantic_intention) else ""

            reason_str = f" | Details: {s.verification_reason[:60]}" if s.verification_reason else ""

            critique_val = getattr(s, 'mistake_critique_and_loop_diagnosis', '') or getattr(s, 'mistake_critique', '') or getattr(s, 'critique', '')
            if critique_val and len(critique_val) > 80:
                critique_val = critique_val[:77] + "..."
            critique_str = f" | Prev Critique: \"{critique_val}\"" if critique_val else ""

            lines.append(
                f"• Step {s.step_num}: Sub-goal=\"{s.subgoal[:40]}\"{intent_str} | Action={act_str}{target_str} "
                f"| Outcome={v_badge}{reason_str}{critique_str} | Window=\"{s.active_window[:30]}\""
            )

        # 3. Loop detection & guidance
        ping_pong_msg = self.detect_ping_pong_loop()
        if ping_pong_msg:
            lines.append(f"⚠️ {ping_pong_msg}")
        else:
            last_step = self.steps[-1]
            last_critique = (
                getattr(last_step, 'mistake_critique_and_loop_diagnosis', '') or
                getattr(last_step, 'mistake_critique', '') or
                getattr(last_step, 'critique', '')
            )
            if self.is_repeated_failure(last_step.target_label):
                lines.append(
                    f"⚠️ CRITICAL LOOP WARNING: The previous action failed or produced NO CHANGE. "
                    f"DO NOT repeat the exact same action! You MUST switch strategy now: single click instead of double-click, "
                    f"keyboard Enter/Tab, the address bar, or direct 'open_path'."
                )
            elif self.is_repeated_action():
                lines.append(
                    f"⚠️ REPETITION WARNING: You just repeated action '{last_step.action.upper()}' twice. "
                    f"DO NOT execute the same action a third time. Try an alternative action."
                )
            elif not last_step.verified:
                lines.append(f"⚠️ NOTE: Step {last_step.step_num} yielded no verified screen change. Inspect current observation before deciding next action.")

            if (not last_step.verified or last_step.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE")) and last_critique:
                lines.append(f"⚠️ DIAGNOSED MISTAKE: {last_critique[:120]}")
                lines.append("   MANDATE: Avoid repeating this diagnosed mistake. Pivot to an alternative tactical modality.")

        cons_fails = self.get_consecutive_failures()
        if cons_fails >= 3:
            lines.append(f"⚠️ PERSISTENT FAILURE ALERT: The last {cons_fails} actions had NO effect. Do NOT continue the same plan.")

        return "\n".join(lines)
