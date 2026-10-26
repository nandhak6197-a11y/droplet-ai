"""
Cognitive Awareness Router for Sequential Checkpoint Trajectory Execution.
Decomposes complex multi-task desktop workflows into sequential checkpoints,
enforces strict step-by-step verification before advancing, guides cleanup at task conclusion,
and prunes background spatial distractors outside active window boundaries.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any

from ..core.models import DiagonalDroplet, WindowContext


@dataclass
class Checkpoint:
    id: int
    title: str
    description: str
    expected_window: str = ""
    action_type: str = "interaction"  # 'launch', 'interaction', 'verify', 'cleanup'
    status: str = "PENDING"           # 'DONE', 'ACTIVE', 'PENDING'
    cleanup: bool = False
    verification_evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "expected_window": self.expected_window,
            "action_type": self.action_type,
            "status": self.status,
            "cleanup": self.cleanup,
            "verification_evidence": self.verification_evidence,
        }


class AwarenessRouter:
    """
    Cognitive Awareness Router.
    Decomposes user queries into ordered sequential checkpoints, monitors screen state
    at each turn, enforces checkpoint completion before step advancement, and eliminates
    cross-window spatial hallucinations.
    """

    KNOWN_APPS = {
        "calculator": ["calc", "calculator"],
        "calc": ["calc", "calculator"],
        "notepad": ["notepad", "text editor", "editor"],
        "chrome": ["chrome", "google chrome", "browser"],
        "edge": ["edge", "msedge", "microsoft edge"],
        "powershell": ["powershell", "terminal"],
        "cmd": ["cmd", "command prompt"],
        "explorer": ["explorer", "file explorer", "files"],
        "camera": ["camera", "webcam"],
        "settings": ["settings", "ms-settings:"],
        "paint": ["paint", "mspaint"],
        "code": ["code", "vscode", "vs code"],
    }

    @classmethod
    def identify_app(cls, text: str) -> Optional[str]:
        """Identifies target application name from natural language text."""
        low = text.lower()
        for canonical, aliases in cls.KNOWN_APPS.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', low):
                    return canonical
        return None

    @classmethod
    def is_nav_drawer_toggle(cls, label: str) -> bool:
        """Identifies if a UI control is a hamburger menu / navigation drawer toggle."""
        if not label:
            return False
        clean = re.sub(r'\[(?:btn|icon|app)\]\s*', '', label.lower()).strip()
        return any(k in clean for k in (
            "open navigation", "close navigation", "toggle navigation",
            "navigation drawer", "nav drawer", "hamburger menu"
        )) or clean == "navigation"

    @classmethod
    def decompose_goal(cls, goal: str) -> List[Checkpoint]:
        """
        Decomposes a user goal into sequential checkpoints.
        Handles chained multi-task queries (e.g. 'open calculator, add 2+2, verify, close')
        and single-task directives. Strips optional CLI/slash prefixes.
        """
        clean_goal = goal.strip()
        while True:
            new_goal = re.sub(r'^/(?:goal|boost|turns(?:\s+\d+)?)\s*', '', clean_goal, flags=re.IGNORECASE).strip()
            if new_goal == clean_goal:
                break
            clean_goal = new_goal
        if not clean_goal:
            return [Checkpoint(id=1, title="Execute Task", description="Complete user request", status="ACTIVE")]

        # Split query into chained workflow clauses using commas, semicolons, and sequential conjunctions
        clauses = []
        raw_clauses = re.split(r'(?:;\s*|,\s*(?:then\s+|and\s+)?|\s+(?:and\s+then|then|after\s+that)\s+)', clean_goal, flags=re.IGNORECASE)
        for rc in raw_clauses:
            rc_clean = rc.strip()
            # Also handle inline 'and' if it connects distinct action verbs
            if re.search(r'\b(?:open|launch|start|run|add|type|calculate|verify|check|close|exit|play|skip)\b', rc_clean, re.IGNORECASE):
                sub_parts = re.split(r'\s+and\s+(?=(?:open|launch|start|run|add|type|calculate|verify|check|close|exit|play|skip|forward|backward)\b)', rc_clean, flags=re.IGNORECASE)
                for sp in sub_parts:
                    if sp.strip():
                        clauses.append(sp.strip())
            elif rc_clean:
                clauses.append(rc_clean)

        if not clauses:
            clauses = [clean_goal]

        checkpoints: List[Checkpoint] = []
        step_id = 1
        target_app = cls.identify_app(clean_goal) or ""

        # Check if cleanup/close is requested anywhere in goal
        has_close = any(re.search(r'\b(?:close|cleanup|exit|quit|shut\s+down)\b', cl, re.IGNORECASE) for cl in clauses)

        for cl in clauses:
            cl_low = cl.lower()

            # 1. Launch / Focus clause
            if re.match(r'^(?:open|launch|start|run|switch\s+to|bring\s+up)\b', cl_low):
                app_in_clause = cls.identify_app(cl_low) or target_app or "Application"
                checkpoints.append(Checkpoint(
                    id=step_id,
                    title=f"Open {app_in_clause.title()}",
                    description=f"Launch or focus {app_in_clause.title()} and verify it is active in foreground",
                    expected_window=app_in_clause,
                    action_type="launch",
                    status="ACTIVE" if step_id == 1 else "PENDING",
                    cleanup=False
                ))
                step_id += 1

            # 2. Calculation / Input clause
            elif re.search(r'\b(?:add|calculate|compute|type|enter|press|input|write)\b', cl_low) or re.search(r'\d+\s*[\+\-\*\/]\s*\d+', cl_low):
                calc_match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+(?:\s*[\+\-\*\/]\s*\d+)*)', cl)
                detail = f"Enter calculation '{calc_match.group(1)}'" if calc_match else f"Perform input: '{cl}'"
                checkpoints.append(Checkpoint(
                    id=step_id,
                    title="Perform Input / Calculation",
                    description=detail,
                    expected_window=target_app,
                    action_type="interaction",
                    status="ACTIVE" if step_id == 1 else "PENDING",
                    cleanup=False
                ))
                step_id += 1

            # 3. Verification clause
            elif re.search(r'\b(?:verify|check|confirm|inspect|read)\b', cl_low):
                checkpoints.append(Checkpoint(
                    id=step_id,
                    title="Verify State / Result",
                    description=f"Verify expected outcome: '{cl}'",
                    expected_window=target_app,
                    action_type="verify",
                    status="ACTIVE" if step_id == 1 else "PENDING",
                    cleanup=False
                ))
                step_id += 1

            # 4. Close / Cleanup clause
            elif re.search(r'\b(?:close|cleanup|exit|quit)\b', cl_low):
                checkpoints.append(Checkpoint(
                    id=step_id,
                    title=f"Close {target_app.title() if target_app else 'Window'}",
                    description=f"Close application and clean up workspace: '{cl}'",
                    expected_window=target_app,
                    action_type="cleanup",
                    status="ACTIVE" if step_id == 1 else "PENDING",
                    cleanup=True
                ))
                step_id += 1

            # 5. Generic interaction clause (e.g. 'play', 'forward', 'backward', 'skip')
            else:
                checkpoints.append(Checkpoint(
                    id=step_id,
                    title=cl.title()[:30],
                    description=f"Execute: '{cl}'",
                    expected_window=target_app,
                    action_type="interaction",
                    status="ACTIVE" if step_id == 1 else "PENDING",
                    cleanup=False
                ))
                step_id += 1

        # If user goal requested calculation or multi-step but didn't explicitly write "verify", insert verification checkpoint before close
        has_verify = any(cp.action_type == "verify" for cp in checkpoints)
        has_interaction = any(cp.action_type == "interaction" for cp in checkpoints)
        if has_interaction and not has_verify:
            # Insert verify checkpoint before cleanup (or at end)
            cleanup_idx = next((i for i, cp in enumerate(checkpoints) if cp.cleanup), len(checkpoints))
            v_cp = Checkpoint(
                id=step_id,
                title="Verify Result",
                description="Verify operation completed successfully with correct output displayed",
                expected_window=target_app,
                action_type="verify",
                status="PENDING",
                cleanup=False
            )
            checkpoints.insert(cleanup_idx, v_cp)
            step_id += 1

        # Re-index checkpoint IDs sequentially
        for idx, cp in enumerate(checkpoints):
            cp.id = idx + 1
            if idx == 0 and cp.status != "DONE":
                cp.status = "ACTIVE"
            elif idx > 0 and checkpoints[idx - 1].status != "DONE":
                cp.status = "PENDING"

        return checkpoints

    @classmethod
    def evaluate_screen_and_advance(
        cls,
        checkpoints: List[Checkpoint],
        current_win_ctx: Optional[WindowContext],
        screen_tokens: Optional[Set[str]] = None,
        last_step: Optional[Any] = None,
    ) -> List[Checkpoint]:
        """
        Evaluates whether the current active checkpoint has been satisfied on the live screen.
        Only advances to Checkpoint N+1 once Checkpoint N is confirmed complete.
        """
        if not checkpoints:
            return checkpoints

        tokens = screen_tokens or set()
        win_title = (current_win_ctx.title if current_win_ctx and current_win_ctx.is_valid else "Desktop").lower()

        # Find currently active checkpoint
        active_idx = next((i for i, cp in enumerate(checkpoints) if cp.status == "ACTIVE"), None)
        if active_idx is None:
            return checkpoints

        curr = checkpoints[active_idx]
        step_verified = False
        evidence = ""

        # Case 1: Launch checkpoint
        if curr.action_type == "launch":
            exp = (curr.expected_window or "").lower()
            aliases = cls.KNOWN_APPS.get(exp, [exp]) if exp else []
            if any(a in win_title for a in aliases if len(a) >= 3):
                step_verified = True
                evidence = f"Target window '{current_win_ctx.title}' active in foreground."
            elif last_step and last_step.action in ("launch_app", "switch_window") and last_step.verified:
                if any(a in win_title for a in aliases if len(a) >= 3) or (current_win_ctx and current_win_ctx.is_valid and current_win_ctx.title != "Desktop"):
                    step_verified = True
                    evidence = f"Window switch confirmed: '{current_win_ctx.title}'."

        # Case 2: Interaction / Calculation checkpoint
        elif curr.action_type == "interaction":
            if last_step and last_step.verified and last_step.verdict in ("CONFIRMED", "CONFIRMED_EXPECTED_TRANSITION", "CONFIRMED_VISUAL_CHANGE"):
                # Guard against false positives: drawer toggling does NOT satisfy calculation/data entry
                is_drawer_toggle = cls.is_nav_drawer_toggle(getattr(last_step, 'target_label', '') or '')
                if not is_drawer_toggle:
                    step_verified = True
                    evidence = f"Input verified: {last_step.verification_reason or last_step.action}"

        # Case 3: Verification checkpoint
        elif curr.action_type == "verify":
            # Check if expected math tokens or outcome appear in screen tokens
            desc_tokens = set(re.findall(r'\w+', curr.description.lower()))
            overlap = desc_tokens.intersection(tokens)
            if overlap or (last_step and last_step.verified):
                step_verified = True
                evidence = f"Output state verified on screen ({', '.join(list(overlap)[:3]) if overlap else 'screen confirmed'})."

        # Case 4: Cleanup / Close checkpoint
        elif curr.action_type == "cleanup":
            exp = (curr.expected_window or "").lower()
            aliases = cls.KNOWN_APPS.get(exp, [exp]) if exp else []
            # Verified if window is closed or switched away to Desktop
            if not any(a in win_title for a in aliases if len(a) >= 3) or win_title == "desktop":
                step_verified = True
                evidence = f"Window closed / clean: active window is '{win_title}'."
            elif last_step and last_step.action in ("close", "minimize", "system_window_op", "hotkey") and last_step.verified:
                step_verified = True
                evidence = "Close action executed successfully."

        if step_verified:
            curr.status = "DONE"
            curr.verification_evidence = evidence
            # Advance next checkpoint to ACTIVE
            if active_idx + 1 < len(checkpoints):
                checkpoints[active_idx + 1].status = "ACTIVE"

        return checkpoints

    @classmethod
    def filter_spatial_droplets(
        cls,
        droplets: List[DiagonalDroplet],
        win_ctx: Optional[WindowContext],
        modal_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> List[DiagonalDroplet]:
        """
        Active window spatial pruning:
        When an application window is active in the foreground (not Desktop),
        strictly filters droplets lying outside the active window rect
        to eliminate clicking background windows (such as background PowerShell or wallpaper).
        """
        if not droplets:
            return []

        has_win = bool(
            win_ctx
            and win_ctx.is_valid
            and win_ctx.title != "Desktop"
            and win_ctx.rect
            and len(win_ctx.rect) == 4
            and win_ctx.rect[2] > win_ctx.rect[0]
            and win_ctx.rect[3] > win_ctx.rect[1]
        )
        if not has_win and not modal_rect:
            return droplets

        # Check bounds
        wx1, wy1, wx2, wy2 = (win_ctx.rect if has_win else (0, 0, 1920, 1080))
        mx1, my1, mx2, my2 = (modal_rect if modal_rect else (-100, -100, -100, -100))

        foreground_droplets = []
        for d in droplets:
            is_in_modal = bool(modal_rect and (mx1 - 10 <= d.center_x <= mx2 + 10 and my1 - 10 <= d.center_y <= my2 + 10))
            is_in_active_win = bool(has_win and (wx1 - 5 <= d.center_x <= wx2 + 5 and wy1 - 5 <= d.center_y <= wy2 + 5))

            if is_in_modal or is_in_active_win:
                foreground_droplets.append(d)

        # If we found droplets inside the active window/modal, return exclusively those!
        # Background window droplets (e.g. PowerShell at (1388, 31)) are completely pruned.
        if foreground_droplets:
            return foreground_droplets

        # Fallback to original droplets only if zero droplets lie inside active window rect
        return droplets

    @classmethod
    def format_checkpoints_for_prompt(cls, checkpoints: List[Checkpoint]) -> str:
        """Formats the sequential checkpoints into a dense, token-capped prompt section."""
        if not checkpoints:
            return ""

        lines = ["=== AWARENESS ROUTER SEQUENTIAL CHECKPOINTS ==="]
        active_cp = None
        for cp in checkpoints:
            badge = "✔ [DONE]" if cp.status == "DONE" else ("► [ACTIVE]" if cp.status == "ACTIVE" else "○ [PENDING]")
            ev = f" — Verified: {cp.verification_evidence}" if cp.verification_evidence else ""
            lines.append(f"  {badge} Step {cp.id}: {cp.title} ({cp.description}){ev}")
            if cp.status == "ACTIVE" and active_cp is None:
                active_cp = cp

        if active_cp:
            lines.append("")
            lines.append(f"MANDATE FOR THIS TURN: You are currently executing Step {active_cp.id}: {active_cp.title}.")
            lines.append("You MUST verify Step is physically satisfied on screen before advancing.")
            if active_cp.cleanup:
                lines.append("Note: This is the final cleanup/close step. Ensure all prior steps are completed.")
            lines.append("Do NOT interact with background windows or unrelated applications.")

        return "\n".join(lines)
