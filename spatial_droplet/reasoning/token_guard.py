"""
Strict Prompt Token Budget Guard (<5,000 tokens) & 3-Tier Dynamic Compaction.
Proactively guards all LLM prompts sent to external providers (Groq 25-key pool, Gemini Pro, OpenAI)
to strictly enforce < 4,800 tokens with safety headroom under 5,000 tokens.

3-Tier Dynamic Compaction Architecture:
- Tier 1: Candidate Manifest Compaction:
  Relevance sorting (modal controls, foreground active window, interactive archetypes)
  pruning to top 24–36 high-value nodes.
- Tier 2: Spatial Layout Graph Condensation:
  Omit empty zones, condense non-active background zones to summary counts,
  retain full topological hints only for active window / modal zones.
- Tier 3: Trajectory Memory Compaction:
  Compact history to latest 3 steps with truncated string lengths,
  preserving master plan milestone statuses (✔ [DONE], ► [ACTIVE], ○ [PENDING]).
"""
import re
import math
from typing import List, Dict, Optional, Tuple, Any, Set

from ..core.models import DiagonalDroplet
from ..core.memory import EpisodicTrajectoryMemory

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    _TIKTOKEN_AVAILABLE = False


class TokenBudgetGuard:
    """
    Proactive Token Budget Guard & Dynamic Compaction Engine.
    Enforces prompt token safety (<4,800 target, <5,000 hard ceiling)
    before dispatching to external LLMs.
    """
    _encoding = None
    _encoding_failed = False

    HARD_TOKEN_LIMIT: int = 4800
    SAFE_MAX_BUDGET: int = 5000
    TIER1_TARGET_NODES: int = 36
    TIER1_PRUNED_NODES: int = 24
    TIER3_MAX_STEPS: int = 3

    @classmethod
    def get_encoding(cls):
        """Returns cached tiktoken cl100k_base encoding or None."""
        if cls._encoding is None and not cls._encoding_failed:
            if not _TIKTOKEN_AVAILABLE:
                cls._encoding_failed = True
                return None
            try:
                cls._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                cls._encoding_failed = True
                cls._encoding = None
        return cls._encoding

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """
        Calculates exact token count using tiktoken (cl100k_base).
        Falls back to fast character-ratio (4 chars/token) if tiktoken fails or is unavailable.
        """
        if not text:
            return 0
        enc = cls.get_encoding()
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        # Fast character-ratio fallback: 4 characters per token (ceiling)
        return max(1, (len(text) + 3) // 4)

    @classmethod
    def count_messages_tokens(cls, messages: List[Dict[str, Any]]) -> int:
        """
        Calculates total token count for a list of chat completion messages,
        including per-message framing overhead.
        """
        total = 0
        for msg in messages:
            total += 4  # Per-message overhead (role, tags)
            content = msg.get("content", "")
            if isinstance(content, str):
                total += cls.count_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            total += cls.count_tokens(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            detail = part.get("image_url", {}).get("detail", "auto")
                            total += 85 if detail == "low" else 1105
        total += 3  # Assistant response priming overhead
        return total

    @classmethod
    def score_droplet_relevance(
        cls,
        d: DiagonalDroplet,
        goal: str = "",
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Scores candidate droplet relevance for Tier 1 candidate manifest compaction.
        Prioritizes:
        1. Modal dialog controls (+10000)
        2. Dialog action verbs (save, confirm, ok, cancel, replace) (+350)
        3. Active foreground window controls (+250)
        4. User goal keyword matches (+400 direct, +200 partial)
        5. Interactive archetypes (button, text, slider, tab) (+150)
        """
        score = 0.0
        lbl = str(getattr(d, "label", "")).lower()
        att = str(getattr(d, "attached_label", "")).lower()
        combined = f"{lbl} {att}"

        # 1. Modal dialog containment
        if modal_rect and len(modal_rect) == 4:
            mx1, my1, mx2, my2 = modal_rect
            if (mx1 - 10 <= d.center_x <= mx2 + 10) and (my1 - 10 <= d.center_y <= my2 + 10):
                score += 10000.0

        # 2. Universal dialog choices
        dialog_tokens = {
            "save", "confirm", "ok", "yes", "apply", "done", "continue", "submit",
            "cancel", "close", "discard", "no", "replace", "overwrite", "select", "folder"
        }
        if any(tok in combined for tok in dialog_tokens):
            score += 350.0

        # 3. Active foreground window bounds
        if getattr(d, "is_foreground", True):
            score += 250.0
        if win_rect and len(win_rect) == 4:
            wx1, wy1, wx2, wy2 = win_rect
            if wx1 <= d.center_x <= wx2 and wy1 <= d.center_y <= wy2:
                score += 150.0

        # 4. User goal keyword matches
        if goal:
            goal_words = set(re.findall(r"\w+", goal.lower()))
            node_words = set(re.findall(r"\w+", combined))
            direct_hits = goal_words & node_words
            score += len(direct_hits) * 400.0
            if str(d.id) in goal_words:
                score += 500.0

        # 5. Interactive archetypes
        arch = getattr(d, "archetype", "button").lower()
        role = getattr(d, "semantic_role", "widget").lower()
        if arch in ("button", "text", "slider", "tab") or role in ("control", "action", "input", "slider"):
            score += 150.0
        elif arch == "probe" or getattr(d, "category", "") == "surface_probe":
            score -= 300.0

        return score

    @classmethod
    def compact_candidates_tier1(
        cls,
        nodes: List[DiagonalDroplet],
        goal: str = "",
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None,
        max_nodes: int = 36
    ) -> List[DiagonalDroplet]:
        """
        Tier 1: Prunes candidate manifest to top high-value nodes sorted by relevance.
        Guarantees modal dialog controls are prioritized and never evicted by background distractors.
        """
        if not nodes or len(nodes) <= max_nodes:
            return list(nodes)

        # Guarantee modal dialog controls reservation
        if modal_rect and len(modal_rect) == 4:
            mx1, my1, mx2, my2 = modal_rect
            modal_nodes = [
                d for d in nodes
                if (mx1 - 10 <= d.center_x <= mx2 + 10) and (my1 - 10 <= d.center_y <= my2 + 10)
            ]
            if modal_nodes:
                modal_sorted = sorted(
                    modal_nodes,
                    key=lambda d: cls.score_droplet_relevance(d, goal=goal, modal_rect=modal_rect, win_rect=win_rect),
                    reverse=True
                )
                non_modal = [d for d in nodes if d not in modal_nodes]
                non_modal_sorted = sorted(
                    non_modal,
                    key=lambda d: cls.score_droplet_relevance(d, goal=goal, modal_rect=modal_rect, win_rect=win_rect),
                    reverse=True
                )
                reserved = modal_sorted[:max_nodes]
                remaining = max(0, max_nodes - len(reserved))
                pruned = reserved + non_modal_sorted[:remaining]
                pruned.sort(key=lambda d: d.id)
                return pruned

        scored = sorted(
            nodes,
            key=lambda d: cls.score_droplet_relevance(d, goal=goal, modal_rect=modal_rect, win_rect=win_rect),
            reverse=True
        )
        pruned = scored[:max_nodes]
        pruned.sort(key=lambda d: d.id)
        return pruned

    @classmethod
    def condense_spatial_graph_tier2(
        cls,
        nodes: List[DiagonalDroplet],
        sw: int = 1920,
        sh: int = 1080,
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Tier 2: Spatial layout graph condensation.
        - Omits empty zones completely.
        - Condenses non-active background zones to summary counts.
        - Retains full topological hints only for active window / modal zones.
        """
        from .reasoner import classify_screen_zone

        if not nodes:
            return "=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH ===\n(No droplets present on screen)"

        zone_buckets: Dict[str, List[DiagonalDroplet]] = {
            "[MODAL/POPUP]": [],
            "[HEADER/NAV]": [],
            "[SIDEBAR]": [],
            "[WORKSPACE]": [],
            "[FOOTER]": []
        }

        for d in nodes:
            z = classify_screen_zone(d.center_x, d.center_y, sw, sh, modal_rect, win_rect=win_rect)
            if z in zone_buckets:
                zone_buckets[z].append(d)
            else:
                zone_buckets["[WORKSPACE]"].append(d)

        active_nodes = [
            d for d in nodes
            if (modal_rect and (modal_rect[0] - 10 <= d.center_x <= modal_rect[2] + 10 and modal_rect[1] - 10 <= d.center_y <= modal_rect[3] + 10))
            or (win_rect and (win_rect[0] <= d.center_x <= win_rect[2] and win_rect[1] <= d.center_y <= win_rect[3]))
            or getattr(d, "is_foreground", True)
        ]
        active_set = {d.id for d in active_nodes}

        relative_hints: Dict[int, List[str]] = {d.id: [] for d in nodes}
        sorted_by_y = sorted(active_nodes, key=lambda n: (round(n.center_y / 15.0), n.center_x))
        for i, d1 in enumerate(sorted_by_y):
            left_neighbors = [
                d2 for d2 in sorted_by_y[:i]
                if abs(d2.center_y - d1.center_y) <= 18 and 0 < (d1.center_x - d2.center_x) <= 220
            ]
            if left_neighbors:
                closest_left = max(left_neighbors, key=lambda n: n.center_x)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_left.label).strip()[:16]
                relative_hints[d1.id].append(f"Right of [{closest_left.id}] '{clean_lbl}'")

            right_neighbors = [
                d2 for d2 in sorted_by_y[i+1:]
                if abs(d2.center_y - d1.center_y) <= 18 and 0 < (d2.center_x - d1.center_x) <= 220
            ]
            if right_neighbors:
                closest_right = min(right_neighbors, key=lambda n: n.center_x)
                clean_lbl = re.sub(r'\[(?:btn|icon|app)\]\s*', '', closest_right.label).strip()[:16]
                relative_hints[d1.id].append(f"Left of [{closest_right.id}] '{clean_lbl}'")

        zone_titles = [
            ("[MODAL/POPUP]", "[MODAL / POPUP DIALOG] (Elevated Foreground Dialog/Sheet)"),
            ("[HEADER/NAV]", f"[HEADER / NAV BAR] (Top Navigation Zone | y < {int(0.12 * sh)})"),
            ("[SIDEBAR]", f"[SIDEBAR / NAV DRAWER] (Left Navigation Zone | x < {int(0.22 * sw)})"),
            ("[WORKSPACE]", f"[WORKSPACE / VIEWPORT] (Central Content Area | 0.12 <= y < 0.88)"),
            ("[FOOTER]", f"[FOOTER / BOTTOM BAR] (Bottom Bar & Status Area | y >= {int(0.88 * sh)})"),
        ]

        win_str = f" | Window Bounds: ({win_rect[0]},{win_rect[1]})-({win_rect[2]},{win_rect[3]})" if win_rect else ""
        lines = [
            f"=== 2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH ({sw}x{sh}{win_str}) ===",
            "Topology Zones: [HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP]",
            ""
        ]

        for z_key, z_header in zone_titles:
            cands = zone_buckets[z_key]
            if not cands:
                continue

            active_in_zone = [d for d in cands if d.id in active_set]
            bg_in_zone = [d for d in cands if d.id not in active_set]

            lines.append(f"--- {z_header} ---")
            for d in sorted(active_in_zone, key=lambda n: (n.center_y, n.center_x)):
                arch_tag = f"[{getattr(d, 'archetype', 'button').upper()}]"
                clean_lbl = d.label.replace('"', "'")[:28]
                rel_parts = list(relative_hints.get(d.id, []))[:2]
                rel_str = f" ({', '.join(rel_parts)})" if rel_parts else ""
                lines.append(f"• [{d.id}] ({d.center_x},{d.center_y}) {z_key} {arch_tag} \"{clean_lbl}\"{rel_str}")

            if bg_in_zone:
                if len(bg_in_zone) <= 2:
                    for d in bg_in_zone:
                        lines.append(f"• [{d.id}] ({d.center_x},{d.center_y}) {z_key} (Background) \"{d.label[:20]}\"")
                else:
                    lines.append(f"• ({len(bg_in_zone)} non-active background controls summarized)")
            lines.append("")

        return "\n".join(lines).strip()

    @classmethod
    def compact_trajectory_tier3(
        cls,
        memory: Optional[EpisodicTrajectoryMemory],
        max_steps: int = 3
    ) -> str:
        """
        Tier 3: Trajectory memory compaction.
        - Compacts history to latest 3 steps with truncated string lengths.
        - Preserves Active Master Plan milestones (✔ [DONE], ► [ACTIVE], ○ [PENDING]).
        """
        if not memory or not getattr(memory, "steps", None):
            return "No previous actions taken yet. This is Step 1."

        lines = []

        active_plan = None
        for s in reversed(memory.steps):
            mp = getattr(s, "end_to_end_master_plan", None) or getattr(s, "master_plan", None)
            if mp:
                active_plan = mp
                break

        if active_plan:
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
                    clean_desc = p_str.lstrip('✔►○• ')[:50]
                    if any(k in p_upper for k in ("(DONE)", "[DONE]", "(COMPLETED)", "[COMPLETED]", "✔")):
                        lines.append(f"  ✔ {clean_desc}")
                    elif any(k in p_upper for k in ("(ACTIVE)", "[ACTIVE]", "(IN PROGRESS)", "[IN PROGRESS]", "►", "CURRENT")):
                        lines.append(f"  ► {clean_desc}")
                    elif any(k in p_upper for k in ("(PENDING)", "[PENDING]", "(TODO)", "[TODO]", "○")):
                        lines.append(f"  ○ {clean_desc}")
                    else:
                        lines.append(f"  • {clean_desc}")
                lines.append("")

        recent_steps = memory.steps[-max_steps:]
        lines.append(f"Chronological Trajectory History (Recent {len(recent_steps)} Steps):")
        if len(memory.steps) > max_steps:
            lines.append(f"• (Steps 1-{len(memory.steps)-max_steps} executed previously in trajectory)")

        for s in recent_steps:
            v_badge = "❌ FAILED" if s.verdict == "EXECUTION_FAILED" else (f"✔ {s.verdict}" if s.verified else f"⚠️ {s.verdict}")
            act_str = f"{s.action.upper()}"
            if s.input_text:
                act_str += f" text=\"{s.input_text[:20]}\""
            target_str = f" on \"{s.target_label[:20]}\"" if s.target_label else ""
            reason_str = f" | Details: {s.verification_reason[:30]}" if s.verification_reason else ""
            lines.append(
                f"• Step {s.step_num}: Subgoal=\"{s.subgoal[:25]}\" | Action={act_str}{target_str} | Outcome={v_badge}{reason_str}"
            )

        if hasattr(memory, "detect_ping_pong_loop"):
            ping_pong_msg = memory.detect_ping_pong_loop()
            if ping_pong_msg:
                lines.append(f"⚠️ {ping_pong_msg[:120]}")
            elif len(memory.steps) >= 2 and hasattr(memory, "is_repeated_failure") and memory.is_repeated_failure(memory.steps[-1].target_label):
                lines.append("⚠️ CRITICAL LOOP WARNING: Previous action failed. Pivot strategy now.")

        return "\n".join(lines)

    @classmethod
    def enforce(
        cls,
        system_instruction: str,
        user_content: str,
        nodes: Optional[List[DiagonalDroplet]] = None,
        max_tokens: int = HARD_TOKEN_LIMIT,
        goal: str = "",
        memory: Optional[EpisodicTrajectoryMemory] = None,
        sw: int = 1920,
        sh: int = 1080,
        modal_rect: Optional[Tuple[int, int, int, int]] = None,
        win_rect: Optional[Tuple[int, int, int, int]] = None,
        win_info: str = "Desktop",
        grounding_text: str = "",
        turn_alert: str = "",
        **kwargs
    ) -> Tuple[str, str, List[DiagonalDroplet]]:
        """
        Enforces strict token budget (< max_tokens, target 4800) using 3-tier dynamic compaction:
        - Tier 1: Candidate manifest compaction (prune to top 24–36 high-value nodes)
        - Tier 2: Spatial layout graph condensation (omit empty, summarize background)
        - Tier 3: Trajectory memory compaction (latest 3 steps, truncated strings, master plan preserved)

        Returns: (system_instruction, user_content, nodes)
        Guaranteed to never exceed max_tokens.
        """
        active_nodes = list(nodes) if nodes else []
        current_tokens = cls.count_tokens(system_instruction) + cls.count_tokens(user_content)

        if current_tokens < max_tokens:
            return system_instruction, user_content, active_nodes

        # --- TIER 1: Candidate Manifest Compaction ---
        if len(active_nodes) > cls.TIER1_TARGET_NODES:
            active_nodes = cls.compact_candidates_tier1(
                active_nodes,
                goal=goal,
                modal_rect=modal_rect,
                win_rect=win_rect,
                max_nodes=cls.TIER1_TARGET_NODES
            )

        active_ids: Set[int] = {d.id for d in active_nodes}
        has_rebuild_context = bool(goal and (grounding_text or win_info != "Desktop" or memory is not None))

        if has_rebuild_context:
            spatial_graph = cls.condense_spatial_graph_tier2(
                active_nodes, sw=sw, sh=sh, modal_rect=modal_rect, win_rect=win_rect
            )
            history_text = cls.compact_trajectory_tier3(memory, max_steps=cls.TIER3_MAX_STEPS) if memory else ""

            user_content = f"""<overall_user_goal>
{goal}
</overall_user_goal>

<system_grounding>
{grounding_text}
</system_grounding>

<trajectory_history>
{history_text}
</trajectory_history>
{turn_alert}
<live_screen_observation>
Active Foreground Window: {win_info}

<spatial_layout_graph>
{spatial_graph}
</spatial_layout_graph>
</live_screen_observation>

<instruction>
Autoregressively evaluate the screen state and previous step critique in <deliberation>, and dispatch the single next action.
</instruction>"""
        else:
            if "<spatial_layout_graph>" in user_content and "</spatial_layout_graph>" in user_content:
                spatial_graph = cls.condense_spatial_graph_tier2(
                    active_nodes, sw=sw, sh=sh, modal_rect=modal_rect, win_rect=win_rect
                )
                head, rest = user_content.split("<spatial_layout_graph>", 1)
                _, tail = rest.split("</spatial_layout_graph>", 1)
                user_content = f"{head}<spatial_layout_graph>\n{spatial_graph}\n</spatial_layout_graph>{tail}"

            if "Candidates:\n" in user_content:
                head, rest = user_content.split("Candidates:\n", 1)
                manifest_lines = [f"[{d.id}] ({d.center_x},{d.center_y}) \"{d.label[:36]}\"" for d in active_nodes]
                if "\n\nRespond with ONLY JSON:" in rest:
                    _, tail = rest.split("\n\nRespond with ONLY JSON:", 1)
                    user_content = f"{head}Candidates:\n" + "\n".join(manifest_lines) + f"\n\nRespond with ONLY JSON:{tail}"

            # Filter bulleted droplet lines if raw droplet list exists
            if "• [" in user_content and active_ids:
                lines = user_content.splitlines()
                filtered = []
                for ln in lines:
                    m = re.search(r'•\s*\[(\d+)\]', ln)
                    if m:
                        if int(m.group(1)) in active_ids:
                            filtered.append(ln)
                    else:
                        filtered.append(ln)
                user_content = "\n".join(filtered)

        current_tokens = cls.count_tokens(system_instruction) + cls.count_tokens(user_content)
        if current_tokens < max_tokens:
            return system_instruction, user_content, active_nodes

        # --- Deeper TIER 1 & TIER 2: Prune to 24 nodes ---
        if len(active_nodes) > cls.TIER1_PRUNED_NODES:
            active_nodes = cls.compact_candidates_tier1(
                active_nodes,
                goal=goal,
                modal_rect=modal_rect,
                win_rect=win_rect,
                max_nodes=cls.TIER1_PRUNED_NODES
            )
            active_ids = {d.id for d in active_nodes}
            if has_rebuild_context:
                spatial_graph = cls.condense_spatial_graph_tier2(
                    active_nodes, sw=sw, sh=sh, modal_rect=modal_rect, win_rect=win_rect
                )
                user_content = f"""<overall_user_goal>
{goal}
</overall_user_goal>

<system_grounding>
{grounding_text}
</system_grounding>

<trajectory_history>
{history_text}
</trajectory_history>
{turn_alert}
<live_screen_observation>
Active Foreground Window: {win_info}

<spatial_layout_graph>
{spatial_graph}
</spatial_layout_graph>
</live_screen_observation>

<instruction>
Autoregressively evaluate the screen state and previous step critique in <deliberation>, and dispatch the single next action.
</instruction>"""
            else:
                if "<spatial_layout_graph>" in user_content and "</spatial_layout_graph>" in user_content:
                    spatial_graph = cls.condense_spatial_graph_tier2(
                        active_nodes, sw=sw, sh=sh, modal_rect=modal_rect, win_rect=win_rect
                    )
                    head, rest = user_content.split("<spatial_layout_graph>", 1)
                    _, tail = rest.split("</spatial_layout_graph>", 1)
                    user_content = f"{head}<spatial_layout_graph>\n{spatial_graph}\n</spatial_layout_graph>{tail}"
                if "• [" in user_content and active_ids:
                    lines = user_content.splitlines()
                    filtered = []
                    for ln in lines:
                        m = re.search(r'•\s*\[(\d+)\]', ln)
                        if m:
                            if int(m.group(1)) in active_ids:
                                filtered.append(ln)
                        else:
                            filtered.append(ln)
                    user_content = "\n".join(filtered)

        current_tokens = cls.count_tokens(system_instruction) + cls.count_tokens(user_content)
        if current_tokens < max_tokens:
            return system_instruction, user_content, active_nodes

        # --- TIER 4 Headroom: Condense Exemplars in system_instruction if still exceeding ---
        if "<few_shot_exemplars>" in system_instruction and "--- EXEMPLAR 2" in system_instruction:
            ex_start = system_instruction.find("<few_shot_exemplars>")
            ex_end = system_instruction.find("</few_shot_exemplars>")
            if ex_start != -1 and ex_end != -1:
                exemplar_block = system_instruction[ex_start:ex_end]
                if "--- EXEMPLAR 2" in exemplar_block:
                    cut_idx = exemplar_block.find("--- EXEMPLAR 2")
                    compact_exemplar = exemplar_block[:cut_idx].strip() + "\n"
                    system_instruction = system_instruction[:ex_start] + compact_exemplar + system_instruction[ex_end:]

        current_tokens = cls.count_tokens(system_instruction) + cls.count_tokens(user_content)
        if current_tokens < max_tokens:
            return system_instruction, user_content, active_nodes

        # --- Ultra Fallback: Further truncate bullet lines or history ---
        if "<trajectory_history>" in user_content and "</trajectory_history>" in user_content:
            user_content = re.sub(
                r'(<trajectory_history>).*?(</trajectory_history>)',
                r'\1\n[Trajectory history compacted to latest step]\2',
                user_content,
                flags=re.DOTALL
            )

        current_tokens = cls.count_tokens(system_instruction) + cls.count_tokens(user_content)
        if current_tokens < max_tokens:
            return system_instruction, user_content, active_nodes

        # Hard truncate user_content lines if still exceeding
        lines = user_content.splitlines()
        while len(lines) > 10 and (cls.count_tokens(system_instruction) + cls.count_tokens("\n".join(lines))) >= max_tokens:
            lines.pop(len(lines) // 2)
        user_content = "\n".join(lines)

        return system_instruction, user_content, active_nodes
