"""
Adversarial Empirical Challenge Test Suite for Milestone 2:
Token Budgeting & Dynamic Compaction Stress Verifier.

Tests TokenBudgetGuard.enforce(...) and its tiers under extreme adversarial loads:
1. 150-200 heavy droplets with verbose multi-line descriptions.
2. Massive trajectory histories (30+ steps) and long goals (1,000+ chars).
3. Token ceiling verification (strictly < 4,800 tokens and < 5,000 hard limit).
4. Modal dialog control preservation (modal_rect) during Tier 1 pruning.
5. Active foreground window control preservation (win_rect).
6. Master plan milestone badge preservation (✔ [DONE], ► [ACTIVE], ○ [PENDING]).
7. Ultra-extreme combined stress load (simultaneous 200 droplets, 35 steps, 1,500-char goal, modal dialog).
"""
import re
from typing import Optional, Any
import pytest
import tiktoken
from unittest.mock import patch

from spatial_droplet.core.models import DiagonalDroplet, TrajectoryStep
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.reasoning.token_guard import TokenBudgetGuard


def count_tokens_cl100k(text: str) -> int:
    """Helper to get ground-truth cl100k_base token count."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def make_droplet(
    id: int,
    center_x: int,
    center_y: int,
    category: str = "button",
    label: str = "Button",
    diag_wave: Optional[int] = None,
    is_foreground: bool = True,
    archetype: str = "button",
    **kwargs
) -> DiagonalDroplet:
    if diag_wave is None:
        diag_wave = center_x + center_y
    return DiagonalDroplet(
        id=id,
        center_x=center_x,
        center_y=center_y,
        category=category,
        label=label,
        diag_wave=diag_wave,
        is_foreground=is_foreground,
        archetype=archetype,
        **kwargs
    )


def make_step(
    step_num: int = 1,
    goal: str = "Default Goal",
    subgoal: str = "Default Subgoal",
    action: str = "click",
    target_id: Optional[int] = 1,
    target_label: str = "Target",
    input_text: Optional[str] = "",
    verified: bool = True,
    verification_reason: str = "Verified",
    active_window: str = "Desktop",
    end_to_end_master_plan: Any = None,
    **kwargs
) -> TrajectoryStep:
    if end_to_end_master_plan is None:
        end_to_end_master_plan = []
    return TrajectoryStep(
        step_num=step_num,
        goal=goal,
        subgoal=subgoal,
        action=action,
        target_id=target_id,
        target_label=target_label,
        input_text=input_text,
        verified=verified,
        verification_reason=verification_reason,
        active_window=active_window,
        end_to_end_master_plan=end_to_end_master_plan,
        **kwargs
    )


class TestTokenBudgetGuardAdversarialDroplets:
    """Adversarial stress testing with 150–200 heavy droplets and verbose labels."""

    def test_200_heavy_droplets_strict_budget_enforcement(self):
        """
        Adversarially feeds 200 heavy droplets with complex multi-sentence labels.
        Verifies prompt tokens strictly stay < 4,800 tokens (< 5,000 ceiling).
        """
        droplets = [
            make_droplet(
                id=i,
                center_x=(i * 37) % 1900 + 10,
                center_y=(i * 23) % 1000 + 20,
                category="button" if i % 3 == 0 else "interactive_widget",
                label=(
                    f"Complex Enterprise UI Component Node #{i}: Sub-panel selector with nested "
                    f"hierarchical contextual attributes, telemetry monitoring status indicators, "
                    f"and dynamic operational parameters for automated execution workflow."
                ),
                archetype="button" if i % 2 == 0 else "input",
                is_foreground=(i % 4 != 0)
            )
            for i in range(200)
        ]

        system_instruction = (
            "<system>\n"
            "Autonomous Cognitive Spatial Desktop Automation Engine operating with 5-stage CoT deliberation.\n"
            "</system>\n"
            "<few_shot_exemplars>\n"
            "--- EXEMPLAR 1 ---\n"
            "Deliberate carefully across user_intent_and_success_criteria, screen_state_and_visual_hierarchy.\n"
            "--- EXEMPLAR 2 ---\n"
            "Evaluate mistake critique and maintain end_to_end_master_plan.\n"
            "</few_shot_exemplars>\n"
        )
        user_content = (
            "<overall_user_goal>Execute diagnostic validation of cloud microservices and verify metrics.</overall_user_goal>\n"
            "<live_screen_observation>\n"
            + "\n".join(f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label}" for d in droplets)
            + "\n</live_screen_observation>"
        )

        initial_tokens = count_tokens_cl100k(system_instruction + user_content)
        assert initial_tokens > 5000, f"Setup must start >5000 tokens, got {initial_tokens}"

        s_out, u_out, n_out = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=droplets,
            max_tokens=4800,
            goal="Execute diagnostic validation of cloud microservices and verify metrics.",
            sw=1920,
            sh=1080
        )

        total_tokens = count_tokens_cl100k(s_out + u_out)
        assert total_tokens < 4800, f"Total tokens {total_tokens} exceeded strict 4,800 target!"
        assert total_tokens < 5000, f"Total tokens {total_tokens} exceeded 5,000 hard ceiling!"
        assert len(n_out) <= 36, f"Tier 1 must prune nodes to <= 36, got {len(n_out)}"

    def test_180_droplets_with_rebuild_context_stays_under_budget(self):
        """
        Feeds 180 heavy droplets with full rebuild context (grounding_text, win_info, memory).
        Verifies rebuilt user_content remains strictly < 4,800 tokens.
        """
        droplets = [
            make_droplet(
                id=i,
                center_x=(i * 41) % 1900 + 10,
                center_y=(i * 31) % 1000 + 20,
                category="button",
                label=f"Action Control Gateway Node #{i} with verified hardware event listeners across multi-tiered architecture",
                archetype="button"
            )
            for i in range(180)
        ]

        memory = EpisodicTrajectoryMemory("Automate invoice processing")
        for step_idx in range(1, 10):
            memory.add_step(make_step(
                step_num=step_idx,
                goal="Automate invoice processing in ERP system and confirm batch entry",
                subgoal=f"Subgoal processing step #{step_idx} with extended deliberation and validation logic",
                action="click",
                target_id=step_idx,
                target_label=f"Invoice Button {step_idx}",
                verified=True,
                verification_reason="State transition verified by absdiff motion detector and window title sensor"
            ))

        # Initial oversized prompt (>5,000 tokens)
        system_instruction = (
            "<system>Operational Invariants 5-Stage Deliberation</system>\n"
            + "<rules>Desktop automation directives and grounding principles.</rules>\n" * 150
        )
        user_content = (
            "<live_screen_observation>\n"
            + "\n".join(f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label}" for d in droplets)
            + "\n</live_screen_observation>"
        )

        initial_tokens = count_tokens_cl100k(system_instruction + user_content)
        assert initial_tokens > 4800, f"Setup must start >4800 tokens, got {initial_tokens}"

        s_out, u_out, n_out = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=droplets,
            max_tokens=4800,
            goal="Automate invoice processing in ERP system and confirm batch entry",
            memory=memory,
            sw=1920,
            sh=1080,
            win_info="ERP Accounting Suite v4.2 - [Active Ledger]",
            grounding_text="OS: Windows 11 Enterprise | Active DPI: 100% | Resolution: 1920x1080"
        )

        total_tokens = count_tokens_cl100k(s_out + u_out)
        assert total_tokens < 4800
        assert total_tokens < 5000
        assert "<spatial_layout_graph>" in u_out
        assert "<trajectory_history>" in u_out


class TestTokenBudgetGuardAdversarialTrajectoryAndGoal:
    """Adversarial stress testing with massive trajectory histories (30+ steps) and long goals."""

    def test_35_trajectory_steps_compaction_and_master_plan_preservation(self):
        """
        Adversarially feeds a 35-step trajectory history with verbose critique and justifications.
        Verifies:
        1. Compaction limits recent steps to 3.
        2. Older steps 1-32 are summarized.
        3. Master plan milestone badges (✔ [DONE], ► [ACTIVE], ○ [PENDING]) are strictly preserved.
        """
        memory = EpisodicTrajectoryMemory("Complete multi-phase enterprise data migration")
        master_plan = [
            "✔ [DONE] Phase 1: Connect to legacy Oracle database and extract schema",
            "✔ [DONE] Phase 2: Validate table integrity and row counts across all 48 tables",
            "► [ACTIVE] Phase 3: Transform records into target JSON schema and stream load",
            "○ [PENDING] Phase 4: Verify foreign key constraints and index generation",
            "○ [PENDING] Phase 5: Generate audit report and notify deployment channel"
        ]

        for i in range(1, 36):
            step = make_step(
                step_num=i,
                goal="Complete multi-phase enterprise data migration",
                subgoal=f"Execute migration sub-task index {i} in pipeline sequence",
                action="click" if i % 2 == 0 else "type_and_enter",
                target_id=i,
                target_label=f"Pipeline Control Widget {i}",
                input_text=f"batch_param_{i}" if i % 2 != 0 else "",
                verified=(i != 14),
                verdict="VERIFIED_TITLE_AND_MOTION" if i != 14 else "EXECUTION_FAILED",
                verification_reason=f"Transition check completed for step {i}",
                active_window="Enterprise Migration Studio v3",
                end_to_end_master_plan=master_plan
            )
            memory.add_step(step)

        compact_hist = TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=3)

        # Verify master plan badges are preserved
        assert "Active Master Plan:" in compact_hist
        assert "✔" in compact_hist
        assert "►" in compact_hist
        assert "○" in compact_hist
        assert "Phase 1: Connect to legacy Oracle database" in compact_hist
        assert "Phase 2: Validate table integrity" in compact_hist
        assert "Phase 3: Transform records" in compact_hist
        assert "Phase 4: Verify foreign key" in compact_hist
        assert "Phase 5: Generate audit report" in compact_hist

        # Verify recent steps constraint
        assert "Recent 3 Steps" in compact_hist
        assert "Steps 1-32 executed previously in trajectory" in compact_hist
        assert "Step 35:" in compact_hist
        assert "Step 34:" in compact_hist
        assert "Step 33:" in compact_hist
        assert "Step 1:" not in compact_hist
        assert "Step 30:" not in compact_hist

    def test_long_goal_1200_chars_with_30_steps_stays_under_budget(self):
        """
        Adversarially feeds a long goal (>1,200 chars) with 30 trajectory steps and 100 droplets.
        Verifies total tokens strictly stay < 4,800.
        """
        long_goal = (
            "You are tasked with a comprehensive autonomous system administration procedure. "
            "First, navigate to the system control panel and inspect background telemetry services. "
            "Verify that all distributed worker daemon threads are running with CPU usage under 45%. "
            "Next, locate the error log directory at C:\\Logs\\Production\\2026-10-22\\ and scan for any fatal stack traces. "
            "If any critical memory leaks or socket connection timeouts are identified, trigger an automated memory dump. "
            "Subsequently, open the database migration console and initiate a staged schema rollback to checkpoint 4096. "
            "Ensure that transactional locks are gracefully released before restarting the service cluster. "
            "Finally, generate a cryptographically signed SHA-256 diagnostic report, attach it to the audit ledger, "
            "and alert the engineering operations center via the automated notification pipeline. "
            "Do not close any elevated administrative consoles until all verification hashes match the master ledger. "
            "Maintain complete audit logs across all subsystem checkpoints to ensure full traceability."
        )
        assert len(long_goal) > 1000, f"Goal must be >1000 chars, got {len(long_goal)}"

        memory = EpisodicTrajectoryMemory(long_goal)
        for i in range(1, 31):
            memory.add_step(make_step(
                step_num=i,
                goal=long_goal[:80],
                subgoal=f"Administrative task step #{i}",
                action="click",
                target_id=i,
                target_label=f"Admin Tool Button {i}",
                verified=True,
                verification_reason="Action verified",
                end_to_end_master_plan=[
                    "✔ [DONE] Milestone 1: Inspect telemetry",
                    "► [ACTIVE] Milestone 2: Scan error logs",
                    "○ [PENDING] Milestone 3: Database schema rollback"
                ]
            ))

        droplets = [
            make_droplet(
                id=i,
                center_x=100 + (i % 10) * 150,
                center_y=100 + (i // 10) * 80,
                category="button",
                label=f"Administrative Dashboard Console Button Index {i}: System Control Point with telemetry verification handler"
            )
            for i in range(120)
        ]

        system_text = "<system>Desktop Automation</system>\n" + "<policy>Strict execution compliance.</policy>\n" * 200
        user_text = f"<goal>{long_goal}</goal>\n" + "\n".join(f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label}" for d in droplets)

        initial_tokens = count_tokens_cl100k(system_text + user_text)
        assert initial_tokens > 4800, f"Expected initial tokens > 4800, got {initial_tokens}"

        s_out, u_out, n_out = TokenBudgetGuard.enforce(
            system_instruction=system_text,
            user_content=user_text,
            nodes=droplets,
            max_tokens=4800,
            goal=long_goal,
            memory=memory,
            sw=1920,
            sh=1080,
            win_info="Administrative Console - [Elevated Privileges]",
            grounding_text="OS: Windows 11 Enterprise | DPI: 100%"
        )

        total_tokens = count_tokens_cl100k(s_out + u_out)
        assert total_tokens < 4800, f"Expected <4800 tokens, got {total_tokens}"
        assert total_tokens < 5000


class TestTokenBudgetGuardModalAndForegroundPreservation:
    """Adversarial stress testing for modal dialog controls and foreground window preservation."""

    def test_modal_dialog_controls_never_dropped_under_heavy_candidate_load(self):
        """
        Adversarially constructs a scenario with 160 droplets:
        - 4 critical controls inside modal dialog (modal_rect=(600, 350, 1100, 750)):
          Save, Cancel, Confirm, Replace.
        - 156 distractor droplets in background/other zones.
        Verifies all 4 modal dialog controls are prioritized and NEVER dropped during Tier 1 pruning.
        """
        modal_rect = (600, 350, 1100, 750)
        win_rect = (200, 150, 1600, 950)

        # 4 Modal Dialog Controls
        modal_controls = [
            make_droplet(id=1, center_x=700, center_y=700, category="button", label="Save", is_foreground=True),
            make_droplet(id=2, center_x=850, center_y=700, category="button", label="Cancel", is_foreground=True),
            make_droplet(id=3, center_x=1000, center_y=700, category="button", label="Confirm", is_foreground=True),
            make_droplet(id=4, center_x=800, center_y=500, category="button", label="Replace existing file", is_foreground=True),
        ]

        # 156 Background / Distractor droplets
        distractors = [
            make_droplet(
                id=10 + i,
                center_x=50 + (i % 15) * 120,
                center_y=50 + (i // 15) * 80,
                category="button" if i % 2 == 0 else "interactive_widget",
                label=f"Background Workspace Widget Node {i}",
                is_foreground=False
            )
            for i in range(156)
        ]

        all_nodes = modal_controls + distractors
        assert len(all_nodes) == 160

        pruned = TokenBudgetGuard.compact_candidates_tier1(
            all_nodes,
            goal="Save the document and replace existing file",
            modal_rect=modal_rect,
            win_rect=win_rect,
            max_nodes=36
        )

        pruned_ids = {d.id for d in pruned}
        for mc in modal_controls:
            assert mc.id in pruned_ids, f"CRITICAL BUG: Modal control [{mc.id}] '{mc.label}' was dropped during Tier 1 pruning!"

    def test_modal_controls_with_unmatched_labels_prioritized_over_distractors_with_keyword_hits(self):
        """
        ADVERSARIAL STRESS:
        Background distractors have labels with 2-3 goal keyword matches.
        Modal controls have generic labels ("OK", "Dismiss", "Retry", "Details") with NO direct goal words.
        Verifies modal controls (+1000 modal boost) are still prioritized over distractors.
        """
        modal_rect = (500, 300, 1000, 700)
        win_rect = (100, 100, 1400, 900)
        goal = "export financial report spreadsheet summary data"

        # Modal controls (labels don't match goal words, but have dialog tokens or pure modal containment)
        modal_ctrl_1 = make_droplet(id=1, center_x=600, center_y=650, category="button", label="OK", is_foreground=True)
        modal_ctrl_2 = make_droplet(id=2, center_x=750, center_y=650, category="button", label="Cancel", is_foreground=True)
        modal_ctrl_3 = make_droplet(id=3, center_x=900, center_y=650, category="button", label="View Details", is_foreground=True)

        # Distractor controls with 2-3 goal words: "export financial report" (+800 to +1200 points)
        distractors = [
            make_droplet(
                id=100 + i,
                center_x=50 + (i % 10) * 100,
                center_y=50 + (i // 10) * 100,
                category="button",
                label=f"export financial report widget {i}",
                is_foreground=False
            )
            for i in range(50)
        ]

        all_nodes = [modal_ctrl_1, modal_ctrl_2, modal_ctrl_3] + distractors
        pruned = TokenBudgetGuard.compact_candidates_tier1(
            all_nodes,
            goal=goal,
            modal_rect=modal_rect,
            win_rect=win_rect,
            max_nodes=24
        )

        pruned_ids = {d.id for d in pruned}
        assert 1 in pruned_ids, "Modal control 'OK' must be retained!"
        assert 2 in pruned_ids, "Modal control 'Cancel' must be retained!"
        assert 3 in pruned_ids, "Modal control 'View Details' must be retained!"

    def test_vulnerability_modal_controls_dropped_when_distractors_have_heavy_goal_keyword_matches(self):
        """
        EMPIRICAL VULNERABILITY TEST:
        Demonstrates that when background distractors have 5+ goal keyword matches (+2,000+ points),
        they outrank and cause modal dialog controls (score 1550) to be dropped during Tier 1 pruning.
        Goal: 'invoice accounting ledger payment balance tax calculate' (7 keywords)
        Modal node score: 1000 (modal) + 250 (fg) + 150 (win) + 150 (btn) = 1550
        Distractor score: 7 * 400 = 2800 (+ 150 btn) = 2950
        Result: Modal node is evictable when there are >= 24 such distractors!
        """
        modal_rect = (500, 300, 1000, 700)
        win_rect = (100, 100, 1400, 900)
        goal = "invoice accounting ledger payment balance tax calculate"

        modal_node = make_droplet(id=1, center_x=600, center_y=600, category="button", label="Proceed", is_foreground=True)
        bg_nodes = [
            make_droplet(id=10 + i, center_x=50, center_y=50, category="button", label="invoice accounting ledger payment balance tax calculate widget", is_foreground=False)
            for i in range(40)
        ]

        pruned = TokenBudgetGuard.compact_candidates_tier1(
            [modal_node] + bg_nodes,
            goal=goal,
            modal_rect=modal_rect,
            win_rect=win_rect,
            max_nodes=24
        )
        pruned_ids = {d.id for d in pruned}
        # Invariant requirement: modal controls must never be dropped.
        # This assert fails under the current implementation, demonstrating the vulnerability.
        assert 1 in pruned_ids, "Modal control [1] 'Proceed' was evicted by keyword-rich background distractors!"

    def test_foreground_window_controls_prioritized_over_background(self):
        """
        Verifies foreground active window controls (win_rect) are scored and prioritized
        over non-foreground background controls.
        """
        win_rect = (300, 200, 1200, 800)

        fg_nodes = [
            make_droplet(id=i, center_x=400 + i * 50, center_y=300, category="button", label=f"App Control {i}", is_foreground=True)
            for i in range(15)
        ]
        bg_nodes = [
            make_droplet(id=100 + i, center_x=50 + i * 30, center_y=950, category="interactive_widget", label=f"Taskbar App {i}", is_foreground=False)
            for i in range(50)
        ]

        pruned = TokenBudgetGuard.compact_candidates_tier1(
            fg_nodes + bg_nodes,
            goal="Perform task in application",
            win_rect=win_rect,
            max_nodes=15
        )

        pruned_ids = {d.id for d in pruned}
        # All 15 foreground window nodes should be selected over background taskbar nodes
        for fg in fg_nodes:
            assert fg.id in pruned_ids, f"Foreground node [{fg.id}] should be retained over background!"


class TestTokenBudgetGuardMasterPlanBadgePreservation:
    """Adversarial stress testing for Active Master Plan badge preservation."""

    @pytest.mark.parametrize("badge,expected_symbol", [
        ("[DONE]", "✔"),
        ("(DONE)", "✔"),
        ("✔", "✔"),
        ("[COMPLETED]", "✔"),
        ("[ACTIVE]", "►"),
        ("(ACTIVE)", "►"),
        ("►", "►"),
        ("[IN PROGRESS]", "►"),
        ("[PENDING]", "○"),
        ("(PENDING)", "○"),
        ("○", "○"),
        ("[TODO]", "○"),
    ])
    def test_all_badge_variations_correctly_classified_and_rendered(self, badge, expected_symbol):
        """Verifies diverse milestone status badge formats map to canonical symbols."""
        memory = EpisodicTrajectoryMemory("Test master plan badges")
        step = make_step(
            step_num=1,
            goal="Test badges",
            subgoal="Subgoal 1",
            action="click",
            verified=True,
            end_to_end_master_plan=[
                f"{badge} Task Description for milestone item"
            ]
        )
        memory.add_step(step)

        compact_hist = TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=3)
        assert expected_symbol in compact_hist
        assert "Task Description for milestone item" in compact_hist

    def test_master_plan_preservation_when_stored_as_string_or_list(self):
        """Verifies master plan parsing handles both multiline string and list of strings."""
        # Test as multiline string
        memory_str = EpisodicTrajectoryMemory("Goal")
        memory_str.add_step(make_step(
            step_num=1,
            goal="Goal",
            subgoal="Subgoal",
            action="click",
            verified=True,
            end_to_end_master_plan="✔ [DONE] Step 1\n► [ACTIVE] Step 2\n○ [PENDING] Step 3"
        ))
        res_str = TokenBudgetGuard.compact_trajectory_tier3(memory_str, max_steps=3)
        assert "✔" in res_str and "Step 1" in res_str
        assert "►" in res_str and "Step 2" in res_str
        assert "○" in res_str and "Step 3" in res_str

        # Test as list of strings
        memory_list = EpisodicTrajectoryMemory("Goal")
        memory_list.add_step(make_step(
            step_num=1,
            goal="Goal",
            subgoal="Subgoal",
            action="click",
            verified=True,
            end_to_end_master_plan=["✔ [DONE] Step 1", "► [ACTIVE] Step 2", "○ [PENDING] Step 3"]
        ))
        res_list = TokenBudgetGuard.compact_trajectory_tier3(memory_list, max_steps=3)
        assert "✔" in res_list and "Step 1" in res_list
        assert "►" in res_list and "Step 2" in res_list
        assert "○" in res_list and "Step 3" in res_list


class TestTokenBudgetGuardUltraExtremeSimultaneousStress:
    """Extreme stress test combining ALL simultaneous stresses."""

    def test_ultra_extreme_stress_combined(self):
        """
        Adversarially combines:
        - 200 heavy droplets with verbose labels.
        - 35 trajectory steps with extensive critique and master plan.
        - 1,500-character goal with multi-stage directives.
        - Active modal dialog with critical confirmation buttons.
        - Active foreground application window bounds.
        - Multi-exemplar system instruction (>1,500 tokens).

        Verifies:
        1. Final prompt strictly stays < 4,800 tokens (< 5,000 ceiling).
        2. Modal dialog controls are retained in active nodes.
        3. Master plan badges (✔, ►, ○) are present in prompt.
        4. No critical structural XML tags (<instruction>, <deliberation>, etc.) are corrupted.
        """
        modal_rect = (550, 320, 1050, 720)
        win_rect = (150, 100, 1500, 950)

        # 4 Modal controls
        modal_nodes = [
            make_droplet(id=1, center_x=650, center_y=670, category="button", label="Commit Changes", is_foreground=True),
            make_droplet(id=2, center_x=800, center_y=670, category="button", label="Discard and Revert", is_foreground=True),
            make_droplet(id=3, center_x=950, center_y=670, category="button", label="Confirm Action", is_foreground=True),
            make_droplet(id=4, center_x=700, center_y=450, category="text", label="Target Branch Name", is_foreground=True),
        ]

        # 196 Other droplets across screen
        other_nodes = [
            make_droplet(
                id=5 + i,
                center_x=(i * 31) % 1900 + 10,
                center_y=(i * 29) % 1000 + 20,
                category="button" if i % 2 == 0 else "interactive_widget",
                label=f"Extreme Telemetry Diagnostic Node Index {i} in Automated Control Infrastructure Panel with hardware telemetry listeners",
                archetype="button" if i % 2 == 0 else "slider",
                is_foreground=(i < 30)
            )
            for i in range(196)
        ]
        all_droplets = modal_nodes + other_nodes
        assert len(all_droplets) == 200

        goal = (
            "Perform an end-to-end repository migration and branch release workflow. "
            "Examine all open pull requests and verify that continuous integration tests pass with 0 errors. "
            "Inspect the active modal dialog to confirm target branch name and commit changes. "
            "If any merge conflicts or unresolved review comments exist, discard and revert the stage immediately. "
            "Otherwise, confirm action and initiate the release pipeline to production cluster nodes. "
            "Ensure that all metrics are logged, audit traces are validated, and the team lead is notified via Slack webhook."
        ) * 3  # ~1,500 chars

        memory = EpisodicTrajectoryMemory(goal)
        for i in range(1, 36):
            memory.add_step(make_step(
                step_num=i,
                goal=goal[:60],
                subgoal=f"Release workflow step {i}",
                action="click" if i % 2 == 0 else "type_and_enter",
                target_id=i,
                target_label=f"Release Widget {i}",
                input_text=f"input_{i}" if i % 2 != 0 else "",
                verified=(i != 12),
                verification_reason=f"Transition verified at step {i}",
                active_window="Git Release Manager v2.1",
                end_to_end_master_plan=[
                    "✔ [DONE] Verify CI test suite passes",
                    "► [ACTIVE] Inspect modal dialog and commit changes",
                    "○ [PENDING] Deploy to production cluster"
                ]
            ))

        system_instruction = (
            "<system>\n"
            "Autonomous Cognitive Spatial Desktop Automation Engine controlling Windows 11.\n"
            "You MUST deliberate across 5 cognitive pillars:\n"
            "1. user_intent_and_success_criteria\n"
            "2. screen_state_and_visual_hierarchy\n"
            "3. mistake_critique_and_loop_diagnosis\n"
            "4. end_to_end_master_plan\n"
            "5. tactical_action_justification\n"
            "</system>\n"
            "<few_shot_exemplars>\n"
            "--- EXEMPLAR 1: Modal Dialog Commit ---\n"
            "When modal dialog is active, prioritize modal controls over background.\n"
            "--- EXEMPLAR 2: Window Management ---\n"
            "When window needs resize, invoke minimize or maximize.\n"
            "</few_shot_exemplars>\n"
        )

        user_content = (
            f"<overall_user_goal>\n{goal}\n</overall_user_goal>\n"
            f"<trajectory_history>\n{TokenBudgetGuard.compact_trajectory_tier3(memory, max_steps=3)}\n</trajectory_history>\n"
            f"<live_screen_observation>\nActive Foreground Window: Git Release Manager v2.1\n"
            f"<spatial_layout_graph>\n"
            + "\n".join(f"• [{d.id}] ({d.center_x},{d.center_y}) {d.label}" for d in all_droplets)
            + "\n</spatial_layout_graph>\n</live_screen_observation>\n"
            "<instruction>\nAutoregressively evaluate the screen state and previous step critique in <deliberation>, and dispatch the single next action.\n</instruction>"
        )

        initial_tokens = count_tokens_cl100k(system_instruction + user_content)
        assert initial_tokens > 4800, f"Expected initial tokens > 4800, got {initial_tokens}"

        s_out, u_out, n_out = TokenBudgetGuard.enforce(
            system_instruction=system_instruction,
            user_content=user_content,
            nodes=all_droplets,
            max_tokens=4800,
            goal=goal,
            memory=memory,
            sw=1920,
            sh=1080,
            modal_rect=modal_rect,
            win_rect=win_rect,
            win_info="Git Release Manager v2.1 [Modal Commit Active]",
            grounding_text="OS: Windows 11 Enterprise | DPI: 100%"
        )

        final_tokens = count_tokens_cl100k(s_out + u_out)
        assert final_tokens < 4800, f"Final tokens {final_tokens} exceeded 4,800!"
        assert final_tokens < 5000, f"Final tokens {final_tokens} exceeded 5,000 ceiling!"

        # Modal controls preservation
        final_ids = {d.id for d in n_out}
        for mc in modal_nodes:
            assert mc.id in final_ids, f"Modal control [{mc.id}] '{mc.label}' must not be dropped!"

        # Master plan badges preservation
        assert "Active Master Plan:" in u_out
        assert "✔" in u_out
        assert "►" in u_out
        assert "○" in u_out

        # Structural integrity
        assert "<instruction>" in u_out
        assert "</instruction>" in u_out
        assert "<overall_user_goal>" in u_out
        assert "</overall_user_goal>" in u_out
