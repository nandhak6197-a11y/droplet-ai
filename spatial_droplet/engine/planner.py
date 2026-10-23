"""
Hierarchical Multi-Step Goal Planner & Self-Correction State Machine (Pillar 3).
Decomposes complex multi-stage desktop tasks into verifiable sub-goals, tracks execution
state, and autonomously recovers from intermediate modal blocks and failed actions.
"""
import time
import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from ..core.models import DiagonalDroplet, WindowContext
from ..vision.settle_sensor import AdaptiveSettleSensor
from ..telemetry.flight_recorder import FlightRecorder, ExecutionTrace, StageTiming


class TaskState(Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING_STEP = "EXECUTING_STEP"
    VERIFYING_STEP = "VERIFYING_STEP"
    RECOVERY = "RECOVERY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PlanStep:
    step_index: int
    instruction: str
    expected_action: str = "click"
    input_text: Optional[str] = None
    expected_outcome: str = ""
    status: str = "PENDING"             # "PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"
    retry_count: int = 0
    max_retries: int = 2
    execution_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalPlan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    is_completed: bool = False
    error: str = ""


class HierarchicalGoalPlanner:
    """
    Decomposes high-level instructions and executes multi-step plans
    with closed-loop verification and autonomous recovery state machines.
    """

    def __init__(self, max_recovery_per_step: int = 2):
        self.max_recovery_per_step = max_recovery_per_step
        self.state = TaskState.IDLE
        self.current_plan: Optional[GoalPlan] = None
        self.recorder = FlightRecorder.get_instance()

    def decompose_goal(self, goal: str) -> GoalPlan:
        """
        Decomposes complex multi-action instructions into an ordered sequence of PlanSteps.
        Handles chained commands (e.g., 'open notepad, type hello, then save').
        """
        clean_goal = goal.strip()
        steps: List[PlanStep] = []

        # Split on sequential conjunctions: 'then', 'and then', 'after that', commas followed by verbs
        delimiters = r'\s+(?:then|and\s+then|after\s+that)\s+|,\s*(?=open|click|type|select|press|close)'
        raw_parts = [p.strip().rstrip(",;.") for p in re.split(delimiters, clean_goal, flags=re.IGNORECASE) if p.strip().rstrip(",;.")]

        if len(raw_parts) <= 1:
            # Single-step goal
            steps.append(PlanStep(
                step_index=0,
                instruction=clean_goal.rstrip(",;."),
                expected_outcome=f"Completed {clean_goal}"
            ))
        else:
            for idx, part in enumerate(raw_parts):
                # Infer action type
                p_lower = part.lower()
                action = "click"
                inp_text = None

                if p_lower.startswith("type ") or p_lower.startswith("write ") or p_lower.startswith("enter "):
                    action = "type"
                    m = re.match(r'^(?:type|write|enter)\s+["\']?(.*?)["\']?$', part, re.IGNORECASE)
                    if m:
                        inp_text = m.group(1).rstrip("'\"")
                elif p_lower.startswith("press ") or p_lower in ("enter", "esc", "tab"):
                    action = "press"
                    inp_text = part.replace("press ", "").strip()
                elif p_lower.startswith("double click"):
                    action = "double_click"
                elif p_lower.startswith("open ") or p_lower.startswith("launch "):
                    action = "open_path"
                    inp_text = part.replace("open ", "").replace("launch ", "").strip()

                steps.append(PlanStep(
                    step_index=idx,
                    instruction=part,
                    expected_action=action,
                    input_text=inp_text,
                    expected_outcome=f"Executed: {part}"
                ))

        plan = GoalPlan(goal=clean_goal, steps=steps)
        self.current_plan = plan
        return plan

    async def execute_plan(
        self,
        goal: str,
        reasoner: Any,
        grabber: Any,
        sprayer: Any,
        actuator: Any,
        verifier: Any,
        sw: int = 1920,
        sh: int = 1080
    ) -> Dict[str, Any]:
        """
        Executes a goal plan through the TaskStateMachine with live self-correction.
        """
        self.state = TaskState.PLANNING
        plan = self.decompose_goal(goal)
        print(f"\n[Planner] [PLAN] Decomposed Goal into {len(plan.steps)} steps:")
        for s in plan.steps:
            print(f"   Step {s.step_index + 1}: \"{s.instruction}\" (Action: {s.expected_action})")

        overall_trace_id = self.recorder.start_trace(goal)

        for step in plan.steps:
            plan.current_step_index = step.step_index
            step.status = "IN_PROGRESS"
            self.state = TaskState.EXECUTING_STEP

            step_success = False

            while step.retry_count <= step.max_retries and not step_success:
                print(f"\n[Planner] [STEP] Executing Step {step.step_index + 1}/{len(plan.steps)}: \"{step.instruction}\" (Attempt {step.retry_count + 1})")
                t_start = time.time()
                timing = StageTiming()

                # 1. Dynamic UI Settle Sensor
                t0 = time.time()
                is_settled, settle_elapsed, frame = AdaptiveSettleSensor.wait_for_settle(
                    grabber_func=grabber.grab_full,
                    timeout_sec=0.50,
                    motion_threshold=0.8
                )
                timing.settle_ms = (time.time() - t0) * 1000

                if frame is None:
                    frame = grabber.grab_full()

                # 2. Spray Droplets
                t0 = time.time()
                win_ctx = None
                try:
                    from ..platform.window import get_active_window_context
                    win_ctx = get_active_window_context()
                except Exception:
                    pass

                droplets = await sprayer.spray(frame, win_ctx=win_ctx)
                timing.spray_ms = (time.time() - t0) * 1000

                # Option 3: Adaptive Long-Term Knowledge Base Injection
                kb_context = self.recorder.query_knowledge(step.instruction)

                # 3. Multimodal Reasoning Decision
                t0 = time.time()
                decision = await reasoner.decide(
                    user_intent=step.instruction,
                    droplets=droplets,
                    window_ctx=win_ctx,
                    screenshot=frame,
                    mode="combined",
                    memory=kb_context  # Option 3 Injection
                )
                timing.reason_ms = (time.time() - t0) * 1000

                target_id = decision.get("target_dot_id") or decision.get("target_droplet_id")
                action_to_dispatch = step.expected_action if step.expected_action not in ("click", "") else (decision.get("action_type") or decision.get("action") or "click")
                target_node = next((d for d in droplets if d.id == target_id), None) if target_id is not None else None

                # Option 1: Autonomous "Scroll & Seek" Vision Engine (Visual Pagination)
                if target_node is None and decision.get("requires_scroll", True):
                    print(f"[Planner] [VISION ENGINE] Target not found in view. Triggering autonomous scroll & seek pagination...")
                    try:
                        actuator.execute(droplet=None, action="scroll", scroll_amount=-300)
                        await asyncio.sleep(0.4)
                        # Re-run capture, spray, and deliberation
                        frame = grabber.grab_full()
                        droplets = await sprayer.spray(frame, win_ctx=win_ctx)
                        decision = await reasoner.decide(
                            user_intent=step.instruction,
                            droplets=droplets,
                            window_ctx=win_ctx,
                            screenshot=frame,
                            mode="combined"
                        )
                        target_id = decision.get("target_dot_id") or decision.get("target_droplet_id")
                        action_to_dispatch = step.expected_action if step.expected_action not in ("click", "") else (decision.get("action_type") or decision.get("action") or "click")
                        target_node = next((d for d in droplets if d.id == target_id), None) if target_id is not None else None
                    except Exception as e:
                        print(f"[Planner] [VISION ENGINE] Error during scroll & seek: {e}")

                # 4. Actuation
                t0 = time.time()
                drag_dest = None
                if decision.get("drag_destination_dot_id"):
                    dest_node = next((d for d in droplets if d.id == decision.get("drag_destination_dot_id")), None)
                    if dest_node:
                        drag_dest = (dest_node.center_x, dest_node.center_y)
                elif decision.get("drag_dest_x") is not None and decision.get("drag_dest_y") is not None:
                    drag_dest = (int(decision.get("drag_dest_x")), int(decision.get("drag_dest_y")))
                    
                exec_result = actuator.execute(
                    droplet=target_node,
                    action=action_to_dispatch,
                    input_text=step.input_text or decision.get("input_text"),
                    win_ctx=win_ctx,
                    target_x=int(decision.get("target_x")) if decision.get("target_x") is not None else (int(decision.get("x")) if decision.get("x") is not None else None),
                    target_y=int(decision.get("target_y")) if decision.get("target_y") is not None else (int(decision.get("y")) if decision.get("y") is not None else None),
                    drag_dest=drag_dest
                )
                timing.actuate_ms = (time.time() - t0) * 1000

                # 5. Verification
                self.state = TaskState.VERIFYING_STEP
                ver_status = exec_result.get("verified", True)
                step.execution_result = exec_result

                trace = ExecutionTrace(
                    trace_id=f"{overall_trace_id}_s{step.step_index}",
                    task_goal=step.instruction,
                    status="SUCCESS" if ver_status else "VERIFICATION_FAILED",
                    total_latency_ms=(time.time() - t_start) * 1000,
                    timings=timing,
                    target_dot_id=target_id,
                    action_type=action_to_dispatch,
                    target_coords=[int(decision.get("target_x") or decision.get("x") or 0), int(decision.get("target_y") or decision.get("y") or 0)],
                    provider=decision.get("provider", "groq"),
                    model=decision.get("model_used", ""),
                    verification_outcome=exec_result
                )
                self.recorder.record_trace(trace)

                if ver_status:
                    step.status = "COMPLETED"
                    step_success = True
                    print(f"[Planner] [OK] Step {step.step_index + 1} succeeded: {exec_result.get('reason', '')}")
                    # Option 3: Adaptive Long-Term Knowledge Base Injection
                    self.recorder.update_knowledge(step.instruction, str(target_id), str(decision.get("reasoning", "")))
                else:
                    step.retry_count += 1
                    self.state = TaskState.RECOVERY
                    print(f"[Planner] [WARN] Step {step.step_index + 1} failed verification: {exec_result.get('reason', '')}")

                    # Self-Correction Protocol
                    if exec_result.get("suggested_pivot"):
                        print(f"[Planner] [RECOVERY] Auto-Recovery pivot: {exec_result.get('suggested_pivot')}")
                        alt_id = exec_result.get("suggested_alternative_id")
                        if alt_id is not None:
                            # Direct runner-up execution attempt
                            alt_node = next((d for d in droplets if d.id == alt_id), None)
                            if alt_node:
                                print(f"[Planner] [RECOVERY] Retrying immediately with runner-up Dot [{alt_id}]...")
                                retry_res = actuator.execute(
                                    droplet=alt_node,
                                    action=action_to_dispatch,
                                    input_text=step.input_text
                                )
                                if retry_res.get("verified", False):
                                    step.status = "COMPLETED"
                                    step_success = True
                                    print(f"[Planner] [OK] Auto-Recovery succeeded with Dot [{alt_id}]!")
                                    break

                    # Modal Escape Hatch: if stuck, press Escape to clear unhandled popup
                    if not step_success and step.retry_count < step.max_retries:
                        print("[Planner] [ESCAPE] Triggering escape hatch (Escape key) to clear potential modal lock...")
                        actuator.execute(droplet=None, action="press", input_text="esc")
                        await asyncio.sleep(0.15)

            if not step_success:
                step.status = "FAILED"
                self.state = TaskState.FAILED
                plan.error = f"Step {step.step_index + 1} ('{step.instruction}') failed after {step.retry_count} attempts."
                print(f"[Planner] [FAILED] Task aborted: {plan.error}")

                # Log post-mortem forensic bundle
                post_frame = grabber.grab_full()
                self.recorder.log_incident(
                    trace=trace,
                    pre_screenshot=frame,
                    post_screenshot=post_frame,
                    deliberation_data=decision.to_dict() if hasattr(decision, 'to_dict') else None
                )

                return {
                    "success": False,
                    "plan": plan,
                    "error": plan.error,
                    "completed_steps": step.step_index
                }

        self.state = TaskState.COMPLETED
        plan.is_completed = True
        print(f"\n[Planner] [COMPLETED] Goal completed successfully: \"{goal}\"")
        return {
            "success": True,
            "plan": plan,
            "completed_steps": len(plan.steps)
        }
