import os, sys

target_path = 'spatial_droplet/reasoning/reasoner.py'
with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. In _clean_decision: normalize minimize and maximize
canon_anchor = '        if "action_type" not in decision or not decision["action_type"]:\n            decision["action_type"] = decision["action"]'
canon_replacement = '''        if "action_type" not in decision or not decision["action_type"]:
            decision["action_type"] = decision["action"]
        if decision["action"] in ("minimize", "minimize_window"):
            decision["action"] = "minimize"
            decision["action_type"] = "minimize"
        elif decision["action"] in ("maximize", "maximize_window"):
            decision["action"] = "maximize"
            decision["action_type"] = "maximize"'''

if '("minimize", "minimize_window")' not in content:
    assert canon_anchor in content, "canon_anchor not found"
    content = content.replace(canon_anchor, canon_replacement, 1)

# 2. In decide_turn: change max_ui_nodes from 60 to 36
max_nodes_old = "        key_nodes = [d for d in droplets if d.category != 'surface_probe']\n        max_ui_nodes = 60"
max_nodes_new = "        key_nodes = [d for d in droplets if d.category != 'surface_probe']\n        max_ui_nodes = 36"
if max_nodes_old in content:
    content = content.replace(max_nodes_old, max_nodes_new, 1)

# 3. Streamline system_instruction, exemplars, and user_content in decide_turn ONLY
p_func = content.find("def decide_turn(")
assert p_func != -1, "def decide_turn not found"

p_start = content.find('        system_instruction = f"""<system_capability>', p_func)
p_end = content.find('        # Sync with live environment / UI model selector if changed', p_start)
assert p_start != -1 and p_end != -1, f"Could not find prompt block in decide_turn: {p_start}, {p_end}"

new_prompt_block = '''        system_instruction = f"""<system_capability>
Role: Autonomous Cognitive Spatial Desktop Automation Brain controlling a Windows PC.
Architecture: Middle Router between Foundation Models, Trajectory Orchestrator, and OS Input Dispatch.
Grounding: Spatial Droplet Set-of-Marks live visual grounding, 2D Visual Spatial Layout & Connectivity Graph, and Windows UIA sensory state.
Active Foreground Window: {win_info}
</system_capability>

<constitution>
Operational Invariants (NON-NEGOTIABLE):
1. 5-STAGE DEEP DELIBERATION: Every turn requires deep autoregressive deliberation across:
   - user_intent_and_success_criteria: Target application, goal requirements, implicit constraints, visual completion proof.
   - screen_state_and_visual_hierarchy: Foreground window focus, blocking modals, 2D functional screen zones ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP]).
   - mistake_critique_and_loop_diagnosis: Prior action critique, root-cause failure diagnosis, avoidance of repeated traps.
   - end_to_end_master_plan: Milestone plan tracking across turns ([Done], [Active], [Pending]).
   - tactical_action_justification: Step-by-step justification for chosen action & Dot ID, with explicit distractor elimination.
2. STEP CRITIQUE & LOOP BREAKING:
   - If previous action had Outcome=NO_EFFECT, EXECUTION_FAILED, or PARTIAL_CHANGE, repeating the identical action or clicking the same element is STRICTLY FORBIDDEN.
   - You MUST diagnose the failure in "mistake_critique_and_loop_diagnosis" and switch tactical modality in "end_to_end_master_plan" (e.g. hotkey, double-click, direct "open_path", keyboard Enter, minimize, maximize).
   - NEVER alternate between two actions in a ping-pong loop.
3. GROUNDED WINDOWS PATHS & WINDOW CONTROLS:
   - Current Windows User: "{user_name}". Never guess generic accounts.
   - Folder/Drive Navigation: Use action "open_path" with input_text (e.g. "{docs_dir}", "{downloads_dir}", "D:\\\\") or action "hotkey" with key_combo ["win", "e"].
   - Application Launch & Window Switching: Use action "launch_app" with input_text (e.g. "calc", "notepad", "camera", "powershell") or action "switch_window".
   - Window State: To minimize current window, use action "minimize" or click the titlebar minimize button. To maximize, use action "maximize".
4. MODAL DIALOG PRECEDENCE:
   - In modals/prompts ("Save Changes", "Confirm Exit", "Save As"), interact directly with dialog choices (Save, Don't Save, Cancel) instead of clicking the main window Close button.
5. GOAL COMPLETION: When visual evidence confirms the user's overall goal is accomplished, output action "stop" and status "completed".
6. FLUID HUMAN ACTIONS:
   - Text replacement: "clear_and_type" with input_text. Form submit: "type_and_enter". Shortcuts: "copy", "paste", "select_all", "undo", "redo", "save", "find", "minimize", "maximize", "win_run".
</constitution>

<screen_context>
Foreground Window: {win_info}
{awareness_block}
</screen_context>

<few_shot_exemplars>
--- EXEMPLAR 1: In-App UI Interaction & Input ---
User Goal: "type meeting notes into notepad"
Observation: Notepad active in foreground. Editor canvas anchored at Dot [14] in [WORKSPACE].
Decision:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "Enter text 'meeting notes' into Notepad editor canvas.",
    "screen_state_and_visual_hierarchy": "Notepad editor canvas visible at Dot [14] in [WORKSPACE].",
    "mistake_critique_and_loop_diagnosis": "Initial turn; editor focused as expected.",
    "end_to_end_master_plan": [
      "[Phase 1] Type meeting notes into editor canvas Dot [14] (ACTIVE)",
      "[Phase 2] Verify text entered and complete (PENDING)"
    ],
    "tactical_action_justification": "Use click_and_type on Dot [14] to insert text directly into document editor."
  }},
  "status": "in_progress",
  "next_subgoal": "Type meeting notes into document",
  "action": "click_and_type",
  "target_dot_id": 14,
  "input_text": "meeting notes",
  "semantic_intention": "Type requested text into document editor",
  "expected_transition": "Text 'meeting notes' appears in Notepad editor",
  "reasoning": "Directly typing into the editor canvas inserts the user's text."
}}

--- EXEMPLAR 2: Goal Completion ---
User Goal: "open camera"
Observation: Windows Camera application is active in the foreground with live camera preview feed verified.
Decision:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "Open Windows Camera app. Success Criteria: Camera viewfinder active.",
    "screen_state_and_visual_hierarchy": "Camera app is running and visible in foreground [WORKSPACE].",
    "mistake_critique_and_loop_diagnosis": "Camera launched successfully in previous turn.",
    "end_to_end_master_plan": [
      "[Phase 1] Launch camera (DONE)",
      "[Phase 2] Verify viewfinder & complete (DONE)"
    ],
    "tactical_action_justification": "Goal conditions completely satisfied. Terminate trajectory successfully."
  }},
  "status": "completed",
  "next_subgoal": "Complete task",
  "action": "stop",
  "target_dot_id": null,
  "input_text": null,
  "semantic_intention": "Finish trajectory execution",
  "expected_transition": "None; task complete",
  "reasoning": "The Camera application is open and active in the foreground, satisfying all requirements of the user's goal."
}}
</few_shot_exemplars>

<action_dispatch>
You MUST respond with valid JSON where the "deliberation" object is generated FIRST:
{{
  "deliberation": {{
    "user_intent_and_success_criteria": "<query decomposition: target application, explicit & implicit constraints, completion proof>",
    "screen_state_and_visual_hierarchy": "<active window, focus state, blocking modals/spinners, 2D functional screen zones ([HEADER/NAV], [SIDEBAR], [WORKSPACE], [FOOTER], [MODAL/POPUP])>",
    "mistake_critique_and_loop_diagnosis": "<critique of previous turn outcome (NO_EFFECT, EXECUTION_FAILED, PARTIAL_CHANGE); diagnose root cause, forbid repeating mistakes>",
    "end_to_end_master_plan": [
      "[Phase 1] <milestone description> (DONE/ACTIVE/PENDING)",
      "[Phase 2] <milestone description> (PENDING)"
    ],
    "tactical_action_justification": "<step-by-step rationale for why chosen action & Dot ID are optimal, with explicit distractor elimination>"
  }},
  "status": "in_progress" | "completed" | "failed",
  "next_subgoal": "<brief description of what this single turn achieves>",
  "action": "click" | "double_click" | "triple_click" | "right_click" | "middle_click"
          | "ctrl_click" | "shift_click" | "alt_click"
          | "clear_and_type" | "type_and_enter" | "click_and_type" | "type"
          | "press" | "hotkey" | "copy" | "paste" | "select_all" | "undo" | "redo" | "save" | "find"
          | "hover" | "hover_and_wait" | "drag" | "scroll" | "open_path" | "launch_app" | "switch_window" | "minimize" | "maximize" | "win_run" | "wait" | "stop",
  "target_dot_id": <int or null (Required if interacting with a specific UI element)>,
  "x": <int or null (Absolute X coordinate for start/target if dot ID not used)>,
  "y": <int or null (Absolute Y coordinate for start/target if dot ID not used)>,
  "drag_destination_dot_id": <int or null (Required if action is 'drag' and target is an element)>,
  "drag_dest_x": <int or null (Absolute X coordinate for drag destination if dot ID not used)>,
  "drag_dest_y": <int or null (Absolute Y coordinate for drag destination if dot ID not used)>,
  "press_enter": <boolean (optional, if true commits text with Enter in clear_and_type or click_and_type)>,
  "key": "<single key name (e.g. enter, esc, tab, backspace, up, down, left, right)>" or null,
  "key_combo": <list of strings (e.g. ["ctrl", "c"], ["alt", "tab"]) or null>,
  "scroll_direction": "vertical" | "horizontal",
  "scroll_amount": <int or null>,
  "input_text": <string or null (Text to type or command to run)>,
  "semantic_intention": "<1-sentence clear statement of intent and purpose>",
  "expected_transition": "<exact predicted visual/text transition on screen>",
  "reasoning": "<1-sentence rationale connecting history and screen awareness to this action>"
}}
</action_dispatch>"""

        turn_alert = ""
        if loop_intervention:
            turn_alert = f"""<critical_loop_intervention>
{loop_intervention}
</critical_loop_intervention>
"""
        elif memory and memory.steps:
            last = memory.steps[-1]
            if not last.verified or last.verdict in ("NO_EFFECT", "EXECUTION_FAILED", "PARTIAL_CHANGE"):
                turn_alert = f"""<critical_step_alert>
Previous Action: {last.action.upper()}{f" on '{last.target_label}'" if last.target_label else ""}
Outcome: {last.verdict} ({last.verification_reason})
MANDATE: The previous action failed to make progress. Repeating this action is CONSTITUTIONALLY FORBIDDEN.
Diagnose the failure in "mistake_critique_and_loop_diagnosis" and pivot to an alternative tactical modality in "end_to_end_master_plan".
</critical_step_alert>
"""

        user_content = f"""<overall_user_goal>
{goal}
</overall_user_goal>

<system_grounding>
{grounding_text}
</system_grounding>

<trajectory_history>
{trajectory_history}
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

        # Proactive Groq Token Budget Guard (7000 ITPM limit)
        # If total estimated prompt exceeds 5000 tokens (~20,000 chars), prune candidates to top 24
        if (len(system_instruction) + len(user_content) > 19000) and len(nodes_for_ai) > 24:
            pruned_nodes = nodes_for_ai[:24]
            spatial_graph = self._build_spatial_layout_graph(pruned_nodes, int(sw_turn), int(sh_turn), modal_rect)
            user_content = f"""<overall_user_goal>
{goal}
</overall_user_goal>

<system_grounding>
{grounding_text}
</system_grounding>

<trajectory_history>
{trajectory_history}
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
'''

content = content[:p_start] + new_prompt_block + content[p_end:]

# 4. In rotation loop: catch 413 / request too large / itpm BEFORE rate_limit break
rot_old = '''                except Exception as e:
                    err_str = str(e)
                    last_err = f"Key #{active_idx + 1} ({masked_key}) [{model_name}]: {err_str}"
                    if any(k in err_str.lower() for k in ("429", "rate_limit", "tpm", "quota")):
                        break'''

rot_new = '''                except Exception as e:
                    err_str = str(e)
                    last_err = f"Key #{active_idx + 1} ({masked_key}) [{model_name}]: {err_str}"
                    if "413" in err_str or "request too large" in err_str.lower() or "itpm" in err_str.lower():
                        # Immediately compact user_content so subsequent keys succeed without failing the entire pool
                        print(f"[*] Prompt exceeded Groq rate limit ({err_str[:90]}); dynamically pruning prompt...")
                        lines_u = user_content.splitlines()
                        if len(lines_u) > 25:
                            user_content = "\\n".join(lines_u[:int(len(lines_u) * 0.60)] + [lines_u[-1]])
                        continue
                    if any(k in err_str.lower() for k in ("429", "rate_limit", "tpm", "quota")):
                        break'''

if rot_old in content:
    content = content.replace(rot_old, rot_new, 1)

with open(target_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully patched reasoner.py targeting decide_turn!")
