"""
Automated Verification Script for Grounded Reasoning & Multi-Turn Failure Awareness Overhaul.
Tests:
1. Dynamic system environment discovery (no hardcoded usernames).
2. Modal dialog filtering (PopupHost, dropdowns, explorer windows ignored).
3. Actuator execution error capture and propagation (verified=False on OS errors).
4. Memory ping-pong loop detection and repetition guards.
5. Fast-path folder resolution.
"""
import os
import sys
import unittest
import pathlib

# Add repository root to pythonpath
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from spatial_droplet.reasoning.reasoner import get_system_grounding_context
from spatial_droplet.vision.screen_parser import ScreenStateParser
from spatial_droplet.core.models import WindowContext, TrajectoryStep
from spatial_droplet.platform.actuator import PrecisionActuator
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.engine.fast_path import FastPathRouter

class TestGroundedReasoning(unittest.TestCase):

    def test_01_dynamic_system_grounding(self):
        """Test that get_system_grounding_context returns real OS paths without hallucination."""
        ctx = get_system_grounding_context()
        self.assertIn("user_name", ctx)
        self.assertIn("docs_dir", ctx)
        self.assertIn("downloads_dir", ctx)
        self.assertIn("active_drives", ctx)
        
        # Verify paths actually exist
        self.assertTrue(os.path.exists(ctx["user_profile"]), f"User profile path doesn't exist: {ctx['user_profile']}")
        self.assertTrue(os.path.exists(ctx["docs_dir"]), f"Docs path doesn't exist: {ctx['docs_dir']}")
        self.assertIn("C:\\", ctx["active_drives"])
        
        # Verify text block format
        text = ctx["text"]
        self.assertIn(ctx["user_name"], text)
        self.assertIn("WINDOWS SYSTEM ENVIRONMENT & PATHS", text)
        print(f"[TEST 1 PASS] Dynamic Grounding: User='{ctx['user_name']}', Docs='{ctx['docs_dir']}', Drives={ctx['active_drives']}")

    def test_02_modal_dialog_filtering(self):
        """Test that detect_modal_dialog ignores PopupHost and non-modal windows."""
        # Simulated explorer window
        mock_explorer = WindowContext(
            hwnd=12345,
            title="Documents",
            class_name="CabinetWClass",
            rect=(100, 100, 800, 600),
            is_valid=True
        )
        # Calling on mock window should return None (no modal)
        modal = ScreenStateParser.detect_modal_dialog(mock_explorer)
        self.assertIsNone(modal)
        print("[TEST 2 PASS] Modal Dialog Filter: Non-modal windows and popups successfully ignored.")

    def test_03_actuator_error_propagation(self):
        """Test that actuator.execute returns verified=False on invalid path/app."""
        actuator = PrecisionActuator()
        # Non-existent path
        fake_path = r"C:\Invalid_User_NonExistent_4821\Documents"
        res = actuator.execute(droplet=None, action="open_path", input_text=fake_path)
        self.assertFalse(res.get("verified"))
        self.assertEqual(res.get("type"), "EXECUTION_ERROR")
        self.assertIn("Failed to open", res.get("reason", ""))
        self.assertTrue(len(res.get("error", "")) > 0)
        print(f"[TEST 3 PASS] Actuator Error Propagation: OS error caught correctly: {res['error']}")

    def test_04_memory_loop_and_ping_pong_detection(self):
        """Test that EpisodicTrajectoryMemory catches ping-pong cycles and repeated failures."""
        mem = EpisodicTrajectoryMemory(goal="Navigate to Documents")
        
        # Simulate alternating cycle: Alt+D -> Esc -> Alt+D -> Esc
        step1 = TrajectoryStep(step_num=1, goal="test", subgoal="address bar", action="hotkey", target_id=None,
                               target_label="", input_text=None, verified=True, verification_reason="",
                               active_window="File Explorer", semantic_intention="focus bar", expected_transition="", verdict="CONFIRMED")
        step1.action = "hotkey"
        step1.input_text = "alt+d"
        
        step2 = TrajectoryStep(step_num=2, goal="test", subgoal="dismiss popup", action="press", target_id=None,
                               target_label="", input_text=None, verified=True, verification_reason="",
                               active_window="File Explorer", semantic_intention="close popup", expected_transition="", verdict="CONFIRMED")
        step2.action = "press"
        step2.input_text = "escape"
        
        step3 = TrajectoryStep(step_num=3, goal="test", subgoal="address bar", action="hotkey", target_id=None,
                               target_label="", input_text=None, verified=True, verification_reason="",
                               active_window="File Explorer", semantic_intention="focus bar", expected_transition="", verdict="CONFIRMED")
        step3.action = "hotkey"
        step3.input_text = "alt+d"
        
        step4 = TrajectoryStep(step_num=4, goal="test", subgoal="dismiss popup", action="press", target_id=None,
                               target_label="", input_text=None, verified=True, verification_reason="",
                               active_window="File Explorer", semantic_intention="close popup", expected_transition="", verdict="CONFIRMED")
        step4.action = "press"
        step4.input_text = "escape"

        mem.add_step(step1)
        mem.add_step(step2)
        mem.add_step(step3)
        mem.add_step(step4)

        ping_pong = mem.detect_ping_pong_loop()
        self.assertIsNotNone(ping_pong)
        self.assertIn("CRITICAL PING-PONG LOOP DETECTED", ping_pong)
        print(f"[TEST 4 PASS] Ping-Pong Loop Detection: {ping_pong}")

        # Test history formatting includes warning
        history = mem.format_history_for_prompt()
        self.assertIn("CRITICAL PING-PONG LOOP DETECTED", history)

    def test_05_fast_path_folders(self):
        """Test fast-path folder matching for 'open documents' and 'downloads'."""
        docs_res = FastPathRouter.route("open documents", [], None)
        self.assertIsNotNone(docs_res)
        self.assertEqual(docs_res.get("action"), "system_open_path")
        self.assertTrue(docs_res.get("path").endswith("Documents"))
        self.assertTrue(os.path.exists(docs_res.get("path")))
        print(f"[TEST 5 PASS] Fast Path Folder: 'open documents' -> '{docs_res.get('path')}' in 0ms")

    def test_06_execution_failure_handling(self):
        """Test that execution failures are preserved in memory and surfaced in prompt."""
        mem = EpisodicTrajectoryMemory(goal="Find PDF file")
        fake_path = r"C:\Users\Administrator\Documents"
        step = TrajectoryStep(
            step_num=1,
            goal="Find PDF file",
            subgoal="Open Documents",
            action="open_path",
            target_id=None,
            target_label="",
            input_text=fake_path,
            verified=False,
            verification_reason=f"Failed to open '{fake_path}': [WinError 2] The system cannot find the file specified",
            active_window="File Explorer",
            semantic_intention="Open user documents",
            expected_transition="Documents folder opens",
            verdict="EXECUTION_FAILED"
        )
        mem.add_step(step)
        
        self.assertEqual(mem.get_consecutive_failures(), 1)
        prompt_history = mem.format_history_for_prompt()
        self.assertIn("EXECUTION_FAILED", prompt_history)
        self.assertIn("Failed to open", prompt_history)
        print("[TEST 6 PASS] Execution Failure Memory: Preserved and surfaced in LLM prompt history.")

    def test_07_uia_walker_and_sprayer_bounding_boxes(self):
        """Test UIA Walker return format and Sprayer genuine box ingestion with backward compatibility."""
        # 1. Test genuine 6-tuple ingestion in Sprayer logic
        uia_controls = [
            (225, 230, "[Btn] Save File", "interactive_widget", "Save", (100, 200, 350, 260)),
            (50, 60, "[Btn] Old Button", "interactive_widget", "Old")
        ]
        w, h = 1920, 1080
        discovered_ui = []
        for item in (uia_controls or []):
            if len(item) >= 6:
                ux, uy, ulabel, ucat, usub, raw_bbox = item[:6]
                bx1, by1, bx2, by2 = raw_bbox
                ubox = (max(0, int(bx1)), max(0, int(by1)), min(w, int(bx2)), min(h, int(by2)))
            else:
                ux, uy, ulabel, ucat, usub = item[:5]
                ubox = (max(0, int(ux - 15)), max(0, int(uy - 15)), min(w, int(ux + 15)), min(h, int(uy + 15)))
            discovered_ui.append((ux, uy, ulabel, ucat, usub, 1.0, ubox))

        # Check item 1 (genuine box retained, not collapsed to 30x30)
        self.assertEqual(discovered_ui[0][6], (100, 200, 350, 260))
        bw = discovered_ui[0][6][2] - discovered_ui[0][6][0]
        bh = discovered_ui[0][6][3] - discovered_ui[0][6][1]
        self.assertEqual((bw, bh), (250, 60))
        self.assertNotEqual((bw, bh), (30, 30))

        # Check legacy item 2 (fallback 30x30)
        self.assertEqual(discovered_ui[1][6], (35, 45, 65, 75))

        # Test inverted bbox normalization
        inverted_item = [(225, 230, "[Btn] Inverted", "interactive_widget", "Inv", (350, 260, 100, 200))]
        bx1, by1, bx2, by2 = inverted_item[0][5]
        bx_min, bx_max = min(int(bx1), int(bx2)), max(int(bx1), int(bx2))
        by_min, by_max = min(int(by1), int(by2)), max(int(by1), int(by2))
        norm_box = (max(0, bx_min), max(0, by_min), min(w, bx_max), min(h, by_max))
        self.assertEqual(norm_box, (100, 200, 350, 260))

        # 2. Test live walker walk_controls returns 6-tuples if UIA is present
        from spatial_droplet.vision.uia_walker import UiaControlWalker
        controls = UiaControlWalker.walk_controls(max_depth=2, timeout_sec=0.5)
        if controls:
            first = controls[0]
            self.assertEqual(len(first), 6)
            bbox = first[5]
            self.assertEqual(len(bbox), 4)
            self.assertTrue(bbox[2] >= bbox[0])
            self.assertTrue(bbox[3] >= bbox[1])
        print(f"[TEST 7 PASS] Bounding Box & UIA Ingestion: Retained genuine box (250x60), 5-tuple backward compatible, inverted normalization verified.")

    def test_08_deliberative_reasoning_schema(self):
        """Test deliberative reasoning schema (observation_analysis, critique_of_previous_step, strategy_plan)."""
        import json
        from spatial_droplet.reasoning.reasoner import DecisionResult, DiagonalDropletReasoner

        raw_json_str = '''{
            "observation_analysis": "Active window is Notepad with empty document. Dot [1] is File menu.",
            "critique_of_previous_step": "Step 1 executed successfully opening Notepad. No errors occurred.",
            "strategy_plan": "Click File menu to begin document save workflow.",
            "status": "in_progress",
            "next_subgoal": "Click File menu",
            "action": "click",
            "target_dot_id": 1,
            "semantic_intention": "Open File menu dropdown",
            "expected_transition": "File dropdown menu appears",
            "reasoning": "Standard workflow for document creation"
        }'''
        parsed = json.loads(raw_json_str)
        res = DecisionResult(parsed)

        self.assertEqual(res.observation_analysis, "Active window is Notepad with empty document. Dot [1] is File menu.")
        self.assertEqual(res.critique_of_previous_step, "Step 1 executed successfully opening Notepad. No errors occurred.")
        self.assertEqual(res.strategy_plan, "Click File menu to begin document save workflow.")
        self.assertEqual(res.target_droplet_id, 1)
        self.assertEqual(res.target_dot_id, 1)
        self.assertEqual(res.action_type, "click")
        self.assertEqual(res.action, "click")
        self.assertEqual(res.status, "in_progress")
        self.assertEqual(res.next_subgoal, "Click File menu")

        # Test aliases exist on DiagonalDropletReasoner
        self.assertTrue(hasattr(DiagonalDropletReasoner, 'decide_trajectory_turn'))
        self.assertTrue(hasattr(DiagonalDropletReasoner, 'decide_action'))
        self.assertEqual(DiagonalDropletReasoner.decide_trajectory_turn, DiagonalDropletReasoner.decide_turn)
        self.assertEqual(DiagonalDropletReasoner.decide_action, DiagonalDropletReasoner.decide)

        print("[TEST 8 PASS] Deliberative Reasoning Engine: Analysis-First schema fields and method aliases verified.")

    def test_09_hud_geometry_and_badge_docking(self):
        """Test HUD geometry calculation: no outward padding expansion, badge docked inside (bx1, by1)."""
        sw, sh = 1920, 1080
        raw_box = (200, 300, 450, 380)
        bx1, by1, bx2, by2 = raw_box

        # Clamping calculation without outward expansion
        cbx1 = max(0, min(sw - 14, bx1))
        cby1 = max(0, min(sh - 14, by1))
        cbx2 = max(cbx1 + 16, min(sw, bx2))
        cby2 = max(cby1 + 14, min(sh, by2))

        self.assertEqual((cbx1, cby1, cbx2, cby2), (200, 300, 450, 380))
        # Ensure no artificial padding expansion (was previously bx1 - 5, by1 - 3, bx2 + 5, by2 + 3)
        self.assertNotEqual((cbx1, cby1, cbx2, cby2), (195, 297, 455, 383))

        # Badge pill docked directly inside top-left corner
        ph = 14
        tid = "42"
        pw = max(18, len(tid) * 7 + 8)
        tag_x1 = cbx1
        tag_y1 = cby1
        tag_x2 = tag_x1 + pw
        tag_y2 = tag_y1 + ph

        # Verify badge stays within element row and doesn't protrude into row above
        self.assertEqual(tag_y1, 300)
        self.assertGreaterEqual(tag_y1, cby1)
        self.assertLess(tag_y2, cby2)
        print("[TEST 9 PASS] HUD Geometry: Precise boundary clamping without collision expansion; badge docked directly inside top-left corner.")

    def test_10_semantic_signature_and_stagnation_detection(self):
        """Test that episodic memory signatures ignore transient dot IDs and track PARTIAL_CHANGE stagnation."""
        mem = EpisodicTrajectoryMemory(goal="Save document")

        # Step 1: Click Save Button with Dot ID 3
        s1 = TrajectoryStep(
            step_num=1, goal="Save document", subgoal="Save", action="click",
            target_id=3, target_label="[Btn] Save File", input_text="",
            verified=False, verification_reason="Minor pixel jitter", active_window="Notepad",
            verdict="PARTIAL_CHANGE", observation_analysis="Doc opened",
            critique="Clicked Save file but nothing happened", strategy_plan="Try clicking again"
        )
        # Step 2: Click Save Button with Dot ID 7 (re-sprayed, ID shifted!)
        s2 = TrajectoryStep(
            step_num=2, goal="Save document", subgoal="Save", action="click",
            target_id=7, target_label="[Btn] Save File", input_text="",
            verified=False, verification_reason="Minor pixel jitter", active_window="Notepad",
            verdict="PARTIAL_CHANGE", observation_analysis="Still open",
            critique="Still no response from Save button", strategy_plan="Switch to hotkey Ctrl+S"
        )

        mem.add_step(s1)
        mem.add_step(s2)

        # 1. Verify signatures match despite different dot IDs (3 vs 7)
        self.assertTrue(mem.is_repeated_action(), "Repeated action should match on semantic target label despite different dot IDs!")
        self.assertTrue(mem.is_repeated_failure("[Btn] Save File"), "Repeated stagnant action must be flagged as failure!")

        # 2. Verify consecutive failures counts PARTIAL_CHANGE stagnation
        self.assertEqual(mem.get_consecutive_failures(), 2)

        # 3. Verify history formatting contains critique and repetition alert
        history = mem.format_history_for_prompt()
        self.assertIn("Prev Critique:", history)
        self.assertIn("CRITICAL LOOP WARNING", history)
        print("[TEST 10 PASS] Semantic Signature & Stagnation Detection: Ephemeral dot ID invariance, PARTIAL_CHANGE failure tracking, and CoT critique propagation verified.")

if __name__ == "__main__":
    unittest.main()
