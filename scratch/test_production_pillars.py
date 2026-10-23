"""
Comprehensive Production Pillar Test Suite:
- Pillar 1: Dynamic UI Settle Sensor
- Pillar 2: Virtual Desktop Topology & Multi-Monitor Input
- Pillar 3: Hierarchical Goal Planner & Decomposer
- Pillar 5: Production Blackbox Flight Recorder & Telemetry
- Pillar 6: Subsystem Circuit Breaker Fault Isolation
"""
import os
import sys
import time
import shutil
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.vision.settle_sensor import AdaptiveSettleSensor
from spatial_droplet.platform.desktop import VirtualDesktopTopology
from spatial_droplet.platform.actuator import Win32HardwareActuator
from spatial_droplet.engine.planner import HierarchicalGoalPlanner, PlanStep, TaskState
from spatial_droplet.telemetry.flight_recorder import FlightRecorder, ExecutionTrace, StageTiming
from spatial_droplet.core.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerRegistry


class TestPillar1SettleSensor(unittest.TestCase):
    def test_motion_delta_identical(self):
        f1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        f2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        delta = AdaptiveSettleSensor.compute_motion_delta(f1, f2)
        self.assertEqual(delta, 0.0)

    def test_motion_delta_differing(self):
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.ones((100, 100, 3), dtype=np.uint8) * 10
        delta = AdaptiveSettleSensor.compute_motion_delta(f1, f2)
        self.assertEqual(delta, 10.0)

    def test_wait_for_settle_stabilization(self):
        frames = [
            np.ones((50, 50, 3), dtype=np.uint8) * 10,
            np.ones((50, 50, 3), dtype=np.uint8) * 20,
            np.ones((50, 50, 3), dtype=np.uint8) * 20,
            np.ones((50, 50, 3), dtype=np.uint8) * 20,
        ]
        idx = [0]
        def mock_grab():
            i = min(idx[0], len(frames) - 1)
            idx[0] += 1
            return frames[i]

        is_settled, elapsed, final_frame = AdaptiveSettleSensor.wait_for_settle(
            grabber_func=mock_grab,
            timeout_sec=0.5,
            min_dwell_sec=0.01,
            check_interval=0.01
        )
        self.assertTrue(is_settled)
        self.assertIsNotNone(final_frame)


class TestPillar2VirtualDesktopTopology(unittest.TestCase):
    def test_virtual_bounds_query(self):
        vx, vy, vw, vh = VirtualDesktopTopology.get_virtual_bounds()
        self.assertGreater(vw, 0)
        self.assertGreater(vh, 0)

    def test_normalized_coords_transformation(self):
        nx, ny, is_v = VirtualDesktopTopology.to_normalized_coords(500, 500)
        self.assertGreaterEqual(nx, 0)
        self.assertLessEqual(nx, 65535)
        self.assertGreaterEqual(ny, 0)
        self.assertLessEqual(ny, 65535)

    def test_actuator_norm_coords(self):
        nx, ny, v_flag = Win32HardwareActuator._to_norm_coords(500, 500)
        self.assertGreaterEqual(nx, 0)
        self.assertLessEqual(nx, 65535)


class TestPillar3GoalPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = HierarchicalGoalPlanner()

    def test_single_step_decomposition(self):
        plan = self.planner.decompose_goal("Click the Save button")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].instruction, "Click the Save button")
        self.assertEqual(plan.steps[0].expected_action, "click")

    def test_multi_step_decomposition(self):
        goal = "Open notepad, then type Hello World, and then press enter"
        plan = self.planner.decompose_goal(goal)
        self.assertEqual(len(plan.steps), 3)

        self.assertEqual(plan.steps[0].instruction, "Open notepad")
        self.assertEqual(plan.steps[0].expected_action, "open_path")

        self.assertIn("type", plan.steps[1].expected_action)
        self.assertEqual(plan.steps[1].input_text, "Hello World")

        self.assertEqual(plan.steps[2].expected_action, "press")
        self.assertEqual(plan.steps[2].input_text, "enter")


class TestPillar5FlightRecorder(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(".", "scratch", "test_incidents")
        self.recorder = FlightRecorder(incident_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_trace_lifecycle(self):
        trace_id = self.recorder.start_trace("Test Goal")
        trace = ExecutionTrace(
            trace_id=trace_id,
            task_goal="Test Goal",
            status="SUCCESS",
            total_latency_ms=120.5,
            timings=StageTiming(capture_ms=10.0, spray_ms=25.0, reason_ms=70.0, actuate_ms=5.0, verify_ms=10.5)
        )
        self.recorder.record_trace(trace)
        self.assertEqual(len(self.recorder.traces), 1)
        summary = self.recorder.get_latency_summary()
        self.assertEqual(summary["trace_count"], 1)
        self.assertAlmostEqual(summary["avg_total_ms"], 120.5, places=1)

    def test_log_incident_dump(self):
        trace = ExecutionTrace(
            trace_id="test_fail_01",
            task_goal="Click broken button",
            status="VERIFICATION_FAILED",
            total_latency_ms=200.0
        )
        pre_img = np.zeros((80, 80, 3), dtype=np.uint8)
        post_img = np.zeros((80, 80, 3), dtype=np.uint8)

        bundle_path = self.recorder.log_incident(
            trace=trace,
            pre_screenshot=pre_img,
            post_screenshot=post_img,
            manifest_text="[1] Button: Broken"
        )
        self.assertIsNotNone(bundle_path)
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "incident_summary.json")))
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "pre_action.png")))
        self.assertTrue(os.path.exists(os.path.join(bundle_path, "screen_droplet_manifest.txt")))


class TestPillar6CircuitBreaker(unittest.TestCase):
    def test_circuit_state_transition(self):
        breaker = CircuitBreaker(name="test_service", failure_threshold=2, cooldown_sec=0.2)
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertTrue(breaker.can_execute())

        # First failure
        breaker.record_failure(Exception("Fail 1"))
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertTrue(breaker.can_execute())

        # Second failure -> should trip OPEN
        breaker.record_failure(Exception("Fail 2"))
        self.assertEqual(breaker.state, CircuitState.OPEN)
        self.assertFalse(breaker.can_execute())

        # Fast execution with fallback
        res = breaker.execute(func=lambda: "Real", fallback=lambda: "Fallback")
        self.assertEqual(res, "Fallback")

        # Wait for cooldown
        time.sleep(0.25)
        self.assertTrue(breaker.can_execute())
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)

        # Success in HALF_OPEN resets to CLOSED
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
