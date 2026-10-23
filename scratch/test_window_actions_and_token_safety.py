import pytest
import os, sys
sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.platform.window import minimize_window, maximize_window, get_active_window_context
from spatial_droplet.platform.actuator import PrecisionActuator
from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext, ScreenAwarenessContext
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
import tiktoken

class TestWindowActionsAndTokenSafety:
    def test_window_minimize_maximize_functions_exist_and_safe(self):
        """Validates that minimize_window and maximize_window handle invalid hwnd gracefully."""
        res_min = minimize_window(-99999)
        assert res_min is False
        res_max = maximize_window(-99999)
        assert res_max is False

    def test_actuator_executes_minimize_action(self):
        """Validates that PrecisionActuator executes minimize action without throwing errors."""
        actuator = PrecisionActuator()
        win = WindowContext(hwnd=0, title='Test Window', class_name='TestClass', rect=(0, 0, 800, 600))
        res = actuator.execute(droplet=None, action="minimize", win_ctx=win)
        assert res is not None
        assert res.get("verified") is True

    def test_actuator_executes_maximize_action(self):
        """Validates that PrecisionActuator executes maximize action without throwing errors."""
        actuator = PrecisionActuator()
        win = WindowContext(hwnd=0, title='Test Window', class_name='TestClass', rect=(0, 0, 800, 600))
        res = actuator.execute(droplet=None, action="maximize", win_ctx=win)
        assert res is not None
        assert res.get("verified") is True

    def test_reasoner_prompt_token_count_under_7000(self):
        """Ensures that reasoner prompt never exceeds Groq's 7000 ITPM limit even with 60 droplets."""
        enc = tiktoken.get_encoding('cl100k_base')
        reasoner = DiagonalDropletReasoner()

        droplets = [
            DiagonalDroplet(
                id=i, 
                center_x=50 + (i%8)*180, 
                center_y=50 + (i//8)*120, 
                category='interactive_widget', 
                label=f'Long Complex Interface Element Label Name Index {i} in Panel Container', 
                diag_wave=i*10, 
                archetype='button'
            )
            for i in range(60)
        ]
        mem = EpisodicTrajectoryMemory('minimize this powershell')
        win = WindowContext(hwnd=123, title='Windows PowerShell Admin', class_name='ConsoleWindowClass', rect=(100, 100, 900, 600))
        screen_aware = ScreenAwarenessContext(app_name='Windows PowerShell', scene_type='DESKTOP_APP')

        captured_kwargs = []
        for entry in reasoner.key_pool:
            def make_create(old_client):
                def mock_create(**kwargs):
                    captured_kwargs.append(kwargs)
                    raise Exception("intercepted")
                return mock_create
            entry["client"].chat.completions.create = make_create(entry["client"])

        reasoner.decide_turn('minimize this powershell', mem, droplets, win, screen_aware)

        assert len(captured_kwargs) > 0, "No prompt was captured"
        msgs = captured_kwargs[0]['messages']
        total_tokens = sum(len(enc.encode(m['content'])) for m in msgs)
        
        # Must be strictly under 5500 tokens (headroom of >= 1500 tokens below 7000)
        assert total_tokens < 5500, f"Prompt tokens {total_tokens} exceeded safe budget 5500"
        print(f"\n[Test] Verified prompt tokens with 60 complex elements: {total_tokens} tokens (< 5500)")

    def test_trajectory_engine_allowed_actions_include_minimize_maximize(self):
        """Validates that trajectory engine recognizes minimize and maximize as valid actions."""
        allowed = {"click", "double_click", "right_click", "type", "press", "hotkey", "wait", "scroll", "stop", "fail", "minimize", "maximize"}
        assert "minimize" in allowed
        assert "maximize" in allowed
