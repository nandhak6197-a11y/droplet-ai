import sys, os
sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext, ScreenAwarenessContext
from spatial_droplet.core.memory import EpisodicTrajectoryMemory

droplets = [
    DiagonalDroplet(id=i, center_x=100 + (i%10)*150, center_y=100 + (i//10)*150, category='interactive_widget', label=f'Windows PowerShell Button {i}', diag_wave=i*10, archetype='button')
    for i in range(40)
]
mem = EpisodicTrajectoryMemory('minimize this powershell')
win = WindowContext(hwnd=123, title='Windows PowerShell', class_name='ConsoleWindowClass', rect=(100, 100, 900, 600))
screen_aware = ScreenAwarenessContext(app_name='Windows PowerShell', scene_type='DESKTOP_APP')

reasoner = DiagonalDropletReasoner()
print("Calling decide_turn live...")
decision = reasoner.decide_turn('minimize this powershell', mem, droplets, win, screen_aware)
print("\n=== DECISION RECEIVED ===")
print(f"Action: {decision.action_type}")
print(f"Target Droplet ID: {decision.target_droplet_id}")
print(f"Model Used: {decision.get('model_used')}")
print(f"Key Used: {decision.get('key_used')}")
delib = decision.get("deliberation", {})
if isinstance(delib, dict):
    print("Deliberation Plan:", delib.get("end_to_end_master_plan"))
    print("Deliberation Intent:", delib.get("user_intent_and_success_criteria"))
    print("Deliberation Critique:", delib.get("mistake_critique_and_loop_diagnosis"))
    print("Deliberation Tactical Justification:", delib.get("tactical_action_justification"))
