import sys, os
sys.path.insert(0, os.path.abspath("."))
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext, ScreenAwarenessContext
from spatial_droplet.core.memory import EpisodicTrajectoryMemory

droplets = [
    DiagonalDroplet(id=i, center_x=100 + (i%10)*150, center_y=100 + (i//10)*150, category='interactive_widget', label=f'Windows PowerShell Button {i}', diag_wave=i*10, archetype='button')
    for i in range(60)
]
mem = EpisodicTrajectoryMemory('minimize this powershell')
win = WindowContext(hwnd=123, title='Windows PowerShell', class_name='ConsoleWindowClass', rect=(100, 100, 900, 600))
screen_aware = ScreenAwarenessContext(app_name='Windows PowerShell', scene_type='DESKTOP_APP')

reasoner = DiagonalDropletReasoner()

captured = []
for entry in reasoner.key_pool:
    def make_create(old_client):
        def mock_create(**kwargs):
            captured.append(kwargs)
            raise Exception("intercepted")
        return mock_create
    entry["client"].chat.completions.create = make_create(entry["client"])

res = reasoner.decide_turn('minimize this powershell', mem, droplets, win, screen_aware)

if captured:
    kwargs = captured[0]
    msgs = kwargs['messages']
    total = sum(len(enc.encode(m['content'])) for m in msgs)
    print(f"Total tokens sent to Groq: {total}")
    for i, m in enumerate(msgs):
        print(f"  {m['role']}: {len(enc.encode(m['content']))} tokens")
else:
    print("Nothing captured, res:", res)
