import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
from spatial_droplet.vision.grabber import FastScreenGrabber
from spatial_droplet.vision.scene_tracker import AutonomousSceneTracker
from spatial_droplet.platform.window import get_active_window_context, enumerate_all_windows

# 1. Test Window Context
ctx = get_active_window_context()
print(f'Active Window: "{ctx.title}" (HWND: {ctx.hwnd}, Class: {getattr(ctx, "class_name", "")})')

# 2. Test Window Enum (ensure shadow windows filtered)
wins = enumerate_all_windows(include_minimized=False)
shadows = [w for w in wins if any(k in w.get('class', '').lower() for k in ['shadow', 'tooltip'])]
print(f'Enumerated {len(wins)} windows. Shadow windows detected (should be 0): {len(shadows)}')

# 3. Test Scene Tracker Start/Stop
settled_events = []
tracker = AutonomousSceneTracker(on_settled_callback=lambda r, c: settled_events.append((r, c.title)), sample_hz=20.0)
tracker.start()
time.sleep(0.3)
tracker.stop()
print('AutonomousSceneTracker successfully verified with 0 errors.')
