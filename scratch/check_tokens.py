import sys, os
sys.path.insert(0, os.path.abspath("."))
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner, get_system_grounding_context
from spatial_droplet.core.models import DiagonalDroplet, WindowContext, ScreenAwarenessContext
from spatial_droplet.core.memory import EpisodicTrajectoryMemory

# Let's inspect the entire system_instruction and user_content from decide_turn in reasoner.py
with open('spatial_droplet/reasoning/reasoner.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Grab system_instruction template
import re
sys_m = re.search(r'system_instruction = f"""(.*?)"""\s+turn_alert =', code, re.DOTALL)
if sys_m:
    sys_text = sys_m.group(1)
    print('System instruction template tokens:', len(enc.encode(sys_text)))
else:
    print('Could not find system_instruction template')

# Grab user_content template
user_m = re.search(r'user_content = f"""(.*?)"""\s+# Sync with live environment', code, re.DOTALL)
if user_m:
    user_text = user_m.group(1)
    print('User content template tokens:', len(enc.encode(user_text)))
else:
    print('Could not find user_content template')
