import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_copilot import IntentScopeRouter

test_cmds = [
    "go to chrome and open some stuff",
    "go to chrome and open some stuff like that",
    "go to chrome and open youtube",
    "go to chrome and open google",
    "open chrome and go to youtube",
    "open chrome and open youtube",
    "open chrome and search youtube for music",
    "go to chrome and open wikipedia",
    "go to chrome and open reddit",
    "go to chrome and open github",
    "open youtube in chrome",
    "open youtube on chrome",
    "go to chrome to open youtube",
    "open chrome and browse youtube",
    "open chrome and search cute cats",
    "open chrome and search for python tutorials",
    "go to chrome and open youtube.",
]

for cmd in test_cmds:
    res = IntentScopeRouter.route(cmd)
    act = res.get("action") if res else None
    summ = res.get("summary") if res else "FAIL"
    print(f"{cmd:45} -> {act} | {summ}")
