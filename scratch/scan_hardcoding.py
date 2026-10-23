import os
import re
from pathlib import Path

root = Path(r"E:\autonew\spatial_droplet")

patterns = [
    (r"\b\d{2,4}\s*<=\s*[a-zA-Z_\.]+\s*<=\s*\d{2,4}\b", "Hardcoded coordinate bounding range"),
    (r"[a-zA-Z_\.]+\s*==\s*\d{3,4}", "Exact coordinate check"),
    (r"(Tum Down|0FOB4CLBT2|1GTGZ3A3V8|6GNDSETVY9)", "Benchmark sample or image hash"),
    (r"auto\s*save", "Task phrase: Auto Save"),
    (r"transition\s*panel", "Task phrase: transition panel"),
    (r"debug\s*panel", "Task phrase: debug panel"),
    (r"right\s*margin", "Task phrase: right margin"),
    (r"cancel\s*the\s*highlight", "Task phrase: cancel highlight"),
    (r"star\s*the\s*(first\s*)?slide", "Task phrase: star slide"),
    (r"fast\s*forward\s*the\s*current\s*video", "Task phrase: fast forward"),
    (r"expand\s*the\s*brush\s*function", "Task phrase: brush function"),
    (r"expand\s*the\s*video\s*to\s*watch", "Task phrase: expand video"),
]

findings = []
for p in root.rglob("*.py"):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line_str = line.strip()
        if line_str.startswith("#"):
            continue
        for pat, desc in patterns:
            if re.search(pat, line_str, re.IGNORECASE):
                findings.append((p.name, idx + 1, desc, line_str))

print(f"Total findings across spatial_droplet: {len(findings)}")
for fname, lnum, desc, text in findings:
    print(f"[{fname}:{lnum}] {desc}")
    print(f"   CODE: {text[:140]}")
