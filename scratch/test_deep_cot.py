"""
Test script for Deep Chain of Thought (CoT) Cognitive Reasoner.
Tests how deep deliberation improves target resolution and distractor rejection on complex GUI tasks.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time

from spatial_droplet.reasoning.groq_rotator import GroqRotator
from spatial_droplet.core.models import DiagonalDroplet, WindowContext

# Initialize Groq rotator
rotator = GroqRotator()
client_entry = rotator.get_client()
client = client_entry["client"]
model_name = "qwen/qwen3.8-27b"

# Define 3 challenging test scenarios with distractors
TEST_SCENARIOS = [
    {
        "name": "Scenario 1: Dropdown Control vs Static Setting Header",
        "instruction": "Unfold the auto save dropdown setting",
        "window": "Settings - Visual Studio Code",
        "manifest": """
• [3] (420, 210) Zone=TOP-CENTER Role=HEAD Type=LABEL Label="Files: Auto Save"
• [4] (420, 245) Zone=MID-CENTER Role=CTRL Type=BUTTON Att="Files: Auto Save" Label="[Dropdown / Value for 'Files: Auto Save'] (afterDelay)"
• [5] (650, 210) Zone=TOP-RIGHT Role=CTRL Type=BUTTON Label="Reset Setting"
• [12] (80, 1050) Zone=BOTTOM-LEFT Role=SYST Type=BUTTON Label="[Btn] Taskbar: Windows Start Button"
""",
        "expected_target": 4
    },
    {
        "name": "Scenario 2: Video Player Fast Forward vs Browser Navigation Forward",
        "instruction": "Fast forward the current video",
        "window": "YouTube - Google Chrome",
        "manifest": """
• [1] (45, 82) Zone=TOP-LEFT Role=NAV Type=BUTTON Label="[Btn] Browser Back Button"
• [2] (78, 82) Zone=TOP-LEFT Role=NAV Type=BUTTON Label="[Btn] Browser Forward / Next Page"
• [15] (450, 920) Zone=BOTTOM-CENTER Role=CTRL Type=BUTTON Label="[Icon] Media: Play / Pause"
• [16] (510, 920) Zone=BOTTOM-CENTER Role=CTRL Type=BUTTON Att="0:45" Label="[Icon] Media: Fast Forward / Next Chapter"
• [22] (1880, 45) Zone=TOP-RIGHT Role=CTRL Type=BUTTON Label="[Btn] Close Browser"
""",
        "expected_target": 16
    },
    {
        "name": "Scenario 3: Right Margin Ruler Marker vs Left Margin Marker",
        "instruction": "Adjust right margin of the Word document",
        "window": "Document1 - Microsoft Word",
        "manifest": """
• [8] (240, 145) Zone=TOP-LEFT Role=SLID Type=SLIDER Label="[Icon] Left Margin Ruler Marker / Indent"
• [9] (960, 145) Zone=TOP-CENTER Role=SLID Type=SLIDER Label="[Icon] Center Tab Stop Ruler Marker"
• [10] (1680, 145) Zone=TOP-RIGHT Role=SLID Type=SLIDER Label="[Icon] Right Margin Ruler Marker / Indent"
• [14] (1880, 40) Zone=TOP-RIGHT Role=CTRL Type=BUTTON Label="[Btn] Close Word"
""",
        "expected_target": 10
    }
]

SYSTEM_PROMPT = """You are an Autonomous Cognitive Spatial Desktop AI Brain controlling a Windows PC.
You are equipped with a DEEP CHAIN OF THOUGHT (CoT) deliberation engine. You do NOT make blind guesses.

DELIBERATION PROTOCOL:
You MUST systematically formulate your reasoning in four structured phases before emitting any action fields:
1. "intent_analysis":
   - Deconstruct the user's instruction: identify the core action verb (click, toggle, type, etc.).
   - Identify the exact target entity type (dropdown, button, slider handle, media control, input box).
   - Identify qualifying constraints (e.g. "right margin" vs "left margin", "video fast forward" vs "browser forward", "dropdown setting" vs "static header").
2. "spatial_anchoring":
   - Identify where on the display the target control belongs based on standard GUI layout conventions (e.g. Media controls at screen bottom near video track; ruler handles on top ruler bar; settings dropdowns directly beneath or beside setting labels).
3. "candidate_evaluation":
   - Review each candidate in the manifest.
   - For each plausible candidate, evaluate why it matches or fails.
   - EXPLICITLY ELIMINATE DISTRACTORS: Explain why false-friend candidates (e.g. static labels, browser controls, opposite orientation handles) are rejected.
4. "action_plan":
   - State the single best target Dot ID and the exact physical action required.

Respond with ONLY valid JSON with deliberation fields FIRST:
{
  "deliberation": {
    "intent_analysis": "<step-by-step breakdown of user intent and entity constraints>",
    "spatial_anchoring": "<analysis of screen spatial zones and expected placement>",
    "candidate_evaluation": "<explicit comparison of candidate IDs and elimination of distractors>",
    "action_plan": "<final deduction of target ID and physical action>"
  },
  "target_dot_id": <int>,
  "action": "click" | "double_click" | "type" | "press" | "scroll" | "drag" | "hover",
  "expected_transition": "<predicted visual change on screen>"
}"""

print("=" * 70)
print(" TESTING DEEP CHAIN OF THOUGHT (CoT) REASONER ACROSS CHALLENGING CASES")
print("=" * 70)

passed = 0
for sc in TEST_SCENARIOS:
    print(f"\n[Running] {sc['name']}")
    print(f"Goal: \"{sc['instruction']}\"")
    user_prompt = f"""ACTIVE WINDOW: {sc['window']}
USER REQUEST: "{sc['instruction']}"

LIVE SCREEN DROPLET MANIFEST:
{sc['manifest']}

Perform deep chain of thought deliberation and choose the exact matching target Dot ID."""

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=800
    )
    t1 = time.time()
    
    raw = resp.choices[0].message.content or ""
    raw = raw.replace("```json", "").replace("```", "").strip()
    fb = raw.find("{")
    lb = raw.rfind("}")
    
    try:
        data = json.loads(raw[fb:lb+1])
        target_id = data.get("target_dot_id")
        delib = data.get("deliberation", {})
        
        is_hit = (target_id == sc["expected_target"])
        if is_hit:
            passed += 1
            status = "✔ PASS (Correct Target)"
        else:
            status = f"❌ FAIL (Expected {sc['expected_target']}, got {target_id})"
            
        print(f"Result: {status} in {(t1-t0)*1000:.1f}ms")
        print(f"  • Intent: {delib.get('intent_analysis', '')[:90]}...")
        print(f"  • Spatial: {delib.get('spatial_anchoring', '')[:90]}...")
        print(f"  • Evaluation: {delib.get('candidate_evaluation', '')[:120]}...")
        print(f"  • Plan: {delib.get('action_plan', '')[:90]}...")
        print(f"  • Selected Target ID: [{target_id}], Action: {data.get('action')}")
    except Exception as e:
        print(f"❌ JSON Parse Error: {e}\nRaw output: {raw[:200]}")

print("\n" + "=" * 70)
print(f" Deep Chain of Thought Score: {passed}/{len(TEST_SCENARIOS)} ({passed/len(TEST_SCENARIOS)*100:.1f}%)")
print("=" * 70)
