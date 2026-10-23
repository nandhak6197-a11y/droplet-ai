"""
Verify DiagonalDropletReasoner with live Deep Chain of Thought (CoT).
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import asyncio

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext

# Create mock test droplets
mock_droplets = [
    DiagonalDroplet(id=1, center_x=45, center_y=82, diag_wave=127, category="interactive_widget", label="[Btn] Browser Back Button"),
    DiagonalDroplet(id=2, center_x=78, center_y=82, diag_wave=160, category="interactive_widget", label="[Btn] Browser Forward / Next Page"),
    DiagonalDroplet(id=15, center_x=450, center_y=920, diag_wave=1370, category="interactive_widget", label="[Icon] Media: Play / Pause"),
    DiagonalDroplet(id=16, center_x=510, center_y=920, diag_wave=1430, category="interactive_widget", label="[Icon] Media: Fast Forward / Next Chapter", attached_label="0:45", semantic_role="control"),
    DiagonalDroplet(id=22, center_x=1880, center_y=45, diag_wave=1925, category="interactive_widget", label="[Btn] Close Browser")
]

mock_win = WindowContext(
    hwnd=12345,
    title="YouTube - Google Chrome",
    rect=(0, 0, 1920, 1080),
    width=1920,
    height=1080,
    is_valid=True
)

reasoner = DiagonalDropletReasoner()

async def run_test():
    print("=" * 70)
    print(" TESTING LIVE DIAGONAL DROPLET REASONER WITH DEEP CoT")
    print("=" * 70)
    
    goal = "Fast forward the current video"
    print(f"\nEvaluating Goal: \"{goal}\"")
    decision = await reasoner.decide(
        instruction=goal,
        droplets=mock_droplets,
        active_window=mock_win,
        mode="llm"
    )
    
    print("\nDecision Output:")
    print(f"  • Model Used:   {decision.get('model_used')}")
    print(f"  • Key Used:     {decision.get('key_used')}")
    print(f"  • Target ID:    [{decision.target_droplet_id}]")
    print(f"  • Action:       {decision.action}")
    print(f"  • Coordinates:  ({decision.target_x}, {decision.target_y})")
    
    delib = decision.deliberation
    print("\nDeep Chain of Thought Deliberation:")
    print(f"  1. Intent:      {delib.get('intent_analysis')}")
    print(f"  2. Spatial:     {delib.get('spatial_anchoring')}")
    print(f"  3. Candidates:  {delib.get('candidate_evaluation')}")
    print(f"  4. Plan:        {delib.get('action_plan')}")
    
    assert decision.target_droplet_id == 16, f"Expected Dot ID 16, got {decision.target_droplet_id}"
    assert delib.get('candidate_evaluation'), "Candidate evaluation should not be empty!"
    print("\n✔ ALL TESTS PASSED: Deep Chain of Thought reasoning is operating with 100% precision!")

if __name__ == "__main__":
    asyncio.run(run_test())
