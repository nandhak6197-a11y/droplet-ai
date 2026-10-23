"""
Interactive Pure Droplet Local Test (Zero LLM, 100% Offline).
Grabs your live screen, discovers UI droplets, and resolves natural language prompts in ~40ms.
"""
import sys
import os
import time
import asyncio
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add project root to sys.path
PROJECT_ROOT = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spatial_droplet.vision.grabber import FastScreenGrabber
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.platform.window import get_active_window_context
from spatial_droplet.platform.actuator import PrecisionActuator

async def run_prompt(user_prompt: str, auto_click: bool = False):
    print("\n" + "=" * 70)
    print(" 🚀 PURE DROPLET ENGINE — LOCAL OFFLINE GROUNDING (ZERO LLM)")
    print("=" * 70)
    
    t0 = time.perf_counter()
    
    # 1. Grab live screen
    print("[1/3] 📸 Grabbing live screen frame...")
    img = FastScreenGrabber.get_instance().grab_full()
    sh, sw = img.shape[:2]
    win_ctx = get_active_window_context()
    
    # 2. Spray droplets with on-device computer vision & OCR
    print("[2/3] ✨ Scanning screen & generating context-colored droplets...")
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    droplets = await sprayer.spray(img, win_ctx=win_ctx)
    
    # Archetype breakdown
    arch_counts = {}
    for d in droplets:
        arch_counts[d.archetype] = arch_counts.get(d.archetype, 0) + 1
    
    print(f"      🎯 Discovered {len(droplets)} total UI droplets across {sw}x{sh} display:")
    for arch, count in sorted(arch_counts.items()):
        print(f"         • {arch.upper():<8}: {count} elements")
        
    # 3. Resolve target purely through deterministic symbolic vision
    print(f"\n[3/3] 🧠 Evaluating prompt: \"{user_prompt}\" (Pure Droplet Mode)...")
    t_reason = time.perf_counter()
    decision = await reasoner.decide(user_prompt, droplets=droplets, window_ctx=win_ctx, mode="pure_droplet")
    dt_ms = (time.perf_counter() - t_reason) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    
    target_d = next((d for d in droplets if d.id == decision.target_droplet_id), None)
    
    print("\n" + "=" * 70)
    print(" 🎯 TARGET ACQUIRED")
    print("=" * 70)
    print(f"  • Target Coordinates:  X = {decision.target_x}, Y = {decision.target_y}")
    print(f"  • Selected Droplet:    ID [{decision.target_droplet_id}]")
    if target_d:
        print(f"  • Functional Type:     {target_d.archetype.upper()} ({target_d.semantic_color})")
        print(f"  • Container Area:      {target_d.container_title or target_d.container_id}")
        print(f"  • Control Label:       \"{target_d.label}\"")
    print(f"  • Reasoning:           {decision.reasoning}")
    print(f"  • Pure Droplet Latency: {dt_ms:.1f} ms  (Total end-to-end: {total_ms:.1f} ms)")
    print("=" * 70)
    
    if auto_click and decision.target_x is not None and decision.target_y is not None:
        print(f"\n🖱️ Dispatching click to ({decision.target_x}, {decision.target_y})...")
        actuator = PrecisionActuator()
        actuator.click(decision.target_x, decision.target_y)
        print("✔ Click dispatched successfully!")

def main():
    prompt = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not prompt:
        prompt = input("Enter prompt (e.g. 'Close window', 'Search', 'Start menu'): ").strip()
    if not prompt:
        prompt = "Close window"
        
    asyncio.run(run_prompt(prompt))

if __name__ == "__main__":
    main()
