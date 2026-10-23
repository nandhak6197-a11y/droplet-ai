"""
================================================================================
       SPATIAL DROPLET — LIVE DESKTOP TERMINAL REASONING & CHAT CONSOLE
================================================================================
Interactive real-time terminal loop running over the LIVE Windows desktop.
Captures real screen, sprays live UIA + WinRT OCR + CLAHE icon droplets,
determines targets via Groq 25-key pool (Qwen 3.8 / GPT-OSS 120B), and
dispatches REAL hardware mouse clicks via Win32 SendInput.
================================================================================
"""

import os
import sys
import time
import math
import asyncio
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_droplet.vision.grabber import FastScreenGrabber
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.platform.window import get_active_window_context
from spatial_droplet.platform.actuator import Win32HardwareActuator, PrecisionActuator
from spatial_droplet.verification.action_verifier import ActionVerifier
from spatial_droplet.engine.planner import HierarchicalGoalPlanner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext

def render_live_som(img: np.ndarray, droplets: List[DiagonalDroplet]) -> np.ndarray:
    """Renders high-visibility Set-of-Marks badges on live screenshot."""
    vis = img.copy()
    h, w = vis.shape[:2]
    for d in droplets:
        if d.category == 'surface_probe':
            continue
        cx, cy = int(d.center_x), int(d.center_y)
        raw_box = getattr(d, 'bbox', None)
        if raw_box and len(raw_box) == 4 and raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
            x1, y1 = max(0, int(raw_box[0])), max(0, int(raw_box[1]))
            x2, y2 = min(w, int(raw_box[2])), min(h, int(raw_box[3]))
        else:
            x1, y1 = max(0, cx - 14), max(0, cy - 10)
            x2, y2 = min(w, cx + 14), min(h, cy + 10)

        # 1px Jev Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (210, 160, 40), 1)

        # Docked badge tag inside top-left
        txt = str(d.id)
        tw = len(txt) * 8 + 8
        th = 13
        bx1, by1 = x1, y1
        bx2, by2 = min(w, x1 + tw), min(h, y1 + th)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (210, 160, 40), 1)
        cv2.putText(vis, txt, (bx1 + 3, by1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Center pinpoint
        cv2.circle(vis, (cx, cy), 2, (0, 220, 255), -1)

    return vis

async def async_main():
    print("\n" + "=" * 76)
    print(" 🚀 SPATIAL DROPLET: LIVE DESKTOP CHAT & HARDWARE ACTUATION CONSOLE")
    print("=" * 76)
    print(" • Operating Mode:    LIVE HARDWARE DISPATCH (Real mouse movements & clicks)")
    print(" • Multi-Key Engine:  25 Groq API Keys with Qwen 3.8 27B / GPT-OSS 120B")
    print(" • Live Sensors:      Windows UIA Tree + WinRT Neural OCR + CLAHE Canny Icons")
    print(" • Real-Time Mesh:    Captures your actual desktop and saves live visual map")
    print("=" * 76)
    print(" Commands:")
    print("   - Type any instruction  : 'open chrome', 'click file menu', 'close window'")
    print("   - Fast Click by Dot ID  : 'click 12', 'c 5', or just '12' for instant 0ms click")
    print("   - Type text             : 'type Hello World' to type into focused field")
    print("   - 'dots'                : Show top visible UI droplets on screen")
    print("   - 'scan' / 'respray'    : Refresh screen capture and respray droplets")
    print("   - 'mode'                : Toggle between LIVE HARDWARE and SAFE SIMULATION")
    print("   - 'exit' / 'q'          : Quit\n")

    grabber = FastScreenGrabber.get_instance()
    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    
    live_mode = True  # DEFAULT TO LIVE MODE

    # Initial live capture and spray
    print("[*] Capturing live desktop screen...")
    frame = grabber.grab_full()
    if frame is None:
        print("[!] Error: Could not capture live screen via GDI/MSS. Trying fallback...")
        try:
            import pyautogui
            frame = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[!] Critical Error: Screen capture failed: {e}")
            return

    sh, sw = frame.shape[:2]
    win_ctx = get_active_window_context()
    print(f"[+] Screen Resolution: {sw}x{sh} | Active Window: '{win_ctx.title if win_ctx else 'Desktop'}'")

    print("[*] Spraying live droplets over active desktop controls...")
    droplets = await sprayer.spray(frame, win_ctx=win_ctx)
    ui_droplets = [d for d in droplets if d.category != 'surface_probe']
    print(f"[+] Discovered {len(ui_droplets)} interactive UI controls on screen!")

    # Save initial visual map
    vis = render_live_som(frame, droplets)
    cv2.imwrite("droplet_live_screen.png", vis)
    try:
        downloads_out = Path.home() / "Downloads" / "droplet_live_screen.png"
        cv2.imwrite(str(downloads_out), vis)
        print(f"[+] Saved live annotated Set-of-Marks map to: {downloads_out}")
    except Exception:
        pass

    while True:
        mode_str = "🔴 LIVE HARDWARE DISPATCH" if live_mode else "🟢 SAFE SIMULATION"
        active_title = win_ctx.title if (win_ctx and win_ctx.title) else "Desktop"
        try:
            prompt = input(f"\n[{mode_str} | App: '{active_title[:24]}'] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down live console.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", "q"):
            print("Shutting down live console. Goodbye!")
            break

        if prompt.lower() == "mode":
            live_mode = not live_mode
            status = "🔴 LIVE HARDWARE DISPATCH" if live_mode else "🟢 SAFE SIMULATION"
            print(f"[!] Switched execution mode to: {status}")
            continue

        if prompt.lower() in ("scan", "respray", "refresh"):
            print("[*] Respraying fresh live screen capture...")
            frame = grabber.grab_full()
            win_ctx = get_active_window_context()
            droplets = await sprayer.spray(frame, win_ctx=win_ctx)
            ui_droplets = [d for d in droplets if d.category != 'surface_probe']
            vis = render_live_som(frame, droplets)
            cv2.imwrite("droplet_live_screen.png", vis)
            try:
                cv2.imwrite(str(Path.home() / "Downloads" / "droplet_live_screen.png"), vis)
            except Exception:
                pass
            print(f"[+] Screen updated! {len(ui_droplets)} controls visible. Map saved.")
            continue

        if prompt.lower() == "dots":
            print("\n" + "=" * 72)
            print(f" LIVE SCREEN CONTROLS ({len(ui_droplets)} total UI elements):")
            print("=" * 72)
            for d in ui_droplets[:35]:
                role = getattr(d, 'semantic_role', 'widget').upper()
                print(f"  • Dot [{d.id:3d}] at ({d.center_x:4d}, {d.center_y:4d}) | {role:7s} | {d.label}")
            if len(ui_droplets) > 35:
                print(f"  ... and {len(ui_droplets) - 35} more controls.")
            print("=" * 72)
            continue

        # Fast direct click: e.g. "12", "click 12", "c 12"
        direct_match = None
        if prompt.isdigit():
            direct_match = int(prompt)
        else:
            m = re.match(r'^(?:click|c)\s+(\d+)$', prompt.lower().strip())
            if m:
                direct_match = int(m.group(1))

        if direct_match is not None:
            target_node = next((d for d in droplets if d.id == direct_match), None)
            if target_node:
                tx, ty = int(target_node.center_x), int(target_node.center_y)
                print(f"[FAST-PATH] Direct click on Dot [{direct_match}] '{target_node.label}' at ({tx}, {ty})")
                if live_mode:
                    Win32HardwareActuator.click(tx, ty, button="left", sw=sw, sh=sh)
                    print(f"  ✔ Hardware mouse click executed at ({tx}, {ty})!")
                else:
                    print(f"  [Simulation] Simulated click at ({tx}, {ty}).")
                continue
            else:
                print(f"[!] Dot [{direct_match}] not found on current screen. Type 'dots' to view available IDs.")
                continue

        # Direct typing command: e.g. "type hello world"
        if prompt.lower().startswith("type "):
            text_to_type = prompt[5:]
            print(f"[KEYBOARD] Typing text: \"{text_to_type}\"")
            if live_mode:
                Win32HardwareActuator.type_text(text_to_type)
                print("  ✔ Text typed via Win32 hardware SendInput!")
            else:
                print(f"  [Simulation] Simulated typing: \"{text_to_type}\"")
            continue

        # Multi-Step Plan Dispatch: e.g. "plan open notepad then type hello" or "open chrome and then click search"
        if any(k in prompt.lower() for k in (" then ", " and then ", " after that ")) or prompt.lower().startswith("plan "):
            goal_to_plan = prompt[5:].strip() if prompt.lower().startswith("plan ") else prompt
            planner = HierarchicalGoalPlanner()
            act = PrecisionActuator(verifier=ActionVerifier(sw=sw, sh=sh))
            await planner.execute_plan(
                goal=goal_to_plan,
                reasoner=reasoner,
                grabber=grabber,
                sprayer=sprayer,
                actuator=act,
                verifier=act.verifier,
                sw=sw,
                sh=sh
            )
            continue

        # Refresh screen right before reasoning to ensure fresh coordinates
        frame = grabber.grab_full()
        win_ctx = get_active_window_context()
        droplets = await sprayer.spray(frame, win_ctx=win_ctx)

        print(f"\n[*] Deliberating with Groq Multi-Key Pool for: \"{prompt}\"...")
        t0 = time.time()
        decision = await reasoner.decide(
            user_intent=prompt,
            droplets=droplets,
            window_ctx=win_ctx,
            screenshot=frame,
            mode="combined"
        )
        elapsed = (time.time() - t0) * 1000

        target_id = decision.get("target_dot_id") or decision.get("target_droplet_id")
        action = decision.get("action_type") or decision.get("action") or "click"
        target_x = decision.get("target_x") if decision.get("target_x") is not None else decision.get("x")
        if target_x is None:
            target_x = getattr(decision, "target_x", None)
        target_y = decision.get("target_y") if decision.get("target_y") is not None else decision.get("y")
        if target_y is None:
            target_y = getattr(decision, "target_y", None)
        reasoning = decision.get("reasoning") or getattr(decision, "reasoning", "Identified optimal interaction target.")
        model_used = decision.get("model_used", "Groq AI Pool")

        print("\n" + "-" * 60)
        print(f" 🤖 COGNITIVE DECISION ({elapsed:.0f}ms | {model_used}):")
        print("-" * 60)
        print(f"  • Action        : {action.upper()}")
        print(f"  • Target Dot ID : [{target_id or 'Coord Snapped'}]")
        print(f"  • Coordinates   : ({target_x}, {target_y})")
        print(f"  • Reasoning     : {reasoning}")
        print("-" * 60)

        if target_x is not None and target_y is not None and not (target_x == 0 and target_y == 0):
            tx, ty = int(target_x), int(target_y)
            if live_mode:
                if action == "double_click":
                    Win32HardwareActuator.double_click(tx, ty, sw=sw, sh=sh)
                    print(f"  ✔ Hardware double-click dispatched at ({tx}, {ty})!")
                elif action == "right_click":
                    Win32HardwareActuator.click(tx, ty, button="right", sw=sw, sh=sh)
                    print(f"  ✔ Hardware right-click dispatched at ({tx}, {ty})!")
                elif action == "hover":
                    Win32HardwareActuator.move_cursor(tx, ty, sw=sw, sh=sh)
                    print(f"  ✔ Cursor moved to ({tx}, {ty})!")
                elif action == "drag":
                    dx = decision.get("drag_dest_x")
                    dy = decision.get("drag_dest_y")
                    if dx is not None and dy is not None:
                        Win32HardwareActuator.drag(tx, ty, int(dx), int(dy), sw=sw, sh=sh)
                        print(f"  ✔ Hardware drag dispatched from ({tx}, {ty}) to ({dx}, {dy})!")
                    else:
                        Win32HardwareActuator.click(tx, ty, button="left", sw=sw, sh=sh)
                        print(f"  ✔ Fallback click dispatched at ({tx}, {ty})!")
                elif action == "type":
                    txt = decision.get("input_text", "")
                    Win32HardwareActuator.click(tx, ty, button="left", sw=sw, sh=sh)
                    time.sleep(0.06)
                    Win32HardwareActuator.type_text(txt)
                    print(f"  ✔ Hardware typed \"{txt}\" at ({tx}, {ty})!")
                else:
                    Win32HardwareActuator.click(tx, ty, button="left", sw=sw, sh=sh)
                    print(f"  ✔ Hardware mouse click dispatched at ({tx}, {ty})!")
            else:
                print(f"  [Simulation] Simulated {action} at ({tx}, {ty}). (Type 'mode' to activate Live Hardware)")
        else:
            print("  [!] Could not determine valid pixel coordinates for this action.")

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
