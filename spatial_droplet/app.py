"""
Spatial Droplet Application Controller & Runtime Coordinator.
Orchestrates the HUD overlay, floating copilot widget, real-time focus sentinel,
fast-path router, and autonomous trajectory engine.
"""

import time
import re
import threading
import asyncio
import ctypes
import tkinter as tk
from typing import Optional, List, Dict

import cv2
import numpy as np

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .core.models import (
        DiagonalDroplet, WindowContext, register_own_hwnd, _OWN_HWN_SET as OWN_HWNDS
    )
    from .platform.dpi import DpiAwarenessManager
    from .platform.desktop import DesktopThreadManager
    from .platform.window import get_active_window_context
    from .platform.actuator import PrecisionActuator, Win32HardwareActuator
    from .vision.grabber import FastScreenGrabber
    from .vision.sprayer import FullBleedLatticeSprayer
    from .vision.scene_tracker import AutonomousSceneTracker
    from .reasoning.reasoner import DiagonalDropletReasoner
    from .verification.action_verifier import ActionVerifier
    from .engine.fast_path import FastPathRouter
    from .engine.trajectory_engine import AutonomousTrajectoryEngine
    from .ui.hud import FullBleedHUD
    from .ui.chat_widget import HoverChatWidget
    from .ui.hotkeys import GlobalHotkeyListener
except (ImportError, ValueError):
    from spatial_droplet.core.models import (
        DiagonalDroplet, WindowContext, register_own_hwnd, _OWN_HWN_SET as OWN_HWNDS
    )
    from spatial_droplet.platform.dpi import DpiAwarenessManager
    from spatial_droplet.platform.desktop import DesktopThreadManager
    from spatial_droplet.platform.window import get_active_window_context
    from spatial_droplet.platform.actuator import PrecisionActuator, Win32HardwareActuator
    from spatial_droplet.vision.grabber import FastScreenGrabber
    from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
    from spatial_droplet.vision.scene_tracker import AutonomousSceneTracker
    from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
    from spatial_droplet.verification.action_verifier import ActionVerifier
    from spatial_droplet.engine.fast_path import FastPathRouter
    from spatial_droplet.engine.trajectory_engine import AutonomousTrajectoryEngine
    from spatial_droplet.ui.hud import FullBleedHUD
    from spatial_droplet.ui.chat_widget import HoverChatWidget
    from spatial_droplet.ui.hotkeys import GlobalHotkeyListener


def grab_frame(sw: int, sh: int) -> np.ndarray:
    """Capture full screen frame via persistent grabber."""
    return FastScreenGrabber.get_instance().grab_full()


def main():
    """Main execution entry point for Spatial Droplet Engine."""
    # Ensure DPI awareness and desktop attachment
    DpiAwarenessManager.enable_per_monitor_dpi()
    DesktopThreadManager.attach_input_desktop()

    print("=" * 76)
    print(" SPATIAL DROPLET ENGINE — FULL-BLEED 3,726 MESH & AUTONOMOUS SCENE SYNC ")
    print("=" * 76)

    root = tk.Tk()
    root.withdraw()

    grabber = FastScreenGrabber.get_instance()
    sw, sh = grabber.sw, grabber.sh

    sprayer = FullBleedLatticeSprayer(step_s=48)
    hud = FullBleedHUD(root, sw, sh)
    reasoner = DiagonalDropletReasoner()
    verifier = ActionVerifier(sw, sh)
    actuator = PrecisionActuator(hud, verifier)
    trajectory_engine = AutonomousTrajectoryEngine(reasoner, sprayer, actuator, hud)
    current_abort_event = threading.Event()

    def on_abort_task():
        current_abort_event.set()
        safe_set_status("🛑 Abort signaled (ESC)", color="#ff1744")
        print("[*] Abort signaled via ESC key.")

    register_own_hwnd(root.winfo_id())
    try:
        user32 = ctypes.windll.user32
        rhwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        register_own_hwnd(rhwnd)
    except Exception:
        pass

    show_mesh_stars = True  # Default to full mesh mode with landmarks & surface probes
    droplets_visible = True  # Visibility state of droplets
    chat_widget = None

    def safe_set_status(text: str, color: str = "#00e5ff"):
        if chat_widget:
            root.after(0, lambda: chat_widget.set_status(text, color))

    def safe_reset_button():
        if chat_widget:
            root.after(0, lambda: chat_widget.reset_button())

    def toggle_droplets_visibility() -> bool:
        nonlocal droplets_visible
        droplets_visible = not droplets_visible
        if droplets_visible:
            hud.set_visible(True)
            if droplets:
                safe_render_hud(droplets, current_win_ctx)
            safe_set_status("👁 Droplets Visible", color="#3fb950")
            print("[HUD] Droplets set to VISIBLE.")
        else:
            clear_hud_sync()
            hud.set_visible(False)
            safe_set_status("👁‍🗨 Droplets Hidden (Click 👁 to see)", color="#9ca3af")
            print("[HUD] Droplets set to HIDDEN.")
        if chat_widget:
            root.after(0, lambda: chat_widget.set_droplets_visible(droplets_visible))
        return droplets_visible

    def safe_render_hud(droplet_list, ctx):
        def _do_render():
            if not droplets_visible or (chat_widget and not chat_widget.is_expanded):
                hud.hide()
                return
            copilot_box = chat_widget.get_all_rects() if chat_widget else None
            hud.render(droplet_list, ctx, copilot_rect=copilot_box, show_surface_probes=show_mesh_stars)
            hud.show()
            # CRITICAL: Always re-assert chat_widget topmost AFTER hud.show()
            # Guarantees chat bar remains cleanly in front of both active foreground apps and full-screen HUD
            if chat_widget:
                chat_widget.ensure_visible()
        root.after(0, _do_render)

    def safe_highlight_target(node):
        root.after(0, lambda: hud.highlight_target(node))

    def safe_clear_target_reticle():
        root.after(0, lambda: hud.clear_target_reticle())

    def clear_hud_sync():
        evt = threading.Event()
        def _do_clear():
            try:
                hud.clear_target_reticle()
                hud.clear()
                hud.hide()
                root.update_idletasks()
            finally:
                evt.set()
        if threading.current_thread() == threading.main_thread():
            _do_clear()
        else:
            root.after(0, _do_clear)
            evt.wait(timeout=0.5)
        time.sleep(0.04)  # 40ms for Windows DWM compositor to flush

    droplets: List[DiagonalDroplet] = []
    droplet_map: Dict[int, DiagonalDroplet] = {}
    is_scanning = False
    pending_rescan = False
    scan_lock = threading.Lock()
    current_win_ctx: Optional[WindowContext] = None

    def perform_scan(show_status=True, target_ctx: Optional[WindowContext] = None):
        nonlocal droplets, droplet_map, is_scanning, pending_rescan, current_win_ctx
        with scan_lock:
            if is_scanning:
                pending_rescan = True
                return
            is_scanning = True

        try:
            current_win_ctx = target_ctx or get_active_window_context()
            if show_status:
                safe_set_status("🔍 Auto-Syncing Screen...", color="#ffb300")

            # 1. Grab 100% clean pristine frame (HUD is hardware-excluded via WDA_EXCLUDEFROMCAPTURE)
            frame = grab_frame(sw, sh)

            # 2. Spray clean frame with WinRT OCR & UIA controls (excluding copilot)
            copilot_box = None
            if chat_widget:
                try:
                    copilot_box = chat_widget.get_all_rects()
                except Exception:
                    copilot_box = None
            droplets = asyncio.run(sprayer.spray(frame, current_win_ctx, copilot_rect=copilot_box))
            droplet_map = {d.id: d for d in droplets}

            # 3. Thread-safely render fresh droplets on main UI thread in-place (no screen flicker)
            safe_render_hud(droplets, current_win_ctx)

            active_cnt = sum(1 for d in droplets if d.category != 'surface_probe')
            app_title = current_win_ctx.title[:18] if (current_win_ctx and current_win_ctx.is_valid and current_win_ctx.title != "Desktop") else "Desktop"
            mode_str = "Full Mesh" if show_mesh_stars else "UI Only"
            safe_set_status(f"✔ Live [{app_title}] ({active_cnt} UI | {mode_str})", color="#3fb950")
        except Exception as e:
            print(f"[Scan Notice]: {e}")
        finally:
            with scan_lock:
                is_scanning = False
                should_rerun = pending_rescan
                pending_rescan = False
            if should_rerun:
                root.after(40, lambda: threading.Thread(target=perform_scan, args=(False,), daemon=True).start())

    # --------------------------------------------------------------------------
    # AUTONOMOUS HARDWARE SCENE TRACKER & SETTLE SENTINEL (20 FPS Direct GDI)
    # --------------------------------------------------------------------------
    last_reported_title = ""

    def on_scene_settled(reason: str, ctx: WindowContext):
        nonlocal current_win_ctx, last_reported_title
        current_win_ctx = ctx
        is_win_switch = "window_switch" in reason or "window_move" in reason
        if is_win_switch:
            sprayer.reset_tracking()
            if ctx.title != last_reported_title:
                last_reported_title = ctx.title
                if ctx.title == "Desktop":
                    print(f"\n[Scene Tracker] Active context: Desktop Wallpaper ({reason})")
                    safe_set_status("✔ Live [Desktop]", color="#3fb950")
                else:
                    print(f"\n[Scene Tracker] Foreground settled: \"{ctx.title}\" (HWND: {ctx.hwnd}) [{reason}]")
                    safe_set_status(f"⛶ Focus: {ctx.title[:20]}", color="#00e5ff")
            # Always ensure chat widget stays topmost when switching between applications
            if chat_widget:
                root.after(0, lambda: chat_widget.ensure_visible())
        threading.Thread(target=perform_scan, args=(False, ctx), daemon=True).start()

    scene_tracker = AutonomousSceneTracker(
        on_settled_callback=on_scene_settled,
        motion_threshold=0.45,
        settle_delay=0.15,
        streaming_interval=0.55,
        sample_hz=20.0
    )
    scene_tracker.start()

    # --------------------------------------------------------------------------
    # Automation Execution
    # --------------------------------------------------------------------------
    def run_automation_task(cmd: str):
        nonlocal current_win_ctx

        # 0. Direct Numeric Landmark Selection (e.g. "5", "click 12", "dot 42")
        # Direct landmark index allows instant clicking of visible droplets
        m_dot = re.match(r'^(?:click\s+|dot\s+|c\s+)?(\d+)$', cmd.strip(), re.IGNORECASE)
        if m_dot:
            target_id = int(m_dot.group(1))
            target_node = droplet_map.get(target_id)
            if target_node:
                safe_highlight_target(target_node)
                safe_set_status(f"🎯 Dot [{target_id}] Click: {target_node.label[:25]}", color="#00e5ff")
                print(f"[Direct Dot Selection] Executing instant click on Dot [{target_id}] '{target_node.label}'")
                time.sleep(0.08)
                actuator.execute(target_node, 'click', None, current_win_ctx)
                safe_set_status(f"✔ Clicked: [{target_id}] {target_node.label[:20]}", color="#3fb950")
                safe_reset_button()
                return

        # 1. Live Context and Droplet Sensory Perception
        if not droplets:
            safe_set_status("🔍 Initializing screen scan...", color="#ffb300")
            perform_scan(show_status=False)

        fresh_ctx = get_active_window_context()
        if fresh_ctx and fresh_ctx.is_valid:
            current_win_ctx = fresh_ctx

        app_title = current_win_ctx.title if (current_win_ctx and current_win_ctx.is_valid) else "Desktop"
        print(f"\n[Copilot Task] >>> Instruction: \"{cmd}\" (Active Context: {app_title})")

        # 2. Check for optional command prefixes: /turns, /goal, /boost (in any combination/order)
        turn_limit = 50
        is_boost = False
        cmd_clean = cmd.strip()
        while True:
            m_turns = re.match(r'^/turns\s+(\d+)\s*(.*)$', cmd_clean, re.IGNORECASE)
            if m_turns:
                try:
                    turn_limit = int(m_turns.group(1))
                except Exception:
                    pass
                cmd_clean = m_turns.group(2).strip()
                continue
            if re.match(r'^/boost\b', cmd_clean, re.IGNORECASE):
                is_boost = True
                cmd_clean = re.sub(r'^/boost\s*', '', cmd_clean, flags=re.IGNORECASE).strip()
                continue
            if re.match(r'^/goal\b', cmd_clean, re.IGNORECASE):
                cmd_clean = re.sub(r'^/goal\s*', '', cmd_clean, flags=re.IGNORECASE).strip()
                continue
            break

        if is_boost and turn_limit == 50:
            turn_limit = 75

        cmd = cmd_clean or cmd


        # 3. Autonomous Cognitive Trajectory Engine (Groq 25-Key Rotator / Qwen 27B / ngrok)
        # 100% of natural language instructions (voice or typed) are dispatched directly to the AI Model
        if not reasoner.key_pool and not getattr(reasoner, "use_ngrok", False):
            safe_set_status("⚠️ No LLM API Key in pool — check Groq keys", color="#ff1744")
            safe_reset_button()
            return

        current_abort_event.clear()

        if getattr(reasoner, "use_ngrok", False):
            active_provider = f"⚡ Kaggle Dual-T4 Qwen ({getattr(reasoner, 'ngrok_model', 'qwen2.5vl:7b')} via ngrok)"
        else:
            groq_m = getattr(reasoner, "preferred_groq_model", None) or os.environ.get("GROQ_PRIMARY_MODEL", "qwen/qwen3.8-27b")
            active_provider = f"⚡ Groq 25-Key Rotator ({groq_m})"
        boost_tag = " [🚀 BOOST MODE ACTIVE]" if is_boost else ""
        print(f"\n🧠 [AI Trajectory Engine] Active Reasoner: {active_provider}{boost_tag}")
        print(f"🎯 Target Goal: \"{cmd}\" (Turns: {turn_limit}, Boost: {is_boost})")
        safe_set_status(f"🧠 {active_provider[:20]}{' 🚀' if is_boost else ''} thinking...", color="#00e5ff")

        try:
            memory = trajectory_engine.run_trajectory(
                goal=cmd,
                initial_droplets=droplets,
                initial_win_ctx=current_win_ctx,
                max_turns=turn_limit,
                status_callback=safe_set_status,
                render_callback=safe_render_hud,
                abort_event=current_abort_event,
                boost=is_boost
            )

            if memory.is_completed:
                safe_set_status(f"✔ Done: {memory.completion_summary[:24]}", color="#3fb950")
            elif current_abort_event.is_set():
                safe_set_status("🛑 Trajectory Cancelled (ESC)", color="#ff1744")
            elif "error" in memory.completion_summary.lower():
                safe_set_status(f"❌ {memory.completion_summary[:24]}", color="#ff1744")
            else:
                safe_set_status(f"✦ Trajectory Ended ({len(memory.steps)} turns)", color="#00e5ff")
        except Exception as traj_err:
            print(f"[Trajectory Error]: {traj_err}")
            safe_set_status(f"❌ Error: {str(traj_err)[:22]}", color="#ff1744")
        finally:
            safe_reset_button()

    def on_submit(cmd: str):
        lower_cmd = cmd.strip().lower()
        if lower_cmd in ("hide", "hide droplets", "hide dots", "hide hud", "hide droplet"):
            if droplets_visible:
                toggle_droplets_visibility()
            else:
                safe_set_status("👁‍🗨 Droplets already hidden", color="#9ca3af")
            safe_reset_button()
            return
        if lower_cmd in ("show", "show droplets", "show dots", "see", "see droplets", "show hud"):
            if not droplets_visible:
                toggle_droplets_visibility()
            else:
                safe_set_status("👁 Droplets already visible", color="#3fb950")
            safe_reset_button()
            return
        if lower_cmd in ("toggle", "toggle droplets", "toggle hud"):
            toggle_droplets_visibility()
            safe_reset_button()
            return
        threading.Thread(target=run_automation_task, args=(cmd,), daemon=True).start()

    def on_toggle():
        nonlocal show_mesh_stars
        show_mesh_stars = not show_mesh_stars
        state_str = "Full Mesh (3,726)" if show_mesh_stars else "UI Landmarks Only"
        safe_set_status(f"✦ Mode: {state_str}", color="#ffb300")
        if droplets:
            safe_render_hud(droplets, current_win_ctx)

    def on_refresh():
        sprayer.reset_tracking()
        threading.Thread(target=perform_scan, args=(True,), daemon=True).start()

    def on_quick_preview(query: str):
        q = (query or "").strip()
        if not q:
            safe_clear_target_reticle()
            safe_set_status("⚡ Type ID (e.g. 5) or button name & Enter (ESC to cancel)", color="#ffb300")
            return None

        candidate = None
        if q.isdigit():
            target_id = int(q)
            candidate = droplet_map.get(target_id)

        if not candidate:
            q_lower = q.lower()
            for d in droplets:
                if d.category != 'surface_probe' and d.label.lower() == q_lower:
                    candidate = d
                    break

        if not candidate:
            q_lower = q.lower()
            for d in droplets:
                if d.category != 'surface_probe' and d.label.lower().startswith(q_lower):
                    candidate = d
                    break

        if not candidate:
            q_lower = q.lower()
            for d in droplets:
                if d.category != 'surface_probe' and q_lower in d.label.lower():
                    candidate = d
                    break

        if candidate:
            safe_highlight_target(candidate)
            safe_set_status(f"🎯 Target: [{candidate.id}] {candidate.label} (Press ENTER to dispatch)", color="#00e5ff")
            return candidate
        else:
            safe_clear_target_reticle()
            safe_set_status(f"⚠️ No match for '{q}'", color="#8b949e")
            return None

    def on_quick_dispatch(candidate: DiagonalDroplet):
        def _dispatch():
            nonlocal current_win_ctx
            print(f"\n[Quick-Select Dispatch] Actuating [{candidate.id}] '{candidate.label}' at ({candidate.center_x}, {candidate.center_y})")
            is_desktop_context = (not current_win_ctx) or (not current_win_ctx.is_valid) or (current_win_ctx.title == "Desktop")
            act = 'double_click' if (is_desktop_context and (candidate.category == 'interactive_app' or getattr(candidate, 'archetype', '') == 'logo')) else 'click'
            ver_res = actuator.execute(candidate, act, None, current_win_ctx)
            if ver_res and ver_res.get("verified"):
                safe_set_status(f"✔ Verified: {candidate.label}", color="#3fb950")
            else:
                safe_set_status(f"✔ Executed: {candidate.label}", color="#3fb950")
            safe_reset_button()
        threading.Thread(target=_dispatch, daemon=True).start()

    def on_global_hotkey():
        if chat_widget:
            root.after(0, lambda: chat_widget.activate_quick_select())

    # Initialize Voice Control Subsystem (Groq Whisper Turbo + 16kHz Audio)
    try:
        from .voice import VoiceControlManager
        voice_manager = VoiceControlManager()
        voice_enabled = voice_manager.is_available
        if voice_enabled:
            print("[*] Voice Control Engine active (Groq Whisper Large V3 Turbo, 25 keys ready)")
    except Exception as ve:
        voice_manager = None
        voice_enabled = False
        print(f"[!] Voice control init notice: {ve}")

    def on_voice_toggle():
        if not voice_manager:
            safe_set_status("⚠️ Voice control unavailable (install sounddevice)", color="#ff1744")
            return

        def _voice_status_cb(text: str, color: str):
            root.after(0, lambda: safe_set_status(text, color=color))

        def _voice_state_cb(recording: bool):
            if chat_widget:
                root.after(0, lambda: chat_widget.set_voice_recording(recording))

        def _voice_result_cb(text: str):
            print(f"\n🎙️ [Voice Input Received]: \"{text}\"")
            def _apply_and_submit():
                if chat_widget:
                    chat_widget.set_voice_recording(False)
                    chat_widget.set_voice_text(text, auto_submit=True)
                else:
                    on_submit(text)
            root.after(0, _apply_and_submit)

        is_now_listening = voice_manager.toggle(
            on_status=_voice_status_cb,
            on_result=_voice_result_cb,
            on_state_change=_voice_state_cb
        )
        if chat_widget:
            chat_widget.set_voice_recording(is_now_listening)

    def on_compact_mode_change(is_expanded: bool):
        if not is_expanded:
            # Auto-clean screen HUD / star spray in compact mascot orb mode
            hud.clear_target_reticle()
            hud.hide()
        else:
            if droplets_visible and droplets:
                safe_render_hud(droplets, current_win_ctx)

    def on_model_change(use_ngrok: bool, model_name: str, display_label: str):
        reasoner.set_active_model(use_ngrok=use_ngrok, model_name=model_name)
        safe_set_status(f"✔ Active Model: {display_label}", color="#3fb950")

    chat_widget = HoverChatWidget(
        root, on_submit, on_toggle, on_refresh,
        on_quick_preview_callback=on_quick_preview,
        on_quick_dispatch_callback=on_quick_dispatch,
        on_abort_callback=on_abort_task,
        on_toggle_droplets_callback=toggle_droplets_visibility,
        on_voice_toggle_callback=on_voice_toggle,
        on_compact_callback=on_compact_mode_change,
        on_model_change_callback=on_model_change
    )

    hotkey_listener = GlobalHotkeyListener(
        on_global_hotkey,
        on_toggle_droplets_callback=lambda: root.after(0, toggle_droplets_visibility),
        on_voice_toggle_callback=lambda: root.after(0, on_voice_toggle)
    )
    hotkey_listener.start()
    # Launch initial screen scan asynchronously so copilot UI appears immediately without freeze
    root.after(100, lambda: threading.Thread(target=perform_scan, args=(True,), daemon=True).start())

    print("\n[*] 3,726 FULL-BLEED STARS READY — LIVE WINDOW-AWARE SYNC ACTIVE!")
    print("[*] Global Vimium Mode: Alt+Space, Ctrl+Shift+Space, Alt+Q or click '⚡ Quick'")
    print("[*] Toggle Droplets: Alt+H or click '👁' in floating chatbox")
    print("[*] Voice Control: Click '🎙️ Mic' or press Alt+V, Ctrl+Shift+V, or F8 to speak")
    print("[*] Automatically detects window changes & resprays clean droplets.")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[*] Terminated by user. Goodbye!")
    finally:
        scene_tracker.stop()
        hotkey_listener.stop()
        if voice_manager and getattr(voice_manager, "is_listening", False):
            voice_manager.recorder.stop()


run_app = main
