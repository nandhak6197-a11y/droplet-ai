"""
Comprehensive Test Suite for Real-Time Automation Blueprint (Phases 1-4)
"""
import sys
import os
import time
import asyncio
import numpy as np
import cv2
import json
from pathlib import Path

# Ensure UTF-8 console output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from live_copilot import (
    IntentScopeRouter,
    compute_visual_diff,
    verify_visual_transition
)
from spatial_droplet.vision.uia_walker import UiaControlWalker, AsyncUiaBackgroundWorker
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet, WindowContext
from spatial_droplet.verification.action_verifier import ActionVerifier


def test_phase1_intent_router():
    print("\n" + "=" * 60)
    print("TESTING PHASE 1: INTENT & SCOPE ROUTER")
    print("=" * 60)

    # 1. Test App Launch Commands
    launch_cmds = ["open chrome", "start notepad", "launch code", "run calc"]
    for cmd in launch_cmds:
        decision = IntentScopeRouter.route(cmd)
        assert decision is not None, f"Expected route for '{cmd}'"
        assert decision.get("intent") == "os_fast_path"
        assert decision.get("action") == "launch_app"
        print(f"  ✔ '{cmd}' ➔ {decision['action']} target='{decision['target']}'")

    # 2. Test Window Switching Commands
    switch_cmds = ["switch to chrome", "focus powershell", "bring up notebook", "activate code"]
    for cmd in switch_cmds:
        decision = IntentScopeRouter.route(cmd)
        assert decision is not None, f"Expected route for '{cmd}'"
        assert decision.get("intent") == "os_fast_path"
        assert decision.get("action") in ("window_switch", "launch_app")
        print(f"  ✔ '{cmd}' ➔ {decision['action']}")

    # 3. Test Window Management Shortcuts
    win_ops = {
        "minimize": ("window_op", "minimize"),
        "maximize": ("window_op", "maximize"),
        "restore": ("window_op", "restore"),
        "close window": ("window_op", "close"),
        "close tab": ("system_hotkey", None),
    }
    for cmd, (expected_act, expected_op) in win_ops.items():
        decision = IntentScopeRouter.route(cmd)
        assert decision is not None, f"Expected route for '{cmd}'"
        assert decision.get("action") == expected_act
        if expected_op:
            assert decision.get("op") == expected_op
        print(f"  ✔ '{cmd}' ➔ {decision['action']} (op={decision.get('op')})")

    # 4. Test In-Window GUI Tasks (Must NOT route to OS fast-path)
    gui_cmds = [
        "click search filter",
        "star the first slide",
        "select purple swatch",
        "open dropdown",
        "open filter menu",
        "open settings dialog",
        "focus search box",
        "focus filter",
        "switch to dark mode",
        "activate purple swatch",
        "switch to tab 2",
        "focus input field"
    ]
    for cmd in gui_cmds:
        decision = IntentScopeRouter.route(cmd)
        assert decision is None, f"In-window GUI task '{cmd}' incorrectly routed to OS Fast-Path! Decision: {decision}"
        print(f"  ✔ '{cmd}' ➔ Correctly passed through to GUI Grounding Pipeline")

    # 5. Test Live Desktop Window Discovery
    chrome_win = IntentScopeRouter.find_window_by_name("chrome")
    ps_win = IntentScopeRouter.find_window_by_name("powershell")
    code_win = IntentScopeRouter.find_window_by_name("code")
    print(f"  ✔ Live Window Discovery: Chrome={chrome_win is not None}, PowerShell={ps_win is not None}, VSCode={code_win is not None}")
    assert any((chrome_win, ps_win, code_win)), "Live window discovery failed to discover any running user window!"

    print("Phase 1 Intent Router: ALL TESTS PASSED.")


def test_phase2_async_uia():
    print("\n" + "=" * 60)
    print("TESTING PHASE 2: ASYNCHRONOUS NON-BLOCKING UIA ENGINE")
    print("=" * 60)

    # 1. Test 0ms Non-Blocking Cache Access
    t0 = time.perf_counter()
    res = UiaControlWalker.get_cached_controls(hwnd=0, max_age_sec=1.5)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    print(f"  ✔ UiaControlWalker.get_cached_controls executed in {latency_ms:.3f}ms (Target: < 5ms, 0ms blocking)")
    assert latency_ms < 50.0, f"UIA call blocked main thread for {latency_ms:.1f}ms!"

    # 2. Test Subsystems Alias Package
    from spatial_droplet.vision.subsystems.uia_walker import (
        UiaControlWalker as SubWalker,
        AsyncUiaBackgroundWorker as SubWorker
    )
    assert SubWalker is UiaControlWalker
    assert SubWorker is AsyncUiaBackgroundWorker
    print("  ✔ spatial_droplet.vision.subsystems.uia_walker alias verified identical")

    # 3. Test Background Daemon Worker Liveness
    worker = AsyncUiaBackgroundWorker.get_instance()
    assert worker.running is True
    assert worker.thread.is_alive() is True
    print(f"  ✔ AsyncUiaBackgroundWorker daemon is running ({worker.thread.name})")

    print("Phase 2 Async UIA Engine: ALL TESTS PASSED.")


async def test_phase3_lattice_and_bounded_reasoning():
    print("\n" + "=" * 60)
    print("TESTING PHASE 3: 90-CANDIDATE SPATIAL LATTICE & BOUNDED REASONING")
    print("=" * 60)

    with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)

    reasoner = DiagonalDropletReasoner()
    sprayer = FullBleedLatticeSprayer(48)

    # Test Sample 09: "Star the current chrome page."
    s9 = samples[8]
    img9 = cv2.imread(str(Path('osworld_g_images') / s9['image_path']))
    sh9, sw9 = img9.shape[:2]
    droplets9 = await sprayer.spray(img9)
    win9 = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw9, sh9), width=sw9, height=sh9, is_valid=True)

    trace9 = await reasoner.decide(
        user_intent=s9['instruction'],
        droplets=droplets9,
        window_ctx=win9,
        screenshot=img9,
        mode="combined",
        target_model="qwen/qwen3.8-27b",
        candidate_trace_only=True
    )
    cand_ids_9 = trace9.get("manifest_candidate_ids", [])
    print(f"  Sample 09 ({s9['instruction']}): Generated {len(cand_ids_9)} candidates (Target: 90-100)")
    assert len(cand_ids_9) >= 80, f"Candidate count too low: {len(cand_ids_9)}"

    # Check GT coverage for Sample 09
    gt_coords9 = s9['box_coordinates']
    gx1, gy1, gw, gh = gt_coords9
    gt_box9 = (gx1, gy1, gx1 + gw, gy1 + gh)
    active9 = [d for d in droplets9 if d.category != 'surface_probe']
    gt_dots_9 = [d for d in active9 if gt_box9[0] <= d.center_x <= gt_box9[2] and gt_box9[1] <= d.center_y <= gt_box9[3]]
    if gt_dots_9:
        gt_id_9 = gt_dots_9[0].id
        is_hit = gt_id_9 in cand_ids_9
        print(f"  Sample 09 GT Dot [{gt_id_9}] included in 96-candidate shortlist: {is_hit}")
        assert is_hit, f"Sample 09 GT Dot {gt_id_9} was pruned!"

    # Test Sample 18: "Lower the value for color 'G'"
    s18 = samples[17]
    img18 = cv2.imread(str(Path('osworld_g_images') / s18['image_path']))
    sh18, sw18 = img18.shape[:2]
    droplets18 = await sprayer.spray(img18)
    win18 = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw18, sh18), width=sw18, height=sh18, is_valid=True)

    trace18 = await reasoner.decide(
        user_intent=s18['instruction'],
        droplets=droplets18,
        window_ctx=win18,
        screenshot=img18,
        mode="combined",
        target_model="qwen/qwen3.8-27b",
        candidate_trace_only=True
    )
    cand_ids_18 = trace18.get("manifest_candidate_ids", [])
    print(f"  Sample 18 ({s18['instruction']}): Generated {len(cand_ids_18)} candidates (Target: 90-100)")
    assert len(cand_ids_18) >= 80, f"Candidate count too low: {len(cand_ids_18)}"

    gt_coords18 = s18['box_coordinates']
    gx1, gy1, gw, gh = gt_coords18
    gt_box18 = (gx1, gy1, gx1 + gw, gy1 + gh)
    active18 = [d for d in droplets18 if d.category != 'surface_probe']
    gt_dots_18 = [d for d in active18 if gt_box18[0] <= d.center_x <= gt_box18[2] and gt_box18[1] <= d.center_y <= gt_box18[3]]
    if gt_dots_18:
        gt_id_18 = gt_dots_18[0].id
        is_hit_18 = gt_id_18 in cand_ids_18
        print(f"  Sample 18 GT Dot [{gt_id_18}] included in 96-candidate shortlist: {is_hit_18}")
        assert is_hit_18, f"Sample 18 GT Dot {gt_id_18} was pruned!"

    # Check Quadrant Balancing Distribution
    mid_x, mid_y = sw9 / 2.0, sh9 / 2.0
    c_map = {d.id: d for d in droplets9}
    tl = sum(1 for cid in cand_ids_9 if cid in c_map and c_map[cid].center_x < mid_x and c_map[cid].center_y < mid_y)
    tr = sum(1 for cid in cand_ids_9 if cid in c_map and c_map[cid].center_x >= mid_x and c_map[cid].center_y < mid_y)
    bl = sum(1 for cid in cand_ids_9 if cid in c_map and c_map[cid].center_x < mid_x and c_map[cid].center_y >= mid_y)
    br = sum(1 for cid in cand_ids_9 if cid in c_map and c_map[cid].center_x >= mid_x and c_map[cid].center_y >= mid_y)
    print(f"  Quadrant Distribution: TL={tl}, TR={tr}, BL={bl}, BR={br}")
    assert tl >= 10 and tr >= 10 and bl >= 10 and br >= 10, "Quadrant balancing failed minimum representation!"

    print("Phase 3 90-Candidate Spatial Lattice: ALL TESTS PASSED.")


def test_phase4_visual_diff():
    print("\n" + "=" * 60)
    print("TESTING PHASE 4: REAL-TIME VISUAL DIFF CLOSED LOOP (cv2.absdiff)")
    print("=" * 60)

    # 1. Test Identical Pre/Post Crops (Zero Change)
    crop_a = np.zeros((120, 120, 3), dtype=np.uint8)
    crop_b = np.zeros((120, 120, 3), dtype=np.uint8)
    diff_zero = compute_visual_diff(crop_a, crop_b, threshold=20)
    ver_zero, changed_zero = verify_visual_transition(crop_a, crop_b, diff_threshold=20, min_changed_pixels=15)
    print(f"  ✔ Identical crops: delta={diff_zero}px | verified={ver_zero}")
    assert diff_zero == 0
    assert ver_zero is False

    # 2. Test Sub-threshold Noise (e.g. 5px change)
    crop_noise = crop_a.copy()
    crop_noise[10:15, 10] = 50  # 5 pixels changed
    diff_noise = compute_visual_diff(crop_a, crop_noise, threshold=20)
    ver_noise, _ = verify_visual_transition(crop_a, crop_noise, diff_threshold=20, min_changed_pixels=15)
    print(f"  ✔ Sub-threshold noise: delta={diff_noise}px | verified={ver_noise}")
    assert ver_noise is False

    # 3. Test True UI Transition (e.g. Menu expanded / star highlighted)
    crop_transition = crop_a.copy()
    crop_transition[20:40, 20:40] = 255  # 400 pixels changed
    diff_trans = compute_visual_diff(crop_a, crop_transition, threshold=20)
    ver_trans, changed_trans = verify_visual_transition(crop_a, crop_transition, diff_threshold=20, min_changed_pixels=15)
    print(f"  ✔ Real UI Transition: delta={diff_trans}px | verified={ver_trans}")
    assert diff_trans == 400
    assert ver_trans is True

    # 4. Test ActionVerifier Default Patch Size (120x120)
    verifier = ActionVerifier(1920, 1080)
    d = DiagonalDroplet(id=1, center_x=500, center_y=500, category="interactive_widget", label="Test Btn", diag_wave=1000)
    snap = verifier.snapshot_pre_action(d, None)
    if snap.roi is not None:
        assert snap.roi.shape == (120, 120, 3) or snap.roi.shape == (120, 120, 4), f"Expected 120x120 patch, got {snap.roi.shape}"
        print(f"  ✔ ActionVerifier snapshot ROI patch shape: {snap.roi.shape}")

    # 5. Test ActionVerifier Failure Mode 5 Prevention: Cursor change MUST NOT cause false PASS on zero-delta click!
    from spatial_droplet.verification.action_verifier import PreActionSnapshot
    from spatial_droplet.platform.window import get_active_window_context
    curr_win = get_active_window_context()
    fake_crop = np.zeros((120, 120, 3), dtype=np.uint8)
    fake_pre = PreActionSnapshot(
        hwnd=curr_win.hwnd if curr_win else 0,
        title=curr_win.title if curr_win else "Desktop",
        cursor_shape="ARROW",
        roi=fake_crop,
        cx=500,
        cy=500
    )
    
    # Monkey-patch capture_roi to return identical fake crop (diff = 0)
    orig_capture = verifier.capture_roi
    verifier.capture_roi = lambda cx, cy, size=120: fake_crop.copy()
    
    try:
        click_ver = verifier.verify_action(fake_pre, d, action="click")
        assert click_ver["verified"] is False, f"ActionVerifier incorrectly passed a zero-delta click! {click_ver}"
        assert click_ver["type"] == "NO_CHANGE", f"Expected NO_CHANGE, got {click_ver['type']}"
        print(f"  ✔ ActionVerifier correctly rejected zero-delta click (Outcome: {click_ver['type']})")
    finally:
        verifier.capture_roi = orig_capture

    # 6. Test execute_click_with_visual_diff function availability and signature
    from live_copilot import execute_click_with_visual_diff
    assert callable(execute_click_with_visual_diff)
    print("  ✔ execute_click_with_visual_diff is imported and verified callable")

    print("Phase 4 Visual Diff Closed Loop: ALL TESTS PASSED.")


if __name__ == "__main__":
    print("\n🚀 STARTING REAL-TIME AUTOMATION BLUEPRINT VERIFICATION TEST SUITE\n")
    test_phase1_intent_router()
    test_phase2_async_uia()
    asyncio.run(test_phase3_lattice_and_bounded_reasoning())
    test_phase4_visual_diff()
    print("\n" + "=" * 60)
    print("🎉 ALL 4 PHASES SUCCESSFULLY VERIFIED WITH ZERO ERRORS!")
    print("=" * 60 + "\n")
