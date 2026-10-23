"""
Comprehensive Verification Test Suite for:
1. Procedural 3D Sphere Mascot Animated Eyes & Task-Reactive Emotions + Hover Attentiveness
2. Modern Glassmorphic Dark UI, Frosted Button Tiles & Compact Companion Pod with Symmetrical Companion Mic
3. Fast-Path Navigation & Resilient Compound Commands with Trailing Punctuation Stripping
4. Whisper Prompt Conditioning (Desktop Automation Terms), Language Locking ('en'), and RMS Silence Gating
"""

import sys
import time
import tkinter as tk
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_droplet.ui.chat_widget import SphereMascotAnimator, HoverChatWidget
from spatial_droplet.voice.transcriber import GroqWhisperTranscriber, DEFAULT_DESKTOP_PROMPT
from spatial_droplet.voice.voice_manager import VoiceControlManager
from live_copilot import IntentScopeRouter
from spatial_droplet.engine.fast_path import FastPathRouter


def test_mascot_animator():
    print("--- [1] Testing SphereMascotAnimator & Sensitive Emotions ---")
    anim = SphereMascotAnimator(base_size=56)
    assert anim.emotion == "idle"

    # Test emotion transitions
    test_states = [
        ("Thinking about next step", False, "thinking", (192, 132, 252)),
        ("⚡ Processing: 'search chrome'...", False, "thinking", (192, 132, 252)),
        ("⚡ Voice: 'go to chrome'...", False, "thinking", (192, 132, 252)),
        ("Executing click on button", False, "executing", (56, 189, 248)),
        ("⚡ Fast-Path: Open Chrome ➜ ...", False, "executing", (56, 189, 248)),
        ("✔ Goal completed successfully", False, "happy", (16, 185, 129)),
        ("❌ Target element not found", False, "confused", (244, 63, 94)),
        ("🎙️ Listening for user speech", True, "listening", (0, 240, 255)),
        ("⚡ Transcribing audio", False, "transcribing", (255, 180, 0)),
        ("Scanning screen landmarks", False, "scanning", (0, 229, 255)),
        ("Vimium Quick-Select Active", False, "vimium", (251, 191, 36)),
        ("✔ Live Auto-Sync", False, "idle", (0, 229, 255)),
    ]

    for status, is_rec, expected_emotion, expected_rgb in test_states:
        anim.set_status(status, is_rec)
        assert anim.emotion == expected_emotion, f"Status '{status}' expected {expected_emotion}, got {anim.emotion}"
        assert anim.target_glow_rgb == expected_rgb, f"Expected {expected_rgb}, got {anim.target_glow_rgb}"

        # Advance ticks to verify lerp stability
        for _ in range(5):
            anim.tick()

        # Render frame
        frame = anim.render_frame(size=56)
        assert frame.size == (56, 56)
        assert frame.mode == "RGBA"
        # Verify frame contains non-transparent pixels
        arr = np.array(frame)
        assert np.count_nonzero(arr[:, :, 3] > 0) > 500, "Rendered mascot frame must contain avatar pixels"

    # Test hover attentiveness
    anim.set_hover(True)
    assert anim.is_hovered is True
    assert anim.target_eye_scale_y > 1.0, "Eyes should perk up on hover"
    anim.set_hover(False)
    assert anim.is_hovered is False

    # Test blinking state machine
    anim.blink_timer = 0.0  # Force blink
    anim.tick()
    assert anim.is_blinking is True, "Blink must activate when timer expires"
    while anim.is_blinking:
        anim.tick()
    assert anim.is_blinking is False, "Blink must conclude cleanly"
    print("✔ SphereMascotAnimator passed all emotion, hover, and rendering tests!")


def test_chat_widget_compact_pod():
    print("\n--- [2] Testing HoverChatWidget & Compact Companion Pod ---")
    root = tk.Tk()
    root.withdraw()

    submitted_cmds = []
    voice_toggled = [False]
    compact_states = []

    def on_submit(cmd):
        submitted_cmds.append(cmd)

    def on_voice():
        voice_toggled[0] = True

    def on_compact(is_expanded):
        compact_states.append(is_expanded)

    widget = HoverChatWidget(
        root,
        on_submit_callback=on_submit,
        on_toggle_callback=lambda: None,
        on_refresh_callback=lambda: None,
        on_voice_toggle_callback=on_voice,
        on_compact_callback=on_compact
    )

    # Initial state: Expanded Capsule
    assert widget.is_expanded is True
    initial_rect = widget.get_rect()
    assert initial_rect[2] - initial_rect[0] in (580, 684), f"Expected 580w or 684w, got {initial_rect}"
    assert initial_rect[3] - initial_rect[1] in (76, 96), f"Expected 76h or 96h, got {initial_rect}"

    # Shrink to Compact Pod (animate=False for deterministic test)
    widget.collapse_to_orb(animate=False)
    assert widget.is_expanded is False
    assert len(compact_states) > 0 and compact_states[-1] is False, "on_compact_callback(False) must be triggered"

    compact_rect = widget.get_rect()
    assert compact_rect[2] - compact_rect[0] in (152, 160), f"Expected 152w or 160w, got {compact_rect}"
    assert compact_rect[3] - compact_rect[1] in (74, 76), f"Expected 74h or 76h, got {compact_rect}"

    # Verify Companion Mic elements exist in compact mode
    assert hasattr(widget, "compact_mic_bg") and widget.compact_mic_bg, "Compact mode must render companion mic background"
    assert hasattr(widget, "compact_mic_glyph") and widget.compact_mic_glyph, "Compact mode must render companion mic glyph"

    # Test Companion Mic trigger
    widget._trigger_voice()
    assert voice_toggled[0] is True, "Clicking companion mic must trigger voice toggle"

    # Test Voice Recording State in Compact Mode
    widget.set_voice_recording(True)
    assert widget.is_voice_recording is True
    assert widget.animator.emotion == "listening"

    # Test in-place status update without canvas churn
    old_bg_tag = widget.bg_tag
    widget.set_status("⚡ Processing: search...", color="#ffb300")
    assert widget.bg_tag == old_bg_tag, "Compact status update must update in-place without rebuilding canvas"

    # Test Voice Transcribed Text auto-submit in Compact Mode
    widget.set_voice_recording(False)
    widget.set_voice_text("go to chrome and open youtube", auto_submit=True)
    assert "go to chrome and open youtube" in submitted_cmds, "Voice result must auto-submit directly in compact mode"

    # Test animation re-entrancy guard
    widget.collapse_to_orb(animate=True)
    assert widget.is_expanded is False
    widget.expand_to_capsule(animate=False)
    assert widget.is_expanded is True
    assert compact_states[-1] is True, "on_compact_callback(True) must be triggered"
    expanded_rect = widget.get_rect()
    assert expanded_rect[2] - expanded_rect[0] in (580, 684)

    widget.win.destroy()
    root.destroy()
    print("✔ HoverChatWidget compact pod & companion mic passed all tests!")


def test_fast_path_compound_routes():
    print("\n--- [3] Testing Fast-Path Intent Router & Compound Commands ---")

    # 1. Compound: "go to chrome and open youtube"
    r1 = IntentScopeRouter.route("go to chrome and open youtube")
    assert r1 is not None, "Compound navigation must be recognized"
    assert r1.get("action") == "system_compound"
    steps1 = r1.get("steps", [])
    assert len(steps1) == 2
    assert steps1[0]["target"] == "chrome"
    assert steps1[0].get("parameters") == "https://www.youtube.com"
    assert steps1[1]["url"] == "https://www.youtube.com"

    # 2. Compound with trailing period (Whisper artifact): "go to chrome and open youtube."
    r1_dot = IntentScopeRouter.route("go to chrome and open youtube.")
    assert r1_dot is not None
    assert r1_dot.get("action") == "system_compound"
    assert r1_dot.get("steps")[1]["url"] == "https://www.youtube.com", "Trailing period must not break YouTube URL mapping"

    # 3. Compound with generic target: "go to chrome and open some stuff like that"
    r2 = IntentScopeRouter.route("go to chrome and open some stuff like that")
    assert r2 is not None
    assert r2.get("action") == "launch_app"
    assert r2.get("target") == "chrome"

    # 4. Inverted pattern: "open youtube in chrome"
    r_inv1 = IntentScopeRouter.route("open youtube in chrome")
    assert r_inv1 is not None
    assert r_inv1.get("action") == "system_compound"
    assert r_inv1.get("steps")[0]["target"] == "chrome"
    assert r_inv1.get("steps")[1]["url"] == "https://www.youtube.com"

    # 5. Inverted pattern: "open youtube on chrome"
    r_inv2 = IntentScopeRouter.route("open youtube on chrome")
    assert r_inv2 is not None
    assert r_inv2.get("action") == "system_compound"
    assert r_inv2.get("steps")[1]["url"] == "https://www.youtube.com"

    # 6. Compound to: "go to chrome to open youtube"
    r_to = IntentScopeRouter.route("go to chrome to open youtube")
    assert r_to is not None
    assert r_to.get("action") == "system_compound"
    assert r_to.get("steps")[1]["url"] == "https://www.youtube.com"

    # 7. Compound browse: "open chrome and browse youtube"
    r_browse = IntentScopeRouter.route("open chrome and browse youtube")
    assert r_browse is not None
    assert r_browse.get("action") == "system_compound"
    assert r_browse.get("steps")[1]["url"] == "https://www.youtube.com"

    # 8. Compound sub-search: "open chrome and search youtube for music"
    r_sub = IntentScopeRouter.route("open chrome and search youtube for music")
    assert r_sub is not None
    assert r_sub.get("action") == "system_compound"
    assert "youtube.com/results?search_query=music" in r_sub.get("steps")[1]["url"]

    # 9. Direct switch / app with trailing period: "open chrome."
    r_app_dot = IntentScopeRouter.route("open chrome.")
    assert r_app_dot is not None
    assert r_app_dot.get("action") in ("window_switch", "launch_app")
    assert r_app_dot.get("target") == "chrome"

    # 10. Web search: "search cute cats on youtube"
    r5 = IntentScopeRouter.route("search cute cats on youtube")
    assert r5 is not None
    assert r5.get("action") == "system_open_url"
    assert "youtube.com/results?search_query=cute+cats" in r5.get("url")

    # 11. Google search: "search machine learning on google"
    r6 = IntentScopeRouter.route("search machine learning on google")
    assert r6 is not None
    assert r6.get("action") == "system_open_url"
    assert "google.com/search?q=machine+learning" in r6.get("url")

    # 12. FastPathRouter integration
    fp_r = FastPathRouter.route("go to chrome and open youtube", [], None)
    assert fp_r is not None
    assert fp_r.get("action") == "system_compound"
    assert len(fp_r.get("steps", [])) == 2

    print("✔ Fast-path compound routing passed all tests!")


def test_voice_prompt_and_silence_gating():
    print("\n--- [4] Testing Whisper Prompt Conditioning & RMS Silence Gating ---")

    # 1. Transcriber default prompt verification
    assert "Chrome" in DEFAULT_DESKTOP_PROMPT
    assert "Camera" in DEFAULT_DESKTOP_PROMPT
    assert "search" in DEFAULT_DESKTOP_PROMPT

    transcriber = GroqWhisperTranscriber()
    import inspect
    sig = inspect.signature(transcriber.transcribe_pcm)
    assert "prompt" in sig.parameters
    assert sig.parameters["language"].default == "en", "Transcriber must default to language='en' for speed and precision"

    # 2. RMS Energy Silence Gating
    mgr = VoiceControlManager()

    # Create synthetic pure silence (RMS ~ 0)
    silent_audio = np.zeros(16000 * 2, dtype=np.int16)
    statuses = []

    def mock_status(txt, col):
        statuses.append(txt)

    # Process recording with silence
    mgr.recorder.stop = lambda: silent_audio
    mgr._process_recording(on_status=mock_status, on_result=lambda t: None)

    assert any("silence" in s.lower() or "too quiet" in s.lower() for s in statuses), \
        f"Silence must be gated before API call, got: {statuses}"

    print("✔ Whisper prompt conditioning and silence gating passed all tests!")


if __name__ == "__main__":
    test_mascot_animator()
    test_chat_widget_compact_pod()
    test_fast_path_compound_routes()
    test_voice_prompt_and_silence_gating()
    print("\n=======================================================")
    print(" ALL COPILOT UI, ANIMATION, & VOICE TESTS PASSED 100%! ")
    print("=======================================================")
