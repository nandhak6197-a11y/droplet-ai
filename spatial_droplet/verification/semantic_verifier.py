"""
Cognitive Semantic State Verifier.
Performs whole-screen semantic verification via WinRT OCR token differential,
typed text confirmation, modal state tracking, and expected transition evaluation.
"""
import re
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

# Type alias for snapshot dictionary
SemanticSnapshot = Dict[str, Any]

try:
    import winocr
    HAS_WINOCR = True
except ImportError:
    HAS_WINOCR = False

from ..core.models import WindowContext
from ..platform.window import get_active_window_context
from ..platform.actuator import get_cursor_shape
from ..vision.grabber import FastScreenGrabber
from ..vision.screen_parser import ScreenStateParser

class SemanticStateVerifier:
    """
    Cognitive Semantic State Verifier (Phase 6 & 8).
    Replaces blind local pixel diffs with whole-screen semantic verification:
      - OCR Text Token Differential (T_post \ T_pre): Detects newly rendered and removed text.
      - Modal Dialog State Transitions: Detects when modal dialogs appear or dismiss.
      - Typed Text Confirmation: Confirms typed string is present on screen.
      - Predictive Hypothesis Evaluation: Verifies if the observed delta matches the reasoner's Expected Visual Transition.
    """
    @classmethod
    def capture_snapshot(cls, win_ctx: Optional[WindowContext] = None) -> Dict[str, Any]:
        frame = FastScreenGrabber.get_instance().grab_full()
        text_lines: List[str] = []
        if HAS_WINOCR:
            try:
                def _run_mta(f):
                    op = winocr.recognize_cv2(f, lang='en')
                    return op.get() if hasattr(op, 'get') else op
                ocr_res = _run_mta(frame)
                if ocr_res and hasattr(ocr_res, 'lines'):
                    text_lines = [l.text.strip() for l in ocr_res.lines if l.text.strip()]
            except Exception:
                pass

        active_ctx = get_active_window_context()
        modal = ScreenStateParser.detect_modal_dialog(active_ctx)
        token_set = set(re.findall(r'\w+', " ".join(text_lines).lower()))

        return {
            "frame": frame,
            "text_lines": text_lines,
            "tokens": token_set,
            "modal": modal,
            "window_title": active_ctx.title if active_ctx else "Desktop",
            "hwnd": active_ctx.hwnd if active_ctx else 0,
            "cursor_shape": get_cursor_shape(),
            "timestamp": time.time()
        }

    @classmethod
    def verify_transition(
        cls,
        pre_snap: Dict[str, Any],
        post_snap: Dict[str, Any],
        action: str,
        input_text: Optional[str],
        semantic_intention: str,
        expected_transition: str
    ) -> Dict[str, Any]:
        pre_tokens = pre_snap.get("tokens", set())
        post_tokens = post_snap.get("tokens", set())

        added_tokens = list(post_tokens - pre_tokens)
        removed_tokens = list(pre_tokens - post_tokens)

        pre_hwnd = pre_snap.get("hwnd")
        post_hwnd = post_snap.get("hwnd")
        window_switched = (pre_hwnd != post_hwnd)
        post_title = post_snap.get("window_title", "")

        pre_modal = pre_snap.get("modal")
        post_modal = post_snap.get("modal")
        modal_opened = (pre_modal is None and post_modal is not None)
        modal_closed = (pre_modal is not None and post_modal is None)

        pre_cursor = pre_snap.get("cursor_shape", "")
        post_cursor = post_snap.get("cursor_shape", "")
        cursor_changed = (pre_cursor != post_cursor)

        # 1. Type action text verification
        if action in ("type", "clear_and_type", "type_and_enter", "click_and_type") and input_text:
            clean_input = input_text.strip().lower()
            input_words = set(re.findall(r'\w+', clean_input))
            if input_words and input_words.issubset(post_tokens):
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Typed text '{input_text[:16]}' confirmed on screen."
                }
            elif any(clean_input in l.lower() for l in post_snap.get("text_lines", [])):
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Typed text '{input_text[:16]}' detected in screen lines."
                }
            elif len(added_tokens) > 0:
                return {
                    "verified": True,
                    "verdict": "PARTIAL_CHANGE",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Screen text changed (+{len(added_tokens)} words), input presence partial."
                }
            else:
                return {
                    "verified": False,
                    "verdict": "NO_EFFECT",
                    "text_added": [],
                    "text_removed": [],
                    "reason": f"Typed text '{input_text[:16]}' did not appear on screen."
                }

        # 1.5 Application launch verification & invalid target guard (e.g. LAUNCH_APP '2')
        if action in ("launch_app", "system_launch_app", "open_app"):
            target_str = (input_text or semantic_intention or "").strip().lower()
            target_clean = re.sub(r'[\s.,!?]+$', '', target_str).strip()

            # Guard against invalid launch targets (numbers, arithmetic expressions, single symbols)
            is_formula = bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', target_clean)) or any(op in target_clean for op in ("+", "*", "=", "/", "<", ">"))
            if not target_clean or target_clean.isdigit() or is_formula or (len(target_clean) <= 2 and not target_clean.isalpha()):
                return {
                    "verified": False,
                    "verdict": "INVALID_ACTION",
                    "text_added": [],
                    "text_removed": [],
                    "reason": f"Invalid launch target '{input_text}': numbers, formulas and symbols cannot be launched as applications."
                }

            # Verify window switch or application title presence in post snapshot
            target_tokens = set(re.findall(r'[a-zA-Z0-9]+', target_clean))
            post_title_tokens = set(re.findall(r'[a-zA-Z0-9]+', post_title.lower()))
            matched_title = bool(target_tokens.intersection(post_title_tokens)) or target_clean in post_title.lower()

            # Guard: If application was ALREADY in foreground before the turn, this was NOT a launch transition!
            pre_title = pre_snap.get("window_title", "")
            pre_title_tokens = set(re.findall(r'[a-zA-Z0-9]+', pre_title.lower()))
            was_already_active = (not window_switched) and (bool(target_tokens.intersection(pre_title_tokens)) or target_clean in pre_title.lower())
            if was_already_active and matched_title:
                return {
                    "verified": False,
                    "verdict": "ALREADY_ACTIVE",
                    "text_added": [],
                    "text_removed": [],
                    "reason": f"Application '{target_clean}' was already active in foreground window '{post_title[:20]}'."
                }

            if window_switched and matched_title:
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Application '{target_clean}' launched and active window is '{post_title[:20]}'."
                }
            elif matched_title and (pre_title.lower() == "desktop" or not pre_title):
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Application '{target_clean}' launched from Desktop into '{post_title[:20]}'."
                }
            return {
                "verified": False,
                "verdict": "NO_EFFECT",
                "text_added": [],
                "text_removed": [],
                "reason": f"Application '{target_clean}' did not open into foreground (active window remains '{post_title[:20]}')."
            }

        # 1.8 Window close verification
        if action in ("close", "close_window"):
            pre_title = pre_snap.get("window_title", "")
            if window_switched or post_title.lower() == "desktop" or post_title != pre_title:
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                    "text_added": added_tokens[:6],
                    "text_removed": removed_tokens[:6],
                    "reason": f"Window '{pre_title[:20]}' closed successfully (active window is '{post_title[:20]}')."
                }
            return {
                "verified": False,
                "verdict": "NO_EFFECT",
                "text_added": [],
                "text_removed": [],
                "reason": f"Window '{pre_title[:20]}' did not close (still active)."
            }

        # 2. Window switch transition
        if window_switched:
            return {
                "verified": True,
                "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Window switched to '{post_title[:20]}'."
            }

        # 3. Modal dialog appearance / dismissal
        if modal_opened:
            return {
                "verified": True,
                "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Modal dialog '{post_modal.title[:20]}' appeared."
            }
        if modal_closed:
            return {
                "verified": True,
                "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Modal dialog dismissed successfully."
            }

        # 4. Expected transition keyword matching
        exp_words = set(re.findall(r'\w+', (expected_transition or "").lower()))
        matched_added = [w for w in added_tokens if w in exp_words]
        if matched_added:
            sample = ", ".join(matched_added[:3])
            return {
                "verified": True,
                "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Predicted change confirmed: [{sample}] appeared."
            }

        # 5. General text addition/removal delta
        if len(added_tokens) >= 2 or len(removed_tokens) >= 2:
            sample = ", ".join(added_tokens[:3]) if added_tokens else ", ".join(removed_tokens[:3])
            return {
                "verified": True,
                "verdict": "CONFIRMED_EXPECTED_TRANSITION",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Screen text changed ({'+' if added_tokens else '-'}{sample})."
            }

        if cursor_changed:
            return {
                "verified": True,
                "verdict": "PARTIAL_CHANGE",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Cursor changed to {post_cursor}."
            }

        if len(added_tokens) == 1:
            return {
                "verified": True,
                "verdict": "PARTIAL_CHANGE",
                "text_added": added_tokens[:6],
                "text_removed": removed_tokens[:6],
                "reason": f"Minor change: '{added_tokens[0]}' appeared."
            }

        # 6. Fallback local pixel diff
        pre_f = pre_snap.get("frame")
        post_f = post_snap.get("frame")
        if pre_f is not None and post_f is not None and pre_f.shape == post_f.shape:
            d1 = cv2.resize(pre_f, (64, 36))
            d2 = cv2.resize(post_f, (64, 36))
            diff = float(np.mean(np.abs(d2.astype(float) - d1.astype(float))))
            if diff > 7.0:
                return {
                    "verified": True,
                    "verdict": "CONFIRMED_VISUAL_CHANGE",
                    "text_added": [],
                    "text_removed": [],
                    "reason": f"Significant visual pixel transition detected (Δ={diff:.1f})."
                }
            elif diff > 2.5:
                return {
                    "verified": False,
                    "verdict": "PARTIAL_CHANGE",
                    "text_added": [],
                    "text_removed": [],
                    "reason": f"Minor pixel delta detected (Δ={diff:.1f}), but no semantic text or window change."
                }

        return {
            "verified": False,
            "verdict": "NO_EFFECT",
            "text_added": [],
            "text_removed": [],
            "reason": "Zero semantic screen or text transition detected."
        }
