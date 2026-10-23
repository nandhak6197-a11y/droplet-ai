"""
Pre/Post Action ROI Delta Sensor, Cursor Verification & Closed-Loop Auto-Pivot.
Performs deterministic closed-loop verification via cv2.absdiff motion delta
and semantic window title change tracking to confirm physical UI transitions.
"""
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..core.models import DiagonalDroplet, WindowContext
from ..platform.actuator import get_cursor_shape
from ..platform.window import get_active_window_context
from ..vision.grabber import FastScreenGrabber


@dataclass
class PreActionSnapshot:
    hwnd: int
    title: str
    cursor_shape: str
    roi: Optional[np.ndarray]
    cx: int
    cy: int
    full_frame: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


class ActionVerifier:
    """
    Closed-loop physical action verifier with multi-signal delta sensing:
      - Active window HWND / title shift (semantic transition confirmation)
      - OS cursor shape transformation (ARROW -> IBEAM / HAND) for hover/move
      - Sub-region pixel difference via cv2.absdiff (diff_pixels > 15 or mean delta > 2.8)
      - Whole-screen frame difference for global actions / peripheral changes
      - Failure recording and automatic candidate/action pivoting
    """

    def __init__(self, sw: int = 1920, sh: int = 1080):
        self.sw = sw
        self.sh = sh
        self.blacklisted_ids: Set[int] = set()
        self.failure_counts: Dict[int, int] = {}
        self.failure_history: List[Dict[str, Any]] = []

    def reset_blacklist(self):
        """Clears blacklisted droplet IDs on task transitions."""
        self.blacklisted_ids.clear()
        self.failure_counts.clear()
        self.failure_history.clear()

    def record_failure(self, droplet_id: Optional[int], action: str, reason: str, delta: float):
        """Records an ineffective action on a droplet and blacklists it."""
        safe_id = droplet_id if droplet_id is not None else -1
        cnt = self.failure_counts.get(safe_id, 0) + 1
        self.failure_counts[safe_id] = cnt
        if safe_id >= 0:
            self.blacklisted_ids.add(safe_id)
        self.failure_history.append({
            "droplet_id": safe_id,
            "action": action,
            "reason": reason,
            "delta": delta,
            "timestamp": time.time()
        })

    def is_blacklisted(self, droplet_id: Optional[int]) -> bool:
        """Checks if a droplet ID has previously failed physical verification."""
        if droplet_id is None:
            return False
        return droplet_id in self.blacklisted_ids

    def suggest_pivot(
        self,
        failed_droplet: Optional[DiagonalDroplet],
        candidates: Optional[List[DiagonalDroplet]] = None,
        action: str = "click",
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Auto-pivots when an action fails physical verification:
        1. If suitable alternative candidates exist nearby, recommends the top candidate.
        2. If same target should be escalated, recommends double-click, press enter, or clear-and-type.
        """
        fx = failed_droplet.center_x if failed_droplet else self.sw // 2
        fy = failed_droplet.center_y if failed_droplet else self.sh // 2
        failed_id = failed_droplet.id if failed_droplet else None

        # Candidate-based pivot
        if candidates:
            valid_cands = [
                c for c in candidates
                if (failed_id is None or c.id != failed_id) and c.id not in self.blacklisted_ids
            ]
            if valid_cands:
                # Rank candidates by spatial proximity and lexical match
                def score_cand(c: DiagonalDroplet) -> float:
                    dist = math.hypot(c.center_x - fx, c.center_y - fy)
                    spatial_score = max(0.0, 1.0 - (dist / 400.0))
                    lexical_score = 0.0
                    if query and getattr(c, "label", None):
                        q_words = query.lower().split()
                        lbl = c.label.lower()
                        matches = sum(1 for w in q_words if w in lbl)
                        lexical_score = matches / max(1, len(q_words))
                    return (lexical_score * 2.0) + spatial_score

                valid_cands.sort(key=score_cand, reverse=True)
                best_alt = valid_cands[0]
                return {
                    "pivot_type": "ALTERNATIVE_CANDIDATE",
                    "target_dot_id": best_alt.id,
                    "target_droplet": best_alt,
                    "suggested_action": "click",
                    "reason": f"Auto-pivot to runner-up Dot [{best_alt.id}] '{getattr(best_alt, 'label', '')}' at ({best_alt.center_x}, {best_alt.center_y})"
                }

        # Action escalation pivot
        act_lower = action.lower()
        if act_lower in ("click", "single_click"):
            return {
                "pivot_type": "ACTION_ESCALATION",
                "target_dot_id": failed_id,
                "suggested_action": "double_click",
                "reason": f"Escalating action on Dot [{failed_id or 'Target'}] to double-click"
            }
        elif act_lower == "double_click":
            return {
                "pivot_type": "ACTION_ESCALATION",
                "target_dot_id": failed_id,
                "suggested_action": "press_enter",
                "reason": f"Escalating action on Dot [{failed_id or 'Target'}] to keyboard Enter"
            }
        elif act_lower in ("type", "fill"):
            return {
                "pivot_type": "ACTION_ESCALATION",
                "target_dot_id": failed_id,
                "suggested_action": "clear_and_type",
                "reason": f"Escalating action on Dot [{failed_id or 'Target'}] to clear_and_type"
            }

        return {
            "pivot_type": "FALLBACK_RETRY",
            "target_dot_id": failed_id,
            "suggested_action": "click",
            "reason": f"Retrying Dot [{failed_id or 'Target'}] with spatial micro-jitter"
        }

    def capture_roi(self, cx: int, cy: int, size: int = 120) -> Optional[np.ndarray]:
        """Captures local sub-region centered around (cx, cy) with bounds safety."""
        half = size // 2
        x1 = max(0, min(self.sw - size, cx - half))
        y1 = max(0, min(self.sh - size, cy - half))
        return FastScreenGrabber.get_instance().grab_box(int(x1), int(y1), int(size), int(size))

    def capture_full_frame(self) -> Optional[np.ndarray]:
        """Captures full screen frame for global transition diffing."""
        return FastScreenGrabber.get_instance().grab_full()

    def snapshot_pre_action(
        self,
        droplet: Optional[DiagonalDroplet] = None,
        win_ctx: Optional[WindowContext] = None,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None
    ) -> PreActionSnapshot:
        """Captures multi-signal baseline state prior to action execution."""
        if droplet:
            cx, cy = droplet.center_x, droplet.center_y
        elif target_x is not None and target_y is not None:
            cx, cy = int(target_x), int(target_y)
        elif win_ctx and win_ctx.rect:
            rx1, ry1, rx2, ry2 = win_ctx.rect
            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
        else:
            cx, cy = self.sw // 2, self.sh // 2

        cur = get_cursor_shape()
        hwnd = win_ctx.hwnd if win_ctx else 0
        title = win_ctx.title if win_ctx else "Desktop"
        roi = self.capture_roi(cx, cy, size=120)
        full_frame = self.capture_full_frame()
        return PreActionSnapshot(
            hwnd=hwnd,
            title=title,
            cursor_shape=cur,
            roi=roi,
            cx=cx,
            cy=cy,
            full_frame=full_frame
        )

    @staticmethod
    def compute_roi_delta(roi_a: np.ndarray, roi_b: np.ndarray, threshold: int = 20) -> Tuple[int, float]:
        """Computes pixel count delta and mean absolute delta via cv2.absdiff."""
        if roi_a is None or roi_b is None or roi_a.shape != roi_b.shape:
            return 0, 0.0

        if HAS_CV2:
            try:
                delta = cv2.absdiff(roi_a, roi_b)
                delta_gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY) if len(delta.shape) == 3 else delta
                diff_pixels = int(np.count_nonzero(delta_gray > threshold))
                mean_delta = float(np.mean(delta_gray))
                return diff_pixels, mean_delta
            except Exception:
                pass

        # Vectorized numpy fallback
        diff = np.abs(roi_a.astype(np.int16) - roi_b.astype(np.int16))
        if len(diff.shape) == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff
        diff_pixels = int(np.count_nonzero(diff_gray > threshold))
        mean_delta = float(np.mean(diff_gray))
        return diff_pixels, mean_delta

    def verify_action(
        self,
        pre: PreActionSnapshot,
        droplet: Optional[DiagonalDroplet],
        action: str,
        candidates: Optional[List[DiagonalDroplet]] = None,
        query: Optional[str] = None,
        post_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-signal closed-loop verification:
        1. Window context change (HWND switch or semantic title change)
        2. Local ROI pixel delta via cv2.absdiff
        3. Whole-screen motion delta via cv2.absdiff
        4. Cursor shape transformation (for hover actions only)
        5. Adaptive polling up to 0.70s for asynchronous repaints
        """
        # Dwell 160ms for initial OS repaint
        time.sleep(0.16)

        post_ctx = get_active_window_context()
        post_hwnd = post_ctx.hwnd if post_ctx else 0
        post_title = post_ctx.title if post_ctx else "Desktop"
        post_cur = get_cursor_shape()
        post_roi = self.capture_roi(pre.cx, pre.cy, size=120)

        # 1. Semantic Window HWND / Title Shift Verification
        if post_hwnd != pre.hwnd:
            return {
                "verified": True,
                "reason": f"Window HWND switched: {pre.hwnd} ➜ {post_hwnd} ('{post_title[:20]}')",
                "delta": 100.0,
                "type": "WINDOW_CHANGE"
            }

        # Check semantic title change (supporting window close/minimize to Desktop)
        clean_pre_title = (pre.title or "").strip()
        clean_post_title = (post_title or "").strip()
        if clean_pre_title != clean_post_title and not (clean_pre_title == "Desktop" and clean_post_title == "Desktop"):
            return {
                "verified": True,
                "reason": f"Window shifted: '{clean_pre_title[:18]}' ➜ '{clean_post_title[:18]}'",
                "delta": 100.0,
                "type": "WINDOW_CHANGE"
            }

        # 2. Local Visual ROI Difference via cv2.absdiff
        diff_pixels, mean_delta = self.compute_roi_delta(pre.roi, post_roi, threshold=20)
        if diff_pixels > 15 or mean_delta > 2.8:
            return {
                "verified": True,
                "reason": f"Visual pixel transition verified via cv2.absdiff (pixels={diff_pixels} > 15, mean={mean_delta:.1f})",
                "delta": diff_pixels,
                "mean_delta": mean_delta,
                "type": "PIXEL_DELTA"
            }

        # 3. Cursor Transformation (ONLY valid for hover/movement actions)
        act_lower = action.lower().strip()
        is_hover_action = act_lower in ("hover", "move_to", "hover_and_wait", "hover_wait")
        if is_hover_action and post_cur != pre.cursor_shape:
            return {
                "verified": True,
                "reason": f"Cursor transformed: {pre.cursor_shape} ➜ {post_cur}",
                "delta": 50.0,
                "type": "CURSOR_CHANGE"
            }

        # 4. Adaptive Polling for Asynchronous UI Transitions (up to 0.70s)
        start_t = time.time()
        max_poll_sec = 0.70
        while (time.time() - start_t) < max_poll_sec:
            time.sleep(0.070)
            poll_ctx = get_active_window_context()
            poll_hwnd = poll_ctx.hwnd if poll_ctx else 0
            poll_title = (poll_ctx.title if poll_ctx else "Desktop").strip()
            poll_cur = get_cursor_shape()

            # Window shift during polling
            if poll_hwnd != pre.hwnd or (clean_pre_title != poll_title and not (clean_pre_title == "Desktop" and poll_title == "Desktop")):
                return {
                    "verified": True,
                    "reason": f"Window shifted: '{clean_pre_title[:18]}' ➜ '{poll_title[:18]}'",
                    "delta": 100.0,
                    "type": "WINDOW_CHANGE"
                }

            poll_roi = self.capture_roi(pre.cx, pre.cy, size=120)
            p_diff_pixels, p_mean_delta = self.compute_roi_delta(pre.roi, poll_roi, threshold=20)
            if p_diff_pixels > 15 or p_mean_delta > 2.8:
                return {
                    "verified": True,
                    "reason": f"Visual pixel delta detected via cv2.absdiff (pixels={p_diff_pixels} > 15, mean={p_mean_delta:.1f})",
                    "delta": p_diff_pixels,
                    "mean_delta": p_mean_delta,
                    "type": "PIXEL_DELTA"
                }

            if is_hover_action and poll_cur != pre.cursor_shape:
                return {
                    "verified": True,
                    "reason": f"Cursor transformed: {pre.cursor_shape} ➜ {poll_cur}",
                    "delta": 50.0,
                    "type": "CURSOR_CHANGE"
                }

        # 5. Whole-Screen Frame Delta Comparison (for non-spatial / peripheral actions)
        post_full = post_frame if post_frame is not None else self.capture_full_frame()
        if pre.full_frame is not None and post_full is not None and pre.full_frame.shape == post_full.shape:
            # Subsample 16x for whole-screen delta
            full_diff_pixels, full_mean_delta = self.compute_roi_delta(
                pre.full_frame[::16, ::16], post_full[::16, ::16], threshold=20
            )
            if full_diff_pixels > 30 or full_mean_delta > 3.0:
                return {
                    "verified": True,
                    "reason": f"Fullscreen transition verified via cv2.absdiff (delta_pixels={full_diff_pixels}, mean={full_mean_delta:.1f})",
                    "delta": full_diff_pixels,
                    "mean_delta": full_mean_delta,
                    "type": "FULLSCREEN_DELTA"
                }

        # 6. Action Verification Failed: Zero Visual or Semantic Change
        safe_id = droplet.id if droplet else None
        self.record_failure(safe_id, action, f"Zero visual change (Δ={diff_pixels}px)", diff_pixels)
        pivot = self.suggest_pivot(droplet, candidates=candidates, action=action, query=query)
        return {
            "verified": False,
            "reason": f"Zero visual change detected via cv2.absdiff (Δ={diff_pixels}px)",
            "delta": diff_pixels,
            "type": "NO_CHANGE",
            "blacklisted_id": safe_id,
            "suggested_pivot": pivot.get("reason"),
            "suggested_alternative_id": pivot.get("target_dot_id"),
            "suggested_action": pivot.get("suggested_action")
        }
