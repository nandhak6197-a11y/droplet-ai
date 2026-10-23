"""
Adaptive Dynamic UI Settle & Visual Stability Sensor (Pillar 1).
Prevents spraying half-rendered animations, CSS transitions, and loading states
by adaptively sensing when screen visual motion drops below the stability threshold.
Utilizes OpenCV cv2.absdiff hardware-accelerated motion sensing with numpy fallback.
"""
import time
from typing import Optional, Tuple, Callable, Any
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class AdaptiveSettleSensor:
    """
    Real-time screen motion and rendering stabilization sensor.
    Replaces static sleep timers with dynamic visual entropy polling.
    """

    @staticmethod
    def compute_motion_delta(
        frame_a: Optional[np.ndarray],
        frame_b: Optional[np.ndarray],
        subsample: int = 8
    ) -> float:
        """
        Calculates normalized mean absolute difference between two image frames
        via fast spatial subsampling and cv2.absdiff (<0.5ms).
        """
        if frame_a is None or frame_b is None:
            return 999.0
        if frame_a.shape != frame_b.shape:
            return 999.0
        if frame_a.size == 0 or frame_b.size == 0:
            return 0.0

        # Subsample frames for sub-millisecond execution
        step = max(1, subsample)
        if len(frame_a.shape) == 3:
            sub_a = frame_a[::step, ::step]
            sub_b = frame_b[::step, ::step]
        else:
            sub_a = frame_a[::step, ::step]
            sub_b = frame_b[::step, ::step]

        if HAS_CV2:
            try:
                diff = cv2.absdiff(sub_a, sub_b)
                if len(diff.shape) == 3:
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                else:
                    diff_gray = diff
                return float(np.mean(diff_gray))
            except Exception:
                pass

        # Vectorized numpy fallback
        diff = np.abs(sub_a.astype(np.int16) - sub_b.astype(np.int16))
        return float(np.mean(diff))

    @classmethod
    def wait_for_settle(
        cls,
        grabber_func: Callable[[], Optional[np.ndarray]],
        roi: Optional[Tuple[int, int, int, int]] = None,
        timeout_sec: float = 0.50,
        min_dwell_sec: float = 0.015,
        motion_threshold: float = 0.75,
        check_interval: float = 0.015
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Polls screen/ROI until rendering motion stabilizes below motion_threshold.
        
        Args:
            grabber_func: Callable returning latest screen frame as np.ndarray
            roi: Optional (x1, y1, x2, y2) bounding box
            timeout_sec: Maximum time to wait for stabilization
            min_dwell_sec: Initial dwell before first frame comparison
            motion_threshold: Mean pixel delta threshold for stability (default: 0.75)
            check_interval: Interval between consecutive frame grabs

        Returns:
            (is_settled, elapsed_time_sec, final_frame)
        """
        start_t = time.time()
        if min_dwell_sec > 0:
            time.sleep(min_dwell_sec)

        prev_frame = grabber_func()
        if prev_frame is None:
            return False, time.time() - start_t, None

        consecutive_stable = 0
        latest_frame = prev_frame

        def _crop_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(x1 + 1, min(w, int(x2)))
            y2 = max(y1 + 1, min(h, int(y2)))
            return frame[y1:y2, x1:x2]

        while (time.time() - start_t) < timeout_sec:
            if check_interval > 0:
                time.sleep(check_interval)
            curr_frame = grabber_func()
            if curr_frame is None:
                continue

            latest_frame = curr_frame

            if roi and len(roi) == 4:
                sub_prev = _crop_roi(prev_frame, roi)
                sub_curr = _crop_roi(curr_frame, roi)
                delta = cls.compute_motion_delta(sub_prev, sub_curr, subsample=2)
            else:
                delta = cls.compute_motion_delta(prev_frame, curr_frame, subsample=8)

            prev_frame = curr_frame

            if delta <= motion_threshold:
                consecutive_stable += 1
                # Fast-exit if completely static (delta <= 0.25) or consecutively stable
                if delta <= 0.25 or consecutive_stable >= 2:
                    return True, time.time() - start_t, latest_frame
            else:
                consecutive_stable = 0

        # Timed out waiting for settle: return latest frame anyway
        return False, time.time() - start_t, latest_frame
