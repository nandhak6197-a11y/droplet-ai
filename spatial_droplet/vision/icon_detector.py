"""
Multi-Scale Visual Icon & Clickable Control Detector.
Fuses true-grayscale CLAHE contrast normalization, Canny edge detection,
and morphological gradient analysis to discover graphical UI controls.
"""
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional
from .clahe import AdaptiveContrastNormalizer

class EnhancedIconDetector:
    """
    Multi-Scale OpenCV-based Visual Icon and Clickable Control Detector (Phase 7 & 8).
    Fuses true-grayscale Canny edge detection with CLAHE adaptive contrast normalization
    to discover small, medium, and wide graphical controls (buttons, tabs, search icons,
    settings gears, hamburgers, navigation arrows, player controls, taskbar widgets)
    even in dark mode and low-contrast flat UI themes.
    """
    @staticmethod
    def detect_icons(frame_bgr: np.ndarray, search_rect: Optional[Tuple[int, int, int, int]] = None,
                     ocr_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
                     existing_points: Optional[List[Tuple[int, int, str, str, str]]] = None) -> List[Tuple[int, int, str, str, str]]:
        h, w = frame_bgr.shape[:2]
        if search_rect:
            rx1, ry1, rx2, ry2 = search_rect
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        roi = frame_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return []

        # 1. True Grayscale & CLAHE Adaptive Contrast Normalization
        gray_raw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = AdaptiveContrastNormalizer.enhance_grayscale(gray_raw)

        # 2. High-Performance Clean Canny Edge Detection with multi-contrast response
        edges = cv2.Canny(gray, 24, 85)

        # 3. Morphological closing to bridge icon strokes and button borders
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close)

        # Contour extraction: RETR_CCOMP captures both outer frames and inner glyphs via hierarchy
        # Single-pass extraction prevents duplicate detection of the same features
        all_contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Extract native edge contours to capture thin slider tracks and scroll rails
        # that get bridged into surrounding panels by morphological closing
        raw_contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        raw_icons = []
        tiny_components = []
        large_containers = []
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1

        for cnt in all_contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Suppress window borders and outer frame boundaries
            if bx <= 2 or by <= 2 or (bx + bw) >= roi_w - 2 or (by + bh) >= roi_h - 2:
                continue
            if bw >= 180 and bh >= 120 and bw <= roi_w * 0.85 and bh <= roi_h * 0.90:
                large_containers.append((rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh))

            # Record small contour fragments for a cheap native-resolution
            # recovery pass. Thin glyph strokes (star/checkbox/chevron) often
            # arrive as adjacent 4-9px contours rather than one large contour.
            if 4 <= bw <= 9 and 4 <= bh <= 9:
                tiny_patch = edges[by:by+bh, bx:bx+bw]
                tiny_density = np.count_nonzero(tiny_patch) / float(bw * bh)
                if 0.12 <= tiny_density <= 0.95:
                    tiny_components.append((rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh))

            aspect = bw / float(bh) if bh > 0 else 0

            # 1. Panel Divider / Splitter Handles (genuine draggable handles >= 4px wide)
            if (4 <= bw <= 18 and 36 <= bh <= 160) and (0.05 <= aspect <= 0.35) and (rx1 + bx > roi_w * 0.08 and rx1 + bx < roi_w * 0.92):
                patch = edges[by:by+bh, bx:bx+bw]
                density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.15 <= density <= 0.90:
                    cx = rx1 + bx + bw // 2
                    cy = ry1 + by + bh // 2
                    raw_icons.append((cx, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "handle"))

            # 2. Horizontal Sliders & Zoom Trackbars (genuine slider tracks with handles)
            elif (60 <= bw <= 1400 and 6 <= bh <= 20) and (aspect >= 3.5):
                patch = edges[by:by+bh, bx:bx+bw]
                density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.05 <= density <= 0.85:
                    cy = ry1 + by + bh // 2
                    left_x = rx1 + bx + max(6, int(bw * 0.025))
                    q1_x = rx1 + bx + max(6, int(bw * 0.25))
                    mid_x = rx1 + bx + bw // 2
                    q3_x = rx1 + bx + bw - max(6, int(bw * 0.25))
                    right_x = rx1 + bx + bw - max(6, int(bw * 0.025))
                    raw_icons.append((left_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_left"))
                    raw_icons.append((q1_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_quarter"))
                    raw_icons.append((mid_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_mid"))
                    raw_icons.append((q3_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_three_quarter"))
                    raw_icons.append((right_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_right"))

            # 3. Vertical Scrollbars & Slider Tracks (genuine scrollbars >= 8px wide)
            elif (4 <= bw <= 32 and 60 <= bh <= 800) and (aspect < 0.25):
                patch = edges[by:by+bh, bx:bx+bw]
                density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.12 <= density <= 0.85:
                    cx = rx1 + bx + bw // 2
                    raw_icons.append((cx, int(ry1 + by + bh * 0.20), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_top"))
                    raw_icons.append((cx, int(ry1 + by + bh * 0.50), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_mid"))
                    raw_icons.append((cx, int(ry1 + by + bh * 0.85), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_bottom"))

            # 4. Standard UI icons, buttons, search glyphs, stars, filter funnels, and small chevrons
            # Captures actionable 6px - 140px icons, modern Fluent glyphs, and compact buttons
            elif (6 <= bw <= 140 and 4 <= bh <= 90 and 0.20 <= aspect <= 4.5):
                # Visual Saliency Filter: Discard flat textureless background noise
                raw_patch = gray_raw[by:by+bh, bx:bx+bw]
                if float(np.std(raw_patch)) < 6.0:
                    continue
                patch = edges[by:by+bh, bx:bx+bw]
                edge_count = np.count_nonzero(patch)
                density = edge_count / float(bw * bh) if (bw * bh) > 0 else 0
                min_edge_thresh = 6 if (bw <= 16 and bh <= 16) else (10 if (bw <= 24 and bh <= 24) else 16)
                if edge_count >= min_edge_thresh and 0.05 <= density <= 0.94:
                    cx = rx1 + bx + bw // 2
                    cy = ry1 + by + bh // 2
                    ibox = (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh)
                    # Visual mass center and snug bounding box refinement
                    nz_positions = np.where(patch > 0)
                    if len(nz_positions[0]) >= 6:
                        M = cv2.moments(patch)
                        if M["m00"] > 0:
                            cx = rx1 + bx + int(round(M["m10"] / M["m00"]))
                            cy = ry1 + by + int(round(M["m01"] / M["m00"]))
                        else:
                            cx = rx1 + bx + int(round(np.mean(nz_positions[1])))
                            cy = ry1 + by + int(round(np.mean(nz_positions[0])))
                        min_py, max_py = int(np.min(nz_positions[0])), int(np.max(nz_positions[0]))
                        min_px, max_px = int(np.min(nz_positions[1])), int(np.max(nz_positions[1]))
                        if (max_px - min_px + 1 >= 5) and (max_py - min_py + 1 >= 4):
                            ibox = (rx1 + bx + min_px, ry1 + by + min_py, rx1 + bx + max_px + 1, ry1 + by + max_py + 1)
                    raw_icons.append((cx, cy, bw, bh, patch, ibox, "icon"))

        # Recover thin linear slider and scrollbar tracks from raw edge contours
        # that were bridged into larger panels by morphological closing
        for cnt in raw_contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bx <= 2 or by <= 2 or (bx + bw) >= roi_w - 2 or (by + bh) >= roi_h - 2:
                continue
            aspect = bw / float(bh) if bh > 0 else 0
            if (60 <= bw <= 1400 and 6 <= bh <= 20) and (aspect >= 3.5):
                patch = edges[by:by+bh, bx:bx+bw]
                density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.05 <= density <= 0.85:
                    if any(abs(rx1 + bx - r[5][0]) <= 6 and abs(ry1 + by - r[5][1]) <= 6 for r in raw_icons if r[6] == "slider_left"):
                        continue
                    cy = ry1 + by + bh // 2
                    left_x = rx1 + bx + max(6, int(bw * 0.025))
                    q1_x = rx1 + bx + max(6, int(bw * 0.25))
                    mid_x = rx1 + bx + bw // 2
                    q3_x = rx1 + bx + bw - max(6, int(bw * 0.25))
                    right_x = rx1 + bx + bw - max(6, int(bw * 0.025))
                    raw_icons.append((left_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_left"))
                    raw_icons.append((q1_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_quarter"))
                    raw_icons.append((mid_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_mid"))
                    raw_icons.append((q3_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_three_quarter"))
                    raw_icons.append((right_x, cy, bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_right"))
            elif (4 <= bw <= 32 and 60 <= bh <= 800) and (aspect < 0.25):
                patch = edges[by:by+bh, bx:bx+bw]
                density = np.count_nonzero(patch) / float(bw * bh) if (bw * bh) > 0 else 0
                if 0.12 <= density <= 0.85:
                    if any(abs(rx1 + bx - r[5][0]) <= 6 and abs(ry1 + by - r[5][1]) <= 6 for r in raw_icons if r[6] == "slider_top"):
                        continue
                    cx = rx1 + bx + bw // 2
                    raw_icons.append((cx, int(ry1 + by + bh * 0.20), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_top"))
                    raw_icons.append((cx, int(ry1 + by + bh * 0.50), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_mid"))
                    raw_icons.append((cx, int(ry1 + by + bh * 0.85), bw, bh, patch, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), "slider_bottom"))

        # Fuse touching small contours into compact glyph controls without
        # resizing the entire screenshot. This retains small-target coverage
        # at approximately the cost of the original single-resolution scan.
        tiny_components.sort(key=lambda b: (b[0], b[1]))
        tiny_clusters = []
        for box in tiny_components:
            x1, y1, x2, y2 = box
            matches = []
            for idx, cluster in enumerate(tiny_clusters):
                cx1, cy1, cx2, cy2 = cluster
                gap_x = max(0, max(cx1, x1) - min(cx2, x2))
                gap_y = max(0, max(cy1, y1) - min(cy2, y2))
                if gap_x <= 3 and gap_y <= 3 and max(cx2, x2) - min(cx1, x1) <= 40 and max(cy2, y2) - min(cy1, y1) <= 40:
                    matches.append(idx)
            if not matches:
                tiny_clusters.append(box)
                continue
            base = matches[0]
            cluster = tiny_clusters[base]
            tiny_clusters[base] = (min(cluster[0], x1), min(cluster[1], y1), max(cluster[2], x2), max(cluster[3], y2))
            for idx in reversed(matches[1:]):
                other = tiny_clusters.pop(idx)
                cluster = tiny_clusters[base]
                tiny_clusters[base] = (min(cluster[0], other[0]), min(cluster[1], other[1]), max(cluster[2], other[2]), max(cluster[3], other[3]))

        # Spatial indexing for fast bounding box overlap checks (replaces O(N*M) scans)
        raw_grid = {}
        grid_bin = 48
        def _add_to_grid(idx, box):
            gx1, gy1 = max(0, box[0] // grid_bin), max(0, box[1] // grid_bin)
            gx2, gy2 = max(0, box[2] // grid_bin), max(0, box[3] // grid_bin)
            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    raw_grid.setdefault((gx, gy), []).append(idx)

        for idx, r in enumerate(raw_icons):
            _add_to_grid(idx, r[5])

        def _box_overlaps_any(box, thresh):
            gx1, gy1 = max(0, box[0] // grid_bin), max(0, box[1] // grid_bin)
            gx2, gy2 = max(0, box[2] // grid_bin), max(0, box[3] // grid_bin)
            seen_indices = set()
            b_area = float(max(1, (box[2] - box[0]) * (box[3] - box[1])))
            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    for idx in raw_grid.get((gx, gy), []):
                        if idx in seen_indices:
                            continue
                        seen_indices.add(idx)
                        eb = raw_icons[idx][5]
                        inter_w = max(0, min(box[2], eb[2]) - max(box[0], eb[0]))
                        inter_h = max(0, min(box[3], eb[3]) - max(box[1], eb[1]))
                        if (inter_w * inter_h) / b_area >= thresh:
                            return True
            return False

        for x1, y1, x2, y2 in tiny_clusters:
            bw, bh = x2 - x1, y2 - y1
            if not (10 <= bw <= 36 and 10 <= bh <= 36 and 0.35 <= bw / float(bh) <= 2.8):
                continue
            local_x1, local_y1 = x1 - rx1, y1 - ry1
            patch = edges[local_y1:local_y1+bh, local_x1:local_x1+bw]
            density = np.count_nonzero(patch) / float(max(1, bw * bh))
            if not (0.08 <= density <= 0.95):
                continue
            box = (x1, y1, x2, y2)
            if _box_overlaps_any(box, 0.70):
                continue
            M = cv2.moments(patch)
            cx = x1 + int(round(M["m10"] / M["m00"])) if M["m00"] > 0 else (x1 + x2) // 2
            cy = y1 + int(round(M["m01"] / M["m00"])) if M["m00"] > 0 else (y1 + y2) // 2
            raw_icons.append((cx, cy, bw, bh, patch, box, "icon"))
            _add_to_grid(len(raw_icons) - 1, box)

        # Intermediate micro-glyph recovery: contours in 10-20px range with
        # strong internal structure that were not admitted by the primary filters.
        # Recovers star badges, small checkboxes, cancel/clear icons.
        for cnt in all_contours:
            bx, by, bw_c, bh_c = cv2.boundingRect(cnt)
            if bx <= 2 or by <= 2 or (bx + bw_c) >= roi_w - 2 or (by + bh_c) >= roi_h - 2:
                continue
            if not (10 <= bw_c <= 20 and 10 <= bh_c <= 20):
                continue
            asp = bw_c / float(bh_c) if bh_c > 0 else 0
            if not (0.5 <= asp <= 2.0):
                continue
            box = (rx1 + bx, ry1 + by, rx1 + bx + bw_c, ry1 + by + bh_c)
            # Skip if already covered by an existing raw_icon
            if _box_overlaps_any(box, 0.50):
                continue
            patch = edges[by:by+bh_c, bx:bx+bw_c]
            density = np.count_nonzero(patch) / float(max(1, bw_c * bh_c))
            if not (0.10 <= density <= 0.90):
                continue
            # Require meaningful internal structure: at least 2 distinct
            # horizontal spans (rules out single-line border fragments)
            row_count = sum(1 for r in range(bh_c) if np.any(patch[r, :] > 0))
            if row_count < bh_c * 0.4:
                continue
            M = cv2.moments(patch)
            cx = rx1 + bx + int(round(M["m10"] / M["m00"])) if M["m00"] > 0 else (box[0] + box[2]) // 2
            cy = ry1 + by + int(round(M["m01"] / M["m00"])) if M["m00"] > 0 else (box[1] + box[3]) // 2
            raw_icons.append((cx, cy, bw_c, bh_c, patch, box, "icon"))
            _add_to_grid(len(raw_icons) - 1, box)

        # Multi-stroke contour fusion: merge adjacent sub-strokes of the same control
        fused_raw = []
        raw_icons.sort(key=lambda r: r[5][0])
        merged_mask = [False] * len(raw_icons)
        for i in range(len(raw_icons)):
            if merged_mask[i]:
                continue
            cx1, cy1, bw1, bh1, p1, box1, kind1 = raw_icons[i]
            ux1, uy1, ux2, uy2 = box1
            for j in range(i + 1, len(raw_icons)):
                b2 = raw_icons[j][5]
                # Early break: sorted by left x coordinate
                if b2[0] - ux2 > 6:
                    break
                if merged_mask[j]:
                    continue
                cx2, cy2, bw2, bh2, p2, box2, kind2 = raw_icons[j]
                gap_x = max(0, max(ux1, box2[0]) - min(ux2, box2[2]))
                gap_y = max(0, max(uy1, box2[1]) - min(uy2, box2[3]))
                if kind1 == kind2:
                    if kind1 == "handle" and abs(cx1 - cx2) <= 4 and gap_y <= 15:
                        ux1 = min(ux1, box2[0])
                        uy1 = min(uy1, box2[1])
                        ux2 = max(ux2, box2[2])
                        uy2 = max(uy2, box2[3])
                        merged_mask[j] = True
                    elif kind1 == "icon" and gap_x <= 2 and gap_y <= 2:
                        new_w = max(ux2, box2[2]) - min(ux1, box2[0])
                        new_h = max(uy2, box2[3]) - min(uy1, box2[1])
                        if new_w > 140 or new_h > 90:
                            continue
                        cur_w = ux2 - ux1
                        cur_h = uy2 - uy1
                        # Prevent merging vertically stacked or horizontally adjacent standalone buttons
                        if (cur_w >= 12 and cur_h >= 12) and (bw2 >= 12 and bh2 >= 12):
                            continue
                        ux1 = min(ux1, box2[0])
                        uy1 = min(uy1, box2[1])
                        ux2 = max(ux2, box2[2])
                        uy2 = max(uy2, box2[3])
                        bw1 = ux2 - ux1
                        bh1 = uy2 - uy1
                        merged_mask[j] = True
            merged_patch = edges[uy1:uy2, ux1:ux2]
            if kind1 == "slider_left":
                merged_cx = int(ux1 + max(6, int((ux2 - ux1) * 0.025)))
                merged_cy = (uy1 + uy2) // 2
            elif kind1 == "slider_quarter":
                merged_cx = int(ux1 + max(6, int((ux2 - ux1) * 0.25)))
                merged_cy = (uy1 + uy2) // 2
            elif kind1 == "slider_right":
                merged_cx = int(ux2 - max(6, int((ux2 - ux1) * 0.025)))
                merged_cy = (uy1 + uy2) // 2
            elif kind1 == "slider_three_quarter":
                merged_cx = int(ux2 - max(6, int((ux2 - ux1) * 0.25)))
                merged_cy = (uy1 + uy2) // 2
            elif kind1 == "slider_mid" and (ux2 - ux1) > (uy2 - uy1):
                merged_cx = (ux1 + ux2) // 2
                merged_cy = (uy1 + uy2) // 2
            elif kind1.startswith("slider_"):
                merged_cx = (ux1 + ux2) // 2
                merged_cy = cy1
            else:
                # Sub-pixel visual center of mass via spatial moments
                M = cv2.moments(merged_patch)
                if M["m00"] > 0:
                    merged_cx = ux1 + int(round(M["m10"] / M["m00"]))
                    merged_cy = uy1 + int(round(M["m01"] / M["m00"]))
                else:
                    merged_cx = (ux1 + ux2) // 2
                    merged_cy = (uy1 + uy2) // 2
            fused_raw.append((merged_cx, merged_cy, ux2 - ux1, uy2 - uy1, merged_patch, (ux1, uy1, ux2, uy2), kind1))

        # Filter out repetitive vertical column lines (e.g. list item borders, table lines)
        handle_items = [item for item in fused_raw if item[6] == "handle"]
        handle_buckets = {}
        for hi in handle_items:
            bk = hi[0] // 8
            handle_buckets[bk] = handle_buckets.get(bk, 0) + 1
        suppressed_handle_cx = {bk for bk, cnt in handle_buckets.items() if cnt >= 3}

        valid_fused = [
            item for item in fused_raw
            if not (item[6] == "handle" and (item[0] // 8 in suppressed_handle_cx or (item[0] - 4) // 8 in suppressed_handle_cx or (item[0] + 4) // 8 in suppressed_handle_cx))
        ]



        # Compact OCR word bounding boxes for fast spatial indexing
        compact_ocr = [
            (ob[0] - 2, ob[1] - 2, ob[2] + 2, ob[3] + 2)
            for ob in ocr_boxes
            if (ob[2] - ob[0] <= 55 and ob[3] - ob[1] <= 30)
        ] if ocr_boxes else []

        detected = []
        for cx, cy, bw, bh, patch, ibox, kind in valid_fused:
            # Avoid duplicate coverage when a visual contour is effectively the
            # OCR glyph itself.  A stricter 85% overlap gate preserves small
            # actionable icons that are adjacent to (or partly inside) labels.
            if compact_ocr:
                iou_suppressed = False
                for cob in compact_ocr:
                    ix1 = max(ibox[0], cob[0]); iy1 = max(ibox[1], cob[1])
                    ix2 = min(ibox[2], cob[2]); iy2 = min(ibox[3], cob[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    icon_area = max(1, (ibox[2] - ibox[0]) * (ibox[3] - ibox[1]))
                    if inter / icon_area >= 0.85:
                        iou_suppressed = True
                        break
                if iou_suppressed:
                    continue
            # Run the same semantic shape classifier on recovered tiny glyphs;
            # replacing these labels with a generic class harms downstream
            # ranking for filters, stars, checkboxes, arrows and tools.
            label = EnhancedIconDetector._classify(patch, cx, cy, w, h, bw, bh, kind)

            # Sample RGB color if detected as Color Swatch / Color Cell
            if ("Color Swatch" in label or "Color Cell" in label) and 0 <= cy < h and 0 <= cx < w:
                b, g, r = frame_bgr[cy, cx]
                hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])
                color_name = ""
                if v_val < 38: color_name = "Black"
                elif s_val < 30 and v_val > 195: color_name = "White"
                elif s_val < 35: color_name = "Gray"
                elif h_val <= 10 or h_val >= 170: color_name = "Red"
                elif 11 <= h_val <= 24: color_name = "Orange"
                elif 25 <= h_val <= 38: color_name = "Yellow"
                elif 39 <= h_val <= 82: color_name = "Green"
                elif 83 <= h_val <= 98: color_name = "Cyan"
                elif 99 <= h_val <= 135: color_name = "Blue"
                elif 136 <= h_val <= 165: color_name = "Purple"
                if color_name:
                    label = f"Color Swatch / Color Cell ({color_name})"

            # Dual Spinbox Stepper Splitting (Phase 8 Generalization)
            # If the icon matches vertical stepper geometry (10-32px wide, 22-60px tall with top/bot symmetry)
            is_stepper_shape = (10 <= bw <= 32 and 22 <= bh <= 60 and 0.35 <= (bw / float(bh)) <= 0.85)
            if (is_stepper_shape or label in ("Icon", "Graphic Element", "Small Graphic", "Button", "Toolbar Tool / Visual Button") or "Spinbox" in label or "Arrow" in label) and kind == "icon":
                if 10 <= bw <= 32 and 22 <= bh <= 60:
                    top_half_px = int(np.sum(patch[:bh//2, :] > 0))
                    bot_half_px = int(np.sum(patch[bh//2:, :] > 0))
                    if top_half_px >= 4 and bot_half_px >= 4:
                        up_cy = cy - bh // 4
                        dn_cy = cy + bh // 4
                        up_box = (ibox[0], ibox[1], ibox[2], cy)
                        dn_box = (ibox[0], cy, ibox[2], ibox[3])
                        detected.append((cx, up_cy, "[Icon] Up Arrow / Increment / Spinbox / Up", "interactive_widget", "Up Arrow / Increment / Spinbox / Up", up_box))
                        detected.append((cx, dn_cy, "[Icon] Down Arrow / Decrement / Spinbox / Down", "interactive_widget", "Down Arrow / Decrement / Spinbox / Down", dn_box))
                        continue

            detected.append((cx, cy, f"[Icon] {label}", "interactive_widget", label, ibox))

        # Recover narrow scrollbar/rail landmarks along the right edge of a
        # large detected panel. The thumb can blend into a colored panel edge,
        # so contour-only detection often sees the panel as one large shape.
        if large_containers:
            vertical = []
            for cx1, cy1, cx2, cy2 in large_containers:
                # Fast strip scan along right panel boundary (<0.1ms vs 224ms full-screen HoughLinesP)
                sx1 = max(0, cx2 - 8 - rx1)
                sx2 = min(roi_w, cx2 + 8 - rx1)
                sy1 = max(0, cy1 - ry1)
                sy2 = min(roi_h, cy2 - ry1)
                if sx2 > sx1 and sy2 - sy1 >= 60:
                    strip = edges[sy1:sy2, sx1:sx2]
                    col_sums = np.sum(strip > 0, axis=0)
                    for c_idx, c_val in enumerate(col_sums):
                        if c_val >= 50:
                            vertical.append((rx1 + sx1 + c_idx, ry1 + sy1, ry1 + sy2))
            rail_markers = []
            for cx1, cy1, cx2, cy2 in large_containers:
                edge_x = cx2
                nearby = [line for line in vertical if abs(line[0] - edge_x) <= 8 and min(line[2], cy2) - max(line[1], cy1) >= 60]
                distinct_x = {line[0] for line in nearby}
                if len(distinct_x) < 2:
                    continue
                rail_y1 = max(cy1, min(line[1] for line in nearby))
                rail_y2 = min(cy2, max(line[2] for line in nearby))
                if rail_y2 - rail_y1 < 80:
                    continue
                for fraction, name in ((0.20, "Upper"), (0.50, "Middle"), (0.85, "Lower")):
                    marker_y = int(rail_y1 + (rail_y2 - rail_y1) * fraction)
                    rail_markers.append((edge_x - 2, marker_y, name, (edge_x - 7, marker_y - 10, edge_x + 3, marker_y + 10)))
            for cx, cy, name, box in rail_markers[:18]:
                if any(math.hypot(cx - item[0], cy - item[1]) <= 8 for item in detected):
                    continue
                detected.append((cx, cy, f"[Icon] Panel Scroll Rail / {name} Segment", "interactive_widget", f"Panel Scroll Rail / {name} Segment", box))

        return detected

    @staticmethod
    def _classify(patch: np.ndarray, cx: int, cy: int, sw: int, sh: int, bw: int = 24, bh: int = 24, kind: str = "icon") -> str:
        if kind == "handle":
            return "Panel Divider / Splitter Handle"
        if kind == "slider_left":
            return "Horizontal Slider / Trackbar Left / Start / Beginning"
        if kind == "slider_quarter":
            return "Horizontal Slider / Trackbar 25% / Left Quarter"
        if kind == "slider_three_quarter":
            return "Horizontal Slider / Trackbar 75% / Right Three Quarter"
        if kind == "slider_right":
            return "Horizontal Slider / Trackbar Right / End"
        if kind == "slider" or (kind == "slider_mid" and bw > bh):
            return "Horizontal Slider / Trackbar Middle / Slider Handle"
        if kind == "slider_top":
            return "Scrollbar / Slider Top"
        if kind == "slider_mid":
            return "Scrollbar / Slider Middle"
        if kind == "slider_bottom":
            return "Scrollbar / Slider Bottom"

        # Titlebar window caption buttons (relative to window top-right corner)
        if cy < 45 and cx > sw - 140:
            if cx > sw - 50: return "Close"
            elif cx > sw - 95: return "Maximize"
            else: return "Minimize"
        # Top-left browser/window navigation (first 10% of width, 3-8% of height)
        if sh * 0.03 < cy < sh * 0.08 and cx < sw * 0.10:
            if cx < sw * 0.02: return "Back"
            elif cx < sw * 0.04: return "Forward"
            else: return "Refresh"

        # Guard against empty patches
        if patch is None or patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
            return "Icon"

        # Check if patch has an enclosing container/button frame
        top_edge = np.sum(patch[:2, :] > 0) / float(bw) if bw > 0 else 0
        bot_edge = np.sum(patch[-2:, :] > 0) / float(bw) if bw > 0 else 0
        left_edge = np.sum(patch[:, :2] > 0) / float(bh) if bh > 0 else 0
        right_edge = np.sum(patch[:, -2:] > 0) / float(bh) if bh > 0 else 0
        has_frame = (bw >= 12 and bh >= 12 and top_edge >= 0.4 and bot_edge >= 0.4 and left_edge >= 0.4 and right_edge >= 0.4)

        inner_patch = patch
        if has_frame:
            my = max(1, int(round(bh * 0.15)))
            mx = max(1, int(round(bw * 0.15)))
            if bh - 2 * my >= 6 and bw - 2 * mx >= 6:
                candidate = patch[my:bh-my, mx:bw-mx]
                if np.sum(candidate > 0) >= 4:
                    inner_patch = candidate

        p24 = cv2.resize(inner_patch, (24, 24), interpolation=cv2.INTER_NEAREST)
        p24_bin = (p24 > 0).astype(np.uint8)
        integ = cv2.integral(p24_bin)

        def bsum(r1, c1, r2, c2):
            return int(integ[r2, c2] - integ[r1, c2] - integ[r2, c1] + integ[r1, c1])

        total_px = int(integ[24, 24])
        aspect = bw / float(bh) if bh > 0 else 1.0
        circ_ratio = abs(bw - bh) / float(max(bw, bh)) if max(bw, bh) > 0 else 0

        arm_tl = bsum(0, 0, 8, 8)
        arm_tr = bsum(0, 16, 8, 24)
        arm_bl = bsum(16, 0, 24, 8)
        arm_br = bsum(16, 16, 24, 24)
        center_core = bsum(9, 9, 15, 15)
        cardinal = bsum(0, 9, 3, 15) + bsum(21, 9, 24, 15) + bsum(9, 0, 15, 3) + bsum(9, 21, 15, 24)

        # 0a. Question Mark / Help / Info ("?")
        if 8 <= bw <= 36 and 10 <= bh <= 38 and (0.45 <= aspect <= 1.15):
            q_top = bsum(0, 0, 8, 24)
            q_mid = bsum(8, 0, 16, 24)
            q_dot = bsum(19, 0, 24, 24)
            q_gap = bsum(16, 0, 19, 24)
            if q_top >= 8 and q_mid >= 4 and q_dot >= 2 and q_gap <= 2:
                return "Question / Help / Info / Prompt"

        # 0b. Underline / Text Formatting ("U" with bottom underline)
        if 8 <= bw <= 36 and 10 <= bh <= 36 and (0.65 <= aspect <= 1.45):
            bot_bar = bsum(19, 0, 24, 24)
            top_struct = bsum(0, 0, 16, 24)
            u_gap = bsum(16, 0, 19, 24)
            if bot_bar >= 8 and top_struct >= 10 and u_gap <= 4:
                return "Underline / Text Format / U"

        # 1. Close / Cross (X) glyph (e.g. close buttons, notification dismiss)
        diag1 = sum(any(p24[i, max(0, min(23, i+k))] > 0 for k in (-1, 0, 1)) for i in range(3, 21))
        diag2 = sum(any(p24[i, max(0, min(23, 23-i+k))] > 0 for k in (-1, 0, 1)) for i in range(3, 21))
        if diag1 >= 5 and diag2 >= 5 and cardinal <= 8 and (arm_tl >= 1 and arm_tr >= 1 and arm_bl >= 1 and arm_br >= 1) and (0.65 <= aspect <= 1.55):
            return "Close / Cross / Dismiss / Delete"

        # 2. Universal 5-Point Star / Favorite / Bookmark (Dual-scale: standard and micro-glyph)
        top_c = bsum(1, 8, 7, 16)
        mid_l = bsum(7, 0, 13, 8)
        mid_r = bsum(7, 16, 13, 24)
        bot_l = bsum(16, 0, 23, 9)
        bot_r = bsum(16, 15, 23, 24)
        val_tl = bsum(0, 0, 6, 7)
        val_tr = bsum(0, 17, 6, 24)
        bot_groove = bsum(21, 10, 24, 14)
        is_star = (
            top_c >= 5 and mid_l >= 5 and mid_r >= 5 and bot_l >= 5 and bot_r >= 5
            and min(val_tl, val_tr) <= 6 and max(val_tl, val_tr) <= 14
            and center_core >= 6
            and bot_groove <= 10
            and total_px <= 200
            and (0.7 <= aspect <= 1.4)
        )
        if is_star:
            return "Star / Favorite / Bookmark"
        # 3. Fullscreen / Maximize (opposite diagonal arrows or 4 corner brackets with empty center)
        quad_tl = int(np.sum(p24[:12, :12] > 0))
        quad_tr = int(np.sum(p24[:12, 12:] > 0))
        quad_bl = int(np.sum(p24[12:, :12] > 0))
        quad_br = int(np.sum(p24[12:, 12:] > 0))
        is_diag_tr_bl = (quad_tr >= 5 and quad_bl >= 5 and quad_tl <= 2 and quad_br <= 2)
        is_diag_tl_br = (quad_tl >= 5 and quad_br >= 5 and quad_tr <= 2 and quad_bl <= 2)
        is_4_brackets = (
            not has_frame and total_px <= 45
            and arm_tl >= 3 and arm_tr >= 3 and arm_bl >= 3 and arm_br >= 3
            and abs(arm_tl - arm_tr) <= 3 and abs(arm_bl - arm_br) <= 3
            and cardinal <= 6 and center_core <= 4
        )
        is_square_window = (bw >= 14 and bh >= 14 and 0.75 <= aspect <= 1.35 and cardinal >= 10 and center_core <= 8)
        is_titlebar_maximize = (cy < 45 and cx > sw * 0.85 and is_square_window)
        if (
            14 <= bw <= 48 and 14 <= bh <= 48
            and (0.75 <= aspect <= 1.30)
            and cy < sh - 55
            and total_px >= 14
            and (is_diag_tr_bl or is_diag_tl_br or is_4_brackets or is_titlebar_maximize)
        ):
            if cx < sw * 0.08 and sh * 0.10 < cy < sh * 0.90:
                return "Expand / Tool Options / Flyout"
            return "Fullscreen / Maximize / Video Fullscreen"

        # 4. Source Control / Git Branch / Version Control (vertical line with branch node)
        git_top = int(np.sum(p24[1:7, 3:10] > 0))
        git_bot = int(np.sum(p24[17:23, 1:9] > 0))
        git_right = int(np.sum(p24[7:15, 13:20] > 0))
        git_shaft = int(np.sum(p24[8:16, 4:8] > 0))
        git_tr_void = int(np.sum(p24[:6, 14:] > 0))
        git_br_void = int(np.sum(p24[18:, 14:] > 0))
        if git_top >= 12 and git_bot >= 12 and git_right >= 14 and git_shaft >= 12 and git_tr_void <= 4 and git_br_void <= 4:
            return "Source Control / Version Control / Git / Branch"

        # 5a. Funnel / Filter / Search Filter (wide top rim, central vertical tube/stem, empty bottom flanks)
        top_rim = int(np.sum(p24[:6, :] > 0))
        bot_stem = int(np.sum(p24[14:, 8:16] > 0))
        bot_flanks = int(np.sum(p24[14:, :8] > 0) + np.sum(p24[14:, 16:] > 0))
        if 12 <= bw <= 38 and 12 <= bh <= 38 and (0.70 <= aspect <= 1.40) and top_rim >= 15 and bot_stem >= 8 and top_rim > 2.0 * bot_flanks:
            return "Filter / Funnel / Search Filter"

        # 5b. Paintbrush / Drawing Tool (diagonal shaft with distinct bulbous tip)
        if 16 <= bw <= 42 and 16 <= bh <= 42:
            diag_anti = sum(p24[i, 23-i] > 0 for i in range(2, 22))
            tip = int(np.sum(p24[13:, :11] > 0))
            if diag_anti >= 10 and tip >= 8:
                return "Paintbrush Tool / Pencil / Brush"

        # 5c. Color Swatch / Palette Box / Color Cell (enclosed 4-sided square with empty/flat interior)
        te = int(np.sum(p24[:4, :] > 0))
        be = int(np.sum(p24[20:, :] > 0))
        le = int(np.sum(p24[:, :4] > 0))
        re = int(np.sum(p24[:, 20:] > 0))
        mid_line_swatch = np.sum(p24[9:15, 3:21] > 0)
        if 10 <= bw <= 45 and 10 <= bh <= 45 and 0.60 <= aspect <= 1.5:
            if te >= 5 and be >= 5 and le >= 5 and re >= 5 and (te + be + le + re) >= 26 and center_core <= 4 and mid_line_swatch <= 8:
                return "Color Swatch / Palette Box / Color Cell / Square Option"

        # 5d. Bullet / Numbered List (Toolbar & Sidebar formatting control: dots on left, text bars on right)
        if 8 <= bw <= 36 and 8 <= bh <= 36 and (0.65 <= aspect <= 1.45):
            col_sums_init = np.sum(p24 > 0, axis=0)
            bullet_left = int(np.sum(col_sums_init[:6]))
            bullet_right = int(np.sum(col_sums_init[8:]))
            bullet_gap_min = int(np.min(col_sums_init[4:9]))
            corner_px_b = int(np.sum(p24[:4, :4] > 0) + np.sum(p24[:4, 20:] > 0) + np.sum(p24[20:, :4] > 0) + np.sum(p24[20:, 20:] > 0))
            if bullet_left >= 8 and bullet_right >= 16 and bullet_gap_min <= 3 and corner_px_b >= 8:
                return "Bullet List / Numbered List"

        # 6. Strikethrough / Text Format icon (glyph with horizontal strike passing through S, toolbar button)
        if 10 <= bw <= 32 and 10 <= bh <= 32 and (0.75 <= aspect <= 1.35):
            mid_line = np.sum(p24[8:16, 3:21] > 0)
            left_strike = np.sum(p24[7:17, 0:4] > 0)
            right_strike = np.sum(p24[7:17, 20:24] > 0)
            top_s = np.sum(p24[:8, :] > 0)
            bot_s = np.sum(p24[16:, :] > 0)
            bl_void = int(np.sum(p24[14:18, :8] > 0))
            max_run = 0
            for r in range(8, 16):
                cur = 0
                for c in range(24):
                    if p24[r, c] > 0: cur += 1; max_run = max(max_run, cur)
                    else: cur = 0
            if mid_line >= 12 and left_strike >= 2 and right_strike >= 2 and top_s >= 8 and bot_s >= 8 and max_run >= 8 and max_run <= 22 and bl_void <= 12 and center_core >= 12:
                return "Strikethrough / Text Format / S"

        # 6e. Carets / Chevrons & Spinbox Steppers (^, v) - strict monotonic tapering only
        if bh <= 25 and bw <= 28 and not has_frame:
            row_spans = []
            for r in range(inner_patch.shape[0]):
                nz = np.where(inner_patch[r, :] > 0)[0]
                if len(nz) > 0:
                    row_spans.append((r, int(nz[-1] - nz[0] + 1)))
            if len(row_spans) >= 3:
                first_w = row_spans[0][1]
                last_w = row_spans[-1][1]
                # True caret: apex at one end, base at other, monotonic widening by >= 1.8x
                if last_w >= 1.8 * max(1, first_w) and last_w >= 6 and first_w <= 5:
                    return "Up Arrow / Increment / Spinbox / Up"
                elif first_w >= 1.8 * max(1, last_w) and first_w >= 6 and last_w <= 5:
                    return "Down Arrow / Dropdown / Decrement / Spinbox / Down"

        # 7. Universal Media Controls (Pause, Fast Forward, Rewind, Play/Debug)
        pause_l = int(np.sum(p24[4:20, 5:10] > 0))
        pause_r = int(np.sum(p24[4:20, 14:19] > 0))
        pause_gap = int(np.sum(p24[4:20, 10:14] > 0))
        if pause_l >= 10 and pause_r >= 10 and pause_gap <= 3 and bh >= 12:
            return "Media: Play / Pause / Pause video"

        col_sums24 = np.sum(p24 > 0, axis=0)
        play_left_base = int(np.sum(p24[4:20, 4:10] > 0))
        play_right_apex = int(np.sum(p24[8:16, 16:22] > 0))
        rewind_left_apex = int(np.sum(p24[8:16, 2:8] > 0))
        rewind_right_base = int(np.sum(p24[4:20, 14:20] > 0))
        left_m = int(np.sum(p24[:, :12] > 0))
        right_m = int(np.sum(p24[:, 12:] > 0))
        top_m = int(np.sum(p24[:12, :] > 0))
        bot_m = int(np.sum(p24[12:, :] > 0))

        if 14 <= bw <= 48 and 14 <= bh <= 45 and (0.68 <= aspect <= 1.45):
            is_playback_zone = (cy > sh * 0.65)
            # Fast Forward: base on left, tall stop bar on right, right-bottom corner open, solid core
            if play_left_base >= 10 and max(col_sums24[17:]) >= 8 and arm_br <= 5 and center_core >= 8:
                return "Media: Fast Forward / Next" if is_playback_zone else "Toolbar Tool / Visual Button / Action Arrow"
            # Rewind: base on right, tall stop bar on left, pointing left, left-bottom corner open, solid core
            if right_m > left_m and rewind_right_base >= 10 and max(col_sums24[:7]) >= 8 and arm_bl <= 5 and center_core >= 8:
                return "Media: Rewind / Previous" if is_playback_zone else "Toolbar Tool / Visual Button / Action Arrow"
            # Play / Run & Debug: triangle pointing right, base on left, apex on right, no right stop bar, right-bottom and top-left corners open
            if left_m >= 1.35 * right_m and play_left_base >= 14 and play_right_apex >= 4 and max(col_sums24[18:]) <= 6 and arm_br <= 4 and int(np.sum(p24[:6, :6] > 0)) <= 4:
                if cx < sw * 0.08 and sh * 0.10 < cy < sh * 0.90:
                    return "Activity Bar: Run and Debug / Debug Panel"
                return "Play / Run and Debug / Start"

        # 8. Carets / Chevrons (Strict triangular tapering with apex and wide base, no vertical side walls)
        if 8 <= bw <= 32 and 6 <= bh <= 24 and not (le >= 4 and re >= 4):
            apex_up = int(np.sum(p24[4:12, 7:17] > 0))
            apex_dn = int(np.sum(p24[12:20, 7:17] > 0))
            top_corners = int(np.sum(p24[:8, :6] > 0) + np.sum(p24[:8, 18:] > 0))
            bot_corners = int(np.sum(p24[16:, :6] > 0) + np.sum(p24[16:, 18:] > 0))
            l_arms = int(np.sum(p24[6:18, 3:10] > 0))
            r_arms = int(np.sum(p24[6:18, 14:21] > 0))

            if (bot_m > 2.0 * top_m and bot_m >= 6 and (arm_bl >= 2 and arm_br >= 2)) or (apex_up >= 8 and l_arms >= 3 and r_arms >= 3 and top_corners <= 3 and apex_up > apex_dn):
                return "Up Arrow / Increment / Spinbox / Up"
            if (top_m > 2.0 * bot_m and top_m >= 6 and (arm_tl >= 2 and arm_tr >= 2)) or (apex_dn >= 8 and l_arms >= 3 and r_arms >= 3 and bot_corners <= 3 and apex_dn > apex_up):
                return "Down Arrow / Dropdown / Decrement / Spinbox / Down"

        # 10. Micro-Star / Favorite Glyphs (small 6-24px 5-point star geometry anywhere on screen)
        # Intrinsic morphological geometry: solid core or hollow star with apex, split feet, and empty bottom groove
        star_corners = int(np.sum(p24[:4, :4] > 0) + np.sum(p24[:4, 20:] > 0) + np.sum(p24[20:, :4] > 0) + np.sum(p24[20:, 20:] > 0))
        if 6 <= bw <= 24 and 6 <= bh <= 24 and abs(bw - bh) <= 5 and not has_frame and star_corners <= 10:
            if bot_groove <= 4 and top_c >= 2 and bot_l >= 2 and bot_r >= 2:
                if center_core >= 18 or (val_tl <= 3 and val_tr <= 3 and mid_l >= 12 and mid_r >= 12):
                    return "Star / Favorite / Bookmark"

        # 10b. Ruler Margin Handle / Marker / Indent Notch along horizontal ruler
        # Horizontal ruler track edge with distinct margin notch flanks or block slider handle
        if sh * 0.15 < cy < sh * 0.35 and (14 <= bh <= 28) and (1.5 <= aspect <= 3.5) and (20 <= bw <= 55):
            ruler_top = int(np.sum(p24[:8, :] > 0))
            ruler_bot_flanks = int(np.sum(p24[16:, :8] > 0) + np.sum(p24[16:, 16:] > 0))
            if ruler_top >= 25 and ruler_bot_flanks >= 6:
                return "Ruler Margin Handle / Marker / Indent"

        # 11. Prohibition / None / Cancel / No Fill (circle with distinct central diagonal slash)
        if 16 <= bw <= 45 and 16 <= bh <= 45 and circ_ratio < 0.20 and te >= 3 and be >= 3 and le >= 3 and re >= 3:
            diag_slash = max(diag1, diag2)
            diag_other = min(diag1, diag2)
            if diag_slash >= 9 and center_core >= 3 and diag_slash > 2.2 * max(1, diag_other) and cardinal >= 10 and diag_other <= 3:
                return "None / Clear / Cancel / No Fill / Transparent"

        # 12. Paintbrush / Drawing Tool (diagonal shaft with distinct bulbous tip)
        if 16 <= bw <= 42 and 16 <= bh <= 42:
            diag_anti = sum(p24[i, 23-i] > 0 for i in range(2, 22))
            tip = np.sum(p24[13:, :11] > 0)
            if diag_anti >= 10 and tip >= 8:
                return "Paintbrush Tool / Pencil / Brush"

        # 13. Directional Navigation Arrows (Strict horizontal tapering with apex)
        if 10 <= bw <= 45 and 8 <= bh <= 35:
            if left_m > 2.2 * right_m and right_m >= 4 and max(col_sums24[:6]) >= 8:
                return "Right Arrow / Forward"
            if right_m > 2.2 * left_m and left_m >= 4 and max(col_sums24[18:]) >= 8:
                return "Left Arrow / Back / Rewind / Previous"

        # 14. Expand / Plus Cross (+)
        if 8 <= bw <= 32 and 8 <= bh <= 32 and abs(bw - bh) <= 8:
            center_cross = np.sum(p24[8:16, :] > 0) + np.sum(p24[:, 8:16] > 0)
            corners = (np.sum(p24[:6, :6] > 0) + np.sum(p24[:6, 18:] > 0) +
                       np.sum(p24[18:, :6] > 0) + np.sum(p24[18:, 18:] > 0))
            if center_cross > 25 and corners <= 6 and (corners / float(center_cross + 1e-5) < 0.15):
                return "Expand / Plus"

        # 15. Filter / Funnel (top row wide, narrows downward)
        top_w = np.sum(p24[2:8, :] > 0)
        bot_w = np.sum(p24[14:22, :] > 0)
        if 0.75 <= aspect <= 1.35 and top_w > 1.5 * bot_w and top_w > 12 and bh <= 36:
            return "Filter / Funnel / Search Filter"

        # 16. Checkmark
        left_tick = sum(p24[8+i, 4+i] > 0 for i in range(5))
        right_tail = sum(p24[15-i, 9+i] > 0 for i in range(8))
        if left_tick >= 3 and right_tail >= 5 and top_c <= 4:
            return "Check / Checkmark / Selected"

        # 17. Circular / Curved Arched Glyphs (Undo / Redo / Refresh / Circular Arrow)
        # Intrinsic morphological geometry: hollow central loop with perimeter curvature and vertical side strokes
        if 10 <= bw <= 42 and 10 <= bh <= 42 and center_core <= 12 and circ_ratio < 0.35:
            top_arc = int(np.sum(p24[:8, 4:20] > 0))
            bot_arc = int(np.sum(p24[16:, 4:20] > 0))
            if (top_arc >= 6 or bot_arc >= 6) and (mid_l >= 4 or mid_r >= 4) and total_px >= 36:
                if abs(arm_tr - arm_tl) <= 2 and abs(bot_r - bot_l) <= 2 and top_arc >= 6 and bot_arc >= 6:
                    return "Toggle / Option Circle / Radio / Checkbox"
                elif arm_tr > arm_tl or (bot_r > bot_l and mid_r > mid_l):
                    return "Redo / Action Forward / Curved Arrow / Undo-Redo"
                elif arm_tl > arm_tr or (bot_l > bot_r and mid_l > mid_r):
                    return "Undo / Action Back / Curved Arrow / Undo-Redo"
                elif cardinal >= 12 and center_core <= 4 and arm_tr >= 4:
                    return "Refresh / Reload / Circular Arrow"
        # 18. Windows / Grid / App Launcher Logo (4 sub-squares with cross gap)
        if 12 <= bw <= 45 and 12 <= bh <= 45 and abs(bw - bh) <= 8:
            q_tl = int(np.sum(p24[2:10, 2:10] > 0))
            q_tr = int(np.sum(p24[2:10, 14:22] > 0))
            q_bl = int(np.sum(p24[14:22, 2:10] > 0))
            q_br = int(np.sum(p24[14:22, 14:22] > 0))
            gap_v = int(np.sum(p24[4:20, 10:14] > 0))
            gap_h = int(np.sum(p24[10:14, 4:20] > 0))
            if q_tl >= 4 and q_tr >= 4 and q_bl >= 4 and q_br >= 4 and gap_v <= 6 and gap_h <= 6:
                return "Start / App Launcher / Grid Logo"

        # 19. Settings / Cog / Gear (radial teeth around circular core with central hole)
        if circ_ratio < 0.25 and 12 <= bw <= 45 and 12 <= bh <= 45:
            if cardinal >= 8 and (arm_tl >= 2 and arm_tr >= 2 and arm_bl >= 2 and arm_br >= 2) and center_core <= 4:
                return "Settings / Preferences / Gear"

        # 20. Power / Standby (circle broken at top with vertical line in middle - system dock/menu only)
        if (cx < 65 or cy > sh - 55 or cy < 45) and 12 <= bw <= 36 and 12 <= bh <= 36 and circ_ratio < 0.20:
            top_gap = int(np.sum(p24[0:3, 7:17] > 0))
            vert_stalk = int(np.sum(p24[0:10, 11:13] > 0))
            if vert_stalk >= 4 and top_gap <= 4:
                return "Power / Standby / Shut Down"

        if bw > bh * 1.6:
            return "Toolbar Tool / Visual Button"

        if 10 <= bw <= 32 and 10 <= bh <= 32 and abs(bw - bh) <= 6:
            return "Toolbar Tool / Visual Button"

        return "Toolbar Tool / Visual Button"


# Backwards compatibility alias
IconShapeDetector = EnhancedIconDetector
