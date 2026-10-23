"""
Full-Bleed Lattice Sprayer with Triple-Sensor Consensus.
Combines WinRT Neural OCR, Windows UI Automation (UIA) Tree, and CLAHE Computer Vision
to place razor-sharp precision droplets with 0ms stale coordinates.
"""
import re
import math
import asyncio
import threading
import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    import winocr
    HAS_WINOCR = True
except ImportError:
    HAS_WINOCR = False

try:
    import ctypes
    from ctypes import wintypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

from ..core.models import DiagonalDroplet, WindowContext
from .icon_detector import EnhancedIconDetector
from .uia_walker import UiaControlWalker

class FullBleedLatticeSprayer:
    """
    Triple-Sensor Consensus Sprayer (Phase 8).
    Orchestrates:
      - Tier 1: Windows UI Automation (UIA) Tree (Confidence: 1.0, contrast-immune)
      - Tier 2: Native WinRT Neural OCR (Confidence: 0.95, resolution-independent)
      - Tier 3: Adaptive CLAHE Computer Vision (Confidence: 0.70, textless icons)
      - Background 45° Diagonal Mesh (Surface Probes for empty space navigation)
    """
    def __init__(self, step_s: int = 48):
        self.step_s = step_s          # 48px
        self.half_s = step_s // 2      # 24px
        self.step_dy = step_s // 2     # 24px -> Exact 45.0° diagonal slope
        self.last_ocr_lines: List[str] = []
        self._last_ui_droplets: List[DiagonalDroplet] = []
        self._last_win_hwnd: Optional[int] = None
        self._spray_lock = threading.RLock()

    def reset_tracking(self):
        """Reset spatial ID persistence memory (e.g. on manual scan or task change)."""
        self._last_ui_droplets = []
        self._last_win_hwnd = None

    @staticmethod
    def _clean_dialog_token(text: str) -> str:
        t = text.strip()
        t_clean = re.sub(r'[^a-zA-Z0-9]', '', t).lower()
        if 'canc' in t_clean or t_clean in ('caneel', 'cance', 'cncl', 'can'):
            return "Cancel"
        if t_clean in ('sav', 'save', 'saved', 'sve'):
            return "Save"
        if t_clean in ('yes', 'ye', 'ys', 'yess'):
            return "Yes"
        if t_clean in ('no', 'n0', 'noo'):
            return "No"
        if t_clean in ('ok', '0k', 'okay', 'okk'):
            return "OK"
        if 'dont' in t_clean or 'dontsave' in t_clean:
            return "Don't Save"
        if 'retry' in t_clean:
            return "Retry"
        if 'ignore' in t_clean:
            return "Ignore"
        if 'abort' in t_clean:
            return "Abort"
        if 'apply' in t_clean:
            return "Apply"
        if 'open' in t_clean:
            return "Open"
        if 'close' in t_clean:
            return "Close"
        if 'next' in t_clean:
            return "Next"
        if 'finish' in t_clean:
            return "Finish"
        if 'browse' in t_clean:
            return "Browse"
        if 'help' in t_clean:
            return "Help"
        return ""

    @staticmethod
    def _micro_ocr_button_patch(crop: np.ndarray) -> str:
        if not HAS_WINOCR or crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 12:
            return ""
        ch, cw = crop.shape[:2]
        mean_val = [int(v) for v in crop.mean(axis=(0, 1))]
        dialog_tokens = {
            'Cancel', 'Save', 'Yes', 'No', 'OK', "Don't Save", 'Retry', 'Ignore',
            'Abort', 'Apply', 'Open', 'Close', 'Next', 'Finish', 'Browse', 'Help'
        }
        best_fallback = ""
        
        scale = 3.0
        pad = int(12 * scale)
        up = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=mean_val)
        
        # Pass 1: Standard high-contrast padded crop
        try:
            op = winocr.recognize_cv2(padded, lang='en')
            res = op.get() if hasattr(op, 'get') else op
            for line in res.lines:
                raw_txt = line.text.strip()
                tok = FullBleedLatticeSprayer._clean_dialog_token(raw_txt)
                if tok in dialog_tokens:
                    return tok
                cleaned_general = re.sub(r'[^\w\s]', '', raw_txt).strip()
                if len(cleaned_general) >= 2 and not best_fallback:
                    best_fallback = cleaned_general
        except Exception:
            pass

        # Pass 2: Inverted pass only if no token found
        try:
            inv = 255 - padded
            op = winocr.recognize_cv2(inv, lang='en')
            res = op.get() if hasattr(op, 'get') else op
            for line in res.lines:
                raw_txt = line.text.strip()
                tok = FullBleedLatticeSprayer._clean_dialog_token(raw_txt)
                if tok in dialog_tokens:
                    return tok
        except Exception:
            pass
                    
        return best_fallback

    @staticmethod
    def _collinear_dialog_peer_inference(
        candidates: List[Tuple[int, int, str, str, str, float, Tuple[int, int, int, int]]]
    ) -> List[Tuple[int, int, str, str, str, float, Tuple[int, int, int, int]]]:
        dialog_tokens = {'Cancel', 'Save', 'Yes', 'No', 'OK', "Don't Save", 'Retry', 'Ignore', 'Abort', 'Apply', 'Open', 'Close'}
        updated = list(candidates)
        n = len(updated)
        for i in range(n):
            cx1, cy1, lbl1, cat1, sub1, conf1, box1 = updated[i]
            b1_tok = sub1.strip()
            
            for j in range(n):
                if i == j:
                    continue
                cx2, cy2, lbl2, cat2, sub2, conf2, box2 = updated[j]
                b2_tok = sub2.strip()
                
                h1 = box1[3] - box1[1]
                h2 = box2[3] - box2[1]
                if abs(cy1 - cy2) <= 8 and abs(h1 - h2) <= 8:
                    gap = box2[0] - box1[2]
                    if 4 <= gap <= 35:
                        # [?] [No] -> infer left is Yes
                        if b2_tok == "No" and (b1_tok not in dialog_tokens or "Visual Button" in lbl1):
                            updated[i] = (cx1, cy1, "[Btn] Yes", "interactive_widget", "Yes", 0.98, box1)
                        # [Yes] [?] -> infer right is No
                        elif b1_tok == "Yes" and (b2_tok not in dialog_tokens or "Visual Button" in lbl2):
                            updated[j] = (cx2, cy2, "[Btn] No", "interactive_widget", "No", 0.98, box2)
                        # [?] [Cancel]
                        elif b2_tok == "Cancel" and (b1_tok not in dialog_tokens or "Visual Button" in lbl1):
                            inferred_name = "Save" if any("save" in str(u[2]).lower() for u in updated) else "OK"
                            updated[i] = (cx1, cy1, f"[Btn] {inferred_name}", "interactive_widget", inferred_name, 0.96, box1)
        return updated

    @staticmethod
    def _fetch_win32_dialog_buttons(
        parent_hwnd: int, screen_w: int, screen_h: int
    ) -> List[Tuple[int, int, str, str, str, float, Tuple[int, int, int, int]]]:
        if not HAS_CTYPES or not parent_hwnd or parent_hwnd == 0:
            return []
        results = []
        try:
            user32 = ctypes.windll.user32
            hwnds_to_check = [parent_hwnd]
            popup = user32.GetLastActivePopup(parent_hwnd)
            if popup and popup != parent_hwnd and user32.IsWindow(popup):
                hwnds_to_check.append(popup)

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

            def _enum_child(h_child, _):
                if not user32.IsWindowVisible(h_child):
                    return True
                class_buf = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(h_child, class_buf, 256)
                cname = class_buf.value
                if cname in ("Button", "CCtrlButton", "CtrlButton"):
                    text_buf = ctypes.create_unicode_buffer(256)
                    user32.GetWindowTextW(h_child, text_buf, 256)
                    txt = text_buf.value.replace("&", "").strip()
                    rc = wintypes.RECT()
                    user32.GetWindowRect(h_child, ctypes.byref(rc))
                    bw = rc.right - rc.left
                    bh = rc.bottom - rc.top
                    if 14 <= bw <= 400 and 10 <= bh <= 80:
                        cx = (rc.left + rc.right) // 2
                        cy = (rc.top + rc.bottom) // 2
                        lbl = f"[Btn] {txt}" if txt else "[Btn] Button"
                        sub = txt or "Button"
                        bbox = (max(0, rc.left), max(0, rc.top), min(screen_w, rc.right), min(screen_h, rc.bottom))
                        results.append((cx, cy, lbl, "interactive_widget", sub, 0.99, bbox))
                return True

            for h_tgt in hwnds_to_check:
                user32.EnumChildWindows(h_tgt, WNDENUMPROC(_enum_child), 0)
        except Exception:
            pass
        return results

    async def spray(
        self,
        frame_bgr: np.ndarray,
        win_ctx: Optional[WindowContext] = None,
        copilot_rect: Optional[Tuple[int, int, int, int]] = None,
        offline_frame: bool = False
    ) -> List[DiagonalDroplet]:
        with self._spray_lock:
            return await self._spray_internal(
                frame_bgr=frame_bgr,
                win_ctx=win_ctx,
                copilot_rect=copilot_rect,
                offline_frame=offline_frame
            )

    async def _spray_internal(
        self,
        frame_bgr: np.ndarray,
        win_ctx: Optional[WindowContext] = None,
        copilot_rect: Optional[Tuple[int, int, int, int]] = None,
        offline_frame: bool = False
    ) -> List[DiagonalDroplet]:
        h, w = frame_bgr.shape[:2]
        curr_hwnd = win_ctx.hwnd if (win_ctx and win_ctx.is_valid) else 0
        if self._last_win_hwnd is not None and self._last_win_hwnd != curr_hwnd:
            self._last_ui_droplets = []
        self._last_win_hwnd = curr_hwnd

        is_on_desktop = (not win_ctx) or (not win_ctx.is_valid) or (win_ctx.title == "Desktop")
        is_progman = bool(win_ctx and getattr(win_ctx, 'class_name', '') in ("Progman", "WorkerW"))

        # Find all visible non-desktop application window bounds on display
        visible_app_rects = []
        if not offline_frame:
            try:
                from ..platform.window import enumerate_all_windows
                for win in enumerate_all_windows(include_minimized=False):
                    if not win.get("is_minimized") and win.get("title") not in ("Desktop", "Program Manager", "Windows Taskbar"):
                        r = win.get("rect")
                        if r and (r[2] - r[0] > 80) and (r[3] - r[1] > 80):
                            visible_app_rects.append(r)
            except Exception:
                pass

        def is_item_on_desktop(x: int, y: int) -> bool:
            # 1. Taskbar is never desktop
            if y >= h - 55:
                return False
            # 2. Inside active foreground window -> NEVER desktop
            if win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop" and win_ctx.rect:
                wx1, wy1, wx2, wy2 = win_ctx.rect
                if wx1 <= x <= wx2 and wy1 <= y <= wy2:
                    return False
            # 3. Inside ANY other visible application window on screen -> NEVER desktop
            for (vx1, vy1, vx2, vy2) in visible_app_rects:
                if vx1 <= x <= vx2 and vy1 <= y <= vy2:
                    return False
            # 4. Desktop wallpaper context (strictly outside all application windows)
            if is_progman:
                return True
            if win_ctx and win_ctx.is_valid and win_ctx.title in ("Desktop", "Program Manager") and not visible_app_rects:
                return True
            return False

        discovered_ui: List[Tuple[int, int, str, str, str, float]] = []  # (cx, cy, label, cat, sub, confidence)
        ocr_word_boxes: List[Tuple[int, int, int, int]] = []

        is_live_desktop = bool(
            win_ctx and win_ctx.is_valid and (
                win_ctx.title in ("Desktop", "Program Manager", "") or
                getattr(win_ctx, 'class_name', '') in ("Progman", "WorkerW")
            )
        )

        # ======================================================================
        # CONCURRENT SENSOR FUSION: UIA, WinRT OCR & CLAHE Icon Detector
        # ======================================================================
        uia_hwnd = win_ctx.hwnd if (win_ctx and win_ctx.is_valid) else None

        async def _fetch_uia():
            if not uia_hwnd or uia_hwnd == 0:
                return []
            try:
                # Phase 2: Decoupled Non-Blocking 0ms cache check (< 1.5s fresh)
                # If crawling deep tree like Chrome or stale, proceeds immediately with pure CV lattice
                return UiaControlWalker.get_cached_controls(hwnd=uia_hwnd, max_age_sec=1.5)
            except Exception:
                return []

        async def _fetch_ocr():
            if not HAS_WINOCR:
                return None
            try:
                def _run_winocr_mta(f):
                    try:
                        op = winocr.recognize_cv2(f, lang='en')
                        return op.get() if hasattr(op, 'get') else op
                    except Exception:
                        return None
                return await asyncio.to_thread(_run_winocr_mta, frame_bgr)
            except Exception:
                return None

        async def _fetch_icons():
            try:
                srect = win_ctx.rect if (win_ctx and win_ctx.is_valid and win_ctx.title not in ("Desktop", "Program Manager")) else None
                return await asyncio.to_thread(
                    EnhancedIconDetector.detect_icons,
                    frame_bgr,
                    search_rect=srect
                )
            except Exception:
                return []

        async def _fetch_win32_dialog():
            if offline_frame or not uia_hwnd or uia_hwnd == 0:
                return []
            try:
                return await asyncio.to_thread(
                    self._fetch_win32_dialog_buttons,
                    uia_hwnd,
                    w,
                    h
                )
            except Exception:
                return []

        uia_controls, ocr_res, detected_icons, win32_buttons = await asyncio.gather(
            _fetch_uia(), _fetch_ocr(), _fetch_icons(), _fetch_win32_dialog()
        )

        # 0. Ingest Native Win32 Dialog Buttons (authoritative ground-truth controls)
        for item in (win32_buttons or []):
            discovered_ui.append(item)

        # 1. Ingest UIA Controls
        for item in (uia_controls or []):
            if len(item) >= 6:
                ux, uy, ulabel, ucat, usub, raw_bbox = item[:6]
                bx1, by1, bx2, by2 = raw_bbox
                bx_min, bx_max = min(int(bx1), int(bx2)), max(int(bx1), int(bx2))
                by_min, by_max = min(int(by1), int(by2)), max(int(by1), int(by2))
                ubox = (max(0, bx_min), max(0, by_min), min(w, bx_max), min(h, by_max))
            else:
                ux, uy, ulabel, ucat, usub = item[:5]
                ubox = (max(0, int(ux - 15)), max(0, int(uy - 15)), min(w, int(ux + 15)), min(h, int(uy + 15)))
            discovered_ui.append((ux, uy, ulabel, ucat, usub, 1.0, ubox))

        # 2. Ingest WinRT OCR Lines
        if ocr_res and hasattr(ocr_res, 'lines'):
            try:
                    self.last_ocr_lines = [l.text.strip() for l in ocr_res.lines if l.text.strip()]
                    for line in ocr_res.lines:
                        txt = line.text.strip()
                        if len(txt) < 2:
                            continue
                        clean = re.sub(r'[^\x20-\x7E]', '', txt).strip()
                        if not clean:
                            continue

                        # Word boxes for icon collision avoidance
                        for wd in line.words:
                            ocr_word_boxes.append((
                                int(wd.bounding_rect.x - 4), int(wd.bounding_rect.y - 4),
                                int(wd.bounding_rect.x + wd.bounding_rect.width + 4),
                                int(wd.bounding_rect.y + wd.bounding_rect.height + 4)
                            ))

                        # Word-level centroids: place dots directly at functional center of each actionable word
                        if len(line.words) > 1:
                            line_min_x = min(wd.bounding_rect.x for wd in line.words)
                            line_max_x = max(wd.bounding_rect.x + wd.bounding_rect.width for wd in line.words)
                            line_min_y = min(wd.bounding_rect.y for wd in line.words)
                            line_max_y = max(wd.bounding_rect.y + wd.bounding_rect.height for wd in line.words)
                            line_cx = int((line_min_x + line_max_x) / 2)
                            line_raw_cy = int((line_min_y + line_max_y) / 2)
                            line_on_desktop = is_item_on_desktop(line_cx, line_raw_cy)

                            is_shortcut = (
                                line_on_desktop and len(line.words) <= 4 and len(clean) <= 32 and
                                not any(sym in clean.lower() for sym in ['/', '\\', '::', '.pth', '.ckpt', 'record:', 'http', 'checkpoint', '-->', '==', 'latest_'])
                            )

                            if is_shortcut:
                                # For genuine desktop application shortcuts (e.g. "Google Chrome", "Recycle Bin", "This PC"):
                                # Exactly ONE clean unified bounding box covering icon and label!
                                icon_cy = max(24, line_raw_cy - 44)
                                app_box = (
                                    int(max(0, min(line_cx - 28, line_min_x - 4))),
                                    int(max(0, icon_cy - 24)),
                                    int(min(w, max(line_cx + 28, line_max_x + 4))),
                                    int(min(h, line_max_y + 4))
                                )
                                discovered_ui.append((line_cx, icon_cy, clean, "interactive_app", clean[:18], 0.98, app_box))
                            else:
                                line_box = (max(0, int(line_min_x) - 4), max(0, int(line_min_y) - 3), min(w, int(line_max_x) + 4), min(h, int(line_max_y) + 3))
                                # Check if words have wide gaps indicating distinct toolbar/dialog/menu items on same line
                                has_wide_gaps = False
                                for idx in range(len(line.words) - 1):
                                    w_left = line.words[idx].bounding_rect.x + line.words[idx].bounding_rect.width
                                    w_right = line.words[idx + 1].bounding_rect.x
                                    if (w_right - w_left) > 20:
                                        has_wide_gaps = True
                                        break

                                if has_wide_gaps:
                                    # Cluster words separated by <= 18px into cohesive phrases (e.g. "Don't Save", "Save As")
                                    # while splitting at genuine toolbar / dialog button gaps (> 20px)
                                    word_clusters = []
                                    curr_cluster = [line.words[0]]
                                    for idx in range(len(line.words) - 1):
                                        w_left = line.words[idx].bounding_rect.x + line.words[idx].bounding_rect.width
                                        w_right = line.words[idx + 1].bounding_rect.x
                                        if (w_right - w_left) > 20:
                                            word_clusters.append(curr_cluster)
                                            curr_cluster = [line.words[idx + 1]]
                                        else:
                                            curr_cluster.append(line.words[idx + 1])
                                    if curr_cluster:
                                        word_clusters.append(curr_cluster)

                                    for cluster in word_clusters:
                                        c_words = [re.sub(r'[^\x20-\x7E]', '', wd.text).strip() for wd in cluster]
                                        c_text = " ".join(w for w in c_words if w).strip()
                                        if not c_text:
                                            continue
                                        c_min_x = min(wd.bounding_rect.x for wd in cluster)
                                        c_max_x = max(wd.bounding_rect.x + wd.bounding_rect.width for wd in cluster)
                                        c_min_y = min(wd.bounding_rect.y for wd in cluster)
                                        c_max_y = max(wd.bounding_rect.y + wd.bounding_rect.height for wd in cluster)
                                        c_cx = int((c_min_x + c_max_x) / 2)
                                        c_cy = int((c_min_y + c_max_y) / 2)
                                        c_box = (max(0, int(c_min_x) - 4), max(0, int(c_min_y) - 3), min(w, int(c_max_x) + 4), min(h, int(c_max_y) + 3))
                                        discovered_ui.append((c_cx, c_cy, c_text, "interactive_widget", c_text[:18], 0.96, c_box))
                                else:
                                    # Cohesive phrase / button / label: ONE unified box-shaped droplet!
                                    discovered_ui.append((line_cx, line_raw_cy, clean, "interactive_widget", clean[:18], 0.96, line_box))
                        else:
                            # Single-word line
                            wd = line.words[0]
                            cx = int(wd.bounding_rect.x + wd.bounding_rect.width / 2)
                            raw_cy = int(wd.bounding_rect.y + wd.bounding_rect.height / 2)
                            is_dt = (
                                is_item_on_desktop(cx, raw_cy) and len(clean) <= 24 and
                                not any(sym in clean.lower() for sym in ['/', '\\', '::', '.pth', '.ckpt', 'record:', 'http', 'checkpoint'])
                            )
                            if is_dt:
                                cy = max(24, raw_cy - 44)
                                cat = "interactive_app"
                                ubox = (
                                    int(max(0, min(cx - 28, wd.bounding_rect.x - 4))),
                                    int(max(0, cy - 24)),
                                    int(min(w, max(cx + 28, wd.bounding_rect.x + wd.bounding_rect.width + 4))),
                                    int(min(h, raw_cy + int(wd.bounding_rect.height / 2) + 4))
                                )
                                discovered_ui.append((cx, cy, clean, cat, clean[:14], 0.98, ubox))
                            else:
                                cy = raw_cy
                                cat = "interactive_widget"
                                ubox = (
                                    max(0, int(wd.bounding_rect.x) - 4),
                                    max(0, int(wd.bounding_rect.y) - 3),
                                    min(w, int(wd.bounding_rect.x + wd.bounding_rect.width) + 4),
                                    min(h, int(wd.bounding_rect.y + wd.bounding_rect.height) + 3)
                                )
                                discovered_ui.append((cx, cy, clean, cat, clean[:14], 0.96, ubox))
            except Exception:
                pass

        # ======================================================================
        # SENSOR 3: Taskbar Win32 Centroids
        # ======================================================================
        if not offline_frame:
            tb_h = 55
            tb_roi = frame_bgr[h - tb_h:h, 0:w]
            tb_gray = cv2.cvtColor(tb_roi, cv2.COLOR_BGR2GRAY)
            tb_grad = cv2.morphologyEx(tb_gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            _, tb_thresh = cv2.threshold(tb_grad, 20, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(tb_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            taskbar_boxes = []
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if 20 <= bw <= 55 and 20 <= bh <= 50:
                    taskbar_boxes.append((bx + bw // 2, h - 25))

            taskbar_boxes.sort(key=lambda item: item[0])
            deduped_tb = []
            for tcx, tcy in taskbar_boxes:
                if not deduped_tb or abs(tcx - deduped_tb[-1][0]) >= 36:
                    if not any(abs(tcx - u[0]) < 24 and abs(tcy - u[1]) < 20 for u in discovered_ui):
                        deduped_tb.append((tcx, tcy))

            for i, (tcx, tcy) in enumerate(deduped_tb):
                if tcx < max(55, int(w * 0.03)):
                    label, sub = "[Btn] Taskbar: Windows Start Button", "Start"
                elif 55 <= tcx < max(140, int(w * 0.08)):
                    label, sub = "[Btn] Taskbar: Windows Search", "Search"
                elif abs(tcx - w // 2) < int(w * 0.12):
                    label, sub = "[Btn] Taskbar: Windows Start Button / Center Dock", "Start"
                elif tcx > w - max(140, int(w * 0.08)):
                    label, sub = "[Btn] Taskbar: System Tray / Clock", "Clock"
                else:
                    label, sub = f"[Btn] Taskbar Pinned App #{i+1}", f"App#{i+1}"
                tbox = (max(0, tcx - 18), max(0, tcy - 18), min(w, tcx + 18), min(h, tcy + 18))
                discovered_ui.append((tcx, tcy, label, "interactive_widget", sub, 0.98, tbox))

        # ======================================================================
        # SENSOR 4: Titlebar Window Caption Buttons Injector
        # ======================================================================
        if win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop" and win_ctx.rect:
            wx1, wy1, wx2, wy2 = win_ctx.rect
            btn_w = 46
            btn_y = max(15, wy1 + 16)
            close_x = min(w - 24, wx2 - btn_w // 2)
            max_x = close_x - btn_w
            min_x = max_x - btn_w
            for cb in [
                (min_x, btn_y, "[Btn] Minimize", "interactive_widget", "Minimize", 1.0),
                (max_x, btn_y, "[Btn] Maximize", "interactive_widget", "Maximize", 1.0),
                (close_x, btn_y, "[Btn] Close", "interactive_widget", "Close", 1.0)
            ]:
                already_has = any(
                    (abs(cb[0] - u[0]) < 48 and abs(cb[1] - u[1]) < 36) or
                    (cb[4].lower() in u[2].lower() and u[1] < wy1 + 55)
                    for u in discovered_ui
                )
                if not already_has:
                    cbox = (max(0, cb[0] - btn_w // 2), max(0, cb[1] - 15), min(w, cb[0] + btn_w // 2), min(h, cb[1] + 15))
                    discovered_ui.append((cb[0], cb[1], cb[2], cb[3], cb[4], 1.0, cbox))

        is_live_desktop = bool(
            win_ctx and win_ctx.is_valid and (
                win_ctx.title in ("Desktop", "Program Manager", "") or
                getattr(win_ctx, 'class_name', '') in ("Progman", "WorkerW")
            )
        )

        # Spatial index for fast collision checking against discovered UIA/OCR controls
        disc_grid = {}
        grid_bin = 64
        for idx, u in enumerate(discovered_ui):
            ub = u[6]
            for gy in range(max(0, ub[1] // grid_bin), min(h // grid_bin, ub[3] // grid_bin) + 1):
                for gx in range(max(0, ub[0] // grid_bin), min(w // grid_bin, ub[2] // grid_bin) + 1):
                    disc_grid.setdefault((gx, gy), []).append(idx)

        # ======================================================================
        # SENSOR 5: Ingest CLAHE Computer Vision Icons (Filtered vs Text)
        # ======================================================================
        if not is_live_desktop and detected_icons:
            # 1. Super-Resolution Micro-OCR for Unlabeled Visual Button Candidates
            if HAS_WINOCR:
                btn_candidates = []
                for idx_ic, ic in enumerate(detected_icons):
                    cx, cy = ic[0], ic[1]
                    lbl = str(ic[2])
                    ibox = ic[5] if len(ic) > 5 else (max(0, cx - 12), max(0, cy - 12), min(w, cx + 12), min(h, cy + 12))
                    bw = ibox[2] - ibox[0]
                    bh = ibox[3] - ibox[1]
                    is_btn_cand = (
                        ("Visual Button" in lbl or "Button" in lbl or "Graphic" in lbl) and
                        (20 <= bw <= 320 and 12 <= bh <= 65 and bw >= bh * 1.15)
                    )
                    if is_btn_cand:
                        overlaps_text = False
                        for gy in range(max(0, ibox[1] // grid_bin), min(h // grid_bin, ibox[3] // grid_bin) + 1):
                            for gx in range(max(0, ibox[0] // grid_bin), min(w // grid_bin, ibox[2] // grid_bin) + 1):
                                for u_idx in disc_grid.get((gx, gy), []):
                                    u = discovered_ui[u_idx]
                                    ub = u[6]
                                    ix1, iy1 = max(ibox[0], ub[0]), max(ibox[1], ub[1])
                                    ix2, iy2 = min(ibox[2], ub[2]), min(ibox[3], ub[3])
                                    if ix2 > ix1 and iy2 > iy1:
                                        if ((ix2 - ix1) * (iy2 - iy1)) / max(1, bw * bh) > 0.25:
                                            overlaps_text = True
                                            break
                                if overlaps_text:
                                    break
                            if overlaps_text:
                                break
                        if not overlaps_text:
                            btn_candidates.append((idx_ic, ibox))

                if btn_candidates:
                    btn_candidates = btn_candidates[:6]
                    async def _run_micro_ocr_cand(cand_info):
                        idx_ic, c_box = cand_info
                        crop = frame_bgr[max(0, c_box[1]):min(h, c_box[3]), max(0, c_box[0]):min(w, c_box[2])]
                        txt = await asyncio.to_thread(self._micro_ocr_button_patch, crop)
                        return (idx_ic, txt)

                    micro_tasks = [_run_micro_ocr_cand(cand) for cand in btn_candidates]
                    micro_res = await asyncio.gather(*micro_tasks)
                    for idx_ic, txt in micro_res:
                        if txt:
                            old_ic = detected_icons[idx_ic]
                            upg_ibox = old_ic[5] if len(old_ic) > 5 else None
                            detected_icons[idx_ic] = (
                                old_ic[0], old_ic[1], f"[Btn] {txt}", "interactive_widget", txt, upg_ibox
                            )

            for ic in detected_icons:
                cx, cy = ic[0], ic[1]
                icon_label = str(ic[2])
                visual_action = icon_label.startswith("[Icon]") or icon_label.startswith("[Btn]")
                ibox = ic[5] if len(ic) > 5 else (max(0, cx - 12), max(0, cy - 12), min(w, cx + 12), min(h, cy + 12))
                overlaps_existing = False
                ibox_area = max(1, (ibox[2] - ibox[0]) * (ibox[3] - ibox[1]))

                cand_indices = set()
                for gy in range(max(0, ibox[1] // grid_bin), min(h // grid_bin, ibox[3] // grid_bin) + 1):
                    for gx in range(max(0, ibox[0] // grid_bin), min(w // grid_bin, ibox[2] // grid_bin) + 1):
                        for u_idx in disc_grid.get((gx, gy), []):
                            cand_indices.add(u_idx)

                for u_idx in cand_indices:
                    u = discovered_ui[u_idx]
                    ub = u[6]
                    if (ub[0] - 4 <= cx <= ub[2] + 4 and ub[1] - 4 <= cy <= ub[3] + 4):
                        if not (visual_action and math.hypot(cx - u[0], cy - u[1]) >= 9):
                            overlaps_existing = True
                            break
                    ix1 = max(ibox[0], ub[0])
                    iy1 = max(ibox[1], ub[1])
                    ix2 = min(ibox[2], ub[2])
                    iy2 = min(ibox[3], ub[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if (inter / ibox_area) > 0.25:
                            ub_h = ub[3] - ub[1]
                            if u[3] == "text" and ub_h <= 32 and (inter / ibox_area) > 0.50:
                                overlaps_existing = True
                                break
                            ub_area = max(1, (ub[2] - ub[0]) * (ub[3] - ub[1]))
                            if ub_area <= 2.2 * ibox_area:
                                is_handle_on_track = (
                                    any(k in icon_label.lower() for k in ["handle", "marker", "indent", "thumb"]) and
                                    (ub[2] - ub[0] >= 60)
                                )
                                if not is_handle_on_track and not (visual_action and (inter / ibox_area) < 0.85):
                                    overlaps_existing = True
                                    break
                if not overlaps_existing:
                    new_idx = len(discovered_ui)
                    conf = 0.98 if icon_label.startswith("[Btn]") else 0.70
                    discovered_ui.append((cx, cy, ic[2], ic[3], ic[4], conf, ibox))
                    for gy in range(max(0, ibox[1] // grid_bin), min(h // grid_bin, ibox[3] // grid_bin) + 1):
                        for gx in range(max(0, ibox[0] // grid_bin), min(w // grid_bin, ibox[2] // grid_bin) + 1):
                            disc_grid.setdefault((gx, gy), []).append(new_idx)

        # Exclude Copilot widget area
        if copilot_rect:
            rect_list = []
            if isinstance(copilot_rect, list):
                rect_list = copilot_rect
            elif isinstance(copilot_rect, tuple):
                if len(copilot_rect) == 4 and all(isinstance(v, (int, float)) for v in copilot_rect):
                    rect_list = [copilot_rect]
                elif len(copilot_rect) > 0 and isinstance(copilot_rect[0], (tuple, list)):
                    rect_list = list(copilot_rect)

            for cr in rect_list:
                if len(cr) == 4:
                    cpx1, cpy1, cpx2, cpy2 = cr
                    discovered_ui = [
                        u for u in discovered_ui
                        if not (cpx1 - 10 <= u[0] <= cpx2 + 10 and cpy1 - 10 <= u[1] <= cpy2 + 10)
                    ]

        # Spatial collinear dialog button action strip peer inference
        discovered_ui = self._collinear_dialog_peer_inference(discovered_ui)

        # ======================================================================
        # BOX-AWARE NON-MAXIMUM SUPPRESSION (IoU / Box Containment / Centroid NMS)
        # ======================================================================
        discovered_ui.sort(key=lambda item: item[5], reverse=True)  # Higher confidence first
        unique_ui = []
        nms_grid = {}
        for u in discovered_ui:
            is_dup = False
            ux, uy, u_lbl, ucat, usub, u_conf, ubox = u
            u_is_slider = any(k in u_lbl.lower() for k in ["slider", "scrollbar", "handle", "divider"])

            cand_ex = set()
            for gy in range(max(0, (uy - 24) // grid_bin), min(h // grid_bin, (uy + 24) // grid_bin) + 1):
                for gx in range(max(0, (ux - 24) // grid_bin), min(w // grid_bin, (ux + 24) // grid_bin) + 1):
                    for ex_idx in nms_grid.get((gx, gy), []):
                        cand_ex.add(ex_idx)

            for ex_idx in cand_ex:
                ex = unique_ui[ex_idx]
                ex_x, ex_y, ex_lbl, ex_cat, ex_sub, ex_conf, ex_box = ex
                ex_is_slider = any(k in ex_lbl.lower() for k in ["slider", "scrollbar", "handle", "divider"])

                if u_is_slider != ex_is_slider and math.hypot(ux - ex_x, uy - ex_y) >= 8:
                    continue

                dx = abs(ux - ex_x)
                dy = abs(uy - ex_y)
                if dx < 16 and dy < 16:
                    if dx <= 6 and 7 <= dy <= 24:
                        continue  # Preserve vertically stacked spinbox arrows
                    is_dup = True
                    break

                is_handle_on_track = (
                    any(k in u_lbl.lower() for k in ["handle", "marker", "indent", "thumb"]) and
                    any(k in ex_lbl.lower() for k in ["track", "slider", "rail"]) and
                    (ex_box[2] - ex_box[0] >= 60)
                ) or (
                    any(k in ex_lbl.lower() for k in ["handle", "marker", "indent", "thumb"]) and
                    any(k in u_lbl.lower() for k in ["track", "slider", "rail"]) and
                    (ubox[2] - ubox[0] >= 60)
                )
                if not is_handle_on_track:
                    ix1 = max(ubox[0], ex_box[0])
                    iy1 = max(ubox[1], ex_box[1])
                    ix2 = min(ubox[2], ex_box[2])
                    iy2 = min(ubox[3], ex_box[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        area_u = max(1, (ubox[2] - ubox[0]) * (ubox[3] - ubox[1]))
                        area_ex = max(1, (ex_box[2] - ex_box[0]) * (ex_box[3] - ex_box[1]))
                        min_area = min(area_u, area_ex)
                        if (inter_area / min_area) > 0.40:
                            is_dup = True
                            break

            if not is_dup:
                new_idx = len(unique_ui)
                unique_ui.append(u)
                for gy in range(max(0, ubox[1] // grid_bin), min(h // grid_bin, ubox[3] // grid_bin) + 1):
                    for gx in range(max(0, ubox[0] // grid_bin), min(w // grid_bin, ubox[2] // grid_bin) + 1):
                        nms_grid.setdefault((gx, gy), []).append(new_idx)

        # Proximity context enrichment for compound semantic clustering
        ocr_items = [u for u in unique_ui if not u[2].startswith("[Icon]") and not u[2].startswith("[Btn]")]
        # Line/phrase headers (items with multiple words or setting keywords)
        header_items = [
            u for u in ocr_items
            if len(u[2].split()) >= 2 and (":" in u[2] or any(k in u[2].lower() for k in ["settings", "options", "preferences", "panel", "search", "config", "tools"]))
        ]

        enriched_ui = []
        for u in unique_ui:
            ux, uy, ulabel, ucat, usub, uconf, ubox = u
            attached_label = ""
            semantic_role = "control" if (ulabel.startswith("[Icon]") or ulabel.startswith("[Btn]")) else "widget"

            if ulabel.startswith("[Icon]"):
                u_on_desktop = is_item_on_desktop(ux, uy)
                # 1. First look for same-row input/action header (e.g. Search settings input bar, Chrome address bar)
                same_row_header = None
                min_row_dx = 999999
                for tx, ty, tlabel, _, _, _, _ in ocr_items:
                    if is_item_on_desktop(tx, ty) != u_on_desktop:
                        continue
                    c_dy = uy - ty
                    c_dx = ux - tx
                    max_track_dx = 480 if any(k in tlabel.lower() for k in [".com", ".org", ".net", ".edu", ".io", "http", "www", "search", "url", "chrome", "firefox", "edge"]) else 130
                    if abs(c_dy) <= 18 and -240 <= c_dx <= max_track_dx and c_dx != 0:
                        dist_x = abs(c_dx)
                        if dist_x < min_row_dx:
                            min_row_dx = dist_x
                            same_row_header = (tx, ty, tlabel, c_dx, c_dy)

                icon_name = ulabel.replace("[Icon] ", "").strip()

                # Check for split-button pairing (primary tool icon on left + dropdown chevron on right)
                split_partner = None
                if any(k in icon_name.lower() for k in ("down arrow", "dropdown", "decrement", "chevron")):
                    for sx, sy, slbl, _, _, _, _ in unique_ui:
                        if slbl.startswith("[Icon]") and slbl != ulabel:
                            s_dx = ux - sx
                            s_dy = abs(uy - sy)
                            if 12 <= s_dx <= 45 and s_dy <= 8:
                                s_name = slbl.replace("[Icon] ", "").strip()
                                if not any(k in s_name.lower() for k in ("down arrow", "dropdown", "decrement", "chevron")):
                                    if "bullet" in s_name.lower() or "numbered" in s_name.lower() or "list" in s_name.lower():
                                        peer_lists = [
                                            px for px, py, plbl, _, _, _, _ in unique_ui
                                            if abs(py - sy) <= 8 and any(k in plbl.lower() for k in ("bullet", "numbered", "list"))
                                            and not any(k in plbl.lower() for k in ("down arrow", "dropdown", "decrement", "chevron"))
                                        ]
                                        peer_lists = sorted(list(set(peer_lists)))
                                        if len(peer_lists) >= 2:
                                            if sx == peer_lists[0]:
                                                s_name = "Bullet List (Non-numeric / Unordered Dots)"
                                            else:
                                                s_name = "Numbered List (Numeric / Ordered 1. 2. 3.)"
                                    split_partner = s_name
                                    break

                if split_partner:
                    ulabel = f"[Icon] Dropdown / Expand Options for '{split_partner}'"
                    semantic_role = "control"
                elif same_row_header:
                    _, _, ctx_lbl, c_dx, c_dy = same_row_header
                    short_lbl = ctx_lbl[:24].strip()
                    attached_label = short_lbl
                    semantic_role = "control"
                    is_clear_or_close = any(k in icon_name.lower() for k in ("close", "cross", "dismiss", "clear", "delete", "cancel"))
                    if "filter" in icon_name.lower() or "funnel" in icon_name.lower():
                        ulabel = f"[Icon] Filter / Search Options for '{short_lbl}'"
                    elif is_clear_or_close:
                        ulabel = f"[Icon] Clear / Close for '{short_lbl}'"
                    elif re.search(r'\b(star|favorite|bookmark)\b', icon_name.lower()):
                        ulabel = f"[Icon] Star / Favorite / Bookmark for '{short_lbl}'"
                    elif "search" in short_lbl.lower() and ("arrow" in icon_name.lower() or "dropdown" in icon_name.lower() or "action" in icon_name.lower() or "icon" in icon_name.lower()):
                        ulabel = f"[Icon] Search Options / Filter for '{short_lbl}'"
                    elif icon_name in ("Icon", "Action", "Small Graphic", "Graphic Element"):
                        pos_str = "Right of" if c_dx > 0 else "Left of"
                        ulabel = f"[Action Control {pos_str} '{short_lbl}']"
                    elif "Handle" in icon_name or "Scrollbar" in icon_name:
                        side = "Left" if ux < w * 0.40 else ("Right" if ux > w * 0.60 else "")
                        prefix = f"{side} " if side else ""
                        ulabel = f"[Icon] {prefix}{icon_name}"
                    elif c_dx < 0 and not any(k in icon_name.lower() for k in ("undo", "redo", "curved arrow")) and (
                        any(k in icon_name.lower() for k in ("arrow", "chevron", "caret", "plus", "add", "toggle", "expand", "triangle", "slider", "handle"))
                        or any(k in short_lbl.lower() for k in ("v ", "^ ", "> ", "< ", "expand", "collapse", "panel", "character", "paragraph", "options", "settings"))
                    ):
                        ulabel = f"[Icon] Expand / Collapse / Toggle Options for '{short_lbl}'"
                    else:
                        ulabel = f"[Icon] {icon_name} for '{short_lbl}'"
                else:
                    # General nearest text with vertical penalty
                    closest_text = None
                    min_d = 999999
                    for tx, ty, tlabel, _, _, _, _ in ocr_items:
                        if is_item_on_desktop(tx, ty) != u_on_desktop:
                            continue
                        dist = math.hypot(ux - tx, (uy - ty) * 2.5)
                        if dist < min_d:
                            min_d = dist
                            closest_text = (tx, ty, tlabel, ux - tx, uy - ty)
                    if "Handle" in icon_name or "Scrollbar" in icon_name:
                        side = "Left" if ux < w * 0.40 else ("Right" if ux > w * 0.60 else "")
                        prefix = f"{side} " if side else ""
                        ulabel = f"[Icon] {prefix}{icon_name}"
                        semantic_role = "control"
                    elif closest_text and min_d < 120 and abs(closest_text[4]) <= 28:
                        _, _, ctx_lbl, c_dx, c_dy = closest_text
                        short_lbl = ctx_lbl[:24].strip()
                        attached_label = short_lbl
                        semantic_role = "control"
                        # Same row to the left of text -> Tree Expander / Section Toggle
                        if abs(c_dy) <= 18 and -95 <= c_dx < 0:
                            if any(k in short_lbl.lower() for k in ["advanced", "option", "detail", "setting", "folder", "tree", "item", "group", "character", "panel", "lists", "format"]):
                                ulabel = f"[Icon] Expand / Toggle for '{short_lbl}'"
                            elif any(k in icon_name.lower() for k in ["expand", "chevron", "triangle", "plus", "toggle", "disclosure", "caret"]) and not any(k in icon_name.lower() for k in ["undo", "redo", "curved"]):
                                ulabel = f"[Icon] Expand / Toggle for '{short_lbl}'"
                            elif -65 <= c_dx < 0:
                                if any(k in icon_name.lower() for k in ["toggle", "option circle", "radio", "checkbox", "circle"]):
                                    ulabel = f"[Toggle / Checkbox for '{short_lbl}']"
                                elif icon_name not in ("Icon", "Action"):
                                    ulabel = f"[Icon] {icon_name} for '{short_lbl}'"
                                else:
                                    ulabel = f"[Checkbox / Toggle for '{short_lbl}']"
                        elif icon_name not in ("Icon", "Action"):
                            ulabel = f"[Icon] {icon_name} (Near '{short_lbl}')"
                        elif abs(c_dy) <= 22 and c_dx > 0:
                            ulabel = f"[Control/Option for '{short_lbl}']" if "search" in short_lbl.lower() else f"[Action Control Right of '{short_lbl}']"
                        elif abs(c_dx) <= 60 and 15 <= c_dy <= 55:
                            ulabel = f"[Dropdown / Value for '{short_lbl}']"
                        else:
                            ulabel = f"{ulabel} (Near '{short_lbl}')"
            else:
                # Text item: check if it is a single-glyph section toggle (e.g. 'v', '>', '^', '+', '-') before a section header
                clean_sym = ulabel.strip().lower()
                is_sym_toggle = clean_sym in ('v', '>', '^', '+', '-', '▼', '▶', '▲', '◄')
                sym_header = None
                if is_sym_toggle:
                    for tx, ty, tlabel, _, _, _, _ in ocr_items:
                        s_dy = uy - ty
                        s_dx = tx - ux
                        if abs(s_dy) <= 15 and 8 <= s_dx <= 140:
                            sym_header = tlabel[:24].strip()
                            break

                if sym_header:
                    attached_label = sym_header
                    semantic_role = "control"
                    ulabel = f"[Icon] Expand / Collapse / Fold Toggle for '{sym_header}'"
                else:
                    # Text item: check if it is directly below a setting header (e.g. dropdown value box below "Files: Auto Save")
                    above_header = None
                    min_above_dy = 999999
                    for tx, ty, tlabel, _, _, _, _ in header_items:
                        c_dy = uy - ty
                        c_dx = ux - tx
                        if 20 <= c_dy <= 75 and abs(c_dx) <= 80 and tlabel != ulabel:
                            if c_dy < min_above_dy:
                                min_above_dy = c_dy
                                above_header = (tx, ty, tlabel, c_dx, c_dy)

                    # Isolated setting option / value box (not surrounded by multiple words on the same line)
                    is_isolated_val = sum(1 for ox, oy, _, _, _, _, _ in ocr_items if abs(oy - uy) <= 8 and abs(ox - ux) <= 150) <= 2

                    if above_header and is_isolated_val and not any(k in ulabel.lower() for k in ["settings", "options", "preferences", "panel", "files:"]):
                        _, _, h_lbl, _, _ = above_header
                        short_h = h_lbl[:24].strip()
                        attached_label = short_h
                        semantic_role = "control"
                        ulabel = f"[Dropdown / Value for '{short_h}'] ({ulabel})"
                    else:
                        lbl_l = ulabel.lower()
                        if any(k in lbl_l for k in ["slider", "zoom", "magnification", "level"]):
                            semantic_role = "slider"
                        elif any(k in lbl_l for k in ["settings", "options", "preferences", "panel", "files:"]):
                            semantic_role = "header"

            enriched_ui.append((ux, uy, ulabel, ucat, usub, uconf, ubox, attached_label, semantic_role))

        # Collinear Grouping & Ordinal Tagging (Phase 8 Generalization)
        # Identifies collinear vertical and horizontal clusters of controls/widgets
        # and assigns ordinal position tags: [Item 1 of N (First / Topmost)], [Item N of N (Last)], etc.
        enriched_list = [list(item) for item in enriched_ui]

        # Universal Vertical Spinbox Stepper Identification (greedy disjoint pairing by smallest dy)
        stepper_candidates = []
        for i in range(len(enriched_list)):
            u1 = enriched_list[i]
            if not u1[2].startswith("[Icon]"):
                continue
            for j in range(i + 1, len(enriched_list)):
                u2 = enriched_list[j]
                if not u2[2].startswith("[Icon]"):
                    continue
                dx = abs(u1[0] - u2[0])
                dy = abs(u1[1] - u2[1])
                if dx <= 8 and 8 <= dy <= 24:
                    stepper_candidates.append((dy, i, j))

        stepper_candidates.sort(key=lambda item: item[0])
        used_stepper_indices = set()
        for dy, i, j in stepper_candidates:
            if i in used_stepper_indices or j in used_stepper_indices:
                continue
            top_idx = i if enriched_list[i][1] < enriched_list[j][1] else j
            bot_idx = j if enriched_list[i][1] < enriched_list[j][1] else i
            top_lbl = enriched_list[top_idx][2]
            bot_lbl = enriched_list[bot_idx][2]
            is_non_stepper = any(
                k in top_lbl.lower() or k in bot_lbl.lower()
                for k in ["color swatch", "color cell", "palette", "checkbox", "slider", "handle", "star", "bullet", "thumb"]
            )
            if is_non_stepper:
                continue
            used_stepper_indices.add(i)
            used_stepper_indices.add(j)
            if any(k in top_lbl for k in ["Arrow", "Spinbox", "Graphic", "Format"]) or top_lbl == "[Icon] Icon":
                att = enriched_list[top_idx][7] or enriched_list[bot_idx][7]
                suffix = f" (Near '{att}')" if att else ""
                enriched_list[top_idx][2] = f"[Icon] Up Arrow / Increment / Spinbox / Up{suffix}"
                enriched_list[top_idx][8] = "control"
            if any(k in bot_lbl for k in ["Arrow", "Spinbox", "Graphic", "Format"]) or bot_lbl == "[Icon] Icon":
                att = enriched_list[bot_idx][7] or enriched_list[top_idx][7]
                suffix = f" (Near '{att}')" if att else ""
                enriched_list[bot_idx][2] = f"[Icon] Down Arrow / Decrement / Spinbox / Down{suffix}"
                enriched_list[bot_idx][8] = "control"

        # Real UI Landmarks at exact centroids with bounding box geometry
        ui_droplet_points = []
        for uix, uiy, uilabel, uicat, uisub, uiconf, uibox, ui_att, ui_role in enriched_list:
            ui_droplet_points.append({
                'x': uix,
                'y': uiy,
                'diag_key': uix + uiy,
                'category': uicat,
                'label': uilabel,
                'subtext': uisub,
                'confidence': uiconf,
                'bbox': uibox,
                'attached_label': ui_att,
                'semantic_role': ui_role,
                'is_landmark': True
            })

        has_fg = bool(win_ctx and win_ctx.is_valid and win_ctx.title != "Desktop")
        wx1, wy1, wx2, wy2 = win_ctx.rect if has_fg else (0, 0, 0, 0)

        # Background Texture / Edge-Variance Masking for Smart Probe Filtering (Accelerated via 4x downsampling)
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sh_small, sw_small = max(1, h // 4), max(1, w // 4)
        small_gray = cv2.resize(gray_frame, (sw_small, sh_small), interpolation=cv2.INTER_AREA)
        grad_x = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        local_energy_small = cv2.boxFilter(mag, -1, (8, 8))

        # Precision 45° Diagonal Mesh (Surface Probes) - Noise-Suppressed & Affordance-Grounded
        surface_probe_points = []
        if not is_live_desktop:
            # Adaptive Lattice Density: when UI is rich (>=30), use 96px lattice; otherwise 72px
            probe_step = 96 if len(ui_droplet_points) >= 30 else 72
            num_rows = int(math.ceil(h / (probe_step // 2))) + 1

            # Fast 32px spatial occupancy grid for O(1) clearance check
            occupied_grid = set()
            for u in ui_droplet_points:
                gx = u['x'] // 32
                gy = u['y'] // 32
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        occupied_grid.add((gx + ox, gy + oy))

            affordance_radius = min(w, h) * 0.22
            aff_r_sq = affordance_radius * affordance_radius

            for r in range(num_rows):
                y = min(r * (probe_step // 2), h - 1)
                off1 = (probe_step // 2) if (r % 2 == 1) else 0
                gy = y // 32

                x1 = off1
                while x1 < w:
                    gx = x1 // 32
                    if (gx, gy) not in occupied_grid:
                        sy = min(sh_small - 1, y // 4)
                        sx = min(sw_small - 1, x1 // 4)
                        patch_energy = float(local_energy_small[sy, sx])
                        in_active_window = has_fg and (wx1 <= x1 <= wx2 and wy1 <= y <= wy2)

                        # Saliency-Grounded Surface Probe Filtering:
                        # 1. Require structural edge energy (2.5 in window, 5.0 outside)
                        # 2. Require Affordance Proximity: Probe must be within reachable radius of a UI control
                        energy_threshold = 2.5 if in_active_window else 5.0
                        if patch_energy >= energy_threshold:
                            near_ui = any(
                                (abs(x1 - u['x']) <= affordance_radius and (x1 - u['x'])**2 + (y - u['y'])**2 <= aff_r_sq)
                                for u in ui_droplet_points
                            ) if ui_droplet_points else True
                            if near_ui:
                                surface_probe_points.append({
                                    'x': x1, 'y': y, 'diag_key': x1 + y,
                                    'category': 'surface_probe',
                                    'label': f'Lattice Probe at ({x1}, {y})',
                                    'subtext': '',
                                    'confidence': 0.10,
                                    'is_landmark': False,
                                    'energy': patch_energy,
                                    'in_window': in_active_window
                                })
                    x1 += probe_step

            # Adaptive Saliency Probe Budgeting:
            # If screen already has rich UI landmarks (>=35), cap probes to 6
            # If screen has moderate UI landmarks (>=15), cap probes to 10
            # If screen is sparse (e.g. empty canvas), allow up to 16
            max_probes = 6 if len(ui_droplet_points) >= 35 else (10 if len(ui_droplet_points) >= 15 else 16)
            if len(surface_probe_points) > max_probes:
                surface_probe_points.sort(key=lambda p: (
                    1 if p['in_window'] else 0,
                    p['energy']
                ), reverse=True)
                surface_probe_points = surface_probe_points[:max_probes]

        # Spatial & Semantic Two-Pass ID Persistence across turns
        ui_droplet_points.sort(key=lambda p: (p['y'], p['x']))
        droplets: List[DiagonalDroplet] = []
        assigned_ids = set()
        unmatched_points = []

        def _clean_match_label(lbl: str) -> str:
            if not lbl:
                return ""
            s = lbl.lower()
            s = re.sub(r'\[item \d+ of \d+.*?\]', '', s)
            s = re.sub(r'\[(?:btn|icon|app)\]\s*', '', s)
            s = re.sub(r'[^a-z0-9]', '', s)
            return s.strip()

        # Two-Pass Matching against previous frame's UI droplets:
        # Pass A: Exact clean label match within generous displacement (up to 120px for continuous scrolling/motion)
        # Pass B: Spatial proximity match for remaining controls / icons within 24px
        if self._last_ui_droplets:
            used_prev_ids = set()
            pass_b_candidates = []

            # Pass A: Label matching
            for pt in ui_droplet_points:
                px, py = pt['x'], pt['y']
                clean_lbl = _clean_match_label(pt.get('label', ''))
                best_prev = None
                best_dist = 120.0
                if len(clean_lbl) >= 2:
                    for pd in self._last_ui_droplets:
                        if pd.id in used_prev_ids:
                            continue
                        if _clean_match_label(pd.label) == clean_lbl:
                            dist = math.hypot(px - pd.center_x, py - pd.center_y)
                            if dist <= best_dist:
                                best_dist = dist
                                best_prev = pd
                if best_prev is not None:
                    used_prev_ids.add(best_prev.id)
                    assigned_ids.add(best_prev.id)
                    pt['id'] = best_prev.id
                else:
                    pass_b_candidates.append(pt)

            # Pass B: Spatial proximity matching for remaining points
            for pt in pass_b_candidates:
                px, py = pt['x'], pt['y']
                best_prev = None
                best_dist = 24.0
                for pd in self._last_ui_droplets:
                    if pd.id in used_prev_ids:
                        continue
                    dist = math.hypot(px - pd.center_x, py - pd.center_y)
                    if dist <= best_dist:
                        best_dist = dist
                        best_prev = pd
                if best_prev is not None:
                    used_prev_ids.add(best_prev.id)
                    assigned_ids.add(best_prev.id)
                    pt['id'] = best_prev.id
                else:
                    unmatched_points.append(pt)
        else:
            unmatched_points = ui_droplet_points

        # Assign fresh sequential IDs to newly materialized points
        next_id = 1
        for pt in unmatched_points:
            while next_id in assigned_ids:
                next_id += 1
            pt['id'] = next_id
            assigned_ids.add(next_id)

        # Dynamically discover panel titles and boundaries from detected dividers and text headers
        left_dividers = [pt['x'] for pt in ui_droplet_points if "Left Panel Divider" in pt.get('label', '')]
        left_divider_x = left_dividers[0] if left_dividers else None

        right_dividers = [pt['x'] for pt in ui_droplet_points if "Right Panel Divider" in pt.get('label', '')]
        right_divider_x = right_dividers[0] if right_dividers else None

        menubar_tokens = {
            "file", "edit", "view", "insert", "format", "tools", "window",
            "help", "selection", "terminal", "go", "run", "options", "preferences"
        }

        left_sidebar_title = ""
        left_sidebar_min_y = 0
        if left_divider_x is not None:
            top_left_headers = [
                (pt['label'].strip(), pt['y']) for pt in ui_droplet_points
                if pt['x'] < left_divider_x and 60 <= pt['y'] <= 350
                and not pt['label'].startswith("[Icon]") and not pt['label'].startswith("[Btn]")
                and len(pt['label'].strip()) >= 3
                and pt['label'].strip().lower() not in menubar_tokens
                and not any(m in pt['label'].lower().split() for m in menubar_tokens if len(pt['label'].split()) > 3)
            ]
            left_sidebar_title = f"Left Sidebar ({top_left_headers[0][0]})" if top_left_headers else "Left Sidebar"
            left_sidebar_min_y = (top_left_headers[0][1] - 15) if top_left_headers else (h * 0.18)

        right_panel_title = ""
        right_panel_min_y = 0
        if right_divider_x is not None:
            top_right_headers = [
                (pt['label'].strip(), pt['y']) for pt in ui_droplet_points
                if pt['x'] > right_divider_x and 60 <= pt['y'] <= 350
                and not pt['label'].startswith("[Icon]") and not pt['label'].startswith("[Btn]")
                and len(pt['label'].strip()) >= 3
                and pt['label'].strip().lower() not in menubar_tokens
                and not any(m in pt['label'].lower().split() for m in menubar_tokens if len(pt['label'].split()) > 3)
            ]
            right_panel_title = f"Right Side Panel ({top_right_headers[0][0]})" if top_right_headers else "Right Side Panel"
            right_panel_min_y = (top_right_headers[0][1] - 15) if top_right_headers else (h * 0.18)

        def get_container_info(cx, cy, screen_w, screen_h):
            if cy >= screen_h - 45:
                return "bottom_bar", "Taskbar / Bottom Bar"
            if cx <= 50 and screen_w >= 1024:
                return "desktop_dock", "Host OS Desktop Launcher / Dock"
            if cy <= 30 and screen_h >= 600:
                return "top_bar", "Host OS Top Bar"
            if left_divider_x is not None and left_sidebar_title and cx < left_divider_x and cy >= left_sidebar_min_y:
                return "left_sidebar", left_sidebar_title
            if right_divider_x is not None and right_panel_title and cx > right_divider_x and cy >= right_panel_min_y:
                return "right_panel", right_panel_title
            return "", ""

        # 3. Construct persistent UI droplets with container scoping and semantic archetype
        for pt in ui_droplet_points:
            is_fg = has_fg and (wx1 <= pt['x'] <= wx2 and wy1 <= pt['y'] <= wy2)
            cid, ctitle = get_container_info(pt['x'], pt['y'], w, h)

            d_arch = pt.get('archetype')
            d_min = pt.get('is_minimized', False)
            d_hwnd = pt.get('hwnd', 0)

            droplets.append(DiagonalDroplet(
                id=pt['id'],
                center_x=pt['x'],
                center_y=pt['y'],
                category=pt['category'],
                label=pt['label'],
                diag_wave=pt['diag_key'],
                display_subtext=pt['subtext'],
                is_foreground=is_fg,
                confidence=pt.get('confidence', 1.0),
                bbox=pt.get('bbox', (pt['x']-12, pt['y']-12, pt['x']+12, pt['y']+12)),
                attached_label=pt.get('attached_label', ''),
                semantic_role=pt.get('semantic_role', 'widget'),
                archetype=d_arch if d_arch else "button",
                container_id=cid,
                container_title=ctitle,
                is_minimized=d_min,
                hwnd=d_hwnd
            ))

        # Save for next frame's spatial tracking
        self._last_ui_droplets = list(droplets)

        # Sort UI droplets by ID for predictable ordering
        droplets.sort(key=lambda d: d.id)

        # Surface probes get sequential IDs starting above max UI droplet ID
        max_ui_id = max((d.id for d in droplets), default=0)
        surface_probe_points.sort(key=lambda p: (p['diag_key'], p['x']))
        start_probe_id = max_ui_id + 1
        for j, pt in enumerate(surface_probe_points):
            is_fg = has_fg and (wx1 <= pt['x'] <= wx2 and wy1 <= pt['y'] <= wy2)
            droplets.append(DiagonalDroplet(
                id=start_probe_id + j,
                center_x=pt['x'],
                center_y=pt['y'],
                category=pt['category'],
                label=pt['label'],
                diag_wave=pt['diag_key'],
                display_subtext=pt['subtext'],
                is_foreground=is_fg,
                confidence=0.10,
                bbox=(pt['x']-12, pt['y']-12, pt['x']+12, pt['y']+12),
                attached_label='',
                semantic_role='probe'
            ))

        return droplets
