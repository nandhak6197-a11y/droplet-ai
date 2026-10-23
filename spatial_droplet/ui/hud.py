"""
Spatial Droplet HUD (Phase 1-8).
Transparent, click-through Tkinter overlay rendering Set-of-Mark reticles,
active window focus frames, and droplet coordinates with zero display lag.
"""

import tkinter as tk
import ctypes
from typing import List, Optional, Tuple

from ..core.models import DiagonalDroplet, WindowContext, register_own_hwnd


class FullBleedHUD:
    """
    Zero-margin, click-through transparent HUD overlay.
    Renders high-visibility Set-of-Mark precision reticles and target locks.
    """

    def __init__(self, root: tk.Tk, screen_w: int, screen_h: int):
        self.root = root
        self.sw = screen_w
        self.sh = screen_h
        self.is_visible = True
        self.target_reticle = None

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.sw}x{self.sh}+0+0")
        self.win.lift()
        self.win.wm_attributes("-topmost", True)
        self.win.wm_attributes("-transparentcolor", "#010101")
        self.win.config(bg="#010101")

        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        self._hwnd = hwnd
        register_own_hwnd(self.win.winfo_id())
        register_own_hwnd(hwnd)
        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_LAYERED = 0x00080000
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED)

        self.canvas = tk.Canvas(self.win, bg="#010101", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def clear(self):
        self.canvas.delete("all")
        self.target_reticle = None

    def hide(self):
        try:
            self.win.withdraw()
        except Exception:
            pass

    def show(self):
        try:
            if self.is_visible:
                self.win.deiconify()
                self.win.lift()
                if hasattr(self, '_hwnd') and self._hwnd:
                    # HWND_TOPMOST (-1), SWP_NOSIZE (1) | SWP_NOMOVE (2) | SWP_NOACTIVATE (16) | SWP_SHOWWINDOW (64)
                    ctypes.windll.user32.SetWindowPos(
                        self._hwnd, -1, 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0010 | 0x0040
                    )
        except Exception:
            pass

    def set_visible(self, visible: bool):
        self.is_visible = visible
        if visible:
            self.show()
        else:
            self.hide()

    def draw_star(self, cx: int, cy: int, r_out: float, r_in: float, color: str):
        cx = max(2, min(self.sw - 2, cx))
        cy = max(2, min(self.sh - 2, cy))
        pts = [
            cx, cy - r_out,
            cx + r_in, cy - r_in,
            cx + r_out, cy,
            cx + r_in, cy + r_in,
            cx, cy + r_out,
            cx - r_in, cy + r_in,
            cx - r_out, cy,
            cx - r_in, cy - r_in
        ]
        return self.canvas.create_polygon(pts, fill=color, outline="")

    def render(
        self,
        droplets: List[DiagonalDroplet],
        win_ctx: Optional[WindowContext] = None,
        copilot_rect: Optional[Tuple[int, int, int, int]] = None,
        show_surface_probes: bool = False
    ):
        if not self.is_visible:
            return
        self.canvas.delete("all")

        # 0. Active Window Focus Frame (Only for real active application windows, NEVER for Desktop)
        user32 = ctypes.windll.user32
        desk_hwnd = user32.GetDesktopWindow() if hasattr(user32, 'GetDesktopWindow') else 0
        is_desk = (
            (not win_ctx) or
            (not win_ctx.is_valid) or
            (win_ctx.title in ("Desktop", "Program Manager")) or
            (getattr(win_ctx, 'class_name', '') in ("Desktop", "Progman", "WorkerW")) or
            (win_ctx.hwnd in (0, desk_hwnd))
        )
        if not is_desk and win_ctx.rect:
            wx1, wy1, wx2, wy2 = win_ctx.rect
            cx1 = max(0, min(self.sw, wx1))
            cy1 = max(0, min(self.sh, wy1))
            cx2 = max(0, min(self.sw, wx2))
            cy2 = max(0, min(self.sh, wy2))
            if cx2 > cx1 + 50 and cy2 > cy1 + 50:
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00e5ff", width=1, dash=(4, 4))
                tag_txt = f" ⛶ FOCUS: {win_ctx.title[:35]} "
                tw = len(tag_txt) * 7 + 10
                by1 = max(4, cy1 - 18)
                by2 = max(20, cy1 - 2)
                self.canvas.create_rectangle(cx1 + 8, by1, cx1 + 8 + tw, by2,
                                             fill="#090d13", outline="#00e5ff", width=1)
                self.canvas.create_text(cx1 + 12, (by1 + by2) // 2, text=tag_txt, fill="#00e5ff",
                                        font=("Segoe UI", 8, "bold"), anchor="w")

        # Parse copilot exclusion rects
        copilot_rects = []
        if copilot_rect:
            if isinstance(copilot_rect, list):
                copilot_rects = [cr for cr in copilot_rect if len(cr) == 4]
            elif isinstance(copilot_rect, tuple):
                if len(copilot_rect) == 4 and all(isinstance(v, (int, float)) for v in copilot_rect):
                    copilot_rects = [copilot_rect]
                elif len(copilot_rect) > 0 and isinstance(copilot_rect[0], (tuple, list)):
                    copilot_rects = [cr for cr in copilot_rect if len(cr) == 4]

        # 1. Surface Stars (Only drawn in Full Mesh mode, subtle slate-cyan, excluded from copilot)
        if show_surface_probes:
            for d in droplets:
                if d.category == "surface_probe":
                    if copilot_rects and any(cr[0] - 6 <= d.center_x <= cr[2] + 6 and cr[1] - 6 <= d.center_y <= cr[3] + 6 for cr in copilot_rects):
                        continue
                    star_col = "#64b5f6" if d.is_foreground else "#29435c"
                    self.draw_star(d.center_x, d.center_y, r_out=2.2, r_in=0.9, color=star_col)

        # 2. Active UI Landmarks
        for d in droplets:
            if d.category in ("interactive_app", "interactive_widget"):
                cx, cy = d.center_x, d.center_y
                if copilot_rects and any(cr[0] - 10 <= cx <= cr[2] + 10 and cr[1] - 10 <= cy <= cr[3] + 10 for cr in copilot_rects):
                    continue

                is_app = (d.category == "interactive_app")
                is_icon = ("[Icon]" in d.label or "[Btn]" in d.label)
                is_nav = any(k in d.label.lower() for k in ("close", "minimize", "maximize", "back", "forward", "refresh", "menu", "taskbar"))

                # Context-Color Visual Taxonomy:
                is_active_element = d.is_foreground or is_desk
                if is_active_element:
                    badge_bg = getattr(d, 'semantic_color', '#00e676')
                    arch = getattr(d, 'archetype', 'button')
                    if arch in ("logo", "label"):
                        text_col = "#ffffff"
                        ring_col = "#1a0022" if arch == "logo" else "#263238"
                    else:
                        text_col = "#000000"
                        ring_col = "#111111"
                else:
                    badge_bg = "#37474f" if is_app else "#263238"
                    text_col = "#eceff1"
                    ring_col = "#000000"

                # Extract and normalize bounding box geometry (Jev Box)
                raw_box = getattr(d, 'bbox', None)
                if raw_box and len(raw_box) == 4:
                    rbx1, rby1, rbx2, rby2 = raw_box
                    rbx1, rbx2 = min(rbx1, rbx2), max(rbx1, rbx2)
                    rby1, rby2 = min(rby1, rby2), max(rby1, rby2)
                    if (rbx2 - rbx1) >= 4 and (rby2 - rby1) >= 4:
                        bx1, by1, bx2, by2 = rbx1, rby1, rbx2, rby2
                    else:
                        bx1, by1, bx2, by2 = cx - 18, cy - 10, cx + 18, cy + 10
                else:
                    bx1, by1, bx2, by2 = cx - 18, cy - 10, cx + 18, cy + 10

                # Bounding box clamped cleanly to display boundaries (no artificial outward expansion)
                bx1 = max(0, min(self.sw - 2, bx1))
                by1 = max(0, min(self.sh - 2, by1))
                bx2 = max(bx1 + 2, min(self.sw, bx2))
                by2 = max(by1 + 2, min(self.sh, by2))
                bw = bx2 - bx1
                bh = by2 - by1

                # 1. Sleek Box-Shaped Spatial Droplet Frame (Jev Box)
                if is_active_element:
                    # Crisp outer bounding frame
                    self.canvas.create_rectangle(bx1, by1, bx2, by2, outline=badge_bg, width=1)
                    # Corner Accent Ticks (tactical corner brackets for sub-pixel boundary clarity)
                    t_len = min(6, max(3, bw // 4, bh // 4))
                    # Top-Left
                    self.canvas.create_line(bx1, by1, bx1 + t_len, by1, fill=badge_bg, width=2)
                    self.canvas.create_line(bx1, by1, bx1, by1 + t_len, fill=badge_bg, width=2)
                    # Top-Right
                    self.canvas.create_line(bx2, by1, bx2 - t_len, by1, fill=badge_bg, width=2)
                    self.canvas.create_line(bx2, by1, bx2, by1 + t_len, fill=badge_bg, width=2)
                    # Bottom-Left
                    self.canvas.create_line(bx1, by2, bx1 + t_len, by2, fill=badge_bg, width=2)
                    self.canvas.create_line(bx1, by2, bx1, by2 - t_len, fill=badge_bg, width=2)
                    # Bottom-Right
                    self.canvas.create_line(bx2, by2, bx2 - t_len, by2, fill=badge_bg, width=2)
                    self.canvas.create_line(bx2, by2, bx2, by2 - t_len, fill=badge_bg, width=2)
                else:
                    # Inactive background controls: subtle muted dash
                    self.canvas.create_rectangle(bx1, by1, bx2, by2, outline="#37474f", width=1, dash=(3, 3))

                # 2. Precision Non-Occluding Tag Badge:
                # For compact controls (icons, buttons, arrows), dock OUTSIDE above the top edge
                # so the interior icon glyph and click target are 100% visible and unoccluded.
                tid = str(d.id)
                pw = max(16, len(tid) * 7 + 6)
                ph = 13
                if bh <= 42 or bw <= 52:
                    tag_y1 = by1 - ph - 2
                    tag_y2 = by1 - 2
                    tag_x1 = max(0, min(self.sw - pw, bx1))
                    tag_x2 = tag_x1 + pw
                    if tag_y1 < 2:  # Flipped below if pinned against top screen border
                        tag_y1 = min(self.sh - ph - 1, by2 + 2)
                        tag_y2 = tag_y1 + ph
                else:
                    # For large cards/containers, dock inside top-left corner
                    tag_x1 = bx1 + 2
                    tag_y1 = by1 + 2
                    tag_x2 = tag_x1 + pw
                    tag_y2 = tag_y1 + ph

                # Drop shadow
                self.canvas.create_rectangle(tag_x1 + 1, tag_y1 + 1, tag_x2 + 1, tag_y2 + 1,
                                             fill="#000000", outline="")
                # Badge pill
                self.canvas.create_rectangle(tag_x1, tag_y1, tag_x2, tag_y2,
                                             fill=badge_bg, outline=ring_col, width=1)
                # Centered number text
                self.canvas.create_text((tag_x1 + tag_x2) // 2, (tag_y1 + tag_y2) // 2,
                                        text=tid, fill=text_col, font=("Consolas", 8, "bold"))

                # 3. High-Contrast Precision Centroid Pinpoint at exact (cx, cy)
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#000000", outline="")
                self.canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=badge_bg, outline="#ffffff" if is_active_element else "#000000", width=1)
                self.canvas.create_rectangle(cx, cy, cx + 1, cy + 1, fill="#ffffff", outline="")

        if self.is_visible:
            try:
                self.win.lift()
                if hasattr(self, '_hwnd') and self._hwnd:
                    ctypes.windll.user32.SetWindowPos(
                        self._hwnd, -1, 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0010 | 0x0040
                    )
            except Exception:
                pass

    def highlight_target(self, target_node: DiagonalDroplet):
        if self.target_reticle:
            for item in self.target_reticle:
                try:
                    self.canvas.delete(item)
                except Exception:
                    pass
            self.target_reticle = None

        raw_box = getattr(target_node, 'bbox', None)
        if raw_box and len(raw_box) == 4:
            rbx1, rby1, rbx2, rby2 = raw_box
            rbx1, rbx2 = min(rbx1, rbx2), max(rbx1, rbx2)
            rby1, rby2 = min(rby1, rby2), max(rby1, rby2)
            if (rbx2 - rbx1) >= 4 and (rby2 - rby1) >= 4:
                bx1, by1, bx2, by2 = rbx1, rby1, rbx2, rby2
                tx = (bx1 + bx2) // 2
                if getattr(target_node, 'category', '') == "interactive_app" and (by2 - by1) > 38:
                    ty = by1 + min(24, max(14, (by2 - by1) // 3))
                else:
                    ty = (by1 + by2) // 2
            else:
                tx, ty = target_node.center_x, target_node.center_y
                bx1, by1, bx2, by2 = tx - 24, ty - 16, tx + 24, ty + 16
        else:
            tx, ty = target_node.center_x, target_node.center_y
            bx1, by1, bx2, by2 = tx - 24, ty - 16, tx + 24, ty + 16

        lx1 = max(0, bx1 - 5)
        ly1 = max(0, by1 - 4)
        lx2 = min(self.sw, bx2 + 5)
        ly2 = min(self.sh, by2 + 4)

        items = []
        # Glowing neon dual-border lock around the target box
        items.append(self.canvas.create_rectangle(lx1, ly1, lx2, ly2, outline="#ff1744", width=2.5, tags="target_reticle"))
        items.append(self.canvas.create_rectangle(lx1 - 2, ly1 - 2, lx2 + 2, ly2 + 2, outline="#00e5ff", width=1, dash=(3, 3), tags="target_reticle"))

        # Heavy tactical corner brackets
        klen = max(6, min(14, (lx2 - lx1) // 3, (ly2 - ly1) // 3))
        # Top-Left
        items.append(self.canvas.create_line(lx1, ly1, lx1 + klen, ly1, fill="#00e5ff", width=3, tags="target_reticle"))
        items.append(self.canvas.create_line(lx1, ly1, lx1, ly1 + klen, fill="#00e5ff", width=3, tags="target_reticle"))
        # Top-Right
        items.append(self.canvas.create_line(lx2, ly1, lx2 - klen, ly1, fill="#00e5ff", width=3, tags="target_reticle"))
        items.append(self.canvas.create_line(lx2, ly1, lx2, ly1 + klen, fill="#00e5ff", width=3, tags="target_reticle"))
        # Bottom-Left
        items.append(self.canvas.create_line(lx1, ly2, lx1 + klen, ly2, fill="#00e5ff", width=3, tags="target_reticle"))
        items.append(self.canvas.create_line(lx1, ly2, lx1, ly2 - klen, fill="#00e5ff", width=3, tags="target_reticle"))
        # Bottom-Right
        items.append(self.canvas.create_line(lx2, ly2, lx2 - klen, ly2, fill="#00e5ff", width=3, tags="target_reticle"))
        items.append(self.canvas.create_line(lx2, ly2, lx2, ly2 - klen, fill="#00e5ff", width=3, tags="target_reticle"))

        # Subtle full-axis tactical guide crosshairs
        items.append(self.canvas.create_line(0, ty, lx1, ty, fill="#ff1744", width=1, dash=(2, 4), tags="target_reticle"))
        items.append(self.canvas.create_line(lx2, ty, self.sw, ty, fill="#ff1744", width=1, dash=(2, 4), tags="target_reticle"))
        items.append(self.canvas.create_line(tx, 0, tx, ly1, fill="#ff1744", width=1, dash=(2, 4), tags="target_reticle"))
        items.append(self.canvas.create_line(tx, ly2, tx, self.sh, fill="#ff1744", width=1, dash=(2, 4), tags="target_reticle"))

        # Sharp laser crosshair ticks at exact pinpoint (tx, ty)
        items.append(self.canvas.create_line(tx - 11, ty, tx - 3, ty, fill="#ff1744", width=2, tags="target_reticle"))
        items.append(self.canvas.create_line(tx + 3, ty, tx + 11, ty, fill="#ff1744", width=2, tags="target_reticle"))
        items.append(self.canvas.create_line(tx, ty - 11, tx, ty - 3, fill="#ff1744", width=2, tags="target_reticle"))
        items.append(self.canvas.create_line(tx, ty + 3, tx, ty + 11, fill="#ff1744", width=2, tags="target_reticle"))

        # Center laser pinpoint dot
        items.append(self.canvas.create_oval(tx - 3, ty - 3, tx + 3, ty + 3, fill="#00e5ff", outline="#ffffff", width=1, tags="target_reticle"))

        # Tactical Target Readout Banner
        clean_lbl = target_node.label[:32]
        tag_text = f"🎯 LOCKED: [{target_node.id}] ({tx}, {ty}) • {clean_lbl}"
        tag_y = ly2 + 14 if ly2 + 25 < self.sh else ly1 - 14
        half_w = len(tag_text) * 4 + 10
        items.append(self.canvas.create_rectangle(tx - half_w, tag_y - 9, tx + half_w, tag_y + 9,
                                                  fill="#090d13", outline="#ff1744", width=1, tags="target_reticle"))
        items.append(self.canvas.create_text(tx, tag_y, text=tag_text, fill="#00e5ff",
                                             font=("Segoe UI", 9, "bold"), tags="target_reticle"))

        self.target_reticle = tuple(items)

    def clear_target_reticle(self):
        try:
            self.canvas.delete("target_reticle")
        except Exception:
            pass
        if self.target_reticle:
            for item in self.target_reticle:
                try:
                    self.canvas.delete(item)
                except Exception:
                    pass
            self.target_reticle = None

    def trigger_ripple(self, x: int, y: int):
        impact_items = []
        # Center tactical impact crosshair and pinpoint
        p_dot = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ff1744", outline="#00e5ff", width=1.5)
        h_l = self.canvas.create_line(x - 12, y, x + 12, y, fill="#00e5ff", width=1.5)
        v_l = self.canvas.create_line(x, y - 12, x, y + 12, fill="#00e5ff", width=1.5)
        impact_items.extend([p_dot, h_l, v_l])

        def step(radius):
            if radius <= 38:
                r_circ = self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                                outline="#00e5ff" if radius % 12 == 0 else "#ff1744",
                                                width=max(1, int(3 - radius / 14)))
                impact_items.append(r_circ)
                self.win.after(40, lambda: self.canvas.delete(r_circ))
                self.win.after(35, lambda: step(radius + 7))
            else:
                self.win.after(160, lambda: [self.canvas.delete(it) for it in impact_items])

        step(6)

    def toggle(self):
        self.is_visible = not self.is_visible
        if self.is_visible:
            self.show()
        else:
            self.hide()
