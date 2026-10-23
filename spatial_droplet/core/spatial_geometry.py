"""
Spatial Geometry Engine for Continuous and Sub-Pixel Coordinate Resolution.
Handles continuous sliders, intra-word character glyph interpolation,
punctuation targeting, and directional/relative bounding box offsets.
"""
import re
from typing import Tuple, Optional
from .models import DiagonalDroplet


class SpatialGeometryEngine:
    """
    Generalized mathematical engine for continuous spatial coordinates
    beyond discrete element centers.
    """

    @staticmethod
    def _clean_display_text(label: str) -> str:
        """Strips metadata prefixes/suffixes like [Btn], [Icon], [Dropdown / Value for ...], (Item 1 of 2) from a droplet label."""
        if not label:
            return ""
        t = re.sub(r'^\[(?:Btn|Icon|App|Text|Input|Cell|Tab|Slider)\]\s*', '', label, flags=re.IGNORECASE)
        t = re.sub(r'^\[Dropdown\s*/\s*Value\s+for\s+[\'"]?.*?[\'"]?\]\s*', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*\[Item\s+\d+\s+of\s+\d+.*?\]', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*\((?:First|Last|Row\s+\d+|Spinbox).*?\)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*(?:for|near|right of|left of)\s+[\'"].*?[\'"]', '', t, flags=re.IGNORECASE)
        return t.strip()

    @staticmethod
    def interpolate_slider(
        bbox: Tuple[int, int, int, int],
        ratio: float,
        orientation: str = "horizontal"
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        ratio = max(0.0, min(1.0, float(ratio)))

        if orientation == "horizontal" or (x2 - x1) >= (y2 - y1):
            target_x = x1 + ratio * (x2 - x1)
            target_y = (y1 + y2) / 2.0
        else:
            target_x = (x1 + x2) / 2.0
            target_y = y1 + ratio * (y2 - y1)

        return target_x, target_y

    @classmethod
    def interpolate_character(
        cls,
        word_bbox: Tuple[int, int, int, int],
        raw_label: str,
        target_char: str,
        target_word: Optional[str] = None
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, word_bbox)
        clean_text = cls._clean_display_text(raw_label)
        if not clean_text:
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        char_clean = target_char.strip()
        text_lower = clean_text.lower()
        char_lower = char_clean.lower()

        # If target_word is specified and clean_text is a multi-word line containing target_word,
        # narrow the character search to target_word's span inside clean_text
        search_start = 0
        if target_word and target_word.lower() in text_lower:
            w_idx = text_lower.find(target_word.lower())
            if w_idx != -1:
                search_start = w_idx

        idx = text_lower.find(char_lower, search_start)
        if idx == -1:
            idx = text_lower.find(char_lower)
        if idx == -1:
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        n_chars = max(1, len(clean_text))
        char_width = (x2 - x1) / float(n_chars)
        target_x = x1 + (idx + 0.5) * char_width
        target_y = (y1 + y2) / 2.0
        return target_x, target_y

    @staticmethod
    def resolve_directional_offset(
        bbox: Tuple[int, int, int, int],
        direction: str,
        distance_px: float = 24.0
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        d = direction.lower()

        if "right" in d and "corner" in d:
            return x2 - 4.0, y2 - 4.0
        elif "left" in d and "corner" in d:
            return x1 + 4.0, y1 + 4.0
        elif "right" in d:
            return x2 + distance_px, (y1 + y2) / 2.0
        elif "left" in d:
            return max(0.0, x1 - distance_px), (y1 + y2) / 2.0
        elif "bottom" in d or "below" in d:
            return (x1 + x2) / 2.0, y2 + distance_px
        elif "top" in d or "above" in d:
            return (x1 + x2) / 2.0, max(0.0, y1 - distance_px)

        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @classmethod
    def parse_and_resolve(
        cls,
        instruction: str,
        droplet: DiagonalDroplet
    ) -> Optional[Tuple[float, float]]:
        """
        Analyzes instruction semantics and droplet metadata to apply
        continuous or sub-pixel coordinate resolution if requested.
        """
        if not droplet.bbox or droplet.bbox == (0, 0, 0, 0):
            return None

        inst = instruction.lower()
        x1, y1, x2, y2 = map(float, droplet.bbox)
        bw = x2 - x1

        # Case 1: Sliders and explicit fractions / percentages
        is_slider_intent = any(k in inst for k in ("slider", "zoom", "trackbar", "scroll", "position"))
        if is_slider_intent and bw >= 24.0:
            # Check explicit positional words FIRST (e.g. "Set the zoom level to 100% by click the middle of the zoom slider")
            if "middle of" in inst or "center of" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.50)
            if "left 1/4" in inst or "first quarter" in inst or "left quarter" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.25)
            if "right 1/4" in inst or "three quarter" in inst or "3/4" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.75)
            if "right end" in inst or "maximum" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.96)
            if "left end" in inst or "minimum" in inst or "beginning" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.04)

            pct_match = re.search(r'(\d+)\s*%', inst)
            if pct_match:
                pct_val = float(pct_match.group(1)) / 100.0
                return cls.interpolate_slider(droplet.bbox, pct_val)

            if "right part" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.85)
            if "left part" in inst:
                return cls.interpolate_slider(droplet.bbox, 0.15)

        # Case 2: Punctuation mark targeting (e.g. 'Click on the exclamation mark of "Big news!"')
        if "exclamation mark" in inst and droplet.label and "!" in droplet.label:
            return cls.interpolate_character(droplet.bbox, droplet.label, "!")
        if "question mark" in inst and droplet.label and "?" in droplet.label:
            return cls.interpolate_character(droplet.bbox, droplet.label, "?")

        # Case 3: Sub-character glyph resolution (e.g. 'Click the letter "t" in the word "virtual"')
        char_match = re.search(r'letter\s+["\']?([a-zA-Z0-9])["\']?(?:\s+(?:in|of)\s+(?:the\s+)?(?:word\s+)?["\']?([a-zA-Z0-9_-]+)["\']?)?', inst)
        if char_match and droplet.label:
            target_char = char_match.group(1)
            target_word = char_match.group(2)
            clean_lbl = cls._clean_display_text(droplet.label)
            if target_char.lower() in clean_lbl.lower():
                return cls.interpolate_character(droplet.bbox, droplet.label, target_char, target_word=target_word)

        # Case 4: Directional / corner offsets
        if ("lower right corner" in inst or "bottom right corner" in inst) and "zoom slider" not in inst:
            return cls.resolve_directional_offset(droplet.bbox, "lower_right_corner")
        if "upper left corner" in inst or "top left corner" in inst:
            return cls.resolve_directional_offset(droplet.bbox, "upper_left_corner")

        return None

    @classmethod
    def resolve_direct_geometry(
        cls,
        instruction: str,
        droplets: list,
        sw_screen: int = 1920,
        sh_screen: int = 1080
    ) -> Optional[Tuple[float, float, DiagonalDroplet, str]]:
        """
        Deterministically resolves continuous status-bar zoom slider positions and
        quoted sub-character / punctuation targets across the droplet pool.
        """
        if not droplets:
            return None
        inst = instruction.lower()

        # 1. Status-bar Zoom Slider Continuous Interpolation
        if "zoom slider" in inst:
            slider_cands = []
            for d in droplets:
                if d.category == "surface_probe" or not getattr(d, "bbox", None):
                    continue
                x1, y1, x2, y2 = map(float, d.bbox)
                bw, bh = x2 - x1, y2 - y1
                lbl = d.label.lower()
                if (
                    d.center_y >= sh_screen * 0.90
                    and d.center_x >= sw_screen * 0.65
                    and 55.0 <= bw <= 185.0
                    and bh <= 28.0
                    and ("slider" in lbl or "trackbar" in lbl)
                ):
                    slider_cands.append((bw, d))
            if slider_cands:
                # Prefer the full trackbar bounding box
                slider_cands.sort(key=lambda item: item[0], reverse=True)
                best_slider = slider_cands[0][1]
                coords = cls.parse_and_resolve(instruction, best_slider)
                if coords:
                    return coords[0], coords[1], best_slider, f"Zoom Slider Direct Geometry ({coords[0]:.1f}, {coords[1]:.1f})"

        # 2. Sub-character Punctuation Mark Targeting (e.g. 'Click on the exclamation mark of "Big news!"')
        if "exclamation mark" in inst or "question mark" in inst:
            punct = "!" if "exclamation mark" in inst else "?"
            quoted = re.findall(r'["\']([^"\']+)["\']', instruction)
            q_words = set(re.findall(r'[a-zA-Z0-9]+', " ".join(quoted).lower()))
            punct_cands = []
            for d in droplets:
                if d.category == "surface_probe" or d.label.startswith("[Icon]") or not getattr(d, "bbox", None):
                    continue
                clean_lbl = cls._clean_display_text(d.label)
                if punct in clean_lbl:
                    lbl_words = set(re.findall(r'[a-zA-Z0-9]+', clean_lbl.lower()))
                    overlap = len(q_words & lbl_words) if q_words else 1
                    if overlap > 0:
                        punct_cands.append((overlap, d))
            if punct_cands:
                punct_cands.sort(key=lambda item: item[0], reverse=True)
                best_d = punct_cands[0][1]
                tx, ty = cls.interpolate_character(best_d.bbox, best_d.label, punct)
                return tx, ty, best_d, f"Sub-character Punctuation '{punct}' in '{best_d.label}'"

        # 3. Quoted Word Sub-Character Letter Targeting (e.g. 'Click the letter "t" in the word "virtual"')
        char_match = re.search(
            r'letter\s+["\']([a-zA-Z0-9])["\']\s+(?:in|of)\s+(?:the\s+)?(?:word\s+)?["\']([a-zA-Z0-9_-]+)["\']',
            instruction,
            flags=re.IGNORECASE,
        )
        if char_match:
            t_char = char_match.group(1)
            t_word = char_match.group(2)
            in_main_canvas = any(k in inst for k in ("in the slide", "on the slide", "in the document"))
            word_cands = []
            for d in droplets:
                if d.category == "surface_probe" or d.label.startswith("[Icon]") or not getattr(d, "bbox", None):
                    continue
                if in_main_canvas and d.center_x < sw_screen * 0.18:
                    continue
                clean_lbl = cls._clean_display_text(d.label)
                clean_words = re.findall(r'[a-zA-Z0-9_-]+', clean_lbl.lower())
                if t_word.lower() in clean_words and t_char.lower() in clean_lbl.lower():
                    exact_match = (clean_lbl.strip().lower() == t_word.lower())
                    word_cands.append((exact_match, d))
            if word_cands:
                word_cands.sort(key=lambda item: item[0], reverse=True)
                best_d = word_cands[0][1]
                tx, ty = cls.interpolate_character(best_d.bbox, best_d.label, t_char, target_word=t_word)
                return tx, ty, best_d, f"Sub-character '{t_char}' in '{t_word}'"

        return None

