"""
Tabular Grid Resolver for Spreadsheet & Data Grid Interfaces.
Detects column headers (A, B, C...), semantic table headers, and row numbers (1, 2, 3...)
to resolve cell coordinates like 'B21', 'E7', 'row 25', 'column F', 'product of row 12',
'second column of the 31st row', or row/column entity intersections.
"""
import re
import math
from typing import List, Tuple, Optional, Dict
from ..core.models import DiagonalDroplet


class TabularGridResolver:
    """
    Generalized 2D Cartesian grid intersector for spreadsheet and tabular interfaces.
    """

    @classmethod
    def find_grid_axes(
        cls,
        droplets: List[DiagonalDroplet]
    ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]]]:
        """
        Scans droplets for spreadsheet column letters and row numbers.
        Returns:
            col_map: { 'A': (cx, cy, width), 'B': (cx, cy, width), ... }
            row_map: { 1: (cx, cy, height), 2: (cx, cy, height), ... }
        """
        col_map: Dict[str, Tuple[float, float, float]] = {}
        row_map: Dict[int, Tuple[float, float, float]] = {}

        letter_candidates = []
        num_candidates = []

        for d in droplets:
            if d.category == "surface_probe":
                continue
            raw_lbl = re.sub(r'\[.*?\]|\(.*?\)', '', d.label).strip()
            # Column headers: single uppercase letters A-Z in upper half of screen
            if re.match(r'^[A-Z]$', raw_lbl) and 120 <= d.center_y <= 420:
                letter_candidates.append((d, raw_lbl))
            # Row headers: integers 1-999 in left gutter (x <= 140) and below toolbar (y >= 160)
            elif re.match(r'^[0-9]{1,3}$', raw_lbl) and d.center_x <= 140 and d.center_y >= 160:
                val = int(raw_lbl)
                if 1 <= val <= 500:
                    num_candidates.append((d, val))

        # Group letters by Y band (within 14px) to find the primary spreadsheet column header bar
        if letter_candidates:
            y_bands: Dict[int, List[Tuple[DiagonalDroplet, str]]] = {}
            for d, col_name in letter_candidates:
                y_key = round(d.center_y / 14.0) * 14
                y_bands.setdefault(y_key, []).append((d, col_name))
            best_y_band = max(y_bands.values(), key=lambda items: len({c for _, c in items}))
            unique_cols = {c for _, c in best_y_band}
            if len(unique_cols) >= 2:
                for d, col_name in sorted(best_y_band, key=lambda x: x[0].center_x):
                    w = (d.bbox[2] - d.bbox[0]) if d.bbox and d.bbox != (0, 0, 0, 0) else 64.0
                    if col_name not in col_map:
                        col_map[col_name] = (float(d.center_x), float(d.center_y), float(w))

        # Group numbers by X band (within 18px) in the left margin to find the row index gutter
        if num_candidates:
            x_bands: Dict[int, List[Tuple[DiagonalDroplet, int]]] = {}
            for d, row_idx in num_candidates:
                x_key = round(d.center_x / 18.0) * 18
                x_bands.setdefault(x_key, []).append((d, row_idx))
            best_x_band = max(x_bands.values(), key=lambda items: len({r for _, r in items}))
            unique_rows = {r for _, r in best_x_band}
            if len(unique_rows) >= 2:
                for d, row_idx in sorted(best_x_band, key=lambda x: x[0].center_y):
                    h = (d.bbox[3] - d.bbox[1]) if d.bbox and d.bbox != (0, 0, 0, 0) else 20.0
                    if row_idx not in row_map:
                        row_map[row_idx] = (float(d.center_x), float(d.center_y), float(h))

        # If row_map is present (spreadsheet confirmed) but OCR detected <2 column letters,
        # anchor column headers relative to the detected row gutter and any single detected letter
        if row_map and len(col_map) < 2:
            sorted_r = sorted(row_map.items(), key=lambda kv: kv[0])
            r_first, (rx_gutter, ry_first, rh_first) = sorted_r[0]
            r_steps = [
                (sorted_r[i + 1][1][1] - sorted_r[i][1][1]) / max(1, sorted_r[i + 1][0] - sorted_r[i][0])
                for i in range(len(sorted_r) - 1)
                if sorted_r[i + 1][0] > sorted_r[i][0]
            ]
            row_step = (sum(r_steps) / len(r_steps)) if r_steps else max(18.0, rh_first)
            col_hdr_y = ry_first - (r_first * row_step)
            if letter_candidates:
                # Use the detected letter closest to col_hdr_y as anchor
                best_lc, best_lchar = min(letter_candidates, key=lambda item: abs(item[0].center_y - col_hdr_y))
                col_step = max(72.0, float(best_lc.bbox[2] - best_lc.bbox[0]) if best_lc.bbox else 85.0)
                for idx_c, ch in enumerate("ABCDEFGH"):
                    cx_val = best_lc.center_x + (ord(ch) - ord(best_lchar)) * col_step
                    col_map[ch] = (float(cx_val), float(best_lc.center_y), float(col_step))
            else:
                col_step = 86.0
                x_col_a = rx_gutter + 58.0
                for idx_c, ch in enumerate("ABCDEFGH"):
                    col_map[ch] = (float(x_col_a + idx_c * col_step), float(col_hdr_y), float(col_step))

        return col_map, row_map

    @classmethod
    def _interpolate_col_x(cls, col_name: str, col_map: Dict[str, Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
        if col_name in col_map:
            return col_map[col_name][0], col_map[col_name][1]
        if len(col_map) >= 2:
            sorted_cols = sorted(col_map.items(), key=lambda kv: ord(kv[0]))
            steps = []
            for i in range(len(sorted_cols) - 1):
                c1, (x1, y1, _) = sorted_cols[i]
                c2, (x2, y2, _) = sorted_cols[i + 1]
                dc = ord(c2) - ord(c1)
                if dc > 0 and (x2 > x1):
                    steps.append((x2 - x1) / dc)
            if steps:
                avg_step = sum(steps) / len(steps)
                ref_c, (ref_x, ref_y, _) = sorted_cols[0]
                return ref_x + (ord(col_name) - ord(ref_c)) * avg_step, ref_y
        return None

    @classmethod
    def _interpolate_row_y(cls, row_num: int, row_map: Dict[int, Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
        if row_num in row_map:
            return row_map[row_num][0], row_map[row_num][1]
        if len(row_map) >= 2:
            sorted_rows = sorted(row_map.items(), key=lambda kv: kv[0])
            steps = []
            for i in range(len(sorted_rows) - 1):
                r1, (x1, y1, _) = sorted_rows[i]
                r2, (x2, y2, _) = sorted_rows[i + 1]
                dr = r2 - r1
                if dr > 0 and (y2 > y1):
                    steps.append((y2 - y1) / dr)
            if steps:
                avg_step = sum(steps) / len(steps)
                closest_r, (ref_x, ref_y, _) = min(sorted_rows, key=lambda kv: abs(kv[0] - row_num))
                return ref_x, ref_y + (row_num - closest_r) * avg_step
        return None

    @classmethod
    def resolve_cell_intent(
        cls,
        instruction: str,
        droplets: List[DiagonalDroplet]
    ) -> Optional[Tuple[float, float, str]]:
        """
        Resolves spreadsheet cell & header references:
          1. Explicit A1-style cell coordinates ('B21 cell', 'E7 cell')
          2. Ordinal column + ordinal row ('second column of the 31st row')
          3. Semantic column + row number ('product of row 12')
          4. Semantic Row Entity in Semantic Column Entity ('Select the Food Bill in August 2020')
          5. Pure column header selection ('Click to select column F')
          6. Pure row header selection ('Click to select row 12', 'Select the row 25')
        """
        inst = instruction.strip()
        inst_lower = inst.lower()
        col_map, row_map = cls.find_grid_axes(droplets)

        if not col_map and not row_map:
            return None

        # 1. Explicit Cell Coordinate (e.g. 'B21', 'E7')
        cell_match = re.search(r'\b([A-Z])([0-9]{1,3})\b', inst)
        if cell_match and ("cell" in inst_lower or col_map):
            col_name = cell_match.group(1).upper()
            row_num = int(cell_match.group(2))
            col_res = cls._interpolate_col_x(col_name, col_map)
            row_res = cls._interpolate_row_y(row_num, row_map)
            if col_res and row_res:
                cx = col_res[0]
                ry = row_res[1]
                return cx, ry, f"Cell {col_name}{row_num} ({cx:.1f}, {ry:.1f})"

        # 2. Ordinal Column + Ordinal Row (e.g. 'second column of the 31st row')
        ord_col_map = {"first": "A", "1st": "A", "second": "B", "2nd": "B", "third": "C", "3rd": "C", "fourth": "D", "4th": "D", "fifth": "E", "5th": "E", "sixth": "F", "6th": "F"}
        m_ord_rc = re.search(r'\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th)\s+column\s+of\s+the\s+(\d+)(?:st|nd|rd|th)?\s+row', inst_lower)
        if m_ord_rc:
            col_name = ord_col_map.get(m_ord_rc.group(1), "B")
            row_num = int(m_ord_rc.group(2))
            col_res = cls._interpolate_col_x(col_name, col_map)
            row_res = cls._interpolate_row_y(row_num, row_map)
            if col_res and row_res:
                return col_res[0], row_res[1], f"Cell {col_name}{row_num} via ordinal row/col"

        # 3. Semantic Column Name + Row Number (e.g. 'Click on the product of row 12')
        m_sem_row = re.search(r'(?:click|select)\s+(?:on\s+)?the\s+([a-zA-Z]+)\s+of\s+row\s+(\d+)\b', inst_lower)
        if m_sem_row:
            col_keyword = m_sem_row.group(1).lower()
            row_num = int(m_sem_row.group(2))
            row_res = cls._interpolate_row_y(row_num, row_map)
            if row_res:
                _, ry = row_res
                header_cands = [
                    d for d in droplets
                    if d.category != "surface_probe" and col_keyword in d.label.lower() and d.center_y < ry
                ]
                if header_cands:
                    best_hdr = min(header_cands, key=lambda d: d.center_y)
                    return float(best_hdr.center_x), float(ry), f"Semantic Cell ({col_keyword}, row {row_num})"

        # 4. Semantic Row Entity in Semantic Column Entity (e.g. 'Select the Food Bill in August 2020')
        m_ent_in = re.search(r'(?:select|click(?:\s+on)?)\s+the\s+([a-z0-9\s]{3,25}?)\s+in\s+([a-z0-9\s]{3,25})$', inst_lower)
        if m_ent_in and row_map:
            ent1 = m_ent_in.group(1).strip()
            ent2 = m_ent_in.group(2).strip()
            e1_cands = [
                d for d in droplets
                if d.category != "surface_probe"
                and ent1 in (f"{d.label} {getattr(d, 'attached_label', '')}").lower()
            ]
            # Prefer pure text header labels over auxiliary [Icon] droplets
            e1_cands.sort(key=lambda d: (d.label.startswith("[Icon]"), d.center_y))

            month_map = {
                "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
                "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
                "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
                "november": 11, "nov": 11, "december": 12, "dec": 12,
            }
            ent2_words = ent2.split()
            matched_month = next((month_map[w] for w in ent2_words if w in month_map), None)
            matched_year = next((w for w in ent2_words if re.match(r'^20\d{2}$', w)), None)

            if e1_cands and matched_month is not None and matched_year is not None:
                col_hdr = e1_cands[0]
                year_cands = [
                    d for d in droplets
                    if d.category != "surface_probe"
                    and matched_year in (f"{d.label} {getattr(d, 'attached_label', '')}").lower()
                ]
                year_cands.sort(key=lambda d: (d.label.startswith("[Icon]"), d.center_x))
                if year_cands:
                    y_year = float(year_cands[0].center_y)
                    # Estimate row height from row_map (default 18.0px in Calc/Excel)
                    sorted_r = sorted(row_map.items(), key=lambda x: x[0])
                    deltas = [
                        (sorted_r[i + 1][1][1] - sorted_r[i][1][1]) / (sorted_r[i + 1][0] - sorted_r[i][0])
                        for i in range(len(sorted_r) - 1)
                        if 0 < (sorted_r[i + 1][0] - sorted_r[i][0]) <= 10
                    ]
                    row_step = float(sorted(deltas)[len(deltas) // 2]) if deltas else 18.0
                    row_step = max(16.0, min(22.0, row_step))
                    target_ry = y_year + (matched_month + 1) * row_step
                    return float(col_hdr.center_x), float(target_ry), f"2D Month-Year Cell ({ent1} @ {ent2})"

            e2_cands = [d for d in droplets if d.category != "surface_probe" and any(w in d.label.lower() for w in ent2.split() if len(w) >= 3)]
            if e1_cands and e2_cands:
                d1 = e1_cands[0]
                d2 = e2_cands[0]
                if abs(d1.center_x - d2.center_x) > 30 and abs(d1.center_y - d2.center_y) > 15:
                    if d1.center_x < d2.center_x:
                        return float(d2.center_x), float(d1.center_y), f"2D Entity Intersection ({ent1} @ {ent2})"
                    else:
                        return float(d1.center_x), float(d2.center_y), f"2D Entity Intersection ({ent2} @ {ent1})"

        # 5. Pure Column Header Selection (e.g. 'Click to select column F')
        col_only_match = re.search(r'(?:select|click(?:\s+on)?)\s+(?:the\s+)?column\s+([A-Z])\b', inst, re.IGNORECASE)
        if col_only_match:
            col_char = col_only_match.group(1).upper()
            col_res = cls._interpolate_col_x(col_char, col_map)
            if col_res:
                cx, cy = col_res
                return float(cx), float(cy), f"Column Header {col_char}"

        # 6. Pure Row Header Selection (e.g. 'Click to select row 12', 'Select the row 25')
        if " of row " not in inst_lower and " in row " not in inst_lower:
            row_only_match = re.search(r'(?:select|click(?:\s+on)?)\s+(?:the\s+)?row\s+([0-9]{1,3})\b', inst_lower)
            if row_only_match:
                row_int = int(row_only_match.group(1))
                row_res = cls._interpolate_row_y(row_int, row_map)
                if row_res:
                    rx, ry = row_res
                    return float(rx), float(ry), f"Row Header {row_int}"

        return None
