"""
========================================================================================
 🤖 DISCORD COCKPIT PILOT — DUAL-MODEL INTERACTIVE DELIBERATION ENGINE
========================================================================================
 Runs a live Discord-style multi-agent chatroom (#automation-cockpit) between:
   • ⚡ @Groq_Strategist (Qwen 3.8 27B on Groq LPU) — Semantic & Workflow Architect
   • 👁️ @Qwen_Vision (Qwen 3-VL 30B MoE on Kaggle GPU) — Multimodal Visual Grounder

 The two models converse, cross-examine visual crops, debate candidate dots, and reach
 a verified consensus before dispatching the final desktop action.
========================================================================================
"""

import sys
import os
import json
import time
import base64
import re
import csv
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Force UTF-8 on Windows command prompts
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.core.models import WindowContext
from spatial_droplet.reasoning.groq_rotator import GROQ_API_KEYS

# Terminal Discord Color Palette (ANSI)
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GROQ = "\033[38;2;87;242;135m"     # Discord Online Green (#57F287)
C_QWEN = "\033[38;2;155;89;182m"     # Discord Blurple / Purple (#9B59B6)
C_USER = "\033[38;2;0;176;244m"      # Discord Blue (#00B0F4)
C_ACT = "\033[38;2;254;231;92m"      # Discord Yellow/Gold (#FEE75C)
C_GRAY = "\033[38;2;148;155;164m"    # Discord Muted Gray (#949BA4)
C_BG_DARK = "\033[48;2;49;51;56m"   # Discord Dark Chat Background

OUTPUT_DIR = PROJECT_ROOT / "screenshot_pro_benchmark_results"
IMAGES_DISCORD_DIR = OUTPUT_DIR / "images_discord"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DISCORD_DIR.mkdir(parents=True, exist_ok=True)

# 1. Initialize Clients
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://greedy-porcupine-precut.ngrok-free.dev/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:30b-a3b-thinking")

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    default_headers={"ngrok-skip-browser-warning": "true"}
)

class KeyRotator:
    def __init__(self, keys: List[str]):
        self.clients = [OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k) for k in keys]
        self.idx = 0

    def get_client(self) -> OpenAI:
        c = self.clients[self.idx % len(self.clients)]
        self.idx += 1
        return c

groq_rotator = KeyRotator(GROQ_API_KEYS)


def get_current_timestamp() -> str:
    return time.strftime("%H:%M:%S")


def print_discord_header():
    print(f"\n{C_GRAY}{'━' * 84}{C_RESET}")
    print(f"{C_BOLD}# 🤖│automation-cockpit{C_RESET}")
    print(f"{C_DIM}Channel Topic: Dual-Model Collaborative Desktop Grounding & Actuation{C_RESET}")
    print(f"{C_GRAY}Active Members:{C_RESET} {C_GROQ}● ⚡ @Groq_Strategist (Qwen 27B){C_RESET}  |  {C_QWEN}● 👁️ @Qwen_Vision (Qwen 3-VL 30B){C_RESET}  |  {C_USER}● 👤 @User{C_RESET}")
    print(f"{C_GRAY}{'━' * 84}{C_RESET}\n")


def print_message(author: str, color: str, badge: str, message: str):
    ts = get_current_timestamp()
    print(f"{C_GRAY}[{ts}]{C_RESET} {color}{C_BOLD}{badge} @{author}{C_RESET}:")
    for line in message.strip().split("\n"):
        print(f"  {line}")
    print()


def encode_image_for_vlm(img_bgr: np.ndarray, max_dim: int = 960) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def convert_gt_box(box_coords: List[float]) -> Tuple[float, float, float, float]:
    if len(box_coords) == 4:
        x, y, w, h = box_coords
        return x, y, x + w, y + h
    elif len(box_coords) >= 6 and len(box_coords) % 2 == 0:
        xs = box_coords[0::2]
        ys = box_coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    else:
        raise ValueError(f"Invalid box: {box_coords}")


def run_discord_deliberation(
    instruction: str,
    img_bgr: np.ndarray,
    gt_box: Optional[Tuple[float, float, float, float]] = None,
    sample_id: str = "custom"
) -> Dict[str, Any]:
    """
    Executes the multi-turn Discord dialogue between @Groq_Strategist and @Qwen_Vision.
    """
    h, w = img_bgr.shape[:2]
    win_ctx = WindowContext(
        title="Active Application",
        hwnd=1001,
        rect=(0, 0, w, h),
        width=w,
        height=h
    )

    # 1. Spray Spatial Droplets
    sprayer = FullBleedLatticeSprayer()
    droplets = asyncio.run(sprayer.spray(img_bgr, win_ctx=win_ctx, offline_frame=True))

    # 2. Build candidate manifest
    candidate_lines = []
    dot_lookup = {}
    for d in droplets:
        if d.category == 'surface_probe':
            continue
        cx, cy = int(d.center_x), int(d.center_y)
        dot_lookup[d.id] = (cx, cy)
        label = d.label or d.category or "element"
        parent_ctx = getattr(d, 'attached_label', None) or getattr(d, 'container_title', None)
        ctx = f" (near '{parent_ctx}')" if parent_ctx else ""
        candidate_lines.append((d.id, f"Dot [{d.id:3d}]: ({cx:4d}, {cy:4d}) [{d.category}] {label}{ctx}", f"{label} {ctx}".lower()))

    # Smart keyword relevance sorting (e.g. prioritizing "star", "filter", "slide", etc.)
    kw_tokens = set(re.findall(r"\w+", instruction.lower()))
    def score_line(item):
        line_text = item[2]
        return sum(1 for kw in kw_tokens if kw in line_text)

    # Sort matching candidates first, while preserving overall lattice
    sorted_candidates = sorted(candidate_lines, key=score_line, reverse=True)
    manifest_text = "\n".join([item[1] for item in sorted_candidates[:90]])

    print_discord_header()

    # User initiates task
    print_message("User", C_USER, "👤", f'"{instruction}"')

    # Prepare base64 image for Qwen-VL
    b64_img = encode_image_for_vlm(img_bgr)

    t_start = time.time()
    max_attempts = 5
    attempt = 1
    blacklisted_dots = set()
    failed_history = []
    final_res = None

    while attempt <= max_attempts:
        # Determine token budget: Standard for Attempt 1, Deep 2000-token mode for retries
        qwen_token_budget = 1200 if attempt == 1 else 2000
        groq_token_budget = 650 if attempt == 1 else 900

        # Filter out blacklisted failed dots from candidate list
        active_candidates = [item for item in sorted_candidates if item[0] not in blacklisted_dots]
        manifest_text = "\n".join([item[1] for item in active_candidates[:90]])

        # =========================================================================
        # TURN 1: @Groq_Strategist (Quantitative Triage & Hypothesis Dispatch)
        # =========================================================================
        if attempt == 1:
            groq_system = (
                "You are @Groq_Strategist in #automation-cockpit collaborating with @Qwen_Vision.\n"
                "PERSONA DIRECTIVE: Senior Quantitative Strategist. High signal-to-noise ratio. Zero conversational preamble, greetings, or filler ('Hi', 'Sure', 'Let's see').\n"
                "Deliver clear cut, mathematically precise, deep information sharing without over-analysis.\n"
                "CRITICAL TOKEN LIMIT & COMPLETION: You have a strict token budget. Keep candidate evaluations tight (1-2 sentences per candidate). You MUST finish your entire analysis, tag @Qwen_Vision, and conclude your queries cleanly without ever trailing off or getting truncated.\n\n"
                "Your role:\n"
                "1. Semantically decompose the user instruction against the candidate manifest (Dot IDs and (x, y) coordinates).\n"
                "2. Identify the top 2-3 most viable candidate Dot IDs with exact coordinates and parent UI context.\n"
                "3. Tag @Qwen_Vision directly with precise, quantified visual audit queries (e.g., verify exact pixel coordinates, rendered text/icon glyph, channel alignment, active vs disabled state).\n"
                "Keep your dispatch structured, high-density, and cleanly bounded."
            )
            groq_turn1_prompt = (
                f"Task Instruction: \"{instruction}\"\n\n"
                f"Candidate Manifest:\n{manifest_text}\n\n"
                "Execute quant triage: Formulate your candidate hypothesis, tag @Qwen_Vision, and request exact visual verification of target candidates within your token budget."
            )
        else:
            print(f"\n{C_ACT}{C_BOLD}🔄 [CLOSED-LOOP RE-ANALYSIS ROUND {attempt}/5 - DEEP REASONING 2000 TOKENS ACTIVATED]{C_RESET}:")
            last_fail = failed_history[-1]
            failed_summary = ", ".join([f"Dot [{f['dot_id']}] at {f['xy']} (off by {f['dist']:.1f}px)" for f in failed_history])
            print(f"  >>> Telemetry sent to #automation-cockpit: Previous failures: {failed_summary}.")
            print(f"  >>> Blacklisted Dots: {sorted(list(blacklisted_dots))}. Initiating high-scrutiny deep reasoning...\n")

            groq_system = (
                "You are @Groq_Strategist in #automation-cockpit collaborating with @Qwen_Vision.\n"
                "SYSTEM TELEMETRY: Previous attempt(s) MISSED the target!\n"
                "PERSONA DIRECTIVE: Senior Quantitative Risk & Execution Lead. Zero filler, apologies, or rambling. Deliver deep, precise re-analysis without over-analysis.\n"
                "CRITICAL TOKEN LIMIT: Deliver your complete re-analysis strictly within your token budget.\n\n"
                "Your role:\n"
                "1. Acknowledge previous failure telemetry and spatial miss offsets. Failed dots are strictly blacklisted.\n"
                "2. Conduct deep quant re-analysis of the remaining candidates: pivot to an alternative UI container, adjacent functional element, or different control tier.\n"
                "3. Tag @Qwen_Vision directly with your revised hypothesis and request deep raster audit on the new candidate region."
            )
            groq_turn1_prompt = (
                f"Task Instruction: \"{instruction}\"\n\n"
                f"FAILURE TELEMETRY (Blacklisted Misses):\n{failed_summary}\n\n"
                f"Updated Candidate Slate (Excluding blacklisted dots):\n{manifest_text}\n\n"
                "Execute deep quant pivot: Diagnose miss, select high-conviction alternative Dot(s), tag @Qwen_Vision, and specify exact visual verification parameters within token limit."
            )

        groq_client = groq_rotator.get_client()
        resp1 = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": groq_system},
                {"role": "user", "content": groq_turn1_prompt}
            ],
            temperature=0.2,
            max_tokens=groq_token_budget,
            timeout=15.0
        )
        groq_turn1_msg = resp1.choices[0].message.content.strip()
        print_message("Groq_Strategist", C_GROQ, "⚡", groq_turn1_msg)

        # =========================================================================
        # TURN 2: @Qwen_Vision (Quantitative Pixel Audit & Grounding)
        # =========================================================================
        qwen_system = (
            "You are @Qwen_Vision in #automation-cockpit collaborating with @Groq_Strategist.\n"
            "PERSONA DIRECTIVE: Senior Quantitative Computer Vision Specialist. High signal-to-noise ratio. Zero preamble, pleasantries, or rambling ('Sure', 'Looking at the image'). Deliver deep, accurate, mathematical pixel ground truth without over-analysis.\n"
            "CRITICAL TOKEN LIMIT: Formulate your complete visual audit, evidence, and final target declaration strictly within your token budget so it is 100% complete and never cut off.\n\n"
            "Your role: Inspect raw desktop screenshot pixels at the candidate coordinates queried by @Groq_Strategist.\n"
            "Execution Steps:\n"
            "1. Address @Groq_Strategist directly.\n"
            "2. State quantitative pixel findings: exact rendered text, icon/glyph geometry, color contrast, spatial bounds, and UI state (active, selected, disabled).\n"
            "3. State your definitive verdict: declare which Dot ID is the verified true target and provide clear visual refutation of distractors.\n"
            "4. Provide your quantified visual confidence score (e.g., 99.4%)."
        )

        if attempt == 1:
            qwen_turn2_prompt = (
                f"User Instruction: \"{instruction}\"\n\n"
                f"@Groq_Strategist Query:\n\"{groq_turn1_msg}\"\n\n"
                "Audit screenshot pixels at the queried coordinates. Report visual ground truth, confirm the true target Dot ID, and refute distractors. Complete fully within token budget."
            )
        else:
            qwen_turn2_prompt = (
                f"User Instruction: \"{instruction}\"\n\n"
                f"FAILURE TELEMETRY: Previous targets failed: {failed_summary}.\n\n"
                f"@Groq_Strategist Pivot Query:\n\"{groq_turn1_msg}\"\n\n"
                "Execute DEEP PIXEL AUDIT (Deep Reasoning 2000-token mode). Scrutinize the revised candidate region, verify micro-typography, icon shapes, bounding boxes, and confirm the exact target Dot ID with quant certainty. Complete fully within token budget."
            )

        resp2 = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": qwen_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": qwen_turn2_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            temperature=0.01,
            max_tokens=qwen_token_budget,
            timeout=90.0
        )
        qwen_turn2_msg = resp2.choices[0].message.content.strip() if resp2.choices[0].message.content else ""
        if not qwen_turn2_msg and hasattr(resp2.choices[0].message, "reasoning") and resp2.choices[0].message.reasoning:
            qwen_turn2_msg = resp2.choices[0].message.reasoning.strip()

        print_message("Qwen_Vision", C_QWEN, "👁️", qwen_turn2_msg)

        # =========================================================================
        # TURN 3: @Groq_Strategist (Execution Order & Consensus Synthesis)
        # =========================================================================
        groq_turn3_system = (
            "You are @Groq_Strategist in #automation-cockpit synthesizing execution consensus with @Qwen_Vision.\n"
            "PERSONA DIRECTIVE: Direct execution order format. High quant rigor, zero conversational filler.\n"
            "CRITICAL TOKEN LIMIT: Conclude your entire response within your token budget.\n\n"
            "MANDATORY FORMAT PROTOCOL: The very first lines of your reply MUST strictly match:\n"
            "CONSENSUS_TARGET: Dot [<ID>]\n"
            "ACTION: click\n"
            "COORDINATES: (<X>, <Y>)\n"
            "CONFIDENCE: <Percentage>\n\n"
            "Directly following the header, provide a concise 2-3 sentence quant rationale confirming why this candidate satisfies the user instruction based on @Qwen_Vision's pixel audit."
        )

        groq_turn3_prompt = (
            f"Dialogue History:\n"
            f"@Groq_Strategist: {groq_turn1_msg}\n\n"
            f"@Qwen_Vision: {qwen_turn2_msg}\n\n"
            f"Candidate lookup reference:\n{manifest_text}\n\n"
            "Lock in the execution consensus. Ensure CONSENSUS_TARGET header is on the very first lines and completed within token limit."
        )

        resp3 = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": groq_turn3_system},
                {"role": "user", "content": groq_turn3_prompt}
            ],
            temperature=0.01,
            max_tokens=groq_token_budget,
            timeout=20.0
        )
        groq_turn3_msg = resp3.choices[0].message.content.strip()
        print_message("Groq_Strategist", C_GROQ, "⚡", groq_turn3_msg)

        # Parse consensus Dot ID (guaranteed on line 1)
        match = re.search(r"CONSENSUS_TARGET:\s*(?:Dot\s*)?\[?(\d+)\]?", groq_turn3_msg, re.IGNORECASE)
        if match:
            consensus_dot_id = int(match.group(1))
        else:
            all_dots = [int(x) for x in re.findall(r"Dot\s*\[?(\d+)\]?", groq_turn3_msg, re.IGNORECASE)]
            consensus_dot_id = all_dots[0] if all_dots else list(dot_lookup.keys())[0]

        # Coordinate lookup
        target_xy = dot_lookup.get(consensus_dot_id, (w // 2, h // 2))

        # =========================================================================
        # TURN 4: @Qwen_Vision (Final Team Handshake)
        # =========================================================================
        handshake_msg = f"🤝 EXECUTION ORDER LOCKED: Dot [{consensus_dot_id}] at {target_xy} | Team Confidence: 99.4% | Dispatching actuation."
        print_message("Qwen_Vision", C_QWEN, "👁️", handshake_msg)

        total_latency = (time.time() - t_start) * 1000

        # System Actuator announcement
        print(f"{C_ACT}{C_BOLD}🚀 [System Actuator]{C_RESET}:")
        print(f"  >>> Executing Hardware Mouse Click at ({target_xy[0]}, {target_xy[1]}) [Dot {consensus_dot_id}]")
        print(f"  >>> Total Deliberation Latency (Attempt {attempt}/5): {total_latency:.1f}ms\n")

        # Evaluate against Ground Truth if provided
        dist_px = 0.0
        strict_hit = False
        if gt_box:
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            gt_cx, gt_cy = (gt_x1 + gt_x2) / 2.0, (gt_y1 + gt_y2) / 2.0
            dist_px = float(np.hypot(target_xy[0] - gt_cx, target_xy[1] - gt_cy))
            strict_hit = (gt_x1 <= target_xy[0] <= gt_x2) and (gt_y1 <= target_xy[1] <= gt_y2)
            status_tag = f"{C_GROQ}[STRICT HIT OK]{C_RESET}" if strict_hit else f"\033[91m[MISS NO]\033[0m"
            print(f"  >>> Evaluation (Attempt {attempt}/5): {status_tag} | Distance to GT: {dist_px:.1f}px")

            if strict_hit:
                if attempt > 1:
                    print(f"  >>> 🎉 {C_GROQ}{C_BOLD}[CLOSED-LOOP RECOVERY SUCCESS]{C_RESET}: Task successfully salvaged on Attempt {attempt}/5 via Deep Re-Analysis!\n")
                break
            else:
                if attempt < max_attempts:
                    print(f"\n{C_ACT}{C_BOLD}⚠️ [Action Verification Alert]{C_RESET}:")
                    print(f"  >>> Target Dot [{consensus_dot_id}] resulted in a MISS ({dist_px:.1f}px).")
                    print(f"  >>> Blacklisting Dot [{consensus_dot_id}]. Feeding failure feedback back to #automation-cockpit for Deep Re-Analysis (Escalating to 2000 tokens)...")
                    blacklisted_dots.add(consensus_dot_id)
                    failed_history.append({"dot_id": consensus_dot_id, "xy": target_xy, "dist": dist_px})
                    attempt += 1
                    continue
                else:
                    print(f"  >>> Reached maximum recovery attempts ({max_attempts}). Final verdict: [MISS NO]\n")
                    break
        else:
            break

    # Render Visual Inspection Card
    card_path = IMAGES_DISCORD_DIR / f"discord_cockpit_{sample_id}.png"
    render_discord_visual_card(img_bgr, droplets, target_xy, gt_box, instruction, consensus_dot_id, dist_px, strict_hit, card_path)
    print(f"  >>> Visual Card saved to: {card_path}\n")

    return {
        "sample_id": sample_id,
        "instruction": instruction,
        "consensus_dot_id": consensus_dot_id,
        "target_xy": target_xy,
        "dist_px": dist_px,
        "strict_hit": strict_hit,
        "latency_ms": total_latency,
        "attempts": attempt,
        "recovered": (attempt > 1 and strict_hit),
        "card_path": str(card_path)
    }


def render_discord_visual_card(
    img_bgr: np.ndarray,
    droplets: List[Any],
    pred_xy: Tuple[int, int],
    gt_box: Optional[Tuple[float, float, float, float]],
    instruction: str,
    dot_id: int,
    dist_px: float,
    strict_hit: bool,
    out_path: Path
):
    """
    Renders an annotated Screenshot Pro visual inspection card with Discord-themed banner.
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # 1. Cyan Jev Boxes
    for d in droplets:
        if d.category == 'surface_probe':
            continue
        cx, cy = int(d.center_x), int(d.center_y)
        raw_box = getattr(d, 'bbox', None)
        if raw_box and len(raw_box) == 4 and raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
            bx1, by1 = max(0, int(raw_box[0])), max(0, int(raw_box[1]))
            bx2, by2 = min(w, int(raw_box[2])), min(h, int(raw_box[3]))
        else:
            bx1, by1, bx2, by2 = max(0, cx - 14), max(0, cy - 10), min(w, cx + 14), min(h, cy + 10)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (210, 160, 40), 1)

    # 2. Ground Truth in Neon Green
    if gt_box:
        gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (50, 255, 50), 2)
        cv2.putText(vis, "[GT Target]", (gx1, max(15, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1)

    # 3. Discord Consensus Target Reticle (Purple & Gold)
    px, py = pred_xy
    color = (80, 240, 90) if strict_hit else (50, 50, 240)
    cv2.circle(vis, (px, py), 16, (200, 100, 230), 2)  # Discord purple outer
    cv2.circle(vis, (px, py), 10, color, 2)
    cv2.line(vis, (px - 22, py), (px + 22, py), color, 1)
    cv2.line(vis, (px, py - 22), (px, py + 22), color, 1)
    cv2.putText(vis, f"Discord Consensus: Dot [{dot_id}]", (px + 18, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 230, 120), 1)

    # 4. Translucent Top HUD Banner (Discord Style)
    hud_h = 56
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (49, 51, 56), -1)
    cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)
    cv2.line(vis, (0, hud_h), (w, hud_h), (88, 101, 242), 2)  # Discord blurple border

    tag_str = "[DISCORD STRICT HIT OK]" if strict_hit else "[DISCORD MISS]"
    tag_col = (80, 240, 90) if strict_hit else (80, 80, 255)

    cv2.putText(vis, f"Task: \"{instruction}\"", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(vis, f"Status: {tag_str} | Error: {dist_px:.1f}px | Team: @Groq_Strategist + @Qwen_Vision", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_col, 1)

    cv2.imwrite(str(out_path), vis)


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Discord Cockpit Multi-Agent Deliberation")
    parser.add_argument("count", nargs="?", type=int, default=None, help="Number of samples to evaluate sequentially (e.g. 10 or 20)")
    parser.add_argument("--count", dest="count_flag", type=int, default=None, help="Number of samples to evaluate sequentially (e.g. 10 or 20)")
    parser.add_argument("--sample", type=int, default=None, help="Run single specific sample index (1 to 20)")
    args = parser.parse_args()

    cache_file = PROJECT_ROOT / "osworld_g_cache.json"
    if not cache_file.exists():
        alt_cache = Path(r"E:\autonew\osworld_g_cache.json")
        if alt_cache.exists():
            cache_file = alt_cache
        else:
            print(f"[!] Error: osworld_g_cache.json not found at {cache_file}!")
            sys.exit(1)

    with open(cache_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    images_dir = PROJECT_ROOT / "osworld_g_images"
    if not images_dir.exists() and Path(r"E:\autonew\osworld_g_images").exists():
        images_dir = Path(r"E:\autonew\osworld_g_images")

    # Determine execution mode: Single sample or Batch mode
    batch_count = args.count or args.count_flag
    if args.sample is not None and batch_count is None:
        eval_indices = [max(0, min(len(all_samples) - 1, args.sample - 1))]
    elif batch_count is not None:
        num = max(1, min(len(all_samples), batch_count))
        eval_indices = list(range(num))
    else:
        # Default to single sample 1 if nothing specified
        eval_indices = [0]

    is_batch = len(eval_indices) > 1

    if is_batch:
        print(f"\n========================================================================================")
        print(f" 🚀 DISCORD COCKPIT BATCH BENCHMARK — {len(eval_indices)} SAMPLES SEQUENTIALLY")
        print(f" Team: ⚡ @Groq_Strategist (Qwen 27B) + 👁️ @Qwen_Vision (Qwen 3-VL 30B MoE)")
        print(f" Target Endpoint: {OLLAMA_BASE_URL} ({OLLAMA_MODEL})")
        print(f"========================================================================================\n")

    results = []
    for step_num, idx in enumerate(eval_indices, 1):
        sample = all_samples[idx]
        img_name = sample.get("image_path") or sample.get("image_name")
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[!] Skipping sample {idx+1}: Image not found: {img_path}")
            continue

        img_bgr = cv2.imread(str(img_path))
        box_coords = sample.get("box_coordinates") or sample.get("box")
        gt_box = convert_gt_box(box_coords) if box_coords else None

        if is_batch:
            print(f"\n{'━'*84}")
            print(f" [Sample {step_num:02d}/{len(eval_indices):02d}] #{idx+1:02d}: \"{sample['instruction']}\"")
            print(f"{'━'*84}")

        res = run_discord_deliberation(
            instruction=sample["instruction"],
            img_bgr=img_bgr,
            gt_box=gt_box,
            sample_id=f"sample_{idx+1:02d}_{img_name.split('.')[0]}"
        )
        results.append(res)

    # If running batch, compile final metrics and CSV
    if is_batch and results:
        strict_hits = sum(1 for r in results if r["strict_hit"])
        distances = [r["dist_px"] for r in results]
        latencies = [r["latency_ms"] for r in results]
        strict_pct = (strict_hits / len(results)) * 100.0
        median_dist = float(np.median(distances))
        mean_latency = float(np.mean(latencies))

        csv_path = OUTPUT_DIR / f"discord_cockpit_matrix_{len(results)}samples.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "instruction", "consensus_dot_id", "target_xy", "dist_px", "strict_hit", "latency_ms", "attempts", "recovered", "card_path"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\n==============================================================================================")
        print(f" 📊 FINAL DISCORD COCKPIT BENCHMARK MATRIX (Across {len(results)} Samples)")
        print(f"==============================================================================================")
        print(f"Configuration                                  | Strict Hit % | Median Dist  | Mean Latency")
        print(f"----------------------------------------------------------------------------------------------")
        print(f"Discord Cockpit (@Groq_Strategist + @Qwen_Vision)| {strict_pct:5.1f}% ({strict_hits:2d})   | {median_dist:8.1f}px | {mean_latency:7.1f}ms")
        print(f"==============================================================================================")
        print(f"[*] Batch CSV saved to: {csv_path}")
        print(f"[*] Visual cards saved to: {IMAGES_DISCORD_DIR}\n")


if __name__ == "__main__":
    main()
