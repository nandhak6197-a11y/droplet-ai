"""
Screenshot Pro Benchmark Suite (20 Samples).
Evaluates GUI Visual Grounding on OSWorld-G Screenshot Samples:
  1. Default / Vanilla Baselines (Direct pixel prediction without spatial droplets)
  2. Spatial Droplet Framework (Precision Jev Box geometry + Set-of-Marks)
  3. Pure Droplet Engine (Deterministic visual-semantic matching, Zero API call)

Generates annotated 'Screenshot Pro' visual inspection images with:
  - Snug Jev Boxes on all candidate UI elements (docked top-left badge tag, centroid dot)
  - Ground Truth bounding box in Neon Green with [GT Box] pill
  - Predicted target click reticle (Green = Strict Hit, Yellow = Tolerant Hit, Red = Miss)
  - Sleek translucent HUD banner with task instruction, status, distance, and latency.
"""

import sys
import os
import json
import csv
import time
import math
import re
import base64
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = r"c:\Users\NANDHA K\Documents\antigravity\amazing-bose"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext
from spatial_droplet.reasoning.groq_rotator import GROQ_API_KEYS

OUTPUT_DIR = Path("screenshot_pro_benchmark_results")
IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Multi-key client pool for maximum throughput and zero rate-limiting
class KeyRotator:
    def __init__(self, keys: List[str]):
        self.clients = [OpenAI(base_url="https://api.groq.com/openai/v1", api_key=k) for k in keys]
        self.idx = 0

    def get_client(self) -> OpenAI:
        c = self.clients[self.idx % len(self.clients)]
        self.idx += 1
        return c

rotator = KeyRotator(GROQ_API_KEYS)

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

def is_point_in_box(px: float, py: float, box: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def distance_to_box(px: float, py: float, box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    dx = max(0.0, x1 - px, px - x2)
    dy = max(0.0, y1 - py, py - y2)
    return math.hypot(dx, dy)

def distance_to_centroid(px: float, py: float, box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return math.hypot(px - cx, py - cy)

# ----------------------------------------------------------------------------
# 1. Vanilla Grounding Predictors (No Droplets)
# ----------------------------------------------------------------------------
def predict_vanilla(model_name: str, instruction: str, img: np.ndarray) -> Tuple[Optional[float], Optional[float], float, str]:
    sh, sw = img.shape[:2]
    client = rotator.get_client()
    t0 = time.perf_counter()

    if "qwen" in model_name:
        scale = 640.0 / max(sw, sh)
        small = cv2.resize(img, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_url = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        prompt = f"""You are a GUI visual grounding agent.
Screen Resolution: {sw}x{sh} pixels.
User Instruction: "{instruction}"
Examine the image carefully and predict the exact screen pixel coordinates (x, y) of the target UI element to click.
Respond with ONLY valid JSON:
{{"x": <int>, "y": <int>, "reasoning": "<short rationale>"}}"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a JSON-only visual grounding API."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]}
                ],
                temperature=0.1,
                max_tokens=150,
                timeout=20.0
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
            start_indices = [i for i, c in enumerate(raw) if c == '{']
            blocks = []
            for start in start_indices:
                stack = 0
                for i in range(start, len(raw)):
                    if raw[i] == '{': stack += 1
                    elif raw[i] == '}': stack -= 1
                    if stack == 0:
                        blocks.append(raw[start:i+1])
                        break
            blocks.sort(key=len, reverse=True)
            for block in blocks:
                try:
                    data = json.loads(block)
                    dt = (time.perf_counter() - t0) * 1000
                    return float(data.get("x", 0)), float(data.get("y", 0)), dt, data.get("reasoning", "")
                except Exception:
                    try:
                        cleaned = re.sub(r',\s*([}\]])', r'\1', block)
                        data = json.loads(cleaned)
                        dt = (time.perf_counter() - t0) * 1000
                        return float(data.get("x", 0)), float(data.get("y", 0)), dt, data.get("reasoning", "")
                    except Exception:
                        pass
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            return None, None, dt, f"Error: {e}"
    else:
        # Text-only LLM
        prompt = f"""You are a GUI grounding agent controlling a Windows desktop ({sw}x{sh} pixels).
User Instruction: "{instruction}"
Predict the estimated screen pixel coordinates (x, y) of the target UI element to click.
Respond with ONLY valid JSON:
{{"x": <int>, "y": <int>, "reasoning": "<short rationale>"}}"""

        call_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a JSON-only API."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150,
            "timeout": 20.0
        }
        if "gpt-oss" in model_name:
            call_kwargs["reasoning_effort"] = "low"

        try:
            resp = client.chat.completions.create(**call_kwargs)
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
            stack = []
            blocks = []
            start = -1
            for i, c in enumerate(raw):
                if c == '{':
                    if not stack: start = i
                    stack.append(i)
                elif c == '}':
                    if stack:
                        stack.pop()
                        if not stack: blocks.append(raw[start:i+1])
            blocks.sort(key=len, reverse=True)
            for block in blocks:
                try:
                    data = json.loads(block)
                    dt = (time.perf_counter() - t0) * 1000
                    return float(data.get("x", 0)), float(data.get("y", 0)), dt, data.get("reasoning", "")
                except Exception:
                    pass
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            return None, None, dt, f"Error: {e}"

    dt = (time.perf_counter() - t0) * 1000
    return None, None, dt, "Failed to parse coordinates"

# ----------------------------------------------------------------------------
# 2. Spatial Droplet Framework Predictors
# ----------------------------------------------------------------------------
async def predict_with_spatial(
    model_override: Optional[str],
    instruction: str,
    img: np.ndarray,
    droplets: list,
    sprayer: FullBleedLatticeSprayer,
    reasoner: DiagonalDropletReasoner
) -> Tuple[Optional[float], Optional[float], float, str, Optional[int]]:
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    t0 = time.perf_counter()

    try:
        if model_override == "pure_droplet":
            dec = await reasoner.decide(user_intent=instruction, droplets=droplets, window_ctx=win, memory=None, mode="pure_droplet")
            dt = (time.perf_counter() - t0) * 1000
            tgt_id = dec.get("target_dot_id") or dec.get("target_droplet_id")
            return dec.target_x, dec.target_y, dt, dec.reasoning, tgt_id
        elif model_override == "kaggle_qwen":
            dec = await reasoner.decide(
                user_intent=instruction,
                droplets=droplets,
                window_ctx=win,
                memory=None,
                screenshot=img,
                mode="combined",
                provider="ngrok"
            )
            dt = (time.perf_counter() - t0) * 1000
            tgt_id = dec.get("target_dot_id") or dec.get("target_droplet_id")
            return dec.target_x, dec.target_y, dt, dec.reasoning, tgt_id
        else:
            dec = await reasoner.decide(user_intent=instruction, droplets=droplets, window_ctx=win, memory=None, screenshot=img, mode="combined", target_model=model_override, provider="groq")
            dt = (time.perf_counter() - t0) * 1000
            tgt_id = dec.get("target_dot_id") or dec.get("target_droplet_id")
            return dec.target_x, dec.target_y, dt, dec.reasoning, tgt_id
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return None, None, dt, f"Error: {e}", None

# ----------------------------------------------------------------------------
# 3. Screenshot Pro Visualizer (Jev Boxes + GT + Prediction + HUD)
# ----------------------------------------------------------------------------
def render_screenshot_pro(
    img: np.ndarray,
    sample_idx: int,
    total_samples: int,
    instruction: str,
    gt_box: Tuple[float, float, float, float],
    pred_x: Optional[float],
    pred_y: Optional[float],
    pred_dot_id: Optional[int],
    strict_hit: bool,
    tolerant_hit: bool,
    dist_c: float,
    latency_ms: float,
    model_name: str,
    reasoning: str,
    droplets: list
) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    # 1. Draw Jev Boxes for all candidate droplets
    active_droplets = [d for d in droplets if d.category != 'surface_probe']
    for d in active_droplets:
        cx, cy = int(d.center_x), int(d.center_y)
        bbox = getattr(d, 'bbox', None)
        if bbox and len(bbox) == 4:
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
            by1, by2 = min(by1, by2), max(by1, by2)
            if (bx2 - bx1) < 8 or (by2 - by1) < 8:
                bx1, by1, bx2, by2 = cx - 14, cy - 10, cx + 14, cy + 10
        else:
            bx1, by1, bx2, by2 = cx - 14, cy - 10, cx + 14, cy + 10

        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w - 1, bx2), min(h - 1, by2)

        # Sleek 1px Jev Box Frame (Dark Cyan / Steel Blue)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (210, 160, 40), 1)

        # Precision Docked Corner Badge Tag (docked inside top-left, no upward collision)
        tid = str(d.id)
        tw = len(tid) * 8 + 8
        th = 14
        tag_x2 = min(w - 1, bx1 + tw)
        tag_y2 = min(h - 1, by1 + th)

        # Badge pill background (Dark Charcoal)
        cv2.rectangle(vis, (bx1, by1), (tag_x2, tag_y2), (28, 24, 20), -1)
        cv2.rectangle(vis, (bx1, by1), (tag_x2, tag_y2), (210, 160, 40), 1)
        # Badge text
        cv2.putText(vis, tid, (bx1 + 3, by1 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Precision Centroid Pinpoint Dot
        cv2.circle(vis, (cx, cy), 2, (0, 220, 255), -1)

    # 2. Draw Ground Truth Target Box (Neon Green)
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(w - 1, gx2), min(h - 1, gy2)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

    gt_lbl = " [GT Target] "
    (gt_w, gt_h), _ = cv2.getTextSize(gt_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    gt_tag_y1 = max(0, gy1 - gt_h - 4)
    cv2.rectangle(vis, (gx1, gt_tag_y1), (gx1 + gt_w, gt_tag_y1 + gt_h + 4), (0, 180, 0), -1)
    cv2.putText(vis, gt_lbl, (gx1, gt_tag_y1 + gt_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

    # 3. Draw Predicted Click Reticle & Vector
    if pred_x is not None and pred_y is not None:
        px, py = int(pred_x), int(pred_y)
        if strict_hit:
            reticle_col = (0, 255, 0)      # Neon Green
            status_text = "STRICT HIT"
        elif tolerant_hit:
            reticle_col = (0, 215, 255)    # Amber Yellow
            status_text = "TOLERANT HIT"
        else:
            reticle_col = (40, 40, 255)    # Bright Crimson Red
            status_text = "MISS"

        # Crosshair Reticle
        cv2.circle(vis, (px, py), 12, reticle_col, 2)
        cv2.circle(vis, (px, py), 3, reticle_col, -1)
        cv2.line(vis, (px - 18, py), (px + 18, py), reticle_col, 1)
        cv2.line(vis, (px, py - 18), (px, py + 18), reticle_col, 1)

        # Connecting line to Ground Truth Centroid
        gt_cx, gt_cy = int((gx1 + gx2) / 2), int((gy1 + gy2) / 2)
        cv2.line(vis, (px, py), (gt_cx, gt_cy), (0, 255, 255), 1, cv2.LINE_AA)

        # Pred Tag
        p_lbl = f" Pred [{pred_dot_id or '?'}] ({px},{py}) "
        (pw, ph), _ = cv2.getTextSize(p_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        p_tag_y = max(18, py - 18)
        cv2.rectangle(vis, (px + 14, p_tag_y - ph), (px + 14 + pw, p_tag_y + 4), (20, 20, 20), -1)
        cv2.rectangle(vis, (px + 14, p_tag_y - ph), (px + 14 + pw, p_tag_y + 4), reticle_col, 1)
        cv2.putText(vis, p_lbl, (px + 14, p_tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        status_text = "NO PREDICTION"
        reticle_col = (50, 50, 255)

    # 4. Sleek Top HUD Banner
    banner_h = 56
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.line(vis, (0, banner_h), (w, banner_h), (60, 60, 60), 1)

    # Banner Text - Line 1: Sample & Task
    header_text = f"SCREENSHOT PRO BENCHMARK | Sample {sample_idx:02d}/{total_samples:02d} | Task: \"{instruction[:75]}\""
    cv2.putText(vis, header_text, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

    # Banner Text - Line 2: Metrics & Status
    status_col = (0, 255, 0) if strict_hit else ((0, 215, 255) if tolerant_hit else (80, 80, 255))
    metrics_text = f"Status: [{status_text}]  |  Distance: {dist_c:.1f}px  |  Model: {model_name}  |  Latency: {latency_ms:.1f}ms  |  Droplets: {len(active_droplets)}"
    cv2.putText(vis, metrics_text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1, cv2.LINE_AA)

    return vis

# ----------------------------------------------------------------------------
# 4. Main Benchmark Suite Execution
# ----------------------------------------------------------------------------
async def run_screenshot_pro_benchmark(max_samples: int = 20):
    print("=" * 88)
    print(f" 🚀 SCREENSHOT PRO BENCHMARK SUITE — {max_samples} SAMPLES")
    print("=" * 88)

    cache_path = Path("osworld_g_cache.json")
    if not cache_path.exists():
        print(f"[!] Error: {cache_path} not found.")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    eval_samples = all_samples[:max_samples]
    print(f"[*] Loaded {len(eval_samples)} test samples from {cache_path}")
    print(f"[*] Initialized Groq Multi-Key Rotator across 25 keys.")

    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()

    # Configurations to benchmark
    configs = [
        # Pure Droplet Engine (Deterministic Zero-API Visual Grounding)
        {"id": "pure_droplet", "name": "Pure Droplet Engine (Zero API)", "type": "spatial", "model": "pure_droplet"},

        # Vanilla / Default Models (No Droplets)
        {"id": "vanilla_qwen27b", "name": "Default Qwen 3.8 27B Vision (No Droplets)", "type": "vanilla", "model": "qwen/qwen3.8-27b"},

        # With Spatial Droplet Framework (New Precision Jev Box & Grounded Deliberation)
        {"id": "spatial_qwen27b", "name": "Qwen 3.8 27B + Spatial Droplets", "type": "spatial", "model": "qwen/qwen3.8-27b"},
        {"id": "kaggle_qwen_spatial", "name": "Kaggle Dual-T4 Qwen (Multimodal) + Spatial Droplets", "type": "spatial", "model": "kaggle_qwen"}
    ]

    stats = {c["id"]: {"strict_hits": 0, "tolerant_hits": 0, "distances": [], "latencies": []} for c in configs}
    detailed_rows = []

    for s_idx, s in enumerate(eval_samples):
        img_name = s["image_path"]
        img_path = Path("osworld_g_images") / img_name
        if not img_path.exists():
            print(f"[!] Warning: Image {img_path} not found, skipping.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_box = convert_gt_box(s["box_coordinates"])
        gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
        instruction = s["instruction"]

        print(f"\n[Sample {s_idx+1:02d}/{len(eval_samples):02d}] \"{instruction[:60]}\"")

        # 1. Spray precision droplets for the screenshot
        t_spray0 = time.perf_counter()
        droplets = await sprayer.spray(img)
        spray_ms = (time.perf_counter() - t_spray0) * 1000

        # Best prediction tracker for annotated screenshot generation
        best_vis_pred = None
        best_vis_dist = 9999.0

        for cfg in configs:
            cid = cfg["id"]
            if cfg["type"] == "vanilla":
                px, py, lat, reason = await asyncio.to_thread(predict_vanilla, cfg["model"], instruction, img)
                tgt_id = None
            else:
                px, py, lat, reason, tgt_id = await predict_with_spatial(cfg["model"], instruction, img, droplets, sprayer, reasoner)

            if px is not None and py is not None and not (px == 0 and py == 0):
                dist_box = distance_to_box(px, py, gt_box)
                dist_c = distance_to_centroid(px, py, gt_box)
                strict = is_point_in_box(px, py, gt_box)
                tolerant = strict or (dist_box <= 24.0)
            else:
                dist_box = 999.0
                dist_c = 999.0
                strict = False
                tolerant = False

            if strict:
                stats[cid]["strict_hits"] += 1
            if tolerant:
                stats[cid]["tolerant_hits"] += 1
            stats[cid]["distances"].append(dist_c)
            stats[cid]["latencies"].append(lat)

            hit_badge = "[STRICT HIT ✔]" if strict else ("[TOLERANT ✔]" if tolerant else "[MISS ❌]")
            print(f"  |-- {cfg['name']:<44} | {hit_badge:<16} | Dist: {dist_c:5.1f}px | Latency: {lat:6.1f}ms")

            detailed_rows.append({
                "sample_idx": s_idx + 1,
                "image_name": img_name,
                "instruction": instruction,
                "config_id": cid,
                "config_name": cfg["name"],
                "strict_hit": strict,
                "tolerant_hit": tolerant,
                "pred_x": px,
                "pred_y": py,
                "gt_cx": gt_cx,
                "gt_cy": gt_cy,
                "dist_centroid": dist_c,
                "dist_box": dist_box,
                "latency_ms": lat,
                "reasoning": reason
            })

            # Prefer spatial model result for the visual annotated output
            if "spatial" in cid or "pure" in cid:
                if dist_c < best_vis_dist:
                    best_vis_dist = dist_c
                    best_vis_pred = {
                        "pred_x": px, "pred_y": py, "pred_dot_id": tgt_id,
                        "strict_hit": strict, "tolerant_hit": tolerant,
                        "dist_c": dist_c, "latency_ms": lat,
                        "model_name": cfg["name"], "reasoning": reason
                    }

        # Render and save the Screenshot Pro visualization
        if best_vis_pred is not None:
            pro_img = render_screenshot_pro(
                img=img,
                sample_idx=s_idx + 1,
                total_samples=len(eval_samples),
                instruction=instruction,
                gt_box=gt_box,
                pred_x=best_vis_pred["pred_x"],
                pred_y=best_vis_pred["pred_y"],
                pred_dot_id=best_vis_pred["pred_dot_id"],
                strict_hit=best_vis_pred["strict_hit"],
                tolerant_hit=best_vis_pred["tolerant_hit"],
                dist_c=best_vis_pred["dist_c"],
                latency_ms=best_vis_pred["latency_ms"],
                model_name=best_vis_pred["model_name"],
                reasoning=best_vis_pred["reasoning"],
                droplets=droplets
            )
            out_img_path = IMAGES_DIR / f"sample_{s_idx+1:02d}_{img_name.split('.')[0]}_pro.png"
            cv2.imwrite(str(out_img_path), pro_img)

        # Incrementally update CSV after every sample
        total = len(eval_samples)
        csv_path = OUTPUT_DIR / f"benchmark_matrix_{total}samples.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
                writer.writeheader()
                writer.writerows(detailed_rows)
        except Exception as csv_err:
            print(f"[!] Warning writing incremental CSV: {csv_err}", flush=True)

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == total:
            print(f"\n--- Live Progress [{s_idx+1}/{total}] ---", flush=True)
            for cfg in configs:
                cid = cfg["id"]
                s_hits = stats[cid]["strict_hits"]
                pct = (s_hits / (s_idx + 1)) * 100
                print(f"    {cfg['name']:<42} Strict: {pct:5.1f}% ({s_hits}/{s_idx+1})", flush=True)
            print("-" * 52 + "\n", flush=True)

    # Print Final Summary Table
    total = len(eval_samples)
    print("\n" + "=" * 94, flush=True)
    print(f" 📊 FINAL SCREENSHOT PRO BENCHMARK MATRIX (Across {total} Samples)", flush=True)
    print("=" * 94, flush=True)
    print(f"{'Configuration':<44} | {'Strict Hit %':<13} | {'Tolerant %':<13} | {'Median Dist':<13} | {'Mean Latency':<12}", flush=True)
    print("-" * 94, flush=True)

    summary_data = []
    for cfg in configs:
        cid = cfg["id"]
        strict_pct = (stats[cid]["strict_hits"] / total) * 100
        tol_pct = (stats[cid]["tolerant_hits"] / total) * 100
        med_dist = np.median(stats[cid]["distances"]) if stats[cid]["distances"] else 0
        mean_lat = np.mean(stats[cid]["latencies"]) if stats[cid]["latencies"] else 0

        print(f"{cfg['name']:<44} | {strict_pct:6.1f}% ({stats[cid]['strict_hits']:2d})   | {tol_pct:6.1f}% ({stats[cid]['tolerant_hits']:2d})   | {med_dist:8.1f}px    | {mean_lat:8.1f}ms", flush=True)
        summary_data.append({
            "name": cfg["name"],
            "type": cfg["type"],
            "strict_pct": round(float(strict_pct), 1),
            "tolerant_pct": round(float(tol_pct), 1),
            "strict_count": stats[cid]["strict_hits"],
            "tolerant_count": stats[cid]["tolerant_hits"],
            "total_samples": total,
            "median_dist_px": round(float(med_dist), 1),
            "mean_latency_ms": round(float(mean_lat), 1)
        })

    print("=" * 94, flush=True)
    print("\n### Screenshot Pro Benchmark Matrix (Markdown Format)\n", flush=True)
    print("| Configuration | Paradigm | Strict Hit % | Tolerant Hit (<=24px) % | Median Dist (px) | Mean Latency (ms) |", flush=True)
    print("| :--- | :--- | :---: | :---: | :---: | :---: |", flush=True)
    for s in summary_data:
        p_type = "Spatial Droplet Framework" if "Spatial" in s["name"] or "Pure" in s["name"] else "Vanilla Baseline"
        print(f"| **{s['name']}** | {p_type} | **{s['strict_pct']}%** ({s['strict_count']}/{total}) | **{s['tolerant_pct']}%** ({s['tolerant_count']}/{total}) | {s['median_dist_px']}px | {s['mean_latency_ms']}ms |", flush=True)
    print(flush=True)

    json_path = OUTPUT_DIR / f"benchmark_summary_{total}samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[*] Detailed per-sample CSV saved to: {csv_path}", flush=True)
    print(f"[*] Summary JSON saved to: {json_path}", flush=True)
    print(f"[*] Visual annotated screenshot images saved to: {IMAGES_DIR}/", flush=True)

if __name__ == "__main__":
    n = 100 if len(sys.argv) <= 1 else int(sys.argv[1])
    asyncio.run(run_screenshot_pro_benchmark(n))
