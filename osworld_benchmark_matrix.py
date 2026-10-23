"""
OSWorld-G Comprehensive Grounding Benchmark Matrix.
Evaluates:
  1. Default / Vanilla Model Grounding (No Droplets, direct coordinate prediction):
     - openai/gpt-oss-20b
     - openai/gpt-oss-120b
     - qwen/qwen3.8-27b (Multimodal Vision)
  2. With Spatial Droplet Framework (Full-Bleed Lattice + Set-of-Marks + Sub-Pixel Snapping):
     - openai/gpt-oss-20b + Spatial Droplets
     - openai/gpt-oss-120b + Spatial Droplets
     - qwen/qwen3.8-27b (Multimodal) + Spatial Droplets
  3. Pure Droplet Engine (Deterministic Invariant Vision, Zero LLM Call)
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
    sys.stdout.reconfigure(encoding="utf-8")
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
                timeout=15.0
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            fb, lb = raw.find('{'), raw.rfind('}')
            if fb != -1 and lb != -1 and lb > fb:
                data = json.loads(raw[fb:lb+1])
                dt = (time.perf_counter() - t0) * 1000
                return float(data.get("x", 0)), float(data.get("y", 0)), dt, data.get("reasoning", "")
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            return None, None, dt, f"Error: {e}"
    else:
        # Text-only model (gpt-oss-120b or gpt-oss-20b)
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
            "timeout": 15.0
        }
        if "gpt-oss" in model_name:
            call_kwargs["reasoning_effort"] = "low"

        try:
            resp = client.chat.completions.create(**call_kwargs)
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            fb, lb = raw.find('{'), raw.rfind('}')
            if fb != -1 and lb != -1 and lb > fb:
                data = json.loads(raw[fb:lb+1])
                dt = (time.perf_counter() - t0) * 1000
                return float(data.get("x", 0)), float(data.get("y", 0)), dt, data.get("reasoning", "")
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            return None, None, dt, f"Error: {e}"

    dt = (time.perf_counter() - t0) * 1000
    return None, None, dt, "Failed to parse coordinates"

# ----------------------------------------------------------------------------
# 2. Spatial Droplet Framework Predictors
# ----------------------------------------------------------------------------
async def predict_with_spatial(model_override: Optional[str], instruction: str, img: np.ndarray, droplets: list, sprayer: FullBleedLatticeSprayer, reasoner: DiagonalDropletReasoner) -> Tuple[Optional[float], Optional[float], float, str]:
    sh, sw = img.shape[:2]
    win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)
    t0 = time.perf_counter()

    if model_override == "pure_droplet":
        dec = await reasoner.decide(user_intent=instruction, droplets=droplets, window_ctx=win, memory=None, mode="pure_droplet")
        dt = (time.perf_counter() - t0) * 1000
        return dec.target_x, dec.target_y, dt, dec.reasoning
    else:
        dec = await reasoner.decide(user_intent=instruction, droplets=droplets, window_ctx=win, memory=None, screenshot=img, mode="combined", target_model=model_override)

    dt = (time.perf_counter() - t0) * 1000
    return dec.target_x, dec.target_y, dt, dec.reasoning



async def run_benchmark(max_samples: int = 15):
    print("=" * 80)
    print(" OSWORLD-G COMPARATIVE BENCHMARK: DEFAULT VS SPATIAL DROPLET FRAMEWORK")
    print("=" * 80)

    cache_path = Path("osworld_g_cache.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Use a diverse, high-value 15-sample suite
    eval_samples = all_samples[:max_samples]
    print(f"[*] Loaded {len(eval_samples)} test samples.")

    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()

    # Configurations to evaluate:
    configs = [
        # Vanilla / Default Models (No Droplets)
        {"id": "vanilla_20b", "name": "Default GPT-OSS 20B (No Droplets)", "type": "vanilla", "model": "openai/gpt-oss-20b"},
        {"id": "vanilla_120b", "name": "Default GPT-OSS 120B (No Droplets)", "type": "vanilla", "model": "openai/gpt-oss-120b"},
        {"id": "vanilla_qwen27b", "name": "Default Qwen 3.8 27B Vision (No Droplets)", "type": "vanilla", "model": "qwen/qwen3.8-27b"},

        # With Spatial Droplet Framework
        {"id": "spatial_20b", "name": "GPT-OSS 20B + Spatial Droplets", "type": "spatial", "model": "openai/gpt-oss-20b"},
        {"id": "spatial_120b", "name": "GPT-OSS 120B + Spatial Droplets", "type": "spatial", "model": "openai/gpt-oss-120b"},
        {"id": "spatial_qwen27b", "name": "Qwen 3.8 27B (Multimodal) + Spatial Droplets", "type": "spatial", "model": "qwen/qwen3.8-27b"},

        # Pure Droplet Engine
        {"id": "pure_droplet", "name": "Pure Droplet Engine (Zero API Call)", "type": "spatial", "model": "pure_droplet"}
    ]

    stats = {c["id"]: {"strict_hits": 0, "tolerant_hits": 0, "distances": [], "latencies": []} for c in configs}
    detailed_rows = []

    for s_idx, s in enumerate(eval_samples):
        img_path = Path("osworld_g_images") / s["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_box = convert_gt_box(s["box_coordinates"])
        gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
        instruction = s["instruction"]

        print(f"\n[Sample {s_idx+1:02d}/{len(eval_samples):02d}] {instruction[:55]}...")

        # Pre-spray droplets once per frame for all spatial configs
        droplets = await sprayer.spray(img)

        for cfg in configs:
            cfg_id = cfg["id"]
            if cfg["type"] == "vanilla":
                px, py, lat, reason = predict_vanilla(cfg["model"], instruction, img)
            else:
                px, py, lat, reason = await predict_with_spatial(cfg["model"], instruction, img, droplets, sprayer, reasoner)

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
                stats[cfg_id]["strict_hits"] += 1
            if tolerant:
                stats[cfg_id]["tolerant_hits"] += 1
            stats[cfg_id]["distances"].append(dist_c)
            stats[cfg_id]["latencies"].append(lat)

            hit_symbol = "[STRICT HIT]" if strict else ("[TOLERANT]" if tolerant else "[MISS]")
            print(f"  |-- {cfg['name']:<48} | {hit_symbol:<14} | Dist: {dist_c:5.1f}px | Latency: {lat:6.1f}ms")

            detailed_rows.append({
                "sample_idx": s_idx + 1,
                "instruction": instruction,
                "config_id": cfg_id,
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

    # Save detailed CSV
    with open("osworld_comparative_matrix_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)

    # Print Final Summary Table
    total = len(eval_samples)
    print("\n" + "=" * 90)
    print(f" FINAL OSWORLD-G BENCHMARK MATRIX (Across {total} Samples)")
    print("=" * 90)
    print(f"{'Configuration':<45} | {'Strict Hit %':<12} | {'Tolerant %':<12} | {'Median Dist':<12} | {'Mean Latency':<12}")
    print("-" * 90)

    summary_data = []
    for cfg in configs:
        cid = cfg["id"]
        strict_pct = (stats[cid]["strict_hits"] / total) * 100
        tol_pct = (stats[cid]["tolerant_hits"] / total) * 100
        med_dist = np.median(stats[cid]["distances"]) if stats[cid]["distances"] else 0
        mean_lat = np.mean(stats[cid]["latencies"]) if stats[cid]["latencies"] else 0

        print(f"{cfg['name']:<45} | {strict_pct:6.1f}% ({stats[cid]['strict_hits']:2d})  | {tol_pct:6.1f}% ({stats[cid]['tolerant_hits']:2d})  | {med_dist:8.1f}px   | {mean_lat:8.1f}ms")
        summary_data.append({
            "name": cfg["name"],
            "type": cfg["type"],
            "strict_pct": strict_pct,
            "tolerant_pct": tol_pct,
            "strict_count": stats[cid]["strict_hits"],
            "tolerant_count": stats[cid]["tolerant_hits"],
            "median_dist": round(float(med_dist), 1),
            "mean_latency": round(float(mean_lat), 1)
        })

    print("=" * 90)
    print("\n### OSWorld-G Benchmark Matrix (Markdown Format)\n")
    print("| Configuration | Paradigm | Strict Hit % | Tolerant Hit (<=24px) % | Median Dist (px) | Mean Latency (ms) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for s in summary_data:
        p_type = "Spatial Droplet Framework" if "Spatial" in s["name"] or "Pure" in s["name"] else "Vanilla Baseline"
        print(f"| **{s['name']}** | {p_type} | **{s['strict_pct']:.1f}%** ({s['strict_count']}/{total}) | **{s['tolerant_pct']:.1f}%** ({s['tolerant_count']}/{total}) | {s['median_dist']}px | {s['mean_latency']}ms |")
    print()

    with open("osworld_comparative_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("[*] Detailed results saved to osworld_comparative_matrix_results.csv")
    print("[*] Summary saved to osworld_comparative_summary.json")

if __name__ == "__main__":
    n_samples = 15 if len(sys.argv) <= 1 else int(sys.argv[1])
    asyncio.run(run_benchmark(n_samples))
