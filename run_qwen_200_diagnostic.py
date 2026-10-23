"""
Spatial Droplet + ngrok Qwen2.5-VL 7B Diagnostic Benchmark (200 Samples).
Evaluates visual grounding accuracy and systematically diagnoses lag / failure root causes:
  1. STRICT_HIT: Click landed strictly inside GT box
  2. TOLERANT_HIT: Click landed within 24px of GT box
  3. REASONER_DISTRACTOR: Target droplet was in Qwen's candidate manifest, but Qwen chose a distractor
  4. CANDIDATE_PRUNED: Target droplet was detected by sprayer, but pruned out before reaching Qwen
  5. SPRAYER_MISS: Target element was never detected by sprayer (contour/OCR/icon gap)
  6. GEOMETRIC_OFFSET: Right element targeted, but click landed outside GT box (> 24px)
"""

import sys
import os
import json
import csv
import re
import time
import math
import asyncio
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import WindowContext

OUTPUT_DIR = Path(os.environ.get("SPATIAL_DIAGNOSTIC_OUTPUT", "diagnostic_200_results"))
START_INDEX = max(0, int(os.environ.get("SPATIAL_DIAGNOSTIC_START", "0")))
TRACE_ONLY = os.environ.get("SPATIAL_CANDIDATE_TRACE_ONLY", "0").strip().lower() in {"1", "true", "yes"}
DIAGNOSTIC_PROVIDER = os.environ.get("SPATIAL_DIAGNOSTIC_PROVIDER", "groq").strip().lower()
if DIAGNOSTIC_PROVIDER not in {"ngrok", "groq"}:
    raise ValueError("SPATIAL_DIAGNOSTIC_PROVIDER must be 'ngrok' or 'groq'")
IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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

def render_diagnostic_image(
    img: np.ndarray,
    sample_idx: int,
    instruction: str,
    gt_box: Tuple[float, float, float, float],
    pred_x: Optional[float],
    pred_y: Optional[float],
    pred_dot_id: Optional[int],
    closest_dot_id: Optional[int],
    closest_dot_dist: float,
    status_category: str,
    dist_c: float,
    latency_ms: float,
    reasoning: str,
    droplets: list
) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw all droplets as subtle boxes
    active_droplets = [d for d in droplets if d.category != 'surface_probe']
    for d in active_droplets:
        cx, cy = int(d.center_x), int(d.center_y)
        bbox = getattr(d, 'bbox', None)
        if bbox and len(bbox) == 4:
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            bx1, by1, bx2, by2 = cx - 14, cy - 10, cx + 14, cy + 10
        bx1, by1 = max(0, min(bx1, bx2)), max(0, min(by1, by2))
        bx2, by2 = min(w - 1, max(bx1, bx2)), min(h - 1, max(by1, by2))

        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 190, 0), 1)
        cv2.circle(vis, (cx, cy), 2, (0, 220, 255), -1)

    # Draw Ground Truth Target Box (Neon Green)
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(w - 1, gx2), min(h - 1, gy2)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(vis, "GT", (gx1, max(20, gy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw Predicted Click
    if pred_x is not None and pred_y is not None:
        px, py = int(pred_x), int(pred_y)
        if status_category == "STRICT_HIT":
            col = (0, 255, 0)
        elif status_category == "TOLERANT_HIT":
            col = (0, 215, 255)
        else:
            col = (30, 30, 255)

        cv2.circle(vis, (px, py), 12, col, 3)
        cv2.circle(vis, (px, py), 4, col, -1)
        cv2.line(vis, (px - 18, py), (px + 18, py), col, 2)
        cv2.line(vis, (px, py - 18), (px, py + 18), col, 2)

        gt_cx, gt_cy = int((gx1 + gx2) / 2), int((gy1 + gy2) / 2)
        cv2.line(vis, (px, py), (gt_cx, gt_cy), (0, 255, 255), 2, cv2.LINE_AA)

    # Top HUD Banner
    banner_h = 52
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.line(vis, (0, banner_h), (w, banner_h), (60, 60, 60), 1)

    txt1 = f"Sample {sample_idx:03d}/200 | Task: \"{instruction[:70]}\""
    cv2.putText(vis, txt1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    status_col = (0, 255, 0) if "HIT" in status_category else (60, 60, 255)
    txt2 = f"Diagnosis: [{status_category}] | Dist: {dist_c:.1f}px | Lat: {latency_ms:.0f}ms | Pred Dot: {pred_dot_id} | Closest Dot: {closest_dot_id} ({closest_dot_dist:.1f}px)"
    cv2.putText(vis, txt2, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_col, 1, cv2.LINE_AA)

    return vis

async def run_diagnostic(total_samples: int = 200):
    print("=" * 88)
    provider_model = "ngrok Qwen2.5-VL 7B" if DIAGNOSTIC_PROVIDER == "ngrok" else "Groq Qwen3.8-27B"
    print(f" 🚀 SPATIAL DROPLET + {provider_model} DIAGNOSTIC BENCHMARK ({total_samples} SAMPLES)")
    print("=" * 88)

    cache_path = Path("osworld_g_cache.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    samples = all_samples[START_INDEX:START_INDEX + total_samples]

    print(f"[*] Loaded {len(samples)} evaluation samples from {cache_path} (starting at index {START_INDEX})")

    sprayer = FullBleedLatticeSprayer(48)
    reasoner = DiagonalDropletReasoner()
    execution_provider = "ngrok" if TRACE_ONLY else DIAGNOSTIC_PROVIDER
    reasoner.use_ngrok = execution_provider == "ngrok"
    print(f"[*] Provider pinned to {execution_provider}; fallback to other providers disabled")

    results = []
    category_counts = {
        "STRICT_HIT": 0,
        "TOLERANT_HIT": 0,
        "REASONER_DISTRACTOR": 0,
        "CANDIDATE_PRUNED": 0,
        "SPRAYER_MISS": 0,
        "GEOMETRIC_OFFSET": 0,
        "MODEL_ERROR": 0,
        "TRACE_ONLY": 0,
    }

    csv_path = OUTPUT_DIR / f"qwen_diagnostic_{total_samples}samples.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = None

    t_bench_start = time.perf_counter()

    for idx, s in enumerate(samples):
        img_name = s["image_path"]
        img_p = Path("osworld_g_images") / img_name
        if not img_p.exists():
            print(f"[!] Warning: Image {img_p} not found, skipping.")
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue

        sh, sw = img.shape[:2]
        gt_box = convert_gt_box(s["box_coordinates"])
        gt_cx = (gt_box[0] + gt_box[2]) / 2.0
        gt_cy = (gt_box[1] + gt_box[3]) / 2.0
        instruction = s["instruction"]

        # Benchmark images are offline screenshots, not the current Windows
        # desktop. Supply a synthetic full-frame window and disable host-only
        # window/taskbar sensors so machine state cannot contaminate results.
        win = WindowContext(hwnd=0, title="OSWorld-G", rect=(0, 0, sw, sh), width=sw, height=sh, is_valid=True)

        # 1. Spray droplets with fresh ID counter per sample
        sprayer = FullBleedLatticeSprayer(48)
        t0_spray = time.perf_counter()
        droplets = await sprayer.spray(img, win_ctx=win, offline_frame=True)
        spray_ms = (time.perf_counter() - t0_spray) * 1000

        # Proposal recall: is there a droplet overlapping, or very near, GT?
        active_droplets = [d for d in droplets if d.category != 'surface_probe']
        dots_in_gt = [d for d in active_droplets if is_point_in_box(d.center_x, d.center_y, gt_box)]
        
        closest_dot = None
        closest_dot_dist = 9999.0
        for d in active_droplets:
            d_box_dist = distance_to_box(d.center_x, d.center_y, gt_box)
            if d_box_dist < closest_dot_dist:
                closest_dot_dist = d_box_dist
                closest_dot = d

        target_dot_ids = [d.id for d in dots_in_gt]
        if not target_dot_ids and closest_dot is not None and closest_dot_dist <= 24.0:
            target_dot_ids = [closest_dot.id]
        sprayer_found_target = bool(target_dot_ids)

        # 2. Run ngrok Qwen2.5-VL 7B + Spatial Droplets
        t0_qwen = time.perf_counter()
        model_used = ""
        try:
            dec = await reasoner.decide(
                user_intent=instruction,
                droplets=droplets,
                window_ctx=win,
                memory=None,
                screenshot=img,
                mode="combined",
                target_model="qwen/qwen3.8-27b",
                provider=execution_provider,
                candidate_trace_only=TRACE_ONLY,
            )
            qwen_ms = (time.perf_counter() - t0_qwen) * 1000
            px, py = dec.target_x, dec.target_y
            model_used = str(dec.get("model_used", ""))
            provider_used = str(dec.get("provider", DIAGNOSTIC_PROVIDER))
            pred_id = dec.get("target_dot_id") or dec.get("target_droplet_id")
            reasoning = dec.get("reasoning", "") or ""
            deliberation = dec.get("deliberation", {}) or {}
        except Exception as e:
            qwen_ms = (time.perf_counter() - t0_qwen) * 1000
            px, py, pred_id, reasoning, deliberation = None, None, None, f"Error: {e}", {}
            provider_used = DIAGNOSTIC_PROVIDER

        # 3. Evaluate Metrics
        if px is not None and py is not None and not (px == 0 and py == 0):
            dist_c = distance_to_centroid(px, py, gt_box)
            dist_b = distance_to_box(px, py, gt_box)
            strict_hit = is_point_in_box(px, py, gt_box)
            tolerant_hit = strict_hit or (dist_b <= 24.0)
        else:
            dist_c = 999.0
            dist_b = 999.0
            strict_hit = False
            tolerant_hit = False

        # 4. Diagnostic Root Cause Categorization
        manifest_cand_ids = dec.get("manifest_candidate_ids") or []
        target_manifest_ids = [dot_id for dot_id in target_dot_ids if dot_id in manifest_cand_ids]
        target_in_manifest = bool(target_manifest_ids)
        target_candidate_rank = min(
            (manifest_cand_ids.index(dot_id) + 1 for dot_id in target_manifest_ids),
            default=None,
        )
        ranked_details = {item["id"]: item for item in dec.get("ranked_candidate_details", [])}
        target_candidate_details = [
            {"rank": (dec.get("ranked_candidate_ids", []).index(dot_id) + 1)
                     if dot_id in dec.get("ranked_candidate_ids", []) else None,
             **ranked_details[dot_id]}
            for dot_id in target_dot_ids if dot_id in ranked_details
        ]
        model_error = False if TRACE_ONLY else (not model_used or bool(re.search(
            r"\b(?:failures across|request too large|rate limit|timed? out|http error|error code:\s*4\d\d|connection error)\b",
            reasoning,
            flags=re.IGNORECASE,
        )))
        if TRACE_ONLY:
            cat = "TRACE_ONLY"
        elif strict_hit:
            cat = "STRICT_HIT"
        elif tolerant_hit:
            cat = "TOLERANT_HIT"
        elif model_error:
            cat = "MODEL_ERROR"
        else:
            if not sprayer_found_target:
                cat = "SPRAYER_MISS"  # Vision detector missed the control
            elif not target_in_manifest:
                cat = "CANDIDATE_PRUNED"  # At-target proposal existed, shortlist omitted it
            elif dist_b <= 60.0:
                cat = "GEOMETRIC_OFFSET"  # Close but missed boundary
            else:
                cat = "REASONER_DISTRACTOR"

        category_counts[cat] += 1

        row = {
            "sample_idx": idx + 1,
            "image_name": img_name,
            "instruction": instruction,
            "category": cat,
            "strict_hit": strict_hit,
            "tolerant_hit": tolerant_hit,
            "dist_centroid": round(dist_c, 2),
            "dist_box": round(dist_b, 2),
            "pred_x": round(px, 1) if px is not None else None,
            "pred_y": round(py, 1) if py is not None else None,
            "pred_dot_id": pred_id,
            "closest_dot_id": closest_dot.id if closest_dot else None,
            "closest_dot_dist": round(closest_dot_dist, 2),
            "closest_dot_label": closest_dot.label if closest_dot else "",
            "target_dot_ids": target_dot_ids,
            "target_in_manifest": target_in_manifest,
            "target_candidate_rank": target_candidate_rank,
            "target_manifest_ids": target_manifest_ids,
            "target_candidate_details": target_candidate_details,
            "manifest_supplied": bool(manifest_cand_ids),
            "model_error": model_error,
            "selection_source": dec.get("selection_source", "manifest_id" if pred_id is not None else "none"),
            "total_droplets": len(droplets),
            "spray_ms": round(spray_ms, 1),
            "qwen_ms": round(qwen_ms, 1),
            "model_used": model_used,
            "provider_used": provider_used,
            "manifest_candidate_ids": dec.get("manifest_candidate_ids", []),
            "ranked_candidate_ids": dec.get("ranked_candidate_ids", []),
            "ranked_candidate_details": dec.get("ranked_candidate_details", []),
            "active_candidate_ids": dec.get("active_candidate_ids", []),
            "reasoning": reasoning[:200],
            "intent_analysis": deliberation.get("intent_analysis", "")[:120],
            "candidate_evaluation": deliberation.get("candidate_evaluation", "")[:120]
        }

        results.append(row)

        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        # Render diagnostic inspection image (memory-safe)
        try:
            vis_img = render_diagnostic_image(
                img=img,
                sample_idx=idx + 1,
                instruction=instruction,
                gt_box=gt_box,
                pred_x=px,
                pred_y=py,
                pred_dot_id=pred_id,
                closest_dot_id=closest_dot.id if closest_dot else None,
                closest_dot_dist=closest_dot_dist,
                status_category=cat,
                dist_c=dist_c,
                latency_ms=qwen_ms,
                reasoning=reasoning,
                droplets=droplets
            )
            out_img_path = IMAGES_DIR / f"sample_{idx+1:03d}_{img_name.split('.')[0]}_{cat.lower()}.png"
            cv2.imwrite(str(out_img_path), vis_img)
            del vis_img
        except BaseException:
            pass

        # Progress log
        badge = "[TRACE]" if TRACE_ONLY else ("✔ [STRICT]" if strict_hit else ("✔ [TOLERANT]" if tolerant_hit else f"❌ [{cat}]"))
        print(f"[{idx+1:03d}/{total_samples:03d}] {badge:<18} | Dist: {dist_c:5.1f}px | Lat: {qwen_ms:6.0f}ms | \"{instruction[:50]}\"", flush=True)

        # Free memory between samples to prevent RAM fragmentation
        del img, droplets
        gc.collect()

        # Pacing delay to avoid Groq rate limit burst across 25 keys
        if not TRACE_ONLY:
            await asyncio.sleep(2.0)

    csv_file.close()

    total_time = time.perf_counter() - t_bench_start
    total_eval = len(results)

    # Compute Aggregate Diagnostics
    strict_count = sum(1 for r in results if r["strict_hit"])
    tolerant_count = sum(1 for r in results if r["tolerant_hit"])
    strict_pct = (strict_count / total_eval) * 100
    tolerant_pct = (tolerant_count / total_eval) * 100
    median_dist = float(np.median([r["dist_centroid"] for r in results]))
    mean_lat = float(np.mean([r["qwen_ms"] for r in results]))

    proposal_recall_count = sum(1 for r in results if r["target_dot_ids"])
    proposal_recall_pct = (proposal_recall_count / total_eval) * 100
    manifest_rows = [r for r in results if r["manifest_supplied"]]
    shortlist_recall_count = sum(1 for r in manifest_rows if r["target_in_manifest"])
    shortlist_recall_pct = (shortlist_recall_count / len(manifest_rows) * 100) if manifest_rows else 0.0
    manifest_budget = max((len(r["manifest_candidate_ids"]) for r in manifest_rows), default=0)
    ranked_targets = [r["target_candidate_rank"] for r in results if r["target_candidate_rank"] is not None]
    mean_target_rank = float(np.mean(ranked_targets)) if ranked_targets else None
    point_snap_count = sum(1 for r in results if r["selection_source"] == "full_droplet_point_snap")

    # Counterfactual retrieval recall at configurable shortlist sizes. In
    # trace-only mode the reasoner exposes the complete ranked pool without
    # making any model request, isolating pruning from VLM selection errors.
    rank_by_sample = {}
    for r in results:
        ranked_ids = r.get("ranked_candidate_ids") or []
        hits = [ranked_ids.index(dot_id) + 1 for dot_id in r["target_dot_ids"] if dot_id in ranked_ids]
        rank_by_sample[r["sample_idx"]] = min(hits) if hits else None
    ranked_samples = [rank for rank in rank_by_sample.values() if rank is not None]
    ranked_recall_pct = (len(ranked_samples) / total_eval * 100) if total_eval else 0.0
    shortlist_recall_at = {}
    for cutoff in (50, 64, 75, 100, 130):
        hits = sum(1 for rank in rank_by_sample.values() if rank is not None and rank <= cutoff)
        shortlist_recall_at[str(cutoff)] = {
            "hits": hits,
            "pct": round(hits / total_eval * 100, 1) if total_eval else 0.0,
        }

    print("\n" + "=" * 90)
    print(f" 📊 NGROK QWEN2.5-VL 7B + SPATIAL DROPLETS DIAGNOSTIC MATRIX ({total_eval} SAMPLES)")
    print("=" * 90)
    if TRACE_ONLY:
        print(" • Model Hit Rates:       skipped (candidate trace made no model call)")
    else:
        print(f" • Strict Hit Rate:       {strict_pct:5.1f}% ({strict_count}/{total_eval})")
        print(f" • Tolerant Hit Rate:     {tolerant_pct:5.1f}% ({tolerant_count}/{total_eval})")
    print(f" • Proposal Recall:       {proposal_recall_pct:5.1f}% ({proposal_recall_count}/{total_eval}) [detected in/near GT]")
    print(f" • Actual Shortlist Recall@{manifest_budget}: {shortlist_recall_pct:5.1f}% ({shortlist_recall_count}/{len(manifest_rows)}) [valid manifests only]")
    if mean_target_rank is not None:
        print(f" • Mean Rank in Sent Slate: {mean_target_rank:5.1f}")
    print(f" • Full-Pool Point Snaps: {point_snap_count:3d}")
    if TRACE_ONLY:
        print(f" • Ranked-Pool Recall:    {ranked_recall_pct:5.1f}% ({len(ranked_samples)}/{total_eval})")
        if ranked_samples:
            print(f" • Mean Full-Pool Rank:   {float(np.mean(ranked_samples)):5.1f}")
        for cutoff, metrics in shortlist_recall_at.items():
            print(f" • Counterfactual Recall@{cutoff:<3}: {metrics['pct']:5.1f}% ({metrics['hits']}/{total_eval})")
    print(f" • Median Error Distance: {median_dist:5.1f} px")
    print(f" • Mean Deliberation Lat: {mean_lat:5.1f} ms")
    print(f" • Total Evaluation Time: {total_time / 60:.1f} minutes")
    print("-" * 90)
    print(" 🔍 ROOT CAUSE BREAKDOWN (WHERE WE ARE LAGGING):")
    print("-" * 90)
    for cat_name, count in category_counts.items():
        pct = (count / total_eval) * 100
        print(f" • {cat_name:<22}: {pct:5.1f}% ({count:3d}/{total_eval})")
    print("=" * 90)

    summary_json = {
        "total_samples": total_eval,
        "strict_hits": None if TRACE_ONLY else strict_count,
        "strict_pct": None if TRACE_ONLY else round(strict_pct, 1),
        "tolerant_hits": None if TRACE_ONLY else tolerant_count,
        "tolerant_pct": None if TRACE_ONLY else round(tolerant_pct, 1),
        "proposal_recall_pct": round(proposal_recall_pct, 1),
        "shortlist_recall_at_manifest_budget_pct": round(shortlist_recall_pct, 1),
        "manifest_candidate_budget": manifest_budget,
        "shortlist_recall_denominator": len(manifest_rows),
        "mean_target_candidate_rank": round(mean_target_rank, 2) if mean_target_rank is not None else None,
        "full_pool_point_snaps": point_snap_count,
        "candidate_trace_only": TRACE_ONLY,
        "ranked_pool_recall_pct": round(ranked_recall_pct, 1),
        "counterfactual_shortlist_recall": shortlist_recall_at,
        "median_dist_px": round(median_dist, 1),
        "mean_latency_ms": round(mean_lat, 1),
        "total_time_seconds": round(total_time, 1),
        "root_cause_breakdown": {
            k: {"count": v, "pct": round((v / total_eval) * 100, 1)} for k, v in category_counts.items()
        }
    }

    json_path = OUTPUT_DIR / f"qwen_diagnostic_{total_samples}samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[*] Diagnostic CSV: {csv_path}")
    print(f"[*] Diagnostic JSON: {json_path}")
    print(f"[*] Visual images: {IMAGES_DIR}/")

if __name__ == "__main__":
    n = 200 if len(sys.argv) <= 1 else int(sys.argv[1])
    asyncio.run(run_diagnostic(n))
