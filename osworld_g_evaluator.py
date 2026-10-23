import sys
import json
import csv
import asyncio
import urllib.request
import time
import math
import traceback
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = r"C:\Users\NANDHA K\Documents\antigravity\amazing-bose"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.memory import EpisodicTrajectoryMemory
from spatial_droplet.core.models import WindowContext


# ============================================================
# CONFIGURATION
# ============================================================

ANNOTATION_URL = (
    "https://huggingface.co/datasets/yyyang/"
    "UI-Grounding-Benchmarks/raw/main/"
    "OSWorld-G/OSWorld-G.json"
)

IMAGE_BASE_URL = (
    "https://huggingface.co/datasets/yyyang/"
    "UI-Grounding-Benchmarks/resolve/main/"
    "OSWorld-G/images/"
)

# None = evaluate every sample
# Set 15 for a quick pilot
MAX_SAMPLES = 15 if len(sys.argv) <= 1 else (None if sys.argv[1].lower() == "all" else int(sys.argv[1]))
EVAL_MODE = sys.argv[2] if len(sys.argv) > 2 else "combined"

# Your additional diagnostic tolerance.
# DO NOT call this an official OSWorld-G metric unless verified.
TOLERANCE_PX = 24.0

SPRAYER_STEP = 24

OUTPUT_CSV = "osworld_g_verified_results.csv"
FAILED_CSV = "osworld_g_failed_cases.csv"

REQUEST_TIMEOUT = 30


# ============================================================
# NETWORK HELPERS
# ============================================================

def download_json(url):
    cache_path = Path("osworld_g_cache.json")
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT
    ) as response:
        data = json.loads(
            response.read().decode("utf-8")
        )
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data


_IMAGE_CACHE = {}

def download_image(url):
    if url in _IMAGE_CACHE:
        return _IMAGE_CACHE[url].copy()

    img_dir = Path("osworld_g_images")
    img_dir.mkdir(exist_ok=True)
    img_name = url.split("/")[-1]
    local_img_path = img_dir / img_name

    if local_img_path.exists() and local_img_path.stat().st_size > 100:
        img = cv2.imread(str(local_img_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            _IMAGE_CACHE[url] = img
            return img.copy()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT
    ) as response:
        raw = response.read()

    try:
        with open(local_img_path, "wb") as f:
            f.write(raw)
    except Exception:
        pass

    arr = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV failed to decode image:\n{url}"
        )

    _IMAGE_CACHE[url] = image
    return image.copy()



# ============================================================
# GROUND-TRUTH HANDLING
# ============================================================

def convert_gt_box(box_coordinates):
    """
    IMPORTANT:

    This evaluator assumes OSWorld-G stores:

        [x, y, width, height]

    and converts it to:

        [x1, y1, x2, y2]

    Verify this against the dataset documentation before
    publishing results.
    """

    if box_coordinates is None:
        raise ValueError(
            "Sample does not contain box_coordinates."
        )

    coords = [float(v) for v in box_coordinates]

    if len(coords) == 4:
        bx, by, bw, bh = coords
        x1 = bx
        y1 = by
        x2 = bx + bw
        y2 = by + bh
        if bw <= 0 or bh <= 0:
            raise ValueError(
                f"Invalid GT dimensions: {box_coordinates}"
            )
        return x1, y1, x2, y2

    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Degenerate polygon GT: {box_coordinates}"
            )
        return x1, y1, x2, y2

    else:
        raise ValueError(
            f"Expected 4 box values or even polygon vertices, got: {box_coordinates}"
        )


def validate_gt_box(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
):
    """
    Sanity check that the annotation approximately belongs
    to the downloaded image coordinate system.
    """

    if x2 <= x1 or y2 <= y1:
        return False

    if x1 < 0 or y1 < 0:
        return False

    if x2 > image_width or y2 > image_height:
        return False

    return True


# ============================================================
# PURE MODEL INFERENCE
# ============================================================

async def predict(
    image,
    instruction,
    sprayer,
    reasoner
):
    """
    =========================================================
    CRITICAL ANTI-LEAKAGE BOUNDARY
    =========================================================

    Ground-truth coordinates MUST NEVER enter this function.

    Inputs available to the model:

        image
        instruction

    plus normal runtime objects generated from them.

    NO:
        box_coordinates
        GT centroid
        target bbox
        nearest GT node
        GT distance

    =========================================================
    """

    height, width = image.shape[:2]

    # --------------------------------------------
    # Perception
    # --------------------------------------------

    sprayer.reset_tracking()
    droplets = await sprayer.spray(image)

    if droplets is None:
        droplets = []

    # --------------------------------------------
    # Runtime context
    # --------------------------------------------

    memory = EpisodicTrajectoryMemory(
        goal=instruction
    )

    window_context = WindowContext(
        hwnd=0,
        title="OSWorld-G Benchmark",
        rect=(0, 0, width, height),
        width=width,
        height=height,
        is_valid=True
    )

    # --------------------------------------------
    # Cognitive grounding
    # --------------------------------------------

    decision = await reasoner.decide(
        user_intent=instruction,
        droplets=droplets,
        window_ctx=window_context,
        memory=memory,
        screenshot=image,
        mode=EVAL_MODE
    )

    # Prediction is FINAL here.
    #
    # Everything below the return statement belongs
    # to the evaluator, not the model.

    return {
        "x": decision.target_x,
        "y": decision.target_y,

        "droplet_id": (
            decision.target_droplet_id
            if hasattr(
                decision,
                "target_droplet_id"
            )
            else None
        ),

        "action": (
            str(decision.action_type)
            if hasattr(
                decision,
                "action_type"
            )
            else None
        ),

        "confidence": (
            decision.confidence
            if hasattr(
                decision,
                "confidence"
            )
            else None
        ),

        "reasoning": (
            str(decision.reasoning)
            if hasattr(
                decision,
                "reasoning"
            )
            else ""
        ),

        "droplet_count": len(droplets)
    }


# ============================================================
# SCORING
# ============================================================

def score_prediction(
    prediction,
    gt_box,
    tolerance_px
):
    """
    Evaluation happens ONLY after model prediction.
    """

    x1, y1, x2, y2 = gt_box

    pred_x = prediction["x"]
    pred_y = prediction["y"]

    # No prediction
    if pred_x is None or pred_y is None:
        return {
            "valid_prediction": False,
            "strict_hit": False,
            "tolerant_hit": False,
            "centroid_distance": None,
            "box_distance": None
        }

    pred_x = float(pred_x)
    pred_y = float(pred_y)

    if not (
        math.isfinite(pred_x)
        and math.isfinite(pred_y)
    ):
        return {
            "valid_prediction": False,
            "strict_hit": False,
            "tolerant_hit": False,
            "centroid_distance": None,
            "box_distance": None
        }

    # --------------------------------------------
    # STRICT point-in-box
    # --------------------------------------------

    strict_hit = (
        x1 <= pred_x <= x2
        and
        y1 <= pred_y <= y2
    )

    # --------------------------------------------
    # GT centroid distance
    # --------------------------------------------

    gt_cx = (x1 + x2) / 2.0
    gt_cy = (y1 + y2) / 2.0

    centroid_distance = math.hypot(
        pred_x - gt_cx,
        pred_y - gt_cy
    )

    # --------------------------------------------
    # Distance from predicted point to GT rectangle
    #
    # IMPORTANT:
    # A point inside the box has box_distance = 0.
    #
    # This is a better tolerance calculation than
    # simply checking centroid distance <= 24.
    # --------------------------------------------

    dx = max(
        x1 - pred_x,
        0,
        pred_x - x2
    )

    dy = max(
        y1 - pred_y,
        0,
        pred_y - y2
    )

    box_distance = math.hypot(
        dx,
        dy
    )

    tolerant_hit = (
        box_distance <= tolerance_px
    )

    return {
        "valid_prediction": True,
        "strict_hit": strict_hit,
        "tolerant_hit": tolerant_hit,
        "centroid_distance": centroid_distance,
        "box_distance": box_distance
    }


# ============================================================
# SINGLE SAMPLE
# ============================================================

async def evaluate_sample(
    sample,
    sample_index,
    sprayer,
    reasoner
):
    instruction = sample.get(
        "instruction",
        ""
    )

    image_path = sample.get(
        "image_path"
    )

    if not image_path:
        raise ValueError(
            "Missing image_path."
        )

    image_url = (
        IMAGE_BASE_URL
        + image_path
    )

    image = download_image(
        image_url
    )

    height, width = image.shape[:2]

    # ========================================================
    # PHASE 1:
    # MODEL PREDICTION
    #
    # Notice that GT has NOT been extracted yet.
    # ========================================================

    start_time = time.perf_counter()

    prediction = await predict(
        image=image,
        instruction=instruction,
        sprayer=sprayer,
        reasoner=reasoner
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    # ========================================================
    # PHASE 2:
    # GROUND TRUTH
    #
    # Only now do we access box_coordinates.
    # ========================================================

    raw_gt = sample.get(
        "box_coordinates"
    )

    gt_box = convert_gt_box(
        raw_gt
    )

    x1, y1, x2, y2 = gt_box

    if not validate_gt_box(
        x1,
        y1,
        x2,
        y2,
        width,
        height
    ):
        raise ValueError(
            "Ground-truth box does not fit "
            "downloaded image coordinate space. "
            f"Image={width}x{height}, "
            f"GT={raw_gt}"
        )

    # ========================================================
    # PHASE 3:
    # SCORING
    # ========================================================

    score = score_prediction(
        prediction,
        gt_box,
        TOLERANCE_PX
    )

    return {
        "index": sample_index,
        "id": sample.get("id"),

        "instruction": instruction,

        "image_path": image_path,
        "image_width": width,
        "image_height": height,

        "gt_x1": x1,
        "gt_y1": y1,
        "gt_x2": x2,
        "gt_y2": y2,

        "gt_width": x2 - x1,
        "gt_height": y2 - y1,

        "pred_x": prediction["x"],
        "pred_y": prediction["y"],

        "droplet_id": prediction[
            "droplet_id"
        ],

        "droplet_count": prediction[
            "droplet_count"
        ],

        "action": prediction[
            "action"
        ],

        "confidence": prediction[
            "confidence"
        ],

        "valid_prediction": score[
            "valid_prediction"
        ],

        "strict_hit": score[
            "strict_hit"
        ],

        "tolerant_hit": score[
            "tolerant_hit"
        ],

        "centroid_distance_px": score[
            "centroid_distance"
        ],

        "box_distance_px": score[
            "box_distance"
        ],

        "latency_ms": elapsed_ms,

        "reasoning": prediction[
            "reasoning"
        ]
    }


# ============================================================
# CSV
# ============================================================

def save_csv(
    filename,
    rows
):
    if not rows:
        return

    keys = list(
        rows[0].keys()
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# REPORT
# ============================================================

def print_report(
    results,
    failures,
    total_dataset_size
):
    evaluated = len(results)

    if evaluated == 0:
        print(
            "\nNo valid samples were evaluated."
        )
        return

    strict_hits = sum(
        1
        for r in results
        if r["strict_hit"]
    )

    tolerant_hits = sum(
        1
        for r in results
        if r["tolerant_hit"]
    )

    valid_predictions = sum(
        1
        for r in results
        if r["valid_prediction"]
    )

    strict_accuracy = (
        strict_hits / evaluated
    ) * 100.0

    tolerant_accuracy = (
        tolerant_hits / evaluated
    ) * 100.0

    distances = [
        r["centroid_distance_px"]
        for r in results
        if r["centroid_distance_px"]
        is not None
    ]

    box_distances = [
        r["box_distance_px"]
        for r in results
        if r["box_distance_px"]
        is not None
    ]

    latencies = [
        r["latency_ms"]
        for r in results
    ]

    mean_centroid = (
        float(np.mean(distances))
        if distances
        else float("nan")
    )

    median_centroid = (
        float(np.median(distances))
        if distances
        else float("nan")
    )

    mean_box_distance = (
        float(np.mean(box_distances))
        if box_distances
        else float("nan")
    )

    mean_latency = float(
        np.mean(latencies)
    )

    median_latency = float(
        np.median(latencies)
    )

    print()
    print("=" * 78)
    print(
        "OSWORLD-G GROUNDING EVALUATION"
    )
    print("=" * 78)

    print(
        f"Dataset samples loaded:      "
        f"{total_dataset_size}"
    )

    print(
        f"Successfully evaluated:      "
        f"{evaluated}"
    )

    print(
        f"Failed / skipped:            "
        f"{len(failures)}"
    )

    print(
        f"Valid model predictions:     "
        f"{valid_predictions}/{evaluated}"
    )

    print("-" * 78)

    print(
        f"Strict Point-in-BBox:        "
        f"{strict_accuracy:.2f}% "
        f"({strict_hits}/{evaluated})"
    )

    print(
        f"Custom <= {TOLERANCE_PX:.0f}px "
        f"Box Tolerance:     "
        f"{tolerant_accuracy:.2f}% "
        f"({tolerant_hits}/{evaluated})"
    )

    print("-" * 78)

    print(
        f"Mean GT-centroid distance:   "
        f"{mean_centroid:.2f}px"
    )

    print(
        f"Median GT-centroid distance: "
        f"{median_centroid:.2f}px"
    )

    print(
        f"Mean distance to GT box:     "
        f"{mean_box_distance:.2f}px"
    )

    print("-" * 78)

    print(
        f"Mean inference latency:      "
        f"{mean_latency:.1f}ms"
    )

    print(
        f"Median inference latency:    "
        f"{median_latency:.1f}ms"
    )

    print("=" * 78)

    print()
    print(
        "NOTE: Strict Point-in-BBox is "
        "reported separately from the custom "
        f"{TOLERANCE_PX:.0f}px tolerance metric."
    )

    print(
        "Do not compare these numbers directly "
        "with another system unless dataset split, "
        "input conditions, and scoring protocol match."
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

async def main():
    print("=" * 78)
    print(
        "OSWORLD-G CLEAN GROUNDING EVALUATOR"
    )
    print("=" * 78)

    print(
        "[*] Downloading official annotations..."
    )

    dataset = download_json(
        ANNOTATION_URL
    )

    print(
        f"[*] Loaded {len(dataset)} samples."
    )

    if MAX_SAMPLES is not None:
        dataset_to_run = dataset[
            :MAX_SAMPLES
        ]
    else:
        dataset_to_run = dataset

    print(
        f"[*] Evaluating "
        f"{len(dataset_to_run)} samples."
    )

    print(
        "[*] Ground truth is isolated from "
        "the predict() function."
    )

    print()

    # --------------------------------------------------------
    # Initialize model components ONCE
    # --------------------------------------------------------

    sprayer = FullBleedLatticeSprayer(
        step_s=SPRAYER_STEP
    )

    reasoner = (
        DiagonalDropletReasoner()
    )

    results = []
    failures = []

    # --------------------------------------------------------
    # Sequential evaluation
    #
    # Sequential is intentional.
    # It avoids:
    #   API rate problems
    #   memory collisions
    #   thread/UIAutomation problems
    #   inconsistent model state
    # --------------------------------------------------------

    for i, sample in enumerate(
        dataset_to_run,
        start=1
    ):
        short_instruction = (
            sample.get(
                "instruction",
                ""
            )
            .replace("\n", " ")
        )

        if len(short_instruction) > 42:
            short_instruction = (
                short_instruction[:42]
                + "..."
            )

        try:
            result = await evaluate_sample(
                sample=sample,
                sample_index=i,
                sprayer=sprayer,
                reasoner=reasoner
            )

            results.append(
                result
            )

            if result["strict_hit"]:
                status = "EXACT HIT"

            elif result["tolerant_hit"]:
                status = "TOLERANT"

            else:
                status = "MISS"

            distance = result[
                "centroid_distance_px"
            ]

            if distance is None:
                distance_text = "N/A"
            else:
                distance_text = (
                    f"{distance:.1f}px"
                )

            print(
                f"[{i:03d}/"
                f"{len(dataset_to_run):03d}] "
                f"{status:<10} | "
                f"Dist: {distance_text:<9} | "
                f"{short_instruction:<45} | "
                f"{result['latency_ms']:.0f}ms",
                flush=True
            )
            save_csv(OUTPUT_CSV, results)

        except Exception as exc:
            failure = {
                "index": i,
                "id": sample.get("id"),
                "instruction": sample.get(
                    "instruction"
                ),
                "image_path": sample.get(
                    "image_path"
                ),
                "error": repr(exc)
            }

            failures.append(
                failure
            )

            print(
                f"[{i:03d}/"
                f"{len(dataset_to_run):03d}] "
                f"ERROR      | "
                f"{short_instruction}"
            )

            print(
                f"             {exc}"
            )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    save_csv(
        OUTPUT_CSV,
        results
    )

    save_csv(
        FAILED_CSV,
        failures
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_report(
        results=results,
        failures=failures,
        total_dataset_size=len(dataset)
    )

    print()
    print(
        f"Raw results saved to: "
        f"{Path(OUTPUT_CSV).resolve()}"
    )

    if failures:
        print(
            f"Failures saved to: "
            f"{Path(FAILED_CSV).resolve()}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
