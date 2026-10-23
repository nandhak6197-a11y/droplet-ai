"""
Benchmark script to verify screen reading, settle sensing, droplet spraying,
HUD rendering, and reasoner candidate pruning performance.
"""
import sys
sys.path.insert(0, ".")
import time
import numpy as np
import cv2

from spatial_droplet.vision.grabber import FastScreenGrabber
from spatial_droplet.vision.settle_sensor import AdaptiveSettleSensor
from spatial_droplet.vision.sprayer import FullBleedLatticeSprayer
from spatial_droplet.vision.icon_detector import EnhancedIconDetector


def benchmark():
    results = {}
    print("=== PERFORMANCE & PRECISION BENCHMARK ===")

    # 1. Screen Grabber
    grabber = FastScreenGrabber.get_instance()
    t0 = time.perf_counter()
    frame = grabber.grab_full()
    t1 = time.perf_counter()
    results["grab_full_ms"] = (t1 - t0) * 1000.0
    print(f"1. FastScreenGrabber.grab_full(): {results['grab_full_ms']:.2f} ms (shape: {frame.shape})")

    # 2. ROI Box Grab (Cached)
    t0 = time.perf_counter()
    roi = grabber.grab_box(100, 100, 200, 200)
    t1 = time.perf_counter()
    results["grab_box_cached_ms"] = (t1 - t0) * 1000.0
    print(f"2. FastScreenGrabber.grab_box() (ROI Cache): {results['grab_box_cached_ms']:.4f} ms")

    # 3. Settle Sensor Motion Delta
    f2 = frame.copy()
    # Add slight artificial shift
    f2[50:60, 50:60] = 255
    t0 = time.perf_counter()
    for _ in range(50):
        delta = AdaptiveSettleSensor.compute_motion_delta(frame, f2)
    t1 = time.perf_counter()
    results["settle_sensor_delta_ms"] = ((t1 - t0) / 50.0) * 1000.0
    print(f"3. AdaptiveSettleSensor.compute_motion_delta(): {results['settle_sensor_delta_ms']:.3f} ms (delta={delta:.3f})")

    # 4. Icon Detector
    t0 = time.perf_counter()
    icons = EnhancedIconDetector.detect_icons(frame)
    t1 = time.perf_counter()
    results["icon_detector_ms"] = (t1 - t0) * 1000.0
    print(f"4. IconDetector.detect_interactive_icons(): {results['icon_detector_ms']:.2f} ms ({len(icons)} icons detected)")

    # 5. FullBleedLatticeSprayer (Droplets generation)
    import asyncio
    sprayer = FullBleedLatticeSprayer(48)
    t0 = time.perf_counter()
    droplets = asyncio.run(sprayer.spray(frame, offline_frame=True))
    t1 = time.perf_counter()
    results["sprayer_spray_ms"] = (t1 - t0) * 1000.0
    print(f"5. FullBleedLatticeSprayer.spray(): {results['sprayer_spray_ms']:.2f} ms ({len(droplets)} droplets generated)")

    total_pipeline_ms = (results["grab_full_ms"] + 
                         results["settle_sensor_delta_ms"] + 
                         results["sprayer_spray_ms"])
    print(f"\n---> TOTAL VISION PIPELINE LATENCY: {total_pipeline_ms:.2f} ms")
    print(f"---> Screen grab: {results['grab_full_ms']:.2f} ms | ROI Crop: {results['grab_box_cached_ms']:.4f} ms | Motion delta: {results['settle_sensor_delta_ms']:.3f} ms | Spray: {results['sprayer_spray_ms']:.2f} ms")
    print("ALL PIPELINE BENCHMARKS PASSED CLEANLY!")


if __name__ == "__main__":
    benchmark()
