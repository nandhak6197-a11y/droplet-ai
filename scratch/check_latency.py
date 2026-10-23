import csv

with open('diagnostic_200_results/qwen_diagnostic_200samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

lats = [float(r['qwen_ms']) for r in rows]
lats.sort()
print(f"Latencies (ms): min={min(lats):.0f}, p25={lats[50]:.0f}, median={lats[100]:.0f}, p75={lats[150]:.0f}, max={max(lats):.0f}")
slow = [r for r in rows if float(r['qwen_ms']) > 15000]
print(f"Samples taking >15s: {len(slow)} / 200")
for r in slow[:8]:
    lat_s = float(r['qwen_ms']) / 1000.0
    print(f"Sample {r['sample_idx']}: Lat={lat_s:.1f}s | {r['reasoning'][:80]}")
