import csv

with open('screenshot_pro_benchmark_results/benchmark_matrix_50samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

qwen_hits = [r for r in rows if r['config_id'] == 'spatial_qwen27b' and r['strict_hit'] == 'True']
print(f"Total Qwen 3.8 27B + Spatial Droplets Strict Hits: {len(qwen_hits)}/50 ({len(qwen_hits)/50*100:.1f}%)")

for h in qwen_hits:
    s_idx = int(h['sample_idx'])
    inst = h['instruction']
    dist = float(h['dist_centroid'])
    lat = float(h['latency_ms'])
    print(f"  • Sample {s_idx:02d}: \"{inst[:55]}\" -> dist={dist:4.1f}px, lat={lat:6.1f}ms")
