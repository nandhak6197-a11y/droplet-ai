import csv
import json
from collections import Counter

csv_path = 'diagnostic_200_results/qwen_diagnostic_200samples.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

cats = Counter(r['category'] for r in rows)
print("=== CATEGORIES ===")
for k, v in cats.items():
    print(f"  {k:<22}: {v} ({v/len(rows)*100:.1f}%)")

pruned = [r for r in rows if r['category'] == 'CANDIDATE_PRUNED']
distractor = [r for r in rows if r['category'] == 'REASONER_DISTRACTOR']
miss = [r for r in rows if r['category'] == 'SPRAYER_MISS']
offset = [r for r in rows if r['category'] == 'GEOMETRIC_OFFSET']

print(f"\n=== CANDIDATE PRUNED ({len(pruned)} samples) - FIRST 25 ===")
for r in pruned[:25]:
    c_dist = float(r['closest_dot_dist'])
    print(f"Sample {int(r['sample_idx']):>3}: TargetDot={r['closest_dot_id']} (dist {c_dist:.1f}px) | Closest='{r['closest_dot_label'][:35]}' | Task: '{r['instruction']}'")

print(f"\n=== REASONER DISTRACTOR ({len(distractor)} samples) - FIRST 25 ===")
for r in distractor[:25]:
    d_c = float(r['dist_centroid'])
    print(f"Sample {int(r['sample_idx']):>3}: Pred={r['pred_dot_id']} vs GT_Dot={r['closest_dot_id']} (err {d_c:.1f}px) | GT_Label='{r['closest_dot_label'][:30]}' | Task: '{r['instruction']}'")

print(f"\n=== SPRAYER MISS ({len(miss)} samples) ===")
for r in miss:
    c_dist = float(r['closest_dot_dist'])
    print(f"Sample {int(r['sample_idx']):>3}: MinDist={c_dist:.1f}px | Task: '{r['instruction']}'")
