import csv
from collections import Counter

with open('diagnostic_200_results/qwen_diagnostic_200samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

pruned = [r for r in rows if r['category'] == 'CANDIDATE_PRUNED']
distractors = [r for r in rows if r['category'] == 'REASONER_DISTRACTOR']
misses = [r for r in rows if r['category'] == 'SPRAYER_MISS']
hits = [r for r in rows if 'HIT' in r['category']]

print(f"Total Samples: {len(rows)}")
print(f"  • Hits (Strict + Tolerant): {len(hits)} ({len(hits)/len(rows)*100:.1f}%)")
print(f"  • Candidate Pruned:         {len(pruned)} ({len(pruned)/len(rows)*100:.1f}%)")
print(f"  • Reasoner Distractor:      {len(distractors)} ({len(distractors)/len(rows)*100:.1f}%)")
print(f"  • Sprayer Miss:             {len(misses)} ({len(misses)/len(rows)*100:.1f}%)")

print("\n--- SAMPLE PRUNED CASES (Target existed near GT but omitted from Top 25) ---")
for r in pruned[:15]:
    idx = int(r['sample_idx'])
    inst = r['instruction']
    c_id = r['closest_dot_id']
    c_dist = float(r['closest_dot_dist'])
    c_lbl = r['closest_dot_label']
    print(f"Sample {idx:03d}: \"{inst[:45]}\"")
    print(f"   -> Closest droplet [{c_id}] was only {c_dist:.1f}px from GT! Label: \"{c_lbl[:50]}\"")

print("\n--- CLOSEST DROPLET DISTANCE DISTRIBUTION IN PRUNED SAMPLES ---")
pruned_dists = [float(r['closest_dot_dist']) for r in pruned]
print(f"  • Exactly in GT (dist=0px): {sum(1 for d in pruned_dists if d == 0.0)}/{len(pruned)} ({sum(1 for d in pruned_dists if d == 0.0)/len(pruned)*100:.1f}%)")
print(f"  • Within 10px of GT:        {sum(1 for d in pruned_dists if d <= 10.0)}/{len(pruned)} ({sum(1 for d in pruned_dists if d <= 10.0)/len(pruned)*100:.1f}%)")
print(f"  • Within 24px of GT:        {sum(1 for d in pruned_dists if d <= 24.0)}/{len(pruned)} ({sum(1 for d in pruned_dists if d <= 24.0)/len(pruned)*100:.1f}%)")
