import csv
from collections import Counter

csv_path = 'diagnostic_200_results/qwen_diagnostic_200samples.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

pruned = [r for r in rows if r['category'] == 'CANDIDATE_PRUNED']

print(f"Total Pruned: {len(pruned)}")

# Group pruned by closest_dot_label base
label_counter = Counter()
for r in pruned:
    lbl = r['closest_dot_label']
    if lbl.startswith('[Icon] '):
        base = lbl[7:].split('/')[0].strip()
    else:
        base = 'Text: ' + lbl[:20].strip()
    label_counter[base] += 1

print("\nTop Pruned Target Droplet Labels:")
for base, count in label_counter.most_common(20):
    print(f"  {base:<35}: {count} samples")

# Let's inspect what percentage of pruned samples had closest_dot_dist == 0.0 (directly inside GT box!)
dist_zero = sum(1 for r in pruned if float(r['closest_dot_dist']) == 0.0)
dist_le_5 = sum(1 for r in pruned if float(r['closest_dot_dist']) <= 5.0)
print(f"\nPruned targets directly inside GT box (dist == 0.0): {dist_zero}/{len(pruned)} ({dist_zero/len(pruned)*100:.1f}%)")
print(f"Pruned targets within 5px of GT box: {dist_le_5}/{len(pruned)} ({dist_le_5/len(pruned)*100:.1f}%)")
