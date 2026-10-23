import csv

with open('diagnostic_200_results/qwen_diagnostic_200samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

distractors = [r for r in rows if r['category'] == 'REASONER_DISTRACTOR']
print(f"Total Reasoner Distractor Failures: {len(distractors)}")

for r in distractors:
    idx = int(r['sample_idx'])
    inst = r['instruction']
    pred_id = r['pred_dot_id']
    dist = float(r['dist_centroid'])
    c_id = r['closest_dot_id']
    c_dist = float(r['closest_dot_dist'])
    c_lbl = r['closest_dot_label']
    reason = r['reasoning']
    print(f"\n[Sample {idx:03d}] \"{inst}\"")
    print(f"   Picked: Dot [{pred_id}] (dist={dist:.1f}px) | Reason: {reason[:80]}")
    print(f"   Target: Dot [{c_id}] (dist={c_dist:.1f}px) | Label: \"{c_lbl}\"")
