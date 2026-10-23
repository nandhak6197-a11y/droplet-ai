import csv

csv_path = 'diagnostic_200_results/qwen_diagnostic_200samples.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

valid_rows = [r for r in rows if not r['reasoning'].startswith('Failures across all Groq keys')]
distractors = [r for r in valid_rows if r['category'] == 'REASONER_DISTRACTOR']

print(f"=== ALL {len(distractors)} REASONER DISTRACTORS ===")
for r in distractors:
    d_c = float(r['dist_centroid'])
    print(f"Sample {int(r['sample_idx']):>3}: Pred={r['pred_dot_id']} vs GT={r['closest_dot_id']} (err {d_c:.1f}px)")
    print(f"  Task: '{r['instruction']}'")
    print(f"  GT Label: '{r['closest_dot_label']}'")
    print(f"  Qwen Reasoning: {r['reasoning'][:140]}")
    print()
