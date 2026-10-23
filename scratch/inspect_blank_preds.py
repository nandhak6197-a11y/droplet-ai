import csv

csv_path = 'diagnostic_200_results/qwen_diagnostic_200samples.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

blank_preds = [r for r in rows if not r.get('pred_dot_id')]
print(f"Total samples with blank pred_dot_id: {len(blank_preds)} out of {len(rows)}")
for r in blank_preds[:15]:
    print(f"Sample {r['sample_idx']}: Task='{r['instruction']}'")
    print(f"   Reasoning: {r['reasoning'][:120]}")
    print(f"   Intent: {r['intent_analysis'][:80]}")
    print()
