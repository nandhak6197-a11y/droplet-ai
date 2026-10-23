import csv
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

with open('diagnostic_200_results/qwen_diagnostic_200samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

distractors = [r for r in rows if r['category'] == 'REASONER_DISTRACTOR']
for r in distractors[:6]:
    print("=" * 80)
    print(f"Sample {r['sample_idx']}: Task='{r['instruction']}'")
    print(f"PredDot={r['pred_dot_id']} vs ClosestDot={r['closest_dot_id']} (err {float(r['dist_centroid']):.1f}px)")
    print(f"ClosestLabel: {r['closest_dot_label']}")
    print(f"Candidate Eval: {r['candidate_evaluation'][:300]}")
