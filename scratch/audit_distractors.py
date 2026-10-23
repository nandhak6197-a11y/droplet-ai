import csv
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('diagnostic_200_results/qwen_diagnostic_200samples.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

completed = [r for r in rows if not r['reasoning'].startswith('Failures across all Groq keys')]
distractors = [r for r in completed if r['category'] == 'REASONER_DISTRACTOR']

print(f"=== DETAILED AUDIT OF {len(distractors)} REASONER_DISTRACTOR CASES ===")
for i, r in enumerate(distractors, 1):
    s_idx = r['sample_idx']
    inst = r['instruction']
    gt_id = r['closest_dot_id']
    gt_lbl = r['closest_dot_label']
    p_id = r['pred_dot_id']
    px, py = r['pred_x'], r['pred_y']
    err = float(r['dist_centroid'])
    cand_eval = r.get('candidate_evaluation', '')[:160].replace('\n', ' ')
    print(f"\n[{i:02d}] Sample {s_idx}: \"{inst}\" (Err: {err:.1f}px)")
    print(f"     GT  [{gt_id}]: \"{gt_lbl}\"")
    print(f"     PRED[{p_id}] at ({px}, {py})")
    print(f"     Candidate Eval: {cand_eval}...")
