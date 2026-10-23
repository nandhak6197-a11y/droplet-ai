import csv

csv_path = 'diagnostic_200_results/qwen_diagnostic_200samples.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

valid_rows = [r for r in rows if not r['reasoning'].startswith('Failures across all Groq keys')]
api_failed_rows = [r for r in rows if r['reasoning'].startswith('Failures across all Groq keys')]

print(f"Total samples: {len(rows)}")
print(f"API Rate Limited (429) samples: {len(api_failed_rows)} ({len(api_failed_rows)/len(rows)*100:.1f}%)")
print(f"Successfully evaluated by Qwen: {len(valid_rows)} ({len(valid_rows)/len(rows)*100:.1f}%)")

from collections import Counter
valid_cats = Counter(r['category'] for r in valid_rows)
print("\n=== ACCURACY ON COMPLETED SAMPLES ===")
for k, v in valid_cats.items():
    print(f"  {k:<22}: {v} ({v/len(valid_rows)*100:.1f}%)")

strict_c = sum(1 for r in valid_rows if r['strict_hit'] == 'True')
tolerant_c = sum(1 for r in valid_rows if r['tolerant_hit'] == 'True')
print(f"\nReal Strict Hit Rate:   {strict_c}/{len(valid_rows)} = {strict_c/len(valid_rows)*100:.1f}%")
print(f"Real Tolerant Hit Rate: {tolerant_c}/{len(valid_rows)} = {tolerant_c/len(valid_rows)*100:.1f}%")

cands_pruned = [r for r in valid_rows if r['category'] == 'CANDIDATE_PRUNED']
distractors = [r for r in valid_rows if r['category'] == 'REASONER_DISTRACTOR']
misses = [r for r in valid_rows if r['category'] == 'SPRAYER_MISS']
print(f"\nReal Root Cause Breakdown on Completed Samples:")
print(f"  • STRICT_HIT:         {strict_c} ({strict_c/len(valid_rows)*100:.1f}%)")
print(f"  • TOLERANT_HIT:       {tolerant_c - strict_c} ({(tolerant_c - strict_c)/len(valid_rows)*100:.1f}%)")
print(f"  • REASONER_DISTRACTOR:{len(distractors)} ({len(distractors)/len(valid_rows)*100:.1f}%)")
print(f"  • CANDIDATE_PRUNED:   {len(cands_pruned)} ({len(cands_pruned)/len(valid_rows)*100:.1f}%)")
print(f"  • SPRAYER_MISS:       {len(misses)} ({len(misses)/len(valid_rows)*100:.1f}%)")
