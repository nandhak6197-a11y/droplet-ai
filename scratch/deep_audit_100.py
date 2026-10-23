import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import json

df = pd.read_csv('screenshot_pro_benchmark_results/benchmark_matrix_100samples.csv')
with open('osworld_g_cache.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)[:100]

kaggle = df[df['config_id'] == 'kaggle_qwen_spatial'].copy()
spatial = df[df['config_id'] == 'spatial_qwen27b'].copy()
pure = df[df['config_id'] == 'pure_droplet'].copy()

print("=== KAGGLE QWEN SPATIAL SUMMARY (100 SAMPLES) ===")
print(f"Strict Hits:   {kaggle['strict_hit'].sum()}/100")
print(f"Tolerant Hits: {kaggle['tolerant_hit'].sum()}/100")
print(f"Misses:        {(~kaggle['tolerant_hit']).sum()}/100")

misses = kaggle[~kaggle['tolerant_hit']]
print("\n=== DISTANCE BREAKDOWN OF THE 55 MISSES ===")
b_24_50 = misses[(misses['dist_box'] > 24) & (misses['dist_box'] <= 50)]
b_50_100 = misses[(misses['dist_box'] > 50) & (misses['dist_box'] <= 100)]
b_100_250 = misses[(misses['dist_box'] > 100) & (misses['dist_box'] <= 250)]
b_250_500 = misses[(misses['dist_box'] > 250) & (misses['dist_box'] <= 500)]
b_500_plus = misses[misses['dist_box'] > 500]

print(f"  24px - 50px  (Borderline / Adjacent Edge):     {len(b_24_50)}")
print(f"  50px - 100px (Neighboring Sibling Control):    {len(b_50_100)}")
print(f"  100px - 250px (Same Toolbar/Dialog Wrong Item): {len(b_100_250)}")
print(f"  250px - 500px (Wrong Panel / Mid-Range Miss):   {len(b_250_500)}")
print(f"  > 500px       (Wrong Window / Fallback / Null): {len(b_500_plus)}")

fallback_cnt = misses['reasoning'].fillna('').str.contains('Grounded fallback', case=False).sum()
error_cnt = misses['reasoning'].fillna('').str.contains('Error|Failed|All 25', case=False).sum()
print(f"\n=== REASONING TRACE SIGNATURES IN MISSES ===")
print(f"  Grounded Fallback (VLM returned null/unparsed ID): {fallback_cnt}")
print(f"  API / Timeout Error:                               {error_cnt}")

same_pred = 0
for i in range(1, 101):
    k_row = kaggle[kaggle['sample_idx'] == i].iloc[0]
    s_row = spatial[spatial['sample_idx'] == i].iloc[0]
    if k_row['pred_x'] == s_row['pred_x'] and k_row['pred_y'] == s_row['pred_y']:
        same_pred += 1
print(f"  Exact same (pred_x, pred_y) between kaggle_qwen_spatial and spatial_qwen27b: {same_pred}/100")

print("\n===SAMPLES 60 TO 100 MISSES===")
for _, row in misses[misses['sample_idx'] >= 60].iterrows():
    s_idx = int(row['sample_idx'])
    s_data = samples[s_idx - 1]
    box = s_data['box_coordinates']
    if len(box) == 4:
        bw, bh = box[2], box[3]
    else:
        xs, ys = box[0::2], box[1::2]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    print(f"[{s_idx:03d}] DistBox={row['dist_box']:5.1f}px | GT_Size={bw:.0f}x{bh:.0f} | Pred=({row['pred_x']:.0f},{row['pred_y']:.0f}) vs GT=({row['gt_cx']:.0f},{row['gt_cy']:.0f}) | Task: {row['instruction'][:65]}")
    print(f"      Reason: {str(row['reasoning'])[:130]}")
