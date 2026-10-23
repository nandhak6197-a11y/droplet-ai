import pandas as pd
import json

df = pd.read_csv('screenshot_pro_benchmark_results/benchmark_matrix_100samples.csv')
kaggle_df = df[df['config_id'] == 'kaggle_qwen_droplets']

total = len(kaggle_df)
strict = (kaggle_df['strict_hit'] == True).sum()
tolerant = (kaggle_df['tolerant_hit'] == True).sum()
failures = kaggle_df[kaggle_df['tolerant_hit'] == False]

print(f"Total: {total}, Strict: {strict}, Tolerant: {tolerant}, Failures: {len(failures)}")

b_0_50 = len(kaggle_df[kaggle_df['dist_box'] < 50])
b_50_150 = len(kaggle_df[(kaggle_df['dist_box'] >= 50) & (kaggle_df['dist_box'] < 150)])
b_150_400 = len(kaggle_df[(kaggle_df['dist_box'] >= 150) & (kaggle_df['dist_box'] < 400)])
b_400_plus = len(kaggle_df[kaggle_df['dist_box'] >= 400])

print(f"Distance buckets:")
print(f"  0 - 50px (Tolerant Hits): {b_0_50}")
print(f"  50 - 150px (Near Misses / Sibling controls): {b_50_150}")
print(f"  150 - 400px (Same panel/window different section): {b_150_400}")
print(f"  400px+ (Opposite panel / wrong region): {b_400_plus}")

# Also compare with pure droplet and vanilla vision on the exact same samples
print("\nCross-paradigm comparison for these 55 failures:")
pure_df = df[df['config_id'] == 'pure_droplet_deterministic']
vanilla_df = df[df['config_id'] == 'vanilla_qwen_vision']
rotator_df = df[df['config_id'] == 'qwen_rotator_droplets']

failure_indices = failures['sample_idx'].tolist()
pure_hits_in_failures = pure_df[pure_df['sample_idx'].isin(failure_indices) & (pure_df['tolerant_hit'] == True)]
print(f"  Pure Droplet got correct where Kaggle failed: {len(pure_hits_in_failures)}")

rotator_hits_in_failures = rotator_df[rotator_df['sample_idx'].isin(failure_indices) & (rotator_df['tolerant_hit'] == True)]
print(f"  Rotator Qwen got correct where Kaggle failed: {len(rotator_hits_in_failures)}")

# Inspect the 55 failures in detail
print("\n--- DETAILED FAILURE BREAKDOWN (First 15) ---")
for idx, row in failures.head(15).iterrows():
    s_idx = row['sample_idx']
    inst = row['instruction']
    dist = row['dist_box']
    pred = (row['pred_x'], row['pred_y'])
    gt = (row['gt_cx'], row['gt_cy'])
    r_text = str(row['reasoning'])[:140].replace('\n', ' ')
    print(f"Sample {s_idx:02d} | Dist: {dist:5.1f}px | Pred: {pred} | GT: {gt}")
    print(f"   Task: {inst}")
    print(f"   Reasoning: {r_text}")
    print()
