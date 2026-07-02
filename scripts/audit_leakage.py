"""
Leakage Audit Script for UnifiedTMIL
Traces all sources of data leakage in the localization pipeline.
"""
import sys
import os
import pickle
import json
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load vocab
import importlib.util
spec = importlib.util.spec_from_file_location("vocab_def", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "vocab_def.py"))
vocab_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocab_mod)
sys.modules['vocab_def'] = vocab_mod
sys.modules['__main__'].Vocab = vocab_mod.Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES  = os.path.join(ROOT, "results")

print("Loading data...")
ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
pos = [b for b in ptx if b["label"] == 1]

def usable(bag):
    n = bag["length"]
    gt = [g for g in bag["gt_idx"] if g < n - 1]
    return n - 1, gt

idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
print(f"Valid positive bags: {len(idx)}")

audit = {}

# ============================================================
# LEAKAGE SOURCE A: cp_reputation tính trên toàn bộ dataset
# ============================================================
print("\n=== LEAKAGE SOURCE A: cp_reputation global vs LOO ===")

# A1: Global cp_reputation (LEAKY version - as in lean_localization_gbm.py)
cp_global = Counter()
for i in idx:
    nc, gt = usable(pos[i])
    ids = pos[i]["input_ids"][:nc]
    for g in gt:
        cp_global[ids[g]] += 1

# A2: Compute for each bag: what is GT's reputation under global vs LOO?
gt_global_reps = []
gt_loo_reps = []
non_gt_global_reps = []
non_gt_loo_reps = []

for i in idx:
    nc, gt = usable(pos[i])
    ids = pos[i]["input_ids"][:nc]
    gt_set = set(gt)
    for j in range(nc):
        cp = ids[j]
        global_rep = cp_global[cp]
        # LOO: subtract this bag's contribution
        loo_rep = global_rep - (1 if j in gt_set else 0)
        if j in gt_set:
            gt_global_reps.append(global_rep)
            gt_loo_reps.append(loo_rep)
        else:
            non_gt_global_reps.append(global_rep)
            non_gt_loo_reps.append(loo_rep)

print(f"GT transactions - global rep: mean={np.mean(gt_global_reps):.3f}, median={np.median(gt_global_reps):.1f}")
print(f"GT transactions - LOO rep:    mean={np.mean(gt_loo_reps):.3f}, median={np.median(gt_loo_reps):.1f}")
print(f"Non-GT transactions - global rep: mean={np.mean(non_gt_global_reps):.3f}")
print(f"Non-GT transactions - LOO rep:    mean={np.mean(non_gt_loo_reps):.3f}")

# Check: how many bags does GT have strictly higher global rep than all non-GT?
perfect_sep_global = 0
perfect_sep_loo = 0
for i in idx:
    nc, gt = usable(pos[i])
    ids = pos[i]["input_ids"][:nc]
    gt_set = set(gt)
    gt_reps_g = [cp_global[ids[g]] for g in gt]
    non_gt_reps_g = [cp_global[ids[j]] for j in range(nc) if j not in gt_set]
    gt_reps_l = [cp_global[ids[g]] - 1 for g in gt]
    non_gt_reps_l = [cp_global[ids[j]] for j in range(nc) if j not in gt_set]
    if gt_reps_g and non_gt_reps_g:
        if min(gt_reps_g) > max(non_gt_reps_g):
            perfect_sep_global += 1
        if min(gt_reps_l) > max(non_gt_reps_l):
            perfect_sep_loo += 1

print(f"\nBags where GT has STRICTLY HIGHER rep than ALL non-GT (global): {perfect_sep_global}/{len(idx)} = {perfect_sep_global/len(idx):.1%}")
print(f"Bags where GT has STRICTLY HIGHER rep than ALL non-GT (LOO):    {perfect_sep_loo}/{len(idx)} = {perfect_sep_loo/len(idx):.1%}")

audit["leakage_A"] = {
    "description": "cp_reputation global vs LOO",
    "gt_global_rep_mean": float(np.mean(gt_global_reps)),
    "gt_loo_rep_mean": float(np.mean(gt_loo_reps)),
    "non_gt_global_rep_mean": float(np.mean(non_gt_global_reps)),
    "non_gt_loo_rep_mean": float(np.mean(non_gt_loo_reps)),
    "perfect_sep_global": perfect_sep_global,
    "perfect_sep_loo": perfect_sep_loo,
    "n_bags": len(idx),
}

# ============================================================
# LEAKAGE SOURCE B: Positional / final-tx leakage
# ============================================================
print("\n=== LEAKAGE SOURCE B: Positional / final-tx leakage ===")

gt_positions = []
bag_sizes = []
gt_in_last2 = 0
gt_is_last = 0
gt_excluded_properly = 0

for i in idx:
    nc, gt = usable(pos[i])
    n = pos[i]["length"]
    bag_sizes.append(nc)
    for g in gt:
        gt_positions.append(g)
        if g >= nc - 2:
            gt_in_last2 += 1
        if g == n - 1:
            gt_is_last += 1
        if g < n - 1:
            gt_excluded_properly += 1

print(f"Total GT transactions: {len(gt_positions)}")
print(f"GT at position == final tx (n-1): {gt_is_last} (should be 0 after exclusion)")
print(f"GT in last 2 positions of candidate set: {gt_in_last2}/{len(gt_positions)} = {gt_in_last2/len(gt_positions):.1%}")
print(f"GT properly excluded from final tx: {gt_excluded_properly}/{len(gt_positions)} = {gt_excluded_properly/len(gt_positions):.1%}")
print(f"Mean bag size (after exclusion): {np.mean(bag_sizes):.1f}")

# Check if GT is always in top-2 of candidate set (positional leakage)
gt_in_top2_by_position = sum(1 for g in gt_positions if g >= max(0, len(gt_positions) - 2))
print(f"\nRecency prior Hit@1: {sum(1 for i in idx for g in usable(pos[i])[1] if g == usable(pos[i])[0]-1) / len(idx):.3f}")

audit["leakage_B"] = {
    "description": "Positional / final-tx leakage",
    "total_gt_txs": len(gt_positions),
    "gt_at_final_tx": gt_is_last,
    "gt_in_last2_candidates": gt_in_last2,
    "gt_in_last2_pct": float(gt_in_last2 / len(gt_positions)),
    "mean_bag_size": float(np.mean(bag_sizes)),
}

# ============================================================
# LEAKAGE SOURCE C: LambdaMART train/test split
# ============================================================
print("\n=== LEAKAGE SOURCE C: LambdaMART / GBM train/test overlap ===")

# In lean_localization_gbm.py, the LOO loop correctly excludes bag h from training
# But the cp_reputation feature is computed GLOBALLY before the LOO loop
# This means the feature for bag h already encodes information from bag h
# Let's verify this by checking the difference

# In the LOO loop in lean_localization_gbm.py:
# cp_pos_counts = compute_cp_reputation(pos, idx)  <- GLOBAL, computed ONCE
# Then for each test bag h:
#   rep[j] = cp_pos_counts[cp] - (1 if BY[i][j]==1 else 0)
# 
# BUT: the LOO subtraction only removes the GT transaction from the CURRENT bag's contribution
# It does NOT remove non-GT transactions from the test bag that happen to be counterparties
# appearing in other bags' GT transactions.
# 
# More critically: the global count includes ALL bags including the test bag.
# The "LOO" fix only subtracts 1 for the GT transaction of the test bag itself.
# This is CORRECT LOO for the GT transaction, but the feature still encodes
# information about which counterparties appear as GT in OTHER bags.
# This is a VALID discriminative signal, not leakage.

# However, in the original lean_localization_gbm.py:
# The cp_reputation is computed as count of times a CP appears as GT across ALL bags.
# This is the GLOBAL count. The "LOO" subtraction only removes the test bag's own GT.
# But the GBM model is trained on ALL OTHER bags with their GLOBAL cp_reputation features.
# When we test on bag h, we use the GLOBAL count minus 1 (for bag h's GT).
# The train features for other bags INCLUDE bag h's contribution to the global count.
# This means the train features are computed with bag h's data included.
# This is a SUBTLE LEAKAGE: the training features for bags i≠h include information
# from bag h (specifically, which CPs in bag h are GT transactions).

print("Checking if train features include test bag's data...")
print("In lean_localization_gbm.py:")
print("  cp_pos_counts = compute_cp_reputation(pos, idx)  # GLOBAL - includes ALL bags")
print("  For train bag i (i != h): rep[j] = cp_pos_counts[cp]")
print("  This includes bag h's GT contribution to the global count!")
print("  => LEAKAGE: train features for bag i include information from test bag h")
print()
print("In step23_fusion_marginal.py (CLEAN version):")
print("  rep_loo_train(tr) uses sum-minus-i trick: each train bag's rep")
print("  is computed WITHOUT its own contribution to the global count.")
print("  For test bag h: cp_reputation(tr, BIDS[h]) computes rep using ONLY train bags.")
print("  => NO LEAKAGE: test bag h's data is completely excluded from all features.")

audit["leakage_C"] = {
    "description": "LambdaMART/GBM train feature leakage via global cp_reputation",
    "lean_gbm_leaky": "cp_reputation computed globally, train features include test bag h's GT data",
    "step23_clean": "rep_loo_train uses sum-minus-i trick, test bag completely excluded",
    "lean_gbm_full_hit1": 1.0,
    "step23_clean_hit1": 0.8316831683168316,
}

# ============================================================
# LEAKAGE SOURCE D: Candidate set
# ============================================================
print("\n=== LEAKAGE SOURCE D: Candidate set analysis ===")

# Check if GT is always the highest-value transaction
gt_is_highest_amt = 0
for i in idx:
    nc, gt = usable(pos[i])
    amts = np.abs(np.asarray(pos[i]["input_amounts"][:nc], float))
    max_pos = int(np.argmax(amts))
    if max_pos in gt:
        gt_is_highest_amt += 1

print(f"GT is highest-amount tx: {gt_is_highest_amt}/{len(idx)} = {gt_is_highest_amt/len(idx):.1%}")

audit["leakage_D"] = {
    "description": "Candidate set analysis",
    "gt_is_highest_amt_count": gt_is_highest_amt,
    "gt_is_highest_amt_pct": float(gt_is_highest_amt / len(idx)),
    "n_bags": len(idx),
}

# ============================================================
# SUMMARY
# ============================================================
print("\n=== LEAKAGE SUMMARY ===")
print(f"Source A (cp_reputation global): lean_gbm_full Hit@1=1.000 vs lean_gbm_no_rep Hit@1=0.752")
print(f"  => cp_reputation is a STRONG discriminative feature")
print(f"  => But in lean_localization_gbm.py, train features include test bag's data (subtle leakage)")
print(f"  => step23_fusion_marginal.py correctly uses rep_loo_train (no leakage), Hit@1=0.832")
print(f"Source B (positional): GT at final tx = {gt_is_last} (properly excluded)")
print(f"Source C (train/test split): lean_gbm uses global cp_rep in train features -> leakage")
print(f"Source D (candidate set): GT is highest-amt in {gt_is_highest_amt/len(idx):.0%} of bags")

audit["summary"] = {
    "main_leakage_source": "cp_reputation computed globally in lean_localization_gbm.py; train features for bag i include test bag h's GT data",
    "clean_pipeline": "step23_fusion_marginal.py with rep_loo_train",
    "clean_hit1": 0.8316831683168316,
    "leaky_hit1": 1.0,
    "expected_clean_range": "0.83-0.88",
}

os.makedirs(RES, exist_ok=True)
json.dump(audit, open(os.path.join(RES, "leakage_audit_data.json"), "w"), indent=2)
print(f"\nSaved leakage audit data to results/leakage_audit_data.json")
