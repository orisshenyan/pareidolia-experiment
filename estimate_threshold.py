# ============================================
# estimate_threshold.py
# 
# Standalone script to compute Bayesian
# psychometric threshold from training data.
#
# Usage:  python estimate_threshold.py
#         (enter subject number when prompted)
# ============================================

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
import pymc as pm

# ============================================
# Settings
# ============================================
PRIOR_MEAN_THRESHOLD = 0.264
PRIOR_SD_THRESHOLD   = 0.085
PRIOR_MEAN_BETA      = 7.6
PRIOR_SD_BETA        = 3.0
TARGET_PERFORMANCE   = 0.65
LOGIT_TARGET         = np.log(TARGET_PERFORMANCE / (1 - TARGET_PERFORMANCE))
RANDOM_SEED          = 31032026
DEFAULT_OPACITY      = 0.28

DATA_ROOT = "data/"


# ============================================
# Get subject number
# ============================================
subject_id = input("Enter subject number: ").strip()

if not subject_id:
    print("[ERROR] No subject number entered.")
    sys.exit(1)

subject_dir = os.path.join(DATA_ROOT, f"sub{subject_id}")

if not os.path.isdir(subject_dir):
    print(f"[ERROR] Subject directory not found: {subject_dir}")
    sys.exit(1)

print(f"\n[INFO] Subject directory: {subject_dir}")


# ============================================
# Find training block CSVs
# ============================================
block_files = sorted(glob.glob(os.path.join(subject_dir, "*_trainingblock*.csv")))

# Exclude practice files
block_files = [f for f in block_files if "_practice" not in os.path.basename(f)]

if not block_files:
    print(f"[ERROR] No training block CSVs found in {subject_dir}")
    print("        (practice files are excluded)")
    sys.exit(1)

print(f"[INFO] Found {len(block_files)} training block file(s):")
for f in block_files:
    print(f"         {os.path.basename(f)}")


# ============================================
# Load & concatenate
# ============================================
all_blocks = pd.concat([pd.read_csv(f) for f in block_files], ignore_index=True)
print(f"\n[INFO] Total trials loaded: {len(all_blocks)}")


# ============================================
# Score accuracy
# ============================================
# Correct = reported left when face was left, or reported right when face was right
all_blocks['Correct'] = (
    ((all_blocks['Direction_Report'] == 1) & (all_blocks['Face_Position_X'] < 0)) |
    ((all_blocks['Direction_Report'] == 2) & (all_blocks['Face_Position_X'] > 0))
).astype(int)

opacities = all_blocks['Opacity'].values.astype(float)
correct   = all_blocks['Correct'].values.astype(int)

print(f"[INFO] Overall accuracy: {correct.mean():.2f}")
print(f"[INFO] Unique opacities: {sorted(np.unique(opacities))}")
print()

# Per-opacity breakdown
print(f"  {'Opacity':>8}  {'Accuracy':>8}  {'N':>5}")
print(f"  {'─'*8}  {'─'*8}  {'─'*5}")
for op in sorted(np.unique(opacities)):
    mask = opacities == op
    print(f"  {op:8.2f}  {correct[mask].mean():8.2f}  {mask.sum():5d}")


# ============================================
# Fit Bayesian psychometric model
# ============================================
print(f"\n{'=' * 55}")
print(f"  FITTING BAYESIAN MODEL")
print(f"{'=' * 55}")

with pm.Model() as model:

    threshold = pm.Normal(
        'threshold',
        mu=PRIOR_MEAN_THRESHOLD,
        sigma=PRIOR_SD_THRESHOLD,
    )

    beta = pm.TruncatedNormal(
        'beta',
        mu=PRIOR_MEAN_BETA,
        sigma=PRIOR_SD_BETA,
        lower=0,
    )

    p_correct = pm.math.sigmoid(
        beta * (opacities - threshold) + LOGIT_TARGET
    )

    obs = pm.Bernoulli('obs', p=p_correct, observed=correct)

    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.99,
        return_inferencedata=True,
        progressbar=True,
        random_seed=RANDOM_SEED,
        initvals={
            'threshold': PRIOR_MEAN_THRESHOLD,  # start at prior mean
            'beta':      PRIOR_MEAN_BETA,        # must be > 0 for TruncatedNormal
        },
    )


# ============================================
# Extract posterior
# ============================================
threshold_samples = trace.posterior['threshold'].values.flatten()
beta_samples      = trace.posterior['beta'].values.flatten()

threshold_median = np.median(threshold_samples)
threshold_mean   = np.mean(threshold_samples)
threshold_sd     = np.std(threshold_samples)
threshold_lower  = np.quantile(threshold_samples, 0.025)
threshold_upper  = np.quantile(threshold_samples, 0.975)

beta_median = np.median(beta_samples)
beta_lower  = np.quantile(beta_samples, 0.025)
beta_upper  = np.quantile(beta_samples, 0.975)

print(f"\n{'=' * 55}")
print(f"  RESULTS — Subject {subject_id}")
print(f"{'=' * 55}")
print(f"  Threshold (65% correct point):")
print(f"    Median:   {threshold_median:.4f}")
print(f"    Mean:     {threshold_mean:.4f}  (SD: {threshold_sd:.4f})")
print(f"    95% CI:   [{threshold_lower:.4f}, {threshold_upper:.4f}]")
print(f"  Slope (beta):")
print(f"    Median:   {beta_median:.2f}")
print(f"    95% CI:   [{beta_lower:.2f}, {beta_upper:.2f}]")
print(f"{'=' * 55}")


# ============================================
# Sanity check
# ============================================
if threshold_median < 0 or threshold_median > 1:
    print(f"\n[WARN] Threshold {threshold_median:.4f} is outside [0, 1]!")
    print(f"       Falling back to default: {DEFAULT_OPACITY}")
    threshold_median = DEFAULT_OPACITY

if threshold_sd > 0.15:
    print(f"\n[WARN] Posterior SD is large ({threshold_sd:.4f}).")
    print(f"       Estimate may be unreliable.")


# ============================================
# Save — plain text (environment-safe)
# ============================================
opac_txt_path = os.path.join(subject_dir, f"sub{subject_id}_prac_opac.txt")
with open(opac_txt_path, 'w') as f:
    f.write(f"{threshold_median}")
print(f"[SAVED] Experiment opacity txt:  {opac_txt_path}")

# ============================================
# Save — full posterior as CSV (not pickle)
# ============================================
posterior_csv_path = os.path.join(subject_dir, f"sub{subject_id}_bayes_posterior.csv")
pd.DataFrame({
    'threshold': threshold_samples,
    'beta':      beta_samples,
}).to_csv(posterior_csv_path, index=False)
print(f"[SAVED] Posterior samples csv:   {posterior_csv_path}")


# ============================================
# Save — full posterior samples
# ============================================
posterior_pkl_path = os.path.join(subject_dir, f"sub{subject_id}_bayes_posterior.pkl")
with open(posterior_pkl_path, 'wb') as f:
    pickle.dump({
        'threshold_samples': threshold_samples,
        'beta_samples':      beta_samples,
        'opacities_used':    opacities,
        'correct':           correct,
        'n_trials':          len(correct),
        'seed':              RANDOM_SEED,
    }, f)
print(f"[SAVED] Posterior samples pkl:   {posterior_pkl_path}")


# ============================================
# Save — summary CSV
# ============================================
summary = pd.DataFrame([{
    'subject':            subject_id,
    'threshold_median':   threshold_median,
    'threshold_mean':     threshold_mean,
    'threshold_sd':       threshold_sd,
    'threshold_lower_95': threshold_lower,
    'threshold_upper_95': threshold_upper,
    'beta_median':        beta_median,
    'beta_lower_95':      beta_lower,
    'beta_upper_95':      beta_upper,
    'n_trials':           len(correct),
    'overall_accuracy':   correct.mean(),
    'n_blocks':           len(block_files),
    'seed':               RANDOM_SEED,
}])

summary_csv_path = os.path.join(subject_dir, f"sub{subject_id}_bayes_threshold.csv")
summary.to_csv(summary_csv_path, index=False)
print(f"[SAVED] Threshold summary csv:   {summary_csv_path}")


# ============================================
# Done
# ============================================
print(f"\n{'=' * 55}")
print(f"  DONE — Use opacity {threshold_median:.4f} for subject {subject_id}")
print(f"{'=' * 55}")