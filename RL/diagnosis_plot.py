"""
Diagnostic: check whether real vs synthetic regression coefficients
are actually aligned before and after the index reassignment.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
REAL_UNNORM_CSV  = "preprocessing/MIMIC/data/unnormalized_train.csv"
REAL_NORM_CSV    = "preprocessing/MIMIC/data/normalized_train.csv"
SYN_RAW_CSV      = "Tuning_MIMIC_penalized/trial_0/synthetic.csv"           # normalized scale
SYN_RESCALED_CSV = "Tuning_MIMIC_penalized/trial_0/synthetic_rescaled.csv"  # unnormalized scale

# Key columns to inspect — edit to match your dataset
CHECK_COLS = [
    'oxygen saturation', 'mean blood pressure', 'creatinine',
    'lactate', 'bilirubin', 'glascow coma scale total',
    'Age', 'platelets'
]

OUT_DIR = Path("scale_diagnostics")
OUT_DIR.mkdir(exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading CSVs...")
df_real_unnorm  = pd.read_csv(REAL_UNNORM_CSV)
df_real_norm    = pd.read_csv(REAL_NORM_CSV)
df_syn_raw      = pd.read_csv(SYN_RAW_CSV)
df_syn_rescaled = pd.read_csv(SYN_RESCALED_CSV)

# ── check 1: summary stats table ─────────────────────────────────────────────
print("\n--- Check 1: Summary statistics ---\n")
rows = []
for col in CHECK_COLS:
    if col not in df_real_unnorm.columns:
        print(f"  Skipping {col} — not found")
        continue
    rows.append({
        'feature':           col,
        'real_unnorm_min':   df_real_unnorm[col].min(),
        'real_unnorm_max':   df_real_unnorm[col].max(),
        'real_unnorm_mean':  df_real_unnorm[col].mean(),
        'real_norm_min':     df_real_norm[col].min(),
        'real_norm_max':     df_real_norm[col].max(),
        'real_norm_mean':    df_real_norm[col].mean(),
        'syn_raw_min':       df_syn_raw[col].min(),
        'syn_raw_max':       df_syn_raw[col].max(),
        'syn_raw_mean':      df_syn_raw[col].mean(),
        'syn_rescaled_min':  df_syn_rescaled[col].min(),
        'syn_rescaled_max':  df_syn_rescaled[col].max(),
        'syn_rescaled_mean': df_syn_rescaled[col].mean(),
    })

df_stats = pd.DataFrame(rows)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df_stats.to_string(index=False))
df_stats.to_csv(OUT_DIR / "scale_summary.csv", index=False)

# ── check 2: are syn_raw values in [0,1]? ─────────────────────────────────────
print("\n--- Check 2: Are synthetic raw values in [0, 1]? ---")
for col in CHECK_COLS:
    if col not in df_syn_raw.columns:
        continue
    mn, mx = df_syn_raw[col].min(), df_syn_raw[col].max()
    in_range = (mn >= -0.05) and (mx <= 1.05)  # small tolerance
    print(f"  {col:40s}  min={mn:.4f}  max={mx:.4f}  in [0,1]: {in_range}")

# ── check 3: does syn_rescaled match unnorm range? ───────────────────────────
print("\n--- Check 3: Does syn_rescaled match unnormalized real range? ---")
for col in CHECK_COLS:
    if col not in df_syn_rescaled.columns:
        continue
    real_min = df_real_unnorm[col].min()
    real_max = df_real_unnorm[col].max()
    syn_min  = df_syn_rescaled[col].min()
    syn_max  = df_syn_rescaled[col].max()
    # Flag if synthetic range extends beyond 20% outside real range
    tol = (real_max - real_min) * 0.2
    ok  = (syn_min >= real_min - tol) and (syn_max <= real_max + tol)
    print(f"  {col:40s}  real=[{real_min:.2f}, {real_max:.2f}]  "
          f"syn=[{syn_min:.2f}, {syn_max:.2f}]  ok: {ok}")

# ── check 4: distribution plots — all four versions side by side ─────────────
print("\n--- Check 4: Plotting distributions ---")
n_cols_plot = 4
n_rows_plot = int(np.ceil(len(CHECK_COLS) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                          figsize=(n_cols_plot * 4, n_rows_plot * 3))
axes = np.array(axes).flatten()

for i, col in enumerate(CHECK_COLS):
    if col not in df_real_unnorm.columns:
        axes[i].set_visible(False)
        continue
    ax = axes[i]
    ax.hist(df_real_unnorm[col].dropna(),  bins=40, alpha=0.4, density=True,
            color='steelblue', label='real unnorm')
    ax.hist(df_real_norm[col].dropna(),    bins=40, alpha=0.4, density=True,
            color='green',     label='real norm')
    ax.hist(df_syn_raw[col].dropna(),      bins=40, alpha=0.4, density=True,
            color='tomato',    label='syn raw')
    ax.hist(df_syn_rescaled[col].dropna(), bins=40, alpha=0.4, density=True,
            color='orange',    label='syn rescaled')
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=6)

for j in range(len(CHECK_COLS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Distribution comparison: real unnorm / real norm / syn raw / syn rescaled',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "distribution_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {OUT_DIR / 'distribution_comparison.png'}")

# ── check 5: is syn_rescaled ≈ real_unnorm or ≈ real_norm? ───────────────────
print("\n--- Check 5: Which does syn_rescaled match more closely? ---")
from scipy.stats import wasserstein_distance
for col in CHECK_COLS:
    if col not in df_syn_rescaled.columns:
        continue
    wd_unnorm = wasserstein_distance(df_real_unnorm[col].dropna().values,
                                     df_syn_rescaled[col].dropna().values)
    wd_norm   = wasserstein_distance(df_real_norm[col].dropna().values,
                                     df_syn_rescaled[col].dropna().values)
    closer_to = "unnorm" if wd_unnorm < wd_norm else "NORM (possible double-norm!)"
    print(f"  {col:40s}  WD_unnorm={wd_unnorm:.4f}  WD_norm={wd_norm:.4f}  "
          f"closer to: {closer_to}")

print(f"\nDone. Results saved to {OUT_DIR}/")


for col in ['vaso', 'fraction inspired oxygen', 'vent']:
    print(f"{col}:")
    print(f"  real mean: {df_real_unnorm[col].mean():.4f}")
    print(f"  syn  mean: {df_syn_rescaled[col].mean():.4f}")
    print(f"  real std:  {df_real_unnorm[col].std():.4f}")
    print(f"  syn  std:  {df_syn_rescaled[col].std():.4f}")


cols = ['vent', 'vaso', 'fraction inspired oxygen']
print("Real correlations:")
print(df_real_unnorm[cols].corr())
print("\nSynthetic correlations:")
print(df_syn_rescaled[cols].corr())