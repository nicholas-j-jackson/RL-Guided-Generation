import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os


def column_wise_correlations(df_real, df_syn, out_dir, H):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_corr = df_real.corr()
    syn_corr  = df_syn.corr().fillna(0)
    real_corr.to_csv(out / "real_corr_matrix.csv")
    syn_corr.to_csv(out / "syn_corr_matrix.csv")
    n = real_corr.shape[1]
    cwc = norm((real_corr - syn_corr).values, "fro") / (n ** 2)# * 1e6
    np.savetxt(out / "column_wise_corr_score.txt", [cwc], header="cwc_score_×1e-6")
    fig, axes = plt.subplots(1, 2, figsize=(25,20), constrained_layout=True)
    for ax, corr, title in zip(axes, (real_corr, syn_corr), ("Real correlation", "Synthetic correlation")):
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(H.NUM_COLS+H.BIN_COLS+H.CAT_COLS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(H.NUM_COLS+H.BIN_COLS+H.CAT_COLS, fontsize=7)
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Pearson r").ax.tick_params(labelsize=9)
    fig.savefig(out / "corr_heatmaps_real_vs_synth.png", dpi=300)
    plt.close(fig)
    return float(cwc)

def classification_utility(df_train, df_test, label_col):
        
    df_train, df_val = train_test_split(df_train, test_size=0.2)

    train_data = lgb.Dataset(df_train.drop(columns=[label_col]), label=df_train[label_col], params={"verbosity": -1})
    val_data = lgb.Dataset(df_val.drop(columns=[label_col]), label=df_val[label_col], params={"verbosity": -1})
    #test_data = lgb.Dataset(df_test.drop(columns=[label_col]), label=df_test[label_col], params={"verbosity": -1})

    num_round = 100
    param = {'metric': 'auc', 'objective': 'binary', 'verbosity':-1}
    bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])

    proba = bst.predict(df_test.drop(columns=label_col))
    pred = (proba >= 0.5).astype(int)

    warnings.filterwarnings("ignore", category=UserWarning)

    auc = roc_auc_score(df_test[label_col], proba)
    acc = accuracy_score(df_test[label_col], pred)
    prec = precision_score(df_test[label_col], pred)
    rec = recall_score(df_test[label_col], pred)
    f1 = f1_score(df_test[label_col], pred)

    return auc, acc


def membership_inference_risk(train_df, test_df, synth_df, H): 
    #  
    for df in (train_df, test_df, synth_df):
        df[H.BIN_COLS] = (df[H.BIN_COLS] >= 0.5).astype(float)
        df[H.CAT_COLS] = (df[H.CAT_COLS] >= 0.5).astype(float)

    # Compute min distance between real records and synthetic ones
    mins = []
    train = []
    
    n_syn = min(20_000, len(synth_df))
    n_real = min(2_500, len(train_df) // 2, len(test_df))

    synthetic_vals = synth_df.sample(n_syn, random_state=0).values
    real_train_vals = train_df.sample(n_real, random_state=0).values
    real_test_vals = test_df.sample(n_real, random_state=0).values

    for i in range(len(real_train_vals)):
        dist = np.linalg.norm(synthetic_vals - real_train_vals[i], axis=1)
        mins.append(dist.min())
        train.append(1)
    
    for i in range(len(real_test_vals)):
        dist = np.linalg.norm(synthetic_vals - real_test_vals[i], axis=1)
        mins.append(dist.min())
        train.append(0) 

    #
    df = pd.DataFrame({'min_dist':mins, 'true':train})
    df = df[df['true'] <= 1]

    # Compute AUC across many thresholds
    Sens = []
    Spec = []
    for threshold in np.linspace(df['min_dist'].min()*0.9, df['min_dist'].max()*1.1, 10_001):
        df['pred'] = (df['min_dist'] < threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(df['true'], df['pred']).ravel()
        spec = tn / (tn + fp)
        sens = recall_score(df['true'], df['pred'])
        prec = precision_score(df['true'], df['pred'])
        acc = accuracy_score(df['true'], df['pred'])

        Sens.append(sens)
        Spec.append(spec)

    Sens = np.array(Sens)
    Spec = np.array(Spec)

    return auc(Sens, 1-Spec)


# DOMIAS MIA (Breugel et al. 2023)
# p_syn and p_ref are each estimated with scipy gaussian_kde and the membership score is the density ratio p_syn(x) / p_ref(x).
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


def _ios_keep_cols(H):
    """Columns to feed the KDE. We drop one dummy from each one-hot categorical
    group: the K dummies in a group sum to 1, so keeping all of them makes the
    data covariance singular and gaussian_kde fails. Numeric and binary columns
    are kept as-is."""
    keep = list(H.NUM_COLS) + list(H.BIN_COLS)
    cat = list(H.CAT_COLS)
    i = 0
    for dim in H.CAT_DIMS:
        keep += cat[i:i + dim][:-1]   # drop the last dummy of each group
        i += dim
    return keep


def _ios_prep_matrix(df, H, keep_cols):
    """Binarize the binary/categorical (one-hot) columns at 0.5, keep numeric
    columns as-is, and return the reduced (non-collinear) column set as a matrix."""
    df = df.copy()
    bin_cat = [c for c in (list(H.BIN_COLS) + list(H.CAT_COLS)) if c in df.columns]
    df[bin_cat] = (df[bin_cat] >= 0.5).astype(float)
    return df[keep_cols].values.astype(float)


def _ios_fit_logdensity(X, seed=0):
    """Fit a Gaussian kernel density estimator to X (rows = samples) and return a
    callable mapping rows -> log-density. Matches the DOMIAS 'kde' estimator
    (scipy.stats.gaussian_kde). If the data covariance is still singular, retry
    once with negligible jitter to make it positive-definite."""
    try:
        kde = gaussian_kde(X.T)
    except np.linalg.LinAlgError:
        rng = np.random.RandomState(seed)
        kde = gaussian_kde((X + rng.normal(0.0, 1e-6, size=X.shape)).T)
    return lambda Q: kde.logpdf(Q.T)


def _ios_tpr_at_fpr(labels, scores, target_fpr):
    """TPR at the largest achievable FPR <= target_fpr. This is the
    'most-confidently-flagged records' view (Carlini et al.): it asks whether
    the records the attack is *surest* about are actually members -- the
    practically-relevant 'worst-case record' lens, not a worst-case adversary."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def inference_on_synthetic_risk(train_df, test_df, synth_df, H, seed=0, n_eval=1000):
    """DOMIAS-style inference-on-synthetic membership inference attack.

    Members are real training records; non-members are held-out real records.
    A disjoint slice of the held-out real data serves as the population
    reference, so no record is both scored and used to fit p_ref. Returns a dict
    with the overall AUC plus TPR at low FPR (the most-exposed-record view).
    Higher density-ratio score => more likely a training member; AUC ~ 0.5 means
    a competent realistic attack cannot distinguish members from non-members.
    """
    rng = np.random.RandomState(seed)
    keep = _ios_keep_cols(H)

    # Fit p_syn on the released synthetic data.
    n_syn = min(20_000, len(synth_df))
    Xs = _ios_prep_matrix(synth_df.sample(n_syn, random_state=seed), H, keep)

    # Held-out real data supplies both the non-member queries and the population
    # reference for p_ref; the two are kept disjoint. We fix the number of member
    # and non-member queries at n_eval each so the evaluation size is identical
    # across datasets and runs; the remaining held-out records fit p_ref.
    perm = rng.permutation(len(test_df))
    n_query = min(n_eval, len(train_df), len(test_df) - 1)
    nonmem_rows = test_df.iloc[perm[:n_query]]          # non-member queries
    ref_rows    = test_df.iloc[perm[n_query:]]          # disjoint population reference

    Xm = _ios_prep_matrix(train_df.sample(n_query, random_state=seed), H, keep)   # members
    Xn = _ios_prep_matrix(nonmem_rows, H, keep)                                   # non-members
    Xref = _ios_prep_matrix(ref_rows, H, keep)                                    # reference

    # Standardize features using statistics from the real records (shouldn't matter for Gaussian KDEs, but DOMIAS uses it)
    scaler = StandardScaler().fit(np.vstack([Xm, Xn, Xref]))
    Xs, Xm, Xn, Xref = (scaler.transform(Xs), scaler.transform(Xm),
                        scaler.transform(Xn), scaler.transform(Xref))

    logp_syn = _ios_fit_logdensity(Xs, seed=seed)
    logp_ref = _ios_fit_logdensity(Xref, seed=seed)

    Xq = np.vstack([Xm, Xn])
    labels = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xn))])
    scores = logp_syn(Xq) - logp_ref(Xq)                                        # DOMIAS log-ratio

    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]

    return {
        "auc": float(roc_auc_score(labels, scores)),
        "tpr_at_fpr_0.1": _ios_tpr_at_fpr(labels, scores, 0.1),
        "tpr_at_fpr_0.01": _ios_tpr_at_fpr(labels, scores, 0.01),
        "n_eval_per_class": int(len(Xm)),
        "n_ref": int(len(Xref)),
    }


from scipy.stats import wasserstein_distance
def plot_numeric_histograms(df_real, df_syn, out_dir, num_cols):
    out = Path(out_dir) / "numeric_histograms"
    out.mkdir(parents=True, exist_ok=True)

    n_cols = 5
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    cols_per_fig = n_cols * 6  # 6 rows per figure max
    chunks = [num_cols[i:i + cols_per_fig] for i in range(0, len(num_cols), cols_per_fig)]

    for fig_idx, chunk in enumerate(chunks):
        n_chunk_rows = int(np.ceil(len(chunk) / n_cols))
        fig, axes = plt.subplots(n_chunk_rows, n_cols, figsize=(n_cols * 3.5, n_chunk_rows * 3))
        axes = np.array(axes).flatten()

        for i, col in enumerate(chunk):
            ax = axes[i]
            ax.hist(df_real[col].dropna(), bins=40, alpha=0.5, label='Real', color='steelblue', density=True)
            ax.hist(df_syn[col].dropna(), bins=40, alpha=0.5, label='Synthetic', color='tomato', density=True)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=8)

        # Hide unused axes
        for j in range(len(chunk), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Continuous Feature Distributions: Real vs. Synthetic', fontsize=12, y=1.01)
        fig.tight_layout()
        suffix = f"_{fig_idx + 1}" if len(chunks) > 1 else ""
        fig.savefig(out / f"numeric_histograms{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_categorical_prevalences(df_real, df_syn, out_dir, bin_cols, cat_cols):
    out = Path(out_dir) / "prevalences"
    out.mkdir(parents=True, exist_ok=True)

    all_cols = bin_cols + cat_cols
    real_prevs = [df_real[col].mean() for col in all_cols]
    syn_prevs = [df_syn[col].mean() for col in all_cols]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Color binary and categorical differently
    colors = ['steelblue'] * len(bin_cols) + ['tomato'] * len(cat_cols)
    for x, y, col, c in zip(real_prevs, syn_prevs, all_cols, colors):
        ax.scatter(x, y, color=c, s=60, zorder=3)
        ax.annotate(col, (x, y), fontsize=6.5, textcoords='offset points', xytext=(5, 3))

    # Diagonal reference line (perfect agreement)
    lim_max = max(max(real_prevs), max(syn_prevs)) * 1.1
    ax.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, label='Perfect agreement', zorder=2)

    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel('Real Prevalence', fontsize=11)
    ax.set_ylabel('Synthetic Prevalence', fontsize=11)
    ax.set_title('Real vs. Synthetic Prevalence: Binary & Categorical Features', fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Binary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=8, label='Categorical'),
        Line2D([0], [0], linestyle='--', color='k', label='Perfect agreement')
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / "prevalences_scatter.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_dwd(df_real, df_syn, num_cols, bin_cols, cat_cols):
    """
    Dimension-Wise Difference (DWD):
    - Binary/categorical: Absolute Prevalence Difference (APD)
    - Continuous: Wasserstein distance, normalized to [0, 1]
    - DWD = sum of all APDs and normalized AWDs
    """

    apds = []
    awds = []

    # APD for binary and categorical columns
    for col in bin_cols + cat_cols:
        real_prev = df_real[col].mean()
        syn_prev = df_syn[col].mean()
        apds.append(abs(real_prev - syn_prev))

    # Wasserstein distance for continuous columns
    raw_wds = []
    for col in num_cols:
        wd = wasserstein_distance(
            df_real[col].dropna().values,
            df_syn[col].dropna().values
        )
        raw_wds.append(wd)

    # Normalize Wasserstein distances to [0, 1] using the range of the real data
    for i, col in enumerate(num_cols):
        col_range = df_real[col].max() - df_real[col].min()
        normalized = raw_wds[i] / col_range if col_range > 0 else 0.0
        awds.append(normalized)

    dwd = sum(apds) + sum(awds)

    return {
        'dwd': dwd,
        'apds': dict(zip(bin_cols + cat_cols, apds)),
        'awds': dict(zip(num_cols, awds)),
        'mean_apd': np.mean(apds) if apds else 0.0,
        'mean_awd': np.mean(awds) if awds else 0.0,
    }

import statsmodels.api as sm

def prepare_mimic(df):
    """
    SOFA-aligned feature selection:
    - Respiratory: PaO2/FiO2 ratio (use SpO2 & FiO2 as proxy), mechanical ventilation
    - Cardiovascular: MAP, vasopressor, lactate
    - Renal: creatinine
    - Coagulation: platelets
    - Hepatic: bilirubin
    - Neurological: GCS
    - Demographics & admission context
    """
    features_df = pd.DataFrame({
        # Respiratory (PaO2/FiO2 is the SOFA input; use available proxies)
        'Oxygen Saturation': df['oxygen saturation'],
        #'Fraction Inspired Oxygen': df['fraction inspired oxygen'],
        'Mechanical Ventilation': df['vent'],

        # Cardiovascular
        'Mean Blood Pressure': df['mean blood pressure'],
        'Vasopressor': df['vaso'],
        'Lactate': df['lactate'],

        # Renal
        'Creatinine': df['creatinine'],

        # Coagulation
        'Platelets': df['platelets'],

        # Hepatic (was missing before)
        'Bilirubin': df['bilirubin'],

        # Neurological (was missing before)
        'GCS Total': df['glascow coma scale total'],

        # Demographics
        'Age': df['Age'],
        'Male': df['Male'],
        'Black/African American': df['Black/African American'],
        'Other Race': df['Other Race'],
    })

    label = df['mortality']
    return features_df, label


def prepare_acs(df):
    features_df = pd.DataFrame({
        'Age': df['Age'],
        'Years of School': df['Years of School'],
        'Male': df['Male'],
        'Asian': df['Asian'],
        'Black/African American': df['Black/African American'],
        'Other Race': df['Other'],
        'Two or More': df['Two or More'],
        # White is reference, dropped
    })
    label = df['Public Assistance Income']
    return features_df, label


def fit_mimic_regression(df):
    features_df, label = prepare_mimic(df)
    
    constant_cols = [c for c in features_df.columns if features_df[c].nunique() <= 1]
    if constant_cols:
        print(f"Skipping regression — zero variance in: {constant_cols}")
        return None, features_df.columns.tolist()

    X = sm.add_constant(features_df, has_constant='add')
    return sm.Logit(label, X).fit(disp=0), features_df.columns.tolist()


def eval_mimic_regression(model, df_test):
    features_df, label = prepare_mimic(df_test)
    X = sm.add_constant(features_df)
    proba = model.predict(X)
    return roc_auc_score(label, proba)

def eval_acs_regression(model, df_test):
    features_df, label = prepare_acs(df_test)
    X = sm.add_constant(features_df, has_constant='add')
    preds = model.predict(X)
    return mean_squared_error(label, preds)

def fit_acs_regression(df):
    features_df, label = prepare_acs(df)
    X = sm.add_constant(features_df, has_constant='add')
    return sm.OLS(label, X).fit(), features_df.columns.tolist()


def fit_mimic_regression_subgroup(df, subgroup_col):
    df_sub = df[df[subgroup_col] > 0.5].copy()
    if len(df_sub) < 100:
        print(f"Warning: only {len(df_sub)} rows for {subgroup_col}")
        return None, None

    features_df, label = prepare_mimic(df_sub)

    # Drop all race columns — within a subgroup they are either all 1s or all 0s
    race_cols = ['Black/African American', 'Other Race']
    features_df = features_df.drop(columns=[c for c in race_cols if c in features_df.columns])

    # Drop any remaining constant columns
    constant_cols = [c for c in features_df.columns if features_df[c].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant columns for {subgroup_col}: {constant_cols}")
        features_df = features_df.drop(columns=constant_cols)

    if features_df.shape[1] == 0:
        return None, None

    try:
        X = sm.add_constant(features_df)
        model = sm.Logit(label, X).fit(disp=0, maxiter=200)
        return model, features_df.columns.tolist()
    except Exception as e:
        print(f"Regression failed for {subgroup_col}: {e}")
        return None, None


def fit_acs_regression_subgroup(df, subgroup_col):
    df_sub = df[df[subgroup_col] > 0.5].copy()
    if len(df_sub) < 100:
        print(f"Warning: only {len(df_sub)} rows for {subgroup_col}")
        return None, None

    features_df, label = prepare_acs(df_sub)

    # Drop all race columns
    race_cols = ['Asian', 'Black/African American', 'Other Race', 'Two or More']
    features_df = features_df.drop(columns=[c for c in race_cols if c in features_df.columns])

    # Drop any remaining constant columns
    constant_cols = [c for c in features_df.columns if features_df[c].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant columns for {subgroup_col}: {constant_cols}")
        features_df = features_df.drop(columns=constant_cols)

    if features_df.shape[1] == 0:
        return None, None

    try:
        X = sm.add_constant(features_df)
        model = sm.OLS(label, X).fit()
        return model, features_df.columns.tolist()
    except Exception as e:
        print(f"Regression failed for {subgroup_col}: {e}")
        return None, None


def regression_analysis(df_train, df_test, df_syn, H):
    out_dir = Path(H.OUT_DIR) / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    if H.DATASET == 'MIMIC':
        fit_fn = fit_mimic_regression
        eval_fn = eval_mimic_regression
    else:
        fit_fn = fit_acs_regression
        eval_fn = eval_acs_regression

    """model_real, features = fit_fn(df_train)
    model_syn, _ = fit_fn(df_syn)

    # Extract coefficients and CIs (drop intercept)
    coefs_real = model_real.params[1:]
    coefs_syn = model_syn.params[1:]
    ci_real = model_real.conf_int().iloc[1:]
    ci_syn = model_syn.conf_int().iloc[1:]"""

    model_real, features = fit_fn(df_train)
    model_syn, _         = fit_fn(df_syn)

    # Evaluate predictive performance on held-out real test set
    real_model_auc = eval_fn(model_real, df_test) if eval_fn else None
    syn_model_auc  = eval_fn(model_syn,  df_test) if eval_fn else None

    if model_real is None or model_syn is None:
        print("Skipping regression analysis — degenerate features")
        return {'mae': None, 'correlation': None, 'summary': None,
                'real_model_auc': None, 'syn_model_auc': None}

    # Align to shared features by name, not position
    shared = [f for f in features if f in model_real.params.index and f in model_syn.params.index]

    coefs_real = model_real.params[shared]
    coefs_syn  = model_syn.params[shared]
    ci_real    = model_real.conf_int().loc[shared]
    ci_syn     = model_syn.conf_int().loc[shared]


    plot_forest(coefs_real, coefs_syn,
                ci_real[0].values, ci_real[1].values,
                ci_syn[0].values, ci_syn[1].values,
                out_dir, H.DATASET)

    coef_mae = np.mean(np.abs(coefs_real.values - coefs_syn.values))
    coef_corr = np.corrcoef(coefs_real.values, coefs_syn.values)[0, 1]

    summary = pd.DataFrame({
        'feature': shared,
        'coef_real': coefs_real.values,
        'coef_syn': coefs_syn.values,
        'ci_low_real': ci_real[0].values,
        'ci_high_real': ci_real[1].values,
        'ci_low_syn': ci_syn[0].values,
        'ci_high_syn': ci_syn[1].values,
        'abs_diff': np.abs(coefs_real.values - coefs_syn.values)
    })
    summary.to_csv(out_dir / f"regression_summary_{H.DATASET.lower()}.csv", index=False)

    return {'mae': coef_mae, 'correlation': coef_corr, 'summary': summary, 'real_model_auc': real_model_auc, 'syn_model_auc':  syn_model_auc}


def subgroup_regression_analysis(df_train, df_syn, H):
    out_dir = Path(H.OUT_DIR) / "regression" / "subgroups"
    out_dir.mkdir(parents=True, exist_ok=True)

    if H.DATASET == 'MIMIC':
        subgroups = {
            'Black/African American': fit_mimic_regression_subgroup,
            'Other Race': fit_mimic_regression_subgroup,
        }
    else:
        subgroups = {
            'Black/African American': fit_acs_regression_subgroup,
            'Other': fit_acs_regression_subgroup,
            'Two or More': fit_acs_regression_subgroup,
        }

    results = {}

    for subgroup_col, fit_fn in subgroups.items():
        #print(subgroup_col, df_train[subgroup_col].sum(), df_syn[subgroup_col].sum())
        model_real, features = fit_fn(df_train, subgroup_col)
        model_syn, _ = fit_fn(df_syn, subgroup_col)

        if model_real is None or model_syn is None:
            print(f"Skipping {subgroup_col} — insufficient data")
            continue

        coefs_real = model_real.params[1:]
        coefs_syn = model_syn.params[1:]
        ci_real = model_real.conf_int().iloc[1:]
        ci_syn = model_syn.conf_int().iloc[1:]

        coefs_real.index = features
        coefs_syn.index = features
        ci_real.index = features
        ci_syn.index = features

        plot_forest(
            coefs_real, coefs_syn,
            ci_real[0].values, ci_real[1].values,
            ci_syn[0].values, ci_syn[1].values,
            out_dir, f"{H.DATASET}_{subgroup_col.replace('/', '_')}"
        )

        mae = np.mean(np.abs(coefs_real.values - coefs_syn.values))
        corr = np.corrcoef(coefs_real.values, coefs_syn.values)[0, 1]

        results[subgroup_col] = {
            'mae': mae,
            'correlation': corr,
            'n_real': int((df_train[subgroup_col] > 0.5).sum()),
            'n_syn': int((df_syn[subgroup_col] > 0.5).sum()),
        }

        #print(f"{subgroup_col}: n_real={results[subgroup_col]['n_real']} | "
        #      f"n_syn={results[subgroup_col]['n_syn']} | "
        #      f"MAE={mae:.4f} | corr={corr:.4f}")

    # Save summary
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(out_dir / f"subgroup_summary_{H.DATASET.lower()}.csv")

    return results


def plot_forest(coefs_real, coefs_syn, ci_low_real, ci_high_real,
                ci_low_syn, ci_high_syn, out_dir, dataset_name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = coefs_real.index.tolist()
    n = len(features)
    y_real = np.arange(n) + 0.15
    y_syn = np.arange(n) - 0.15

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.7)))

    # Real coefficients
    ax.scatter(coefs_real.values, y_real, color='steelblue', zorder=4, s=60, label='Real')
    ax.hlines(y_real, ci_low_real, ci_high_real, color='steelblue', linewidth=2, alpha=0.7)

    # Synthetic coefficients
    ax.scatter(coefs_syn.values, y_syn, color='tomato', zorder=4, s=60, label='Synthetic')
    ax.hlines(y_syn, ci_low_syn, ci_high_syn, color='tomato', linewidth=2, alpha=0.7)

    # Reference line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Coefficient (Standardized)', fontsize=11)
    ax.set_title(f'{dataset_name}: Real vs. Synthetic Regression Coefficients', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out / f"forest_plot_{dataset_name.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)



def evaluate_model(df_train, df_test, df_syn, H): 
    cols = H.NUM_COLS+H.BIN_COLS+H.CAT_COLS
    os.makedirs(H.OUT_DIR, exist_ok=True)

    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = column_wise_correlations(df_train[cols], df_syn[cols], f"{H.OUT_DIR}/correlations", H) 

    # Univariate plots and DWD
    plot_numeric_histograms(df_train[H.NUM_COLS], df_syn[H.NUM_COLS], H.OUT_DIR, H.NUM_COLS)
    plot_categorical_prevalences(df_train, df_syn, H.OUT_DIR, H.BIN_COLS, H.CAT_COLS)
    dwd_results = compute_dwd(df_train[cols], df_syn[cols], H.NUM_COLS, H.BIN_COLS, H.CAT_COLS)

    #train on synthetic data and test on real data to hold out 
    s2r_auc, s2r_acc = classification_utility(df_syn[cols], df_test[cols], H.LABEL)

    # Membership inference via the inference-on-synthetic (DOMIAS-style) attack.
    # This is the reported MIA; fixed 1000 members / 1000 non-members so the
    # evaluation size is identical across datasets, seeds, and fractions.
    try:
        ios = inference_on_synthetic_risk(df_train[cols], df_test[cols], df_syn[cols], H, n_eval=1000)
    except Exception as e:
        ios = {"auc": float('nan'), "tpr_at_fpr_0.1": float('nan'),
               "tpr_at_fpr_0.01": float('nan'), "error": str(e)}
    mem_auc = ios['auc']

    # Regression analysis
    regression_results = regression_analysis(df_train, df_test, df_syn, H)
    #subgroup_results = subgroup_regression_analysis(df_train, df_syn, H)



    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"--- Fidelity ---\n")
        f.write(f"CWC: {cwc:.4f}\n")
        f.write(f"DWD: {dwd_results['dwd']:.4f} (mean APD: {dwd_results['mean_apd']:.4f}, mean AWD: {dwd_results['mean_awd']:.4f})\n")
        f.write(f"\n--- Utility ---\n")
        f.write(f"S2R AUC: {s2r_auc:.4f} | S2R Acc: {s2r_acc:.4f}\n")
        f.write(f"Regression MAE: {regression_results['mae']:.4f} | Coef Correlation: {regression_results['correlation']:.4f}\n")
        if regression_results.get('real_model_auc') is not None:
            f.write(f" | Real Model AUC: {regression_results['real_model_auc']:.4f} | Syn Model AUC: {regression_results['syn_model_auc']:.4f}\n")
        f.write(f"\n--- Privacy ---\n")
        f.write(f"Membership Inference AUC: {mem_auc:.4f}\n")
        f.write(f"TPR@FPR=0.1: {ios['tpr_at_fpr_0.1']:.4f} | "
                f"TPR@FPR=0.01: {ios['tpr_at_fpr_0.01']:.4f}\n")
        f.write(f"\n--- Hyperparameters ---\n")
        f.write(f"BATCH: {H.BATCH}, NOISE_DIM: {H.NOISE_DIM}, N_CRITIC: {H.DISC_STEPS}, GP_COEFF: {H.GRADIENT_PENALTY}, DISC_LR: {H.D_LR}, GEN LR: {H.G_LR}, G_HIDDEN_DIM: {H.G_H}, D_HIDDEN_DIM: {H.D_H}\n")

    return s2r_auc, cwc, mem_auc, dwd_results, regression_results