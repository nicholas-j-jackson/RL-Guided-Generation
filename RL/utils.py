import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np 
import torch

import os
import sys
from contextlib import redirect_stdout


# Clean column names - remove special characters and leading numbers that balance doesn't like
def clean_column_names(df):
    df.columns = df.columns.str.replace('^', '_', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    # If column starts with a number, prefix with 'col_'
    df.columns = ['col_' + str(col) if str(col)[0].isdigit() else str(col) for col in df.columns]
    return df


#Clean up several of the column names and map onto discrete categories
def ACS_clean_df(df):
    race_cols = ['Asian', 'Black/African American', 'Other', 'Two or More', 'White']
    
    df['Race'] = df[race_cols].idxmax(axis=1)
    df['Race'] = pd.Categorical(df['Race'], categories=race_cols)

    cit_cols = ['Born Citizen', 'Naturalized', 'Not a citizen']
    df['Citizenship'] = df[cit_cols].idxmax(axis=1)
    df['Citizenship'] = pd.Categorical(df['Citizenship'], categories=cit_cols)

    return df


# Logit for receiving public assistance income
def ACS_Assitance_Logit(df, verbose=True):

    logit_formula = """ Public_Health_Insurance ~  Age +
                                                    Years_of_School +
                                                    Male +
                                                    Employed +
                                                    C(Citizenship, Treatment(reference='Not a citizen')) +
                                                    C(Race, Treatment(reference='White'))
                    """

    model = smf.glm(formula=logit_formula, 
                    data=df.rename(columns={x:x.replace(' ', '_' ) for x in df.columns}), 
                    family=sm.families.Binomial(),
                    var_weights=df['PWGTP']).fit()


    odds_ratios = np.exp(model.params)

    # Get confidence intervals for odds ratios
    conf_int = np.exp(model.conf_int())
    conf_int.columns = ['OR_Lower_95%', 'OR_Upper_95%']

    def add_stars(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        elif p_value < 0.1:
            return '.'
        else:
            return ''

    # Combine into interpretable table
    results = pd.DataFrame({
        'Coefficient': model.params,
        'Odds_Ratio': odds_ratios,
        'OR_Lower_95%': conf_int['OR_Lower_95%'],
        'OR_Upper_95%': conf_int['OR_Upper_95%'],
        'P_value': model.pvalues,
        'Sig': model.pvalues.apply(add_stars)
    })

    # Add interpretation column
    results['Interpretation'] = results.apply(
        lambda row: f"{(row['Odds_Ratio']-1)*100:.1f}% increase in odds{row['Sig']}" if row['Odds_Ratio'] > 1 
        else f"{(1-row['Odds_Ratio'])*100:.1f}% decrease in odds{row['Sig']}",
        axis=1
    )

    if verbose:
        print(model.summary())
        print('\n\n\n\n')
        print(results["Interpretation"])

    return model, results


# Gamma GLM to model amount of assistance (conditional on receiving any)
def ACS_Assitance_Gamma(df):
    df_recipients = df[df['Public_Assistance_Binary'] == 1].copy()

    print(f"Number of recipients: {len(df_recipients)}")
    print(f"Assistance amount summary:\n{df_recipients['Public_Assistance_Income'].describe()}")

    # Gamma GLM (for positive, right-skewed data)
    gamma_formula = """Public_Assistance_Income ~   Age +
                                                    Citizen +
                                                    Years_of_School +
                                                    Male +
                                                    Employed +
                                                    C(Race, Treatment(reference='White'))
                    """

    model_gamma = smf.glm(
        formula=gamma_formula,
        data=df_recipients,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=df_recipients['PWGTP']
    ).fit()

    print(model_gamma.summary())

    # For Gamma GLM with log link, exponentiate to get multiplicative effects
    multiplicative_effects = np.exp(model_gamma.params)

    # 95% CIs
    conf_int = np.exp(model_gamma.conf_int())
    conf_int.columns = ['Effect_Lower_95%', 'Effect_Upper_95%']

    def add_stars(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        elif p_value < 0.1:
            return '.'
        else:
            return ''

    # Combine into interpretable table
    results_gamma = pd.DataFrame({
        'Coefficient': model_gamma.params,
        'Multiplicative_Effect': multiplicative_effects,
        'Effect_Lower_95%': conf_int['Effect_Lower_95%'],
        'Effect_Upper_95%': conf_int['Effect_Upper_95%'],
        'P_value': model_gamma.pvalues,
        'Sig': model_gamma.pvalues.apply(add_stars)
    })

    # Add interpretation column
    results_gamma['Interpretation'] = results_gamma.apply(
        lambda row: f"{(row['Multiplicative_Effect']-1)*100:.1f}% higher assistance amount{row['Sig']}" if row['Multiplicative_Effect'] > 1 
        else f"{(1-row['Multiplicative_Effect'])*100:.1f}% lower assistance amount{row['Sig']}",
        axis=1
    )

    print(results_gamma)

    return model_gamma, results_gamma


import numpy as np
from scipy import stats

def wald_test_glm(real, synth, coef_names=None):
    # coefficients
    beta_real = real.params
    beta_synth = synth.params

    # covariance from real data
    cov_real = real.cov_params()

    # Perform Wald on subset (optional)
    if coef_names is not None:
        beta_real = beta_real.loc[coef_names]
        beta_synth = beta_synth.loc[coef_names]
        cov_real = cov_real.loc[coef_names, coef_names]


    # Wald statistic
    delta = (beta_synth - beta_real).to_numpy()
    cov = cov_real.to_numpy()

    cov_inv = np.linalg.inv(cov)
    wald_stat = delta.T @ cov_inv @ delta

    # Degrees of freedom
    df = len(delta)

    # p-value
    p_value = 1 - stats.chi2.cdf(wald_stat, df)

    return {
        "wald_stat": float(wald_stat),
        "df": df,
        "p_value": float(p_value),
        "delta_beta": delta
    }


# Convert samples of from a tensor to dataframe for processing with statsmodels
def to_dataframe(rows, H):
    cols = H.NUM_COLS + H.BIN_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df = pd.DataFrame(rows.cpu().detach().numpy(), columns=cols)
    else: 
        df = pd.DataFrame(rows.detach().numpy(), columns=cols)
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    df_rescaled = df.copy()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_rescaled[col] = (1.0 - df[col]) * xmin + df[col] * xmax
        
    df_rescaled['Age'] = df_rescaled['Age'].round(0)
    df_rescaled['Age'] = df_rescaled['Age'].round(0)
    return df, df_rescaled


# Estimate propensity score via 
def discriminator_ipw_weights(D, x, max_weight=50.0, eps=1e-6):
    logits = D(x, head='ipw').squeeze()
    p = torch.sigmoid(logits)

    w = p / (1.0 - p + eps)
    w = torch.clamp(w, max=max_weight)

    return w.squeeze()


def fit_glm_approx_weights(rows, w, H):
    df_syn, df_syn_rescaled = to_dataframe(rows, H)

    # Store synthetic data as df
    df_syn = ACS_clean_df(df_syn)

    # Perform IPW weighting on synthetic data (using real data as target weights)
    model_syn, results_syn = ACS_Assitance_Logit(df_syn, verbose=False)

    return torch.tensor(model_syn.params.values, dtype=torch.float32)
    


def compute_wald_penalty(D, rows, H):

    # IPW weights from discriminator
    w = discriminator_ipw_weights(D, rows, max_weight=50.0).cpu()

    # Fit synthetic GLM (small batch → fast)
    beta_syn = fit_glm_approx_weights(rows, w, H)

    delta = beta_syn - H.WALD_STATE['beta_real']
    wald_penalty = delta @ H.WALD_STATE['inv_cov_real'] @ delta
    wald_penalty = wald_penalty / H.WALD_STATE['beta_real'].numel()

    return wald_penalty.to(H.DEVICE)


def log_demographics(race_cols, p_syn, it, H):
    
    with open(f"{H.OUT_DIR}/losses/group_stats.txt", "a") as f:
        msg = (
            [f"ITER {it}"] +
            [f"{race_cols[i]} {p_syn[i].item()*100:.2f}"
             for i in range(len(race_cols))]        )
        f.write(" | ".join(msg) + "\n")
    return 


def compute_group_proportions(rows, H, it):
    df_syn, df_syn_rescaled = to_dataframe(rows, H)

    # Store as df
    df_syn = ACS_clean_df(df_syn)
    B = len(df_syn)

    race_cols = ['Asian', 'Black/African American', 'Other', 'Two or More', 'White']

    # Compute proportions
    p_real = torch.tensor(H.STATE['p_real'], device=H.DEVICE, dtype=torch.float)        # shape: [B]
    p_syn = torch.tensor(df_syn[race_cols].mean().values, device=H.DEVICE, dtype=torch.float)   # shape: [B]

    log_demographics(race_cols, p_syn, it, H)

    return p_real, p_syn

def compute_representativeness_reward(p_syn, p_real, H):
       kl_div = (p_syn * (p_syn / (p_real + 1e-8)).log()).sum()
       return -kl_div  # single scalar for batch


def next_real_batch(real_iter, loader, device):
    try:
        batch, = next(real_iter)
    except StopIteration:
        real_iter = iter(loader)
        batch, = next(real_iter)
    return batch.to(device), real_iter
