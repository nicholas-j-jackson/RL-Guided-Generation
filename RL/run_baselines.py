from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from pathlib import Path
import pandas as pd
from RL.evaluation import evaluate_model, classification_utility

import optuna

def synthcity_objective(trial, df_train, df_test, H, model_name):
    if model_name == 'ctgan':
        kwargs = {
            'generator_n_layers_hidden': trial.suggest_int('g_layers', 1, 3),
            'generator_n_units_hidden': trial.suggest_categorical('g_units', [64, 128, 256]),
            'discriminator_n_layers_hidden': trial.suggest_int('d_layers', 1, 3),
            'discriminator_n_units_hidden': trial.suggest_categorical('d_units', [64, 128, 256]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
            'n_iter': trial.suggest_categorical('n_iter', [300, 500, 1000]),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
    elif model_name == 'ddpm':
        kwargs = {
            'n_iter': trial.suggest_categorical('n_iter', [5000, 10000, 20000]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'num_timesteps': trial.suggest_categorical('num_timesteps', [500, 1000]),
            'n_layers_hidden': trial.suggest_int('n_layers', 2, 4),
            'n_units_hidden': trial.suggest_categorical('n_units', [128, 256, 512]),
        }

    try:
        loader = GenericDataLoader(df_train, target_column=H.LABEL, discrete_features=H.BIN_COLS + H.CAT_COLS, sensitive_features=[])
        syn_model = Plugins().get(model_name, **kwargs)
        syn_model.fit(loader)
        df_syn = syn_model.generate(count=len(df_train)).dataframe()
        df_syn = df_syn[H.NUM_COLS + H.BIN_COLS + H.CAT_COLS]
        s2r_auc, _ = classification_utility(df_syn, df_test, H.LABEL)
        return s2r_auc
    except Exception:
        return 0.0


def tune_synthcity_model(df_train, df_test, H, model_name, n_trials=25):
    study = optuna.create_study(
        study_name=f"{H.DATASET}_{model_name}",
        storage=f"sqlite:///{H.OUT_DIR}/baselines/{model_name}_tuning.db",
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    study.optimize(
        lambda trial: synthcity_objective(trial, df_train, df_test, H, model_name),
        n_trials=n_trials
    )
    return study.best_params


def main(H, n_trials=25):
    base_dir = Path(H.OUT_DIR) / "baselines"

    df_train = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]

    results = {}

    for model_name in ['ctgan']:#, 'ddpm']:
        print(f"Tuning {model_name}...")
        best_params = tune_synthcity_model(df_train, df_test, H, model_name, n_trials=n_trials)
        print(f"Best params for {model_name}: {best_params}")

        print(f"Training {model_name} with best params...")
        model_dir = base_dir / model_name
        H_baseline = H.override(OUT_DIR=str(model_dir))

        try:
            loader = GenericDataLoader(df_train, target_column=H.LABEL)
            syn_model = Plugins().get(model_name, **best_params)
            syn_model.fit(loader)
            df_syn = syn_model.generate(count=len(df_train)).dataframe()
            df_syn = df_syn[H.NUM_COLS + H.BIN_COLS + H.CAT_COLS]
            df_syn.to_csv(model_dir / f"synthetic_{model_name}.csv", index=False)
            results[model_name] = evaluate_model(df_train, df_test, df_syn, H_baseline)
        except Exception as e:
            print(f"{model_name} failed: {e}")
            results[model_name] = None

    return results