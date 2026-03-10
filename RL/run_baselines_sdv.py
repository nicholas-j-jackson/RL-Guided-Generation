import torch
import os
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from RL.evaluation import evaluate_model, classification_utility
from RL.hyperparams import HyperParams_ACS, HyperParams_HCUP, HyperParams_MIMIC
import argparse


def build_metadata(df_train, H):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    for col in H.BIN_COLS + H.CAT_COLS:
        metadata.update_column(col, sdtype='categorical')
    return metadata


def train_sdv_baseline(df_train, H, out_dir, model_name, **kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(df_train, H)

    if model_name == 'ctgan':
        synthesizer = CTGANSynthesizer(metadata, **kwargs)
    elif model_name == 'tvae':
        synthesizer = TVAESynthesizer(metadata, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    synthesizer.fit(df_train)

    df_syn = synthesizer.sample(num_rows=len(df_train))
    df_syn = df_syn[H.NUM_COLS + H.BIN_COLS + H.CAT_COLS]
    df_syn.to_csv(out_dir / f"synthetic_{model_name}.csv", index=False)

    return df_syn

def parse_dim(s):
    """Convert '128' to (128, 128) for SDV."""
    d = int(s)
    return (d, d)


def sdv_objective(trial, df_train, df_test, H, model_name):
    if model_name == 'ctgan':
        g_dim = trial.suggest_int('generator_dim', 64, 512, step=64)
        d_dim = trial.suggest_int('discriminator_dim', 64, 512, step=64)
        kwargs = {
            'epochs': 100,
            'batch_size': 256,
            'generator_dim': (g_dim, g_dim),
            'discriminator_dim': (d_dim, d_dim),
            'generator_lr': trial.suggest_float('generator_lr', 1e-5, 1e-3, log=True),
            'discriminator_lr': trial.suggest_float('discriminator_lr', 1e-5, 1e-3, log=True),
            'discriminator_steps': trial.suggest_categorical('discriminator_steps', [1, 5, 10]),
            'verbose': True,
            'pac': 64,
        }
    elif model_name == 'tvae':
        c_dim = trial.suggest_int('compress_dims', 64, 512, step=64)
        d_dim = trial.suggest_int('decompress_dims', 64, 512, step=64)
        kwargs = {
            'epochs': 100,
            'batch_size': 256,
            'compress_dims': (c_dim, c_dim),
            'decompress_dims': (d_dim, d_dim),
            'l2scale': trial.suggest_float('l2scale', 1e-6, 1e-3, log=True),
        }

    #try:
    trial_dir = Path(H.OUT_DIR) / "tuning" / f"trial_{trial.number}"
    df_syn = train_sdv_baseline(df_train, H, trial_dir, model_name, **kwargs)
    s2r_auc, _ = classification_utility(df_syn, df_test, H.LABEL)
    return s2r_auc
    #except Exception as e:
    #    print(f"Trial failed: {e}")
    #    return 0.0


def tune_sdv_baseline(df_train, df_test, H, model_name, n_trials=25):
    os.makedirs(Path(H.OUT_DIR), exist_ok=True)

    study = optuna.create_study(
        study_name=f"{H.DATASET}_{model_name}_tuning",
        storage=f"sqlite:///{H.OUT_DIR}/{model_name}_tuning.db",
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    study.optimize(
        lambda trial: sdv_objective(trial, df_train, df_test, H, model_name),
        n_trials=n_trials
    )
    print(f"{model_name} best trial: AUC={study.best_trial.value:.4f} | params={study.best_trial.params}")
    return study.best_trial.params


def run_baselines(H, n_trials=25):
    base_dir = Path(H.OUT_DIR) / "baselines"

    df_train = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    #df_train = df_train.sample(min(5000, len(df_train)), random_state=0)
    df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]

    results = {}

    for model_name in ['ctgan', 'tvae']:
        model_dir = base_dir / model_name
        H_baseline = H.override(OUT_DIR=str(model_dir))
        params_path = model_dir / "best_params.json"

        # Load existing best params if study already has enough trials
        best_params = None
        db_path = model_dir / f"{model_name}_tuning.db"
        try:
            study = optuna.load_study(
                study_name=f"{H.DATASET}_{model_name}_tuning",
                storage=f"sqlite:///{db_path}",
            )
            if len(study.trials) >= n_trials:
                best_params = study.best_trial.params
                print(f"{model_name}: found {len(study.trials)} existing trials, skipping tuning.")
                print(f"  Best params: {best_params}")
            else:
                print(f"{model_name}: found {len(study.trials)} trials, need {n_trials} — continuing tuning.")
        except Exception as e:
            print(f"{model_name}: could not load existing study ({e}), tuning from scratch.")

        if best_params is None:
            print(f"\nTuning {model_name}...")
            best_params = tune_sdv_baseline(df_train, df_test, H_baseline, model_name, n_trials=n_trials)

        # Save best params
        import json
        best_params['epochs'] = 250

        # Convert dim params back to tuples for SDV
        if model_name == 'ctgan':
            best_params['generator_dim'] = (best_params['generator_dim'], best_params['generator_dim'])
            best_params['discriminator_dim'] = (best_params['discriminator_dim'], best_params['discriminator_dim'])
        elif model_name == 'tvae':
            best_params['compress_dims'] = (best_params['compress_dims'], best_params['compress_dims'])
            best_params['decompress_dims'] = (best_params['decompress_dims'], best_params['decompress_dims'])

        model_dir.mkdir(parents=True, exist_ok=True)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"Training final {model_name} with best params...")
        final_dir = model_dir / "final"
        H_final = H.override(OUT_DIR=str(final_dir))

        try:
            df_syn = train_sdv_baseline(df_train, H, final_dir, model_name, **best_params)
            results[model_name] = evaluate_model(df_train, df_test, df_syn, H_final)
            print(f"{model_name} final: S2R AUC = {results[model_name][0]:.4f}")
        except Exception as e:
            print(f"{model_name} final training failed: {e}")
            results[model_name] = None

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--DATASET", choices=["ACS", "HCUP", "MIMIC"], required=True)
    p.add_argument("--n_trials", type=int, default=10)
    return vars(p.parse_args())


if __name__ == '__main__':
    args = parse_args()

    if args['DATASET'] == 'ACS':
        H_obj = HyperParams_ACS
        data_path = "preprocessing/ACS/data"
    elif args['DATASET'] == 'MIMIC':
        H_obj = HyperParams_MIMIC
        data_path = "preprocessing/MIMIC/data"
    else:
        H_obj = HyperParams_HCUP
        data_path = "preprocessing/HCUP/data"

    H = H_obj().override(
        DATASET=args['DATASET'],
        DATA_PATH=data_path,
        OUT_DIR=f"Baselines_{args['DATASET']}",
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
        NPY_PATH=f"{data_path}/min_max_log.npy",
    )

    results = run_baselines(H, n_trials=args['n_trials'])

    print("\n=== Final Results ===")
    for model_name, result in results.items():
        if result is not None:
            print(f"{model_name}: S2R AUC={result[0]:.4f}, CWC={result[1]:.4f}, MIA={result[2]:.4f}")
        else:
            print(f"{model_name}: FAILED")