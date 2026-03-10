import torch, torch.nn as nn, torch.nn.functional as F
import os 
import optuna
from RL.evaluation import evaluate_model# classification_utility, membership_inference_risk, get_column_wise_correlations
from RL.hyperparams import HyperParams_ACS, HyperParams_HCUP, HyperParams_MIMIC
import pandas as pd
from RL.train_funcs import train, train_regression_penalized_per_sample
import os 
import numpy as np
from pathlib import Path
import json

def objective(trial:optuna.Trial, study_name, data_path, H_obj): 
    #define parameters 
    #BATCH = trial.suggest_categorical("batch", [64, 128, 256, 512])
    #NOISE_DIM = trial.suggest_categorical("noise_dim", [32, 64, 128])
    PPO_EPOCHS = trial.suggest_categorical("ppo_epochs", [1, 3, 5, 10])
    DISC_STEPS = trial.suggest_categorical("disc_steps", [5, 10]) 
    MEAN_PENALTY_SCALE = trial.suggest_float("mean_penalty", 0, 0.2, step=0.1)
    GRADIENT_PENALTY = trial.suggest_int("gradient_penalty", 0, 10, step = 2.5) #
    #USE_TANH = trial.suggest_categorical("use_tanh", [True, False])
    G_LR = trial.suggest_float('g_lr', 1e-5, 1e-3, log=True) #
    D_LR = trial.suggest_float('d_lr', 1e-5, 1e-3, log=True) #
    G_H = trial.suggest_categorical("g_h", [64, 128, 256])
    D_H = trial.suggest_categorical("d_h", [64, 128, 256])
    G_NUM_LAYERS = trial.suggest_int("g_n", 1, 5, step=2)
    D_NUM_LAYERS = trial.suggest_int("d_n", 1, 5, step=2)
    #ITERS = trial.suggest_categorical("iters", [10_000, 50_000]) #[10_000, 50_000])
    
    
    H = H_obj()
    H = H.override(
        OUT_DIR = f"{study_name}/trial_{trial.number}", 
        DATA_PATH = f"{data_path}", 
        RESULT_CSV = f"{study_name}/results.csv", 
        RUN_NAME = f"trial_{trial.number}", 
        DATASET = H.DATASET, 
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu', 
        ITERS = 10_000, 
        NPY_PATH = f"{data_path}/min_max_log.npy", 
        BATCH = 1024, 
        NOISE_DIM = 64, 
        GRADIENT_PENALTY = GRADIENT_PENALTY,
        G_LR = G_LR, 
        G_H = G_H, 
        D_LR = D_LR, 
        D_H = D_H, 
        G_NUM_LAYERS = G_NUM_LAYERS,
        D_NUM_LAYERS = D_NUM_LAYERS,
        DISC_STEPS = DISC_STEPS, 
        USE_TANH = True, 
        PPO_EPOCHS = PPO_EPOCHS, 
        MEAN_PENALTY_SCALE = MEAN_PENALTY_SCALE, 
        VF_COEF = 0.5, 
        CLIP_EPS = 0.1, 
        ENT_BETA = 1e-3, 
        EPS  = 1e-6, 
    ) 
    os.makedirs(H.OUT_DIR, exist_ok=True)
    
    df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    H = H.override(NUM_SAMPLES=len(df_train))

    #
    #race_cols = ['Asian', 'Black/African American', 'Other', 'Two or More', 'White']
    #H = H.override(STATE = {'p_real': df_train[race_cols].mean().values.reshape(-1)})

    #
    real = torch.tensor(df_train.values, dtype=torch.float32)
    #sampler = RandomSampler(real, replacement=True, num_samples=H.ITERS)
    #loader = DataLoader(real, batch_size=H.BATCH, shuffle=True, num_workers=2, pin_memory=True, sampler=sampler) 

    try:
        df_syn = train(real, H)

        df_real = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
        df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
        
        df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
        s2r_auc, cwc, mem_auc, dwd, regression_results = evaluate_model(df_real, df_test, df_syn, H)   

        return s2r_auc
    except Exception as ex:
        return -1

def objective_regression_penalized(trial: optuna.Trial, study_name, data_path, H_obj, dataset):
    
    # Load best hyperparameters from base model tuning
    H = H_obj()
    H = H.load(f"Tuning_{dataset}/best_hyperparams.json")

    # Tune only regression-specific parameters
    REGRESSION_LAMBDA = trial.suggest_float('regression_lambda', 1e-4, 1e2, log=True)
    ANNEAL_START = trial.suggest_categorical('anneal_start', [0.4, 0.6, 0.8]) #[0.5, 0.7, 0.8, 0.9])

    H = H.override(
        OUT_DIR=f"{study_name}/trial_{trial.number}",
        DATA_PATH=data_path,
        RESULT_CSV=f"{study_name}/results.csv",
        DATASET=dataset,
        RUN_NAME=f"trial_{trial.number}",
        REGRESSION_LAMBDA=REGRESSION_LAMBDA,
        ANNEAL_START=ANNEAL_START,
    )
    os.makedirs(H.OUT_DIR, exist_ok=True)

    print(H.DATASET)

    df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    H = H.override(NUM_SAMPLES=len(df_train))
    real = torch.tensor(df_train.values.astype(np.float32), dtype=torch.float32)

    # Resume from base model checkpoint at ANNEAL_START
    resume_iter = int(ANNEAL_START * H.ITERS)
    base_study_name = f"{H.DATASET}/results/50/"
    base_ckpt = base_study_name + f"iter_{resume_iter}"

    if not Path(base_ckpt).exists():
        ckpt_dirs = sorted(Path(base_study_name).glob("iter_*"), key=lambda p: int(p.name.split('_')[1]))
        if not ckpt_dirs:
            raise ValueError(f"No checkpoints found in {base_study_name}")
        base_ckpt = str(ckpt_dirs[-1])
        resume_iter = int(Path(base_ckpt).name.split('_')[1])

    df_syn = train_regression_penalized_per_sample(
        real, H,
        eval_every=None,
        resume_iter=resume_iter,
        base_ckpt=base_ckpt,
    )

    df_real = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]

    s2r_auc, cwc, mem_auc, dwd, reg_results = evaluate_model(df_real, df_test, df_syn, H)

    # Log individual metrics for analysis
    trial.set_user_attr('s2r_auc', s2r_auc)
    trial.set_user_attr('syn_model_auc', reg_results['syn_model_auc'])
    trial.set_user_attr('reg_corr', reg_results['correlation'])

    return s2r_auc, reg_results['syn_model_auc']


import argparse
def parse_args():
    p = argparse.ArgumentParser()    
    p.add_argument("--DATASET", choices=["ACS", 'HCUP', 'MIMIC'], required=True)
    p.add_argument("--PENALIZED", action="store_true", default=False)
    return vars(p.parse_args())


def main(): 
    args = parse_args() 

    print(args['DATASET'])

    # Select appropriate hyperparam object
    if args['DATASET'] == 'ACS':
        H = HyperParams_ACS
    elif args['DATASET'] == "MIMIC":
        H = HyperParams_MIMIC
    else:
        H = HyperParams_HCUP
    
    # Directories and study name
    study_name = f"Tuning_{args['DATASET']}"
    data_path = f"/home/nick/RL_guided/preprocessing/{args['DATASET']}/data"

    if args['PENALIZED']:
        study_name += '_penalized'
    
    base_dir = f"{study_name}"
    os.makedirs(base_dir, exist_ok=True) 


    # Run optimization
    if not args['PENALIZED']:
        # Create study
        study = optuna.create_study(
            study_name=f"{study_name}",
            storage=f"sqlite:///{study_name}.db",       
            load_if_exists=True,
            direction= "maximize", 
            sampler=optuna.samplers.TPESampler(constant_liar=True)
        )

        study.optimize(lambda trial: objective(trial, study_name, data_path, H), n_trials = 10, n_jobs=1)#, n_jobs=10)        

        # Create a HyperParams instance with the best trial's parameters
        best_trial = study.best_trial
        H_best = H().override(
            OUT_DIR=f"{base_dir}/best_trial", 
            DATA_PATH=data_path, 
            RESULT_CSV=f"{base_dir}/results.csv", 
            RUN_NAME=f"best_trial_{best_trial.number}", 
            DATASET=args['DATASET'],
            DEVICE='cuda' if torch.cuda.is_available() else 'cpu', 
            NPY_PATH=f"{data_path}/min_max_log.npy", 
            G_LR=best_trial.params["g_lr"], 
            G_H=best_trial.params["g_h"], 
            D_LR=best_trial.params["d_lr"], 
            D_H=best_trial.params["d_h"], 
            G_NUM_LAYERS=best_trial.params["g_n"],
            DISC_STEPS=best_trial.params["disc_steps"], 
            PPO_EPOCHS=best_trial.params["ppo_epochs"], 
            MEAN_PENALTY_SCALE=best_trial.params["mean_penalty"], 
            VF_COEF=0.5, 
            CLIP_EPS=0.1, 
            ENT_BETA=1e-3, 
            EPS=1e-6, 
        )

        # Save the best parameters to a file
        H_best_path = f"{base_dir}/best_hyperparams.json"
        H_best.save(H_best_path)
        print(f"Best trial hyperparameters saved to {H_best_path}")


    else:
        
        # Create study
        study = optuna.create_study(
            study_name=f"{study_name}",
            storage=f"sqlite:///{study_name}.db",       
            load_if_exists=True,            
            directions=['maximize', 'maximize'],  
            sampler=optuna.samplers.NSGAIISampler()  # standard for multi-objective opt.
        )

        study.optimize(lambda trial: objective_regression_penalized(trial, study_name, data_path, H, args['DATASET']), n_trials = 20, n_jobs=1)#, n_jobs=10)        
        
        
        pareto_trials = study.best_trials  # returns all Pareto-optimal trials
        print(f"Found {len(pareto_trials)} Pareto-optimal trials")

        pareto_dir = f"{base_dir}/pareto_trials"
        os.makedirs(pareto_dir, exist_ok=True)

        
        for trial in [study.trials[6]]:
            trial_dir = f"{pareto_dir}/trial_{trial.number}"
            os.makedirs(trial_dir, exist_ok=True)

            print(f"Trial {trial.number}: reg_mae={trial.values[0]:.4f}, s2r_auc={trial.values[1]:.4f}")
            print(f"  Params: {trial.params}\n\n")


            H_trial = H().load(f"Tuning_{args['DATASET']}/best_hyperparams.json")
            H_trial = H_trial.override(
                OUT_DIR=trial_dir,
                DATA_PATH=data_path,
                RESULT_CSV=f"{pareto_dir}/results.csv",
                RUN_NAME=f"pareto_trial_{trial.number}",
                DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
                REGRESSION_LAMBDA=trial.params['regression_lambda'],
                ANNEAL_START=trial.params['anneal_start'],
            )

            H_trial.save(f"{trial_dir}/hyperparams.json")

            # Save a summary of the objectives alongside the hyperparams
            summary = {
                'trial_number': trial.number,
                'reg_mae': trial.values[0],
                's2r_auc': trial.values[1],
                'params': trial.params,
            }
            with open(f"{trial_dir}/summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"Trial {trial.number}: reg_mae={trial.values[0]:.4f}, s2r_auc={trial.values[1]:.4f} | saved to {trial_dir}")

        # Also save a combined CSV of the full Pareto front for easy inspection
        pareto_df = pd.DataFrame([{
            'trial_number': t.number,
            'reg_mae': t.values[0],
            's2r_auc': t.values[1],
            **t.params
        } for t in pareto_trials])
        pareto_df.to_csv(f"{pareto_dir}/pareto_front.csv", index=False)
        print(f"Pareto front saved to {pareto_dir}/pareto_front.csv")
        

if __name__ == '__main__':
    main() 

