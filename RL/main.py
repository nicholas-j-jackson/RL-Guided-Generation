import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
from pathlib import Path
import numpy as np 
import argparse 
from RL.hyperparams import HyperParams_ACS, HyperParams_HCUP, HyperParams_MIMIC
from RL.evaluation import evaluate_model
from RL.models import build_models
from RL.train_funcs import *
import random
from tqdm import tqdm
from RL.utils import *

def set_seed(seed: int = 42) -> None:
    random.seed(seed)                      
    np.random.seed(seed)          
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 


torch.autograd.set_detect_anomaly(True)


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--SEED", type=int, required=True)
    p.add_argument("--PARAM_PATH", type=str, default="/home/nick/RL_guided/Tuning_ACS/best_hyperparams.json")
    p.add_argument("--DATASET", choices=["ACS", "HCUP", 'MIMIC'], required=True)
    p.add_argument("--EVAL_ONLY", action='store_true')
    p.add_argument("--PENALIZED", action='store_true')
    p.add_argument("--RESUME_ITER", type=int, required=False, default=None)
    p.add_argument("--FRAC", type=float, required=False, default=1.0)
    #p.add_argument("--OUT_DIR", type=str, required=True )
    #p.add_argument("--RESULT_CSV", type=str, required=True)
    #p.add_argument("--DATA_PATH", type=str, default="/home/nick/synth_misinfo/preprocessing/ACS/data")
    #p.add_argument("--ITERS", type=int, required=True)
    #p.add_argument("--DATA_SIZE", type=float, required=True)
    #add more here if you want to override other hps. 
    return vars(p.parse_args())


def main(): 
    #hyperparameter set up
    global H 
    args = parse_args() 
    PARAM_PATH = args.pop("PARAM_PATH")

    if args['DATASET'] == 'ACS':
        H = HyperParams_ACS()  
    elif args['DATASET'] == 'MIMIC':
        H = HyperParams_MIMIC()    
    else:
        H = HyperParams_HCUP()  
    
    H = H.load(PARAM_PATH).override(**args)
    H = H.override(
        OUT_DIR = f"{args['DATASET']}/results/{H.SEED}", 
        RESULT_CSV = f"{args['DATASET']}/results/results.csv",
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu", 
    )

    #train
    set_seed(H.SEED)

    # normalized data for training
    df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
        
    # Subsample if FRAC < 1.0
    if args['FRAC'] < 1.0:
        df_train = df_train.sample(frac=args['FRAC'], random_state=H.SEED)
        print(f"Using {len(df_train)} rows ({args['FRAC']*100:.0f}% of training data)")

    H = H.override(NUM_SAMPLES=len(df_train))

    # un-normalized training and test data
    df_real = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+ H.BIN_COLS+H.CAT_COLS]
    
    
    if not args['EVAL_ONLY']:
        # Training
        real = torch.tensor(df_train.values.astype(np.float32), dtype=torch.float32)#.to(H.DEVICE)
        frac_str = f"_frac{args['FRAC']}" if args['FRAC'] < 1.0 else ""

        if not args['PENALIZED']:
            train_func = train
            H = H.override(OUT_DIR=f"{H.DATASET}/results/{H.SEED}{frac_str}")
        else:
            train_func = train_regression_penalized_per_sample
            H = H.override(OUT_DIR=f"{H.DATASET}/results/penalized/{H.SEED}{frac_str}")            


        os.makedirs(H.OUT_DIR, exist_ok=True)
        print(H)

        resume_iter = int(H.ANNEAL_START * H.ITERS)
        base_study_name = f"{H.DATASET}/results/{H.SEED}/"
        base_ckpt = base_study_name + f"iter_{resume_iter}"

        df_syn = train_func(real, H, eval_every=5_000, resume_iter=H.RESUME_ITER, base_ckpt=base_ckpt)


    else: 
        frac_str = f"_frac{args['FRAC']}" if args['FRAC'] < 1.0 else ""

        if not args['PENALIZED']:
            H = H.override(OUT_DIR=f"{H.DATASET}/results/{H.SEED}{frac_str}")
        else:
            H = H.override(OUT_DIR=f"{H.DATASET}/results/penalized/{H.SEED}{frac_str}")

        df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]

    # Eval synthetic data
    r2r_auc, real_cwc, _, _, _ = evaluate_model(df_real, df_test, df_real, H)
    print("Real data", r2r_auc)

    s2r_auc, cwc, mem_auc, _, _ = evaluate_model(df_real, df_test, df_syn, H)
    print("Synthetic data", s2r_auc)

    # Regression analysis
    # if args['DATASET'] == "ACS":
            # Regression analysis on real data
            # Prep data
            #prepped_df = ACS_clean_df(df_real.copy())
            #model, results = ACS_Assitance_Logit(prepped_df, verbose=False)
    
    #     df_real = ACS_clean_df(df_real)
    #     df_syn = ACS_clean_df(df_syn)
    #     df_test = ACS_clean_df(df_test)
    # 
    #     # Perform IPW weighting on synthetic data (using real data as target weights)
    #     df_syn_prepped = ipw(df_real, df_syn)
    #     
    #     # Model
    #     syn_model, syn_results = ACS_Assitance_Logit(df_syn_prepped)
    #     #

    #results.to_csv(f"{H.OUT_DIR}/real_regression.csv")
    #syn_results.to_csv(f"{H.OUT_DIR}/synth_regression.csv")

    return



if __name__ == '__main__':
    
   

    main() 

