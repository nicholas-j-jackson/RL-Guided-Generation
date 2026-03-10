

@echo off

for %%f in (1.0) do (
    for %%s in (50) do (
        python -m RL.main --SEED %%s --DATASET ACS --FRAC %%f --PARAM_PATH C:\Users\nick\RL_guided\Tuning_ACS\best_hyperparams.json --PENALIZED
    )
)