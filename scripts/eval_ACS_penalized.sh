#!/usr/bin/env bash
# Re-run evaluation only (no training) on the already-trained ACS RLSYN+REG models,
# regenerating eval.txt with the inference-on-synthetic (DOMIAS) membership attack.
for frac in 1.0 0.5 0.25 0.1; do
    for seed in {50..52}; do
        if [ "$frac" = "1.0" ]; then suffix=""; else suffix="_frac${frac}"; fi
        rm -f "ACS/results/penalized/${seed}${suffix}/eval.txt"
        python -m RL.main --SEED $seed --DATASET ACS --FRAC $frac --EVAL_ONLY --PENALIZED \
            --PARAM_PATH /home/nick/RL_guided/Tuning_ACS_penalized/pareto_trials/trial_6/hyperparams.json
    done
done
