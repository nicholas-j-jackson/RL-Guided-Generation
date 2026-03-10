
for frac in 1.0 0.5 0.25 0.1; do
    for seed in {50..52}; do

        CUDA_VISIBLE_DEVICES=5 python -m RL.main --SEED $seed --DATASET ACS --FRAC $frac --PARAM_PATH /home/nick/RL_guided/Tuning_ACS_penalized/pareto_trials/trial_6/hyperparams.json --PENALIZED

    done
done