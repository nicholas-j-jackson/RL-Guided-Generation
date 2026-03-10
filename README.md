# RLSyn+Reg: Reward-Guided Generation for Scientifically Meaningful Synthetic Data

Code for the paper "Reward-Guided Generation for Scientifically Meaningful Synthetic Data" (Jackson et al., AMIA 2026).

## Overview

RLSyn+Reg extends the [RLSyn] framework with a regression-based auxiliary reward that guides a generative model to preserve regression coefficients and predictive performance from real data. We evaluate the approach on two tabular biomedical datasets: MIMIC-III (ICU admissions) and the American Community Survey (ACS).

## Repository Structure
```
├── preprocessing/
│   ├── MIMIC/               # MIMICExtractEasy submodule + preprocessing notebook
│   └── ACS/                 # ACS data loading notebook
├── scripts/                 # Training and evaluation scripts
├── Analysis.ipynb           # Figure generation
```

## Reproducing Results

### 1. Preprocessing

**MIMIC-III**: First complete extraction using the `MIMICExtractEasy` submodule in `preprocessing/MIMIC/`, then run `preprocessing/MIMIC/preprocessing_MIMIC.ipynb`. Access to MIMIC-III requires credentialed access via [PhysioNet](https://physionet.org/).

**ACS**: Run `preprocessing/ACS/load_data.ipynb`.

### 2. Training and Evaluation

Hyperparameter tuning has already been run. To reproduce the main results, run the scripts in `scripts/`:
```bash
bash scripts/run_mimic.sh
bash scripts/run_acs.sh
```

### 3. Figures

Open and run `Analysis.ipynb` to reproduce all figures from the paper.

## Citation

If you use this code, please cite:

**This work:**
```bibtex
% Preprint citation coming soon
```

**RLSyn (Espinosa-Dice et al., 2025):**
```bibtex
@misc{espinosa-dice_reinforcement_2025,
  title={Reinforcement Learning for Synthetic Data Generation},
  author={Espinosa-Dice, Natalia and others},
  year={2025},
  eprint={2512.21395},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2512.21395}
}
```