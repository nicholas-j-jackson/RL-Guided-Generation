import time
import torch, torch.nn as nn, torch.nn.functional as F
from RL.models import build_models
from tqdm import tqdm
from RL.utils import *
from RL.evaluation import evaluate_model, fit_acs_regression, fit_mimic_regression
from pathlib import Path

import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

# Ignore PerfectSeparationWarning that tells us that the model fit to the synthetic data is bad (we know it is, we're penalizing it)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def validate_and_log(G, D, H, iter):
    
    df_real = pd.read_csv(f"{H.DATA_PATH}/unnormalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    df_test = pd.read_csv(f"{H.DATA_PATH}/unnormalized_val.csv")[H.NUM_COLS+ H.BIN_COLS+H.CAT_COLS]
    
    # Generate data
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _, _ = G.sample(z)
    
    # Format as df
    df_syn, df_syn_rescaled = to_dataframe(synthetic, H)

    H_temp = H.override(OUT_DIR=H.OUT_DIR+'/iter_'+str(iter))

    r2r_auc, real_cwc, _, _, _ = evaluate_model(df_real, df_test, df_real, H_temp)
    s2r_auc, cwc, mem_auc, dwd, reg_syn = evaluate_model(df_real, df_test, df_syn_rescaled, H_temp)

    print(f"Iter: {iter:.4f} | Real: {r2r_auc:.4f} | Synthetic {s2r_auc:.4f}")

    torch.save(G.state_dict(), f'{H_temp.OUT_DIR}/G.pth')
    torch.save(D.state_dict(), f'{H_temp.OUT_DIR}/D.pth')


def train(dataset, H, eval_every=None, resume_iter=None, base_ckpt=None):

    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)

    if resume_iter is not None:
        resume_from = Path(H.OUT_DIR+f'/iter_{resume_iter}/')
        G.load_state_dict(torch.load(resume_from / 'G.pth', map_location=H.DEVICE))
        D.load_state_dict(torch.load(resume_from / 'D.pth', map_location=H.DEVICE))
        opt_G.load_state_dict(torch.load(resume_from / 'opt_G.pth', map_location=H.DEVICE))
        opt_D.load_state_dict(torch.load(resume_from / 'opt_D.pth', map_location=H.DEVICE))
        print(f"Resuming from iteration {resume_iter}")


    from torch.utils.data import DataLoader, TensorDataset
    # Before training loop
    loader = DataLoader(TensorDataset(dataset), batch_size=H.BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    real_iter = iter(loader)

    #ppo training loop
    for it in tqdm(range(H.ITERS)):

        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, v_old, _, _ = G.sample(z)

        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        
        # Compute reward
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()

        #compute and smooth advantage 
        adv = rewards - v_old 
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        #mean penalty 
        #with torch.no_grad():
        #    target_mean = dataset.mean(0, keepdim=True).to(H.DEVICE)
        #num_fake = rows[:, :len(H.NUM_COLS)]
        #mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()

        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi + H.VF_COEF * loss_v - H.ENT_BETA * entropy
            #loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            
        # discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            #rand_idx = torch.randint(len(dataset), size=(H.BATCH,))
            #real_batch = dataset[rand_idx]
            real_batch = real_batch.to(H.DEVICE, non_blocking=True)

            #fresh fake batch
            fake_batch, _, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not WGAN
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        if eval_every is not None and it % eval_every == 0:
            validate_and_log(G, D, H, it)


    #Generate
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _, _ = G.sample(z)
    
    # Format as df
    df_syn, df_syn_rescaled = to_dataframe(synthetic, H)

    # Save
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    df_syn_rescaled.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    torch.save(G.state_dict(), f'{H.OUT_DIR}/G.pth')
    torch.save(D.state_dict(), f'{H.OUT_DIR}/D.pth')
    torch.save(opt_G.state_dict(), f'{H.OUT_DIR}/opt_G.pth')
    torch.save(opt_D.state_dict(), f'{H.OUT_DIR}/opt_D.pth')
    return df_syn_rescaled




def train_regression_penalized_per_sample(dataset, H, eval_every=None, resume_iter=None, base_ckpt=None):
    print("Training penalized model")
    # Load model and adjust checkpoint
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)

    if resume_iter is not None:
        resume_from = Path("/".join(base_ckpt.split('/')[:-1])) # Path(H.OUT_DIR.replace('penalized/','')) #Path("/".join(base_ckpt.split('/')[:-1]))
        print(resume_from)
        G.load_state_dict(torch.load(resume_from / 'G.pth', map_location=H.DEVICE))
        D.load_state_dict(torch.load(resume_from / 'D.pth', map_location=H.DEVICE))
        opt_G.load_state_dict(torch.load(resume_from / 'opt_G.pth', map_location=H.DEVICE))
        opt_D.load_state_dict(torch.load(resume_from / 'opt_D.pth', map_location=H.DEVICE))
        print(f"Resuming from iteration {resume_iter}")
    else:
        resume_iter = 0

    #H = H.override(OUT_DIR=f"{H.DATASET}/results/penalized/{H.SEED}")#, ITERS=10_000)


    # Fit regression on real data once before training
    df_real = pd.read_csv(f"{H.DATA_PATH}/normalized_train.csv")[H.NUM_COLS+H.BIN_COLS+H.CAT_COLS]
    if H.DATASET == 'MIMIC':
        reg_model, reg_features = fit_mimic_regression(df_real)
        reg_features_in_cols = ['oxygen saturation', 'vent', 'mean blood pressure', 'vaso', 'lactate', 'creatinine', 'platelets', 'bilirubin', 'glascow coma scale total', 'Age', 'Male', 'Black/African American', 'Other Race']

    else:
        reg_model, reg_features = fit_acs_regression(df_real)
        reg_features_in_cols = ['Age', 'Years of School', 'Male', 'Asian', 'Black/African American', 'Other', 'Two or More']

    real_coefs = torch.tensor(reg_model.params[1:].values, dtype=torch.float32).to(H.DEVICE)
    intercept = torch.tensor(reg_model.params[0], dtype=torch.float32).to(H.DEVICE)


    # Precompute feature and label indices into the synthetic row tensor
    all_cols = H.NUM_COLS + H.BIN_COLS + H.CAT_COLS
    feature_indices = torch.tensor([all_cols.index(f) for f in reg_features_in_cols], dtype=torch.long).to(H.DEVICE)

    if H.DATASET == 'MIMIC':
        label_idx = all_cols.index('mortality')
        label_bin_idx = H.BIN_COLS.index('mortality')
    else:
        label_idx = all_cols.index('Public Assistance Income')
        label_num_idx = H.NUM_COLS.index('Public Assistance Income')

    for it in tqdm(range(resume_iter, H.ITERS)):

        # Rollout
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, v_old, bin_probs, mu_num = G.sample(z)
        rows = rows.detach()
        logp_old = logp_old.detach()
        v_old = v_old.detach()

        # Discriminator reward
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()

            # Regression consistency penalty — annealed in during second half of training
            regression_weight = H.REGRESSION_LAMBDA * max(0.0, (it - H.ANNEAL_START) / (H.ITERS - H.ANNEAL_START))
            
            if regression_weight > 0:
                syn_features = rows[:, feature_indices]
                syn_label = rows[:, label_idx]

                # Compute penalty
                if H.DATASET == 'MIMIC':
                    p_hat = torch.sigmoid(intercept + (syn_features * real_coefs).sum(dim=1)).detach()
                    syn_mortality_prob = bin_probs[:, label_bin_idx]  # no second forward pass

                    # Squared difference in probability space (e.g., penalize generation probability, not actualized value)
                    reg_penalty = (syn_mortality_prob - p_hat).pow(2)

                else:
                    # Need h from a forward pass through the core
                    y_hat = intercept + (syn_features * real_coefs).sum(dim=1)
                    syn_income_mean = mu_num[:, label_num_idx]

                    reg_penalty = (syn_income_mean - y_hat).pow(2)

                # Negate so higher consistency = higher reward
                rewards = rewards - regression_weight * reg_penalty

        # Advantages
        adv = rewards - v_old
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Mean penalty
        with torch.no_grad():
            target_mean = dataset.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()

        # PPO update
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n
            loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi + H.VF_COEF * loss_v - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        # Discriminator update
        for d_it in range(H.DISC_STEPS):
            rand_idx = torch.randint(len(dataset), size=(H.BATCH,))
            real_batch = dataset[rand_idx].to(H.DEVICE)
            fake_batch, _, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) \
                   + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        if eval_every is not None and it % eval_every == 0:
            validate_and_log(G, D, H, it)

    # Final generation
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _, _ = G.sample(z)
    df_syn, df_syn_rescaled = to_dataframe(synthetic, H)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    df_syn_rescaled.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    torch.save(G.state_dict(), f'{H.OUT_DIR}/G.pth')
    torch.save(D.state_dict(), f'{H.OUT_DIR}/D.pth')
    torch.save(opt_G.state_dict(), f'{H.OUT_DIR}/opt_G.pth')
    torch.save(opt_D.state_dict(), f'{H.OUT_DIR}/opt_D.pth')
    return df_syn_rescaled