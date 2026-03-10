import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Independent, TransformedDistribution, Categorical
from torch.distributions.transforms import TanhTransform
import torch.nn.functional as F


#generator / policy 

def build_models(H): 
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()

            #
            layers = []
            in_dim = H.NOISE_DIM
            for _ in range(H.G_NUM_LAYERS):
                layers.append(nn.Linear(in_dim, H.G_H))
                layers.append(nn.ReLU())
                in_dim = H.G_H

            self.core = nn.Sequential(*layers)

            #define mu and sigma for numeric cols 
            self.mu = nn.Linear(H.G_H, len(H.NUM_COLS))
            self.log_sigma = nn.Parameter(torch.zeros(len(H.NUM_COLS)))  

            #binary cols head
            self.bin_logits = nn.Linear(H.G_H, len(H.BIN_COLS))

            # categorical cols heads
            self.cat_heads = nn.ModuleList([nn.Linear(H.G_H, num_classes) for num_classes in H.CAT_DIMS])

            #value network 
            self.v = nn.Linear(H.G_H, 1)
            
        def sample(self, z):
            h = self.core(z)
            outputs = []

            # Numeric columns
            sigma = self.log_sigma.exp()
            mu_h = self.mu(h)
            num_dist = Normal(mu_h, sigma)

            if H.USE_TANH: 
                tanh_dist =  TransformedDistribution(num_dist, [TanhTransform(cache_size=1)])
                #rsample interally implements z = u + oe
                num_unit = tanh_dist.rsample() #(-1, 1)
                num = 0.5 * (num_unit + 1) #(0, 1) 
                num = num.clamp(H.EPS, 1.0 - H.EPS) #to avoid bugs 
                logp = tanh_dist.log_prob(num_unit).sum(-1) #e computed without explicitly storing it
            else: 
                #rsample interally implements z = u + oe
                num = num_dist.rsample() 
                logp = num_dist.log_prob(num).sum(-1)

            outputs.append(num)

            # Binary columns
            logits = self.bin_logits(h)
            bin_dist = Independent(Bernoulli(logits=logits), 1)
            bin = bin_dist.sample() 
            logp_bin = bin_dist.log_prob(bin) 
            logp += logp_bin
            bin_probs = torch.sigmoid(logits)
            outputs.append(bin)

            # Categorical columns
            cat_samples = []
            for cat_head in self.cat_heads:
                cat_logits = cat_head(h)  # (batch, num_classes)
                cat_dist = Categorical(logits=cat_logits)
                cat_sample = cat_dist.sample()  # (batch,) - class indices
                
                # Convert to one-hot for output
                cat_onehot = F.one_hot(cat_sample, num_classes=cat_logits.size(1)).float()
                cat_samples.append(cat_onehot)
                
                logp += cat_dist.log_prob(cat_sample)

            if cat_samples:
                cat_all = torch.cat(cat_samples, dim=1)
                outputs.append(cat_all)

            # Combine data types
            row = torch.cat(outputs, dim=1)


            #value head 
            val = self.v(h).squeeze()
            return row, logp, val, bin_probs, mu_h

        def eval_action(self, z, row):
            h = self.core(z)

            # Numeric columns 
            sigma = self.log_sigma.exp()
            if not H.USE_TANH: 
                num = row[:, :len(H.NUM_COLS)]
                logp = Normal(self.mu(h), sigma).log_prob(num).sum(-1) 
            else:  
                num_scaled = row[:, :len(H.NUM_COLS)].clamp(H.EPS, 1.0-H.EPS) #avoids bugs 
                num_unit = 2 * num_scaled - 1 
                base_dist = Normal(self.mu(h), sigma) 
                tanh_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
                logp = tanh_dist.log_prob(num_unit).sum(-1) 

            # Binary Columns 
            logits = self.bin_logits(h)
            bin = row[:, len(H.NUM_COLS):len(H.NUM_COLS) + len(H.BIN_COLS)]              
            logp += Independent(Bernoulli(logits=logits), 1).log_prob(bin)

            # Categorical Columns
            col_idx = len(H.NUM_COLS) + len(H.BIN_COLS)
            for i, cat_head in enumerate(self.cat_heads):
                num_classes = H.CAT_DIMS[i]
                
                # Extract one-hot from row
                cat_onehot = row[:, col_idx:col_idx + num_classes]
                
                # Convert one-hot to class indices
                cat_indices = cat_onehot.argmax(dim=1)
                
                # Compute log probability
                cat_logits = cat_head(h)
                cat_dist = Categorical(logits=cat_logits)
                logp += cat_dist.log_prob(cat_indices)
                
                col_idx += num_classes

            return logp, self.v(h).squeeze()


    # Discriminator
    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            h = H.D_H
            self.fc = nn.Sequential(nn.Linear(len(H.NUM_COLS) + len(H.BIN_COLS) + len(H.CAT_COLS), h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2))
            self.gan_head = nn.Linear(h, 1)
        def forward(self, row):
            h = self.fc(row)
            return self.gan_head(h)


    return Policy().to(H.DEVICE), Disc().to(H.DEVICE)


