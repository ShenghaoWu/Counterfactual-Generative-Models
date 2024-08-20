'''
Adapted from https://github.com/TeaPearce/Conditional_Diffusion_MNIST

'''


import itertools
import sys
from typing import Dict, Tuple
import argparse
import scipy as sci
import scipy.io as sio
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import logging
import time
import matplotlib as mpl
from matplotlib import colors
import os
from utils import *
from models import *
from simulation import *
from scipy.special import softmax
from sklearn.neighbors import KernelDensity
import pickle

def treatment2id(A):
    # input: A: [nsamples, d] array
    d= len(A[0])
    A_unique = list(itertools.product([0, 1], repeat=d))
    labels = np.zeros(len(A),dtype='int')
    for i, a in enumerate(A_unique):
        if d == 1:
            inds  = np.where(A==a)[0]#for d=1
        else:
            inds  = np.where(np.all(A==a,axis=1))#for d>1
        labels[inds] = int(i)
    return(labels)


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }



class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextAE(nn.Module):
    def __init__(self, n_feat = 256, n_hidden=256, n_classes=8):
        super(ContextAE, self).__init__()

        self.n_feat = n_feat
        self.n_classes = n_classes
        self.input_size = 1
        self.hidden_dim = n_hidden

        self.encode = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size,
                                                            self.n_feat),nn.GELU())])
        self.encode.extend([nn.Sequential(nn.Linear(self.n_feat, self.n_feat),nn.GELU(),
                                          nn.Linear(self.n_feat, self.n_feat),nn.GELU(),
                                          nn.Linear(self.n_feat, self.hidden_dim, nn.GELU()))])


        self.decode = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim*3,
                                                            self.n_feat),nn.GELU())])
        self.decode.extend([nn.Sequential(nn.Linear(self.n_feat, self.n_feat),nn.GELU(),
                                          nn.Linear(self.n_feat, self.n_feat),nn.GELU(),
                                          nn.Linear(self.n_feat, self.input_size))])


        self.timeembed1 = EmbedFC(1, self.hidden_dim)
        self.contextembed1 = EmbedFC(n_classes, self.hidden_dim)

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        for layer in self.encode:
            x = layer(x)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.hidden_dim)
        temb1 = self.timeembed1(t).view(-1, self.hidden_dim)

        # could concatenate the context embedding here instead of adaGN
        x = torch.cat((x, temb1, cemb1), 1)
        for layer in self.decode:
            x = layer(x)
        return x




class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, iptw= False, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.iptw = iptw
        if iptw:
            self.loss_mse = nn.MSELoss(reduction='none')
        else:
            self.loss_mse = nn.MSELoss()

    def forward(self, x, c, iptw):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        
        if self.iptw:
            return  ((torch.mean(self.loss_mse(noise,pred),dim=(1)))*iptw).mean()
        else:
            return self.loss_mse(noise, pred)

    def sample(self, n_sample, d, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,np.power(2,d)).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1)

            # double batch
            x_i = x_i.repeat(2,1)
            t_is = t_is.repeat(2,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i


def main(args):

    print(f'Training {args.model_name}, iptw is {args.is_iptw}')
    fold=0
    data_params={'d':args.d,
                'X_dim':1,
                'Y_dim':1,
                'S': 2000,
                'T':50,
                'noise_sigma': 0.001,
                'seed':0,
                'is_plot':1
            }
    n_epoch = args.n_epoch
    batch_size = 256
    n_T = 400 # 500
    device = "cpu"
    n_classes = np.power(2,data_params['d'])
    n_feat = args.nfeat # 128 ok, 256 better (but slower)
    nhidden = args.nhidden
    lrate = args.lr
    save_model = True
    save_dir = 'model/'
    os.makedirs(save_dir,  exist_ok = True) 
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    # A: [n, d]
    # Y: [n, ]
    # f: [n, d]
    with open('data/syn_1d_demo.pickle','rb') as handle:
        dat = pickle.load(handle)

    outcome_data = dat[f'dat_d{args.d}']
    print(outcome_data.keys())
    outcome_data['A_bar'] = treatment2id(outcome_data['A']).astype('int')
    outcome_data['iptw'] = np.prod(1/(np.multiply(outcome_data['f'],outcome_data['A'])+np.multiply(1-outcome_data['f'],1-outcome_data['A'])),axis=1)
    y = outcome_data['Y']    
    y=(y - y.min()) / (y.max() - y.min()) #doesnt need to rescale bc already in [0,1]

    y = y[:,None]
    print(f"Y (image):{y.shape}\n"+
            f"A (treatment):{outcome_data['A_bar'].shape}\n"+
            f"iptw (weights):{outcome_data['iptw'].shape}\n")

    ndata = len(outcome_data['A_bar'])
    train_subset, val_subset = random_split(
            list(zip(torch.from_numpy(outcome_data['A_bar']),
                        torch.Tensor(clip_w(outcome_data['iptw'],0.0001,99.9999,True)),
                        torch.Tensor(y))),[int(ndata * 0.9), ndata-int(ndata * 0.9)]
        )

    trainloader = torch.utils.data.DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, num_workers=5)

    print(f'number of training samples:{len(trainloader)*batch_size}\n')
    a,iptw,y = next(iter(trainloader))
    print(f'a,iptw,y shape:{a.shape, iptw.shape, y.shape}')


    ddpm = DDPM(nn_model=ContextAE(n_feat=n_feat, n_hidden = nhidden, n_classes=n_classes),betas=(1e-4, 0.02), n_T=n_T, device=device, iptw=args.is_iptw, drop_prob=0.1)
    ddpm.to(device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        pbar = tqdm(trainloader)
        loss_ema = None
        for a, iptw, y in pbar:
            optim.zero_grad()
            y = y.to(device)
            a = a.to(device)
            loss = ddpm(y, a, iptw.to(device))
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if save_model:
            
            torch.save(ddpm.state_dict(), os.path.join(save_dir, f"{args.model_name}.pth"))
            print("saved model at" + os.path.join(save_dir, f"{args.model_name}.pth"))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running script arguments"
    )
    parser.add_argument("-n","--model_name",type=str, help="model name", default="diffusion")
    parser.add_argument("-w","--is_iptw", help="if using IPTW, default = false", action="store_true")
    parser.add_argument("-l","--lr", help="learning rate", type=float, default = 1e-4)
    parser.add_argument("-f","--nfeat", help="feature dim", type=int, default = 256)
    parser.add_argument("-e","--n_epoch", help="number of training epochs", type=int, default = 20)
    parser.add_argument("-hh","--nhidden", help="hidden dim", type=int, default = 256)
    parser.add_argument("-d",help="d", type=int, default = 3)

    
    args = parser.parse_args()
    main(args)
