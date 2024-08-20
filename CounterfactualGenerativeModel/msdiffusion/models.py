import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sci
from tqdm import tqdm
from torch.nn import Parameter
import scipy.io as sio
import scipy.stats as ss
import itertools
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import time
import matplotlib as mpl
from matplotlib import colors
import os
from torch.utils.data import random_split
import torch.optim as optim
from functools import partial
import warnings
from utils import *
import collections
from torch.utils.data import DataLoader
from torch.optim import Adam
import sys
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)
font = {
    'family' : 'serif',
    'weight' : 'normal',
    'size'   : 10}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 2
npg_color_palettes = ['#3c5488','#e64b35','#00a087','#dc0000'
                     ,'#4dbbd5','#b09c85','#f39b7f','#91d1c2'
                     ,'#7e6148','#8491b4']




def load_model(config,data_params):
    if 'propensity_numerator' in config['model']:
        propensity_network_params={'d':data_params['d'],
                             'X_dim':data_params['X_dim'],
                             'Y_dim':data_params['Y_dim'],
                             'layer_width': config['layer_width'],
                             'n_layer':config['n_layer'],
                             'seed':data_params['seed'],
                            }
        net = PropensityNetworkNumerator(propensity_network_params)
        
    elif 'propensity_denominator' in config['model']:
        
        propensity_network_params={'d':data_params['d'],
                             'X_dim':data_params['X_dim'],
                             'Y_dim':data_params['Y_dim'],
                             'layer_width': config['layer_width'],
                             'n_layer':config['n_layer'],
                             'seed':data_params['seed'],
                            }
        net = PropensityNetworkDenominator(propensity_network_params)
    elif 'covariate' in config['model']:
        covariate_network_params={'d':data_params['d'],
                         'X_dim':data_params['X_dim'],
                         'Y_dim':data_params['Y_dim'],
                         'layer_width': config['layer_width'],
                         'n_layer':config['n_layer'],
                         'seed':data_params['seed'],
                        }
        net = CovariateNetwork(covariate_network_params)
    
   
    
    
    elif 'cvae' in config['model']  or 'mscvae' in config['model']:
        cvae_network_params={'d':data_params['d'],
                     'X_dim':data_params['X_dim'],
                     'Y_dim':data_params['Y_dim'],
                     'hidden_size':config['hidden_size'],
                     'layer_width': config['layer_width'],
                     'n_layer':config['n_layer'],
                     'seed':data_params['seed'],
                     'embedding_dim':config['embedding_dim']
                        }
        net = CVAE(cvae_network_params)
    elif 'msm' in config['model']:
        msm_params={'d':data_params['d'],
                     'X_dim':data_params['X_dim'],
                     'Y_dim':data_params['Y_dim'],
                     'layer_width': config['layer_width'],
                     'n_layer':config['n_layer'],
                     'seed':data_params['seed']}
        net = MSM(msm_params)
        
    elif 'gnet' in config['model']:
        gnet_config = {
            "batch_size"    : 400,
            "seq_len"       : data_params['d'],
            "x_dim"         : data_params['X_dim'],
            "y_dim"         : data_params['Y_dim'],
            "a_dim"         : 1,
            "r_dim"         : config['r_dim'],
            "mlp_dim"       : config['mlp_dim']
            }


        net = gnet(gnet_config)
    savename = savename_from_config(config)
    print('../model/'+savename+'.pt')
    
    try:
        net.load_state_dict(torch.load('../model/'+savename+'.pt'))
    except:
        checkpoint = torch.load('../model/'+savename+'.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
    return(net)


def obtain_samples(config, data_params,nsamples):
    net = load_model(config,data_params)
    d = data_params['d']
    

    samples = []
    
    if 'cvae' in config['model']  or 'mscvae' in config['model']:
        A_unique = list(itertools.product([0, 1], repeat=d))
        A_unique = treatment2id(np.array(A_unique))
        for a in A_unique:
            with torch.no_grad():
                tmp = net.decode(torch.Tensor(np.random.randn(nsamples,config['hidden_size'])),
                                 torch.Tensor([a]).repeat(nsamples,1).type(torch.int64))
                samples.append(tmp.detach().numpy()) 
                
    elif 'msm' in config['model'] :
        A_unique = list(itertools.product([0, 1], repeat=d))
        for a in A_unique:
            with torch.no_grad():
                tmp = net(torch.Tensor([a]).repeat(nsamples,1))
                samples.append(tmp.detach().numpy()) 
    elif 'gnet' in config['model'] :
        A_unique = list(itertools.product([0, 1], repeat=d))
        for a in A_unique:
            X_gnet, Y_gnet = net.forward(torch.Tensor(a).repeat(nsamples,1)[:,:,None])
            samples.append(Y_gnet.detach().numpy().squeeze())         
    return(samples)

    
def treatment2id(A):
    # input: A: [nsamples, d] array
    d= len(A[0])
    A_unique = list(itertools.product([0.0, 1.0], repeat=d))
    labels = np.zeros(len(A))
    for i, a in enumerate(A_unique):
        if d == 1:
            inds  = np.where(A==a)[0]#for d=1
        else:
            inds  = np.where(np.all(A==a,axis=1))#for d>1
        labels[inds] = i
    return(labels)
    

    
def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

def vae_loss_fn(x, recon_x, mu, logvar,w_train,beta=1, loss_type='bce'):
    batch_size = x.shape[0]
    if  w_train is None: # for the unweighted loss
        if loss_type=='bce':
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE + KLD*beta)/batch_size
    else: # for the weighted loss
        if loss_type=='bce':
            BCE = (F.binary_cross_entropy(recon_x, x, reduction='none')*w_train[:,None]).sum()
        else:
            BCE = (F.mse_loss(recon_x, x, reduction='none')*w_train[:,None]).sum()
        KLD = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp())*w_train[:,None])
        return (BCE + KLD*beta)/batch_size



    
    
def train_CVAE(config,data_params,data, muted = True,overwrite=True):
    '''
    Train the CVAE/MSCVAE
    Params
    *config: training configurations
    *data: training/val data
    *data_params: parameters for the data generation
    *is_msm: =0 for CVAE, =1 for MSCVAE
   
    '''
    # if model exists, the user decide whether to retrain the model
    savename = savename_from_config(config)
    if os.path.exists('../model/'+savename+'.pt'):
        print("Trained model already exits, overwrite.")
        #overwrite = input("Trained model already exits, overwrite it? (Y/N)")
        if not overwrite:
            return
                        
    # Configure the NN
    cvae_network_params={'d':data_params['d'],
                 'X_dim':data_params['X_dim'],
                 'Y_dim':data_params['Y_dim'],
                 'hidden_size':config['hidden_size'],
                 'layer_width': config['layer_width'],
                 'n_layer':config['n_layer'],
                 'reweight':config['reweight'],
                 'embedding_dim':config['embedding_dim'],
                 'seed':data_params['seed'],
                 'beta':config['beta'],
                 'recon_loss':config['recon_loss']
                    }
    if cvae_network_params['Y_dim'] >500:
        net = CVAE(cvae_network_params)
    else:
        net = CVAE(cvae_network_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['epochs']/2), gamma=0.5)
    ndata = len(data['Y'])
   
    train_subset, val_subset = random_split(
        list(zip(torch.Tensor(data['A_bar']),
                 torch.Tensor(clip_w(data['iptw'],0.01,99.99,True)),
                 torch.Tensor(data['Y']))),[int(ndata * 0.9), ndata-int(ndata * 0.9)]
    )
    if config['reweight'] == 1:
        
        y_train_indices = train_subset.indices
        y_train = [data['A_bar'][i] for i in y_train_indices]
        counter = collections.Counter(y_train)
        sample_count = np.array(
            [counter[t] for t in y_train])
        sample_count = len(y_train)/sample_count
        samples_weight = torch.from_numpy(sample_count)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                                         len(samples_weight),replacement = True)
        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=int(config["batch_size"]),  sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True)
    
    train_losses = []
    val_losses = []
    running_loss = 0.0
    epoch_steps = 0
    if os.path.exists('../model/'+savename+'.pt'):
        checkpoint = torch.load('../model/'+savename+'.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

    with tqdm(desc="epoch", total=config['epochs']) as pbar_outer:  
        for epoch in range(config['epochs']):  # loop over the dataset multiple times
            
            for i, data in enumerate(trainloader,0):
                a, iptw, y= data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                y_recon,mu,logvar =  net(y,a.type(torch.int64))
                
                if 'mscvae' in config['model']:
                    loss = vae_loss_fn(y, y_recon, mu, logvar, iptw,config['beta'],config['recon_loss'])
                else:
                    loss = vae_loss_fn(y, y_recon, mu, logvar, None,config['beta'],config['recon_loss'])
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i%100 ==99:
                    train_losses.append(running_loss / epoch_steps)
                    running_loss = 0.0
                    epoch_steps = 0
            # Validation loss
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    a, iptw, y = data
                    y_recon,mu,logvar =  net(y,a.type(torch.int64))

                    if 'mscvae' in config['model']:
                        loss = vae_loss_fn(y, y_recon, mu, logvar, iptw,config['beta'],config['recon_loss'])
                    else:
                        loss = vae_loss_fn(y, y_recon, mu, logvar, None,config['beta'],config['recon_loss'])
                    val_loss += loss.cpu().numpy()
                    val_steps += 1
            val_losses.append(val_loss/val_steps)
            #scheduler.step()
            pbar_outer.update(1)
            if not muted and len(val_losses) and len(train_losses)>0:
                print(
                    "epoch %d, train loss: %.3f, val loss: %.3f"
                    % (epoch + 1, train_losses[-1], val_losses[-1])
                )
            
            
    
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses':val_losses
            }, '../model/'+savename+'.pt')
    
    output_dict = {'config':config,'data_params':data_params,'train_losses':train_losses,'val_losses':val_losses}
    sio.savemat('../model/'+savename+'.mat',mdict=output_dict)
    print("Finished Training")
    return(output_dict)




class CVAE(nn.Module):
    def __init__(self, params):
    #def __init__(self, input_size, labels_length,hidden_size=20, layer_width = 128):
        super(CVAE, self).__init__()
        d = params['d']
        self.hidden_size = params['hidden_size']
        self.input_size=params['Y_dim']
        
        self.layer_width = params['layer_width']
        self.n_layer = params['n_layer']
        self.embedding_dim = params['embedding_dim']
        
        if self.embedding_dim>0:
            self.embedding = nn.Embedding(np.power(2,d), self.embedding_dim)
            self.labels_length=self.embedding_dim
        else:
            self.labels_length=np.power(2,d)
            
        
        
        self.encode_shared_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size+self.labels_length,
                                                                    self.layer_width),nn.ReLU())])
        self.encode_shared_layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.ReLU())
                                   for i in range(1, self.n_layer-1)])
        
        
        
        
        self.encode_mu_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_width, self.hidden_size))])

        self.encode_sigma_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.layer_width, self.hidden_size))])
        
      
        self.decode_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_size+self.labels_length,
                                                                    self.layer_width),nn.ReLU())])
        self.decode_layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.ReLU())
                                   for i in range(1, self.n_layer-1)])
        self.decode_layers.append(nn.Sequential(nn.Linear(self.layer_width, self.input_size)))
    
    def encode(self, x, labels):
        x = x.view(-1, self.input_size)
        if self.embedding_dim>0:
            labels = torch.squeeze(self.embedding(labels))
        else:
            labels =  idx2onehot(labels, self.labels_length)
        
        x = torch.cat((x, labels), 1)
        for layer in self.encode_shared_layers:
            x = layer(x)
        return self.encode_mu_layers[0](x), self.encode_sigma_layers[0](x)
        
    def decode(self, z, labels):
        if self.embedding_dim>0:
            labels = torch.squeeze(self.embedding(labels))
        else:
            labels =  idx2onehot(labels, self.labels_length)
        z = torch.cat((z, labels), 1)
        for layer in self.decode_layers:
            z = layer(z)
        return F.sigmoid(z)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    def forward(self,x,labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, labels)
        return x, mu, logvar
    
    
    
    
    
    
    
    
class PropensityNetworkNumerator(torch.nn.Module):
    def __init__(self, params):
        super(PropensityNetworkNumerator, self).__init__()
        d = params['d']
        self.input_size = d-1
        self.output_size = 1
        self.layer_width = params['layer_width']
        self.n_layer = params['n_layer']
        
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size, self.layer_width),nn.Sigmoid())])
        self.layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.Sigmoid()) for i in range(1, self.n_layer-1)])
        self.layers.append(nn.Sequential(nn.Linear(self.layer_width, self.output_size),nn.Sigmoid()))
    def forward(self,x):  
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return (x.squeeze())
    
class PropensityNetworkDenominator(torch.nn.Module):
    def __init__(self, params):
        super(PropensityNetworkDenominator, self).__init__()
        d = params['d']
        self.input_size = d-1 + d*params['X_dim']
        self.output_size = 1
        self.layer_width = params['layer_width']
        self.n_layer = params['n_layer']
        
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size, self.layer_width),nn.Sigmoid())])
        self.layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.Sigmoid()) for i in range(1, self.n_layer-1)])
        self.layers.append(nn.Sequential(nn.Linear(self.layer_width, self.output_size),nn.Sigmoid()))
    def forward(self,x):  
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return (x.squeeze())
    

    


class CovariateNetwork(torch.nn.Module):
    def __init__(self, params):
        super(CovariateNetwork, self).__init__()
        d = params['d']
        self.input_size = d-1 + (d-1)*params['X_dim']
        self.output_size = params['X_dim']
        self.layer_width = params['layer_width']
        self.n_layer = params['n_layer']
        
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size, self.layer_width),nn.Sigmoid())])
        self.layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.Sigmoid()) for i in range(1, self.n_layer-1)])
        self.layers.append(nn.Sequential(nn.Linear(self.layer_width, self.output_size)))
    def forward(self,x):  
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return (x.squeeze())






def train_propensity(config,data_params,data, muted = True,overwrite=True):
    '''
    Train a propensity network
    Input
    ------
    config: training configurations
    data: training/val data
    data_params: parameters for the data generation
    numerator: if True, train P(A|\bar{A}),else, train P(A|\bar{X},\bar{A})
    '''
    # if model exists, the user decide whether to retrain the model
    savename = savename_from_config(config)
    if os.path.exists('../model/'+savename+'.pt'):
        print("Trained model already exits, overwrite.")
        #overwrite = input("Trained model already exits, overwrite it? (Y/N)")
        if not overwrite:
            return
        
        
    # Configure the NN
    propensity_network_params={'d':data_params['d'],
                             'X_dim':data_params['X_dim'],
                             'Y_dim':data_params['Y_dim'],
                             'layer_width': config['layer_width'],
                             'n_layer':config['n_layer'],
                             'seed':data_params['seed'],
                            }

    if config['model'] == 'propensity_numerator':
        net = PropensityNetworkNumerator(propensity_network_params)
    else:
        net = PropensityNetworkDenominator(propensity_network_params)


    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['epochs']/2), gamma=0.5)
    
    ndata = len(data['A'])
    if config['model'] == 'propensity_numerator':
        if data_params['d']==1:
            raise ValueError('when d=1, the numerator is simply P(A=a)! ')  
        train_subset, val_subset = random_split(
            list(zip(torch.Tensor(data['A_bar']),torch.Tensor(data['A']))), [int(ndata * 0.8), ndata-int(ndata * 0.8)]
        )
    else:
        if data_params['d']>1:
            train_subset, val_subset = random_split(
                list(zip(torch.Tensor(np.concatenate((data['X_bar'],data['A_bar']),1)),
                         torch.Tensor(data['A']))),[int(ndata * 0.8), ndata-int(ndata * 0.8)]
            )
        else:
            train_subset, val_subset = random_split(
                list(zip(torch.Tensor(data['X_bar']),
                         torch.Tensor(data['A']))),[int(ndata * 0.8), ndata-int(ndata * 0.8)]
            )
    
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True)
    

    train_losses = []
    val_losses = []
    with tqdm(desc="epoch", total=config['epochs']) as pbar_outer:  

        for epoch in range(config['epochs']):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i%100 ==99:
                    train_losses.append(running_loss / epoch_steps)
                    running_loss = 0.0
                    epoch_steps = 0
              
            # Validation loss
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1
            if not muted and len(val_losses) and len(train_losses)>0:
                print(
                    "epoch %d, train loss: %.3f, val loss: %.3f"
                    % (epoch + 1, train_losses[-1], val_losses[-1])
                )
            val_losses.append(val_loss/val_steps)
            scheduler.step()
            pbar_outer.update(1)
        
        
    savename = savename_from_config(config)
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses':val_losses
            }, '../model/'+savename+'.pt')
    output_dict = {'config':config,'data_params':data_params,'train_losses':train_losses,'val_losses':val_losses}
    sio.savemat('../model/'+savename+'.mat',mdict=output_dict)
    print("Finished Training")
    return output_dict
    



class MSM(nn.Module):
    def __init__(self, params):
    #def __init__(self, input_size, labels_length,hidden_size=20, layer_width = 128):
        super(MSM, self).__init__()
        d = params['d']
        self.input_size = d
        self.layer_width = params['layer_width']
        self.n_layer = params['n_layer']
        self.output_size = params['Y_dim']
        
        
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.input_size,self.layer_width),nn.ReLU())])
        self.layers.extend([nn.Sequential(nn.Linear(self.layer_width, self.layer_width),nn.ReLU())
                                   for i in range(1, self.n_layer-1)])
        if self.output_size >10:
            self.layers.extend([nn.Sequential(nn.Linear(self.layer_width,self.output_size),nn.Sigmoid())])
        else:
            self.layers.extend([nn.Sequential(nn.Linear(self.layer_width,self.output_size),nn.Sigmoid())])
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x
        
        

class gnet(nn.Module):
    '''
    what is the output?
    '''
    def __init__(self, config) -> None:
        super(gnet, self).__init__()
        self.batch_size = config["batch_size"]
        self.seq_len = config["seq_len"]
        self.x_dim = config["x_dim"]
        self.y_dim = config["y_dim"]
        self.a_dim = config["a_dim"]
        self.r_dim = config["r_dim"]
        self.mlp_dim = config["mlp_dim"]
        self.MSELoss = nn.MSELoss()

        self.x_sigma = torch.zeros([1])
        self.y_sigma = torch.zeros([1])

        # Blocks
        rblock = nn.Sequential(
            nn.Linear(self.x_dim + self.a_dim + self.r_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.r_dim)
        )

        xblock = nn.Sequential(
            nn.Linear(self.r_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.x_dim)
        )

        # Nets
        self.rnet = nn.ModuleList(
            [rblock for i in range(self.seq_len)]
        )
        self.xnet = nn.ModuleList(
            [xblock for i in range(self.seq_len)]
        )

        self.ynet = nn.Sequential(
            nn.Linear(self.r_dim, self.mlp_dim), # More input, finer outcome but more training time
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.y_dim)
        )

    def forward(self, a : torch.Tensor, init_x = None):
        '''
        Used in Inference Procedure
        Give a complete sequence of a, returns inferenced x and y
        init_x is the initial covaraite x
        '''
        # assert a.shape == torch.Size([self.batch_size, self.seq_len, self.a_dim]), f"Check treatment sequence should be of shape {torch.Size([self.seq_len])}!"
        # assert (init_x == torch.Size([self.batch_size, self.seq_len, self.x_dim])) | (init_x == None), f"Check the init_x input!"

        r = torch.zeros((a.shape[0], 1, self.r_dim))       # Initialization
        if init_x is None:
            x = torch.zeros((a.shape[0], 1, self.x_dim))
        else:
            x = init_x

        x_list = []    
        for i in range(self.seq_len):
            mix_input = torch.cat((a[:, i, :].unsqueeze(1), x, r), 2).float()    # [batch_size, 1, r_dim + x_dim + a_dim]
            r = self.rnet[i](mix_input)                     # [batch_size, 1, r_dim]
            x = self.xnet[i](r)                             # [batch_size, 1, x_dim]
            x_list.append(x.squeeze(1))                     # append [batch_size, x_dim]

        y_hat = self.ynet(r) + self.y_sigma * torch.randn([a.shape[0], 1, self.y_dim])  # [batch_size, 1, y_dim]
        x_hat = torch.stack(x_list, 1)                      # [batch_size, seq_len, x_dim]

        return x_hat, y_hat                # [batch_size, seq_len, x_dim] and [batch_size, 1, y_dim]              
    
    def loss(self, data):
        '''
        Used in Training Procedure
        Give a complete sequence of y, x and a, returns the loss of the model.
        y, x and a are of shape [batch_size, seq_len, y_dim/x_dim/a_dim]
        '''
        y_ori = data[:, :, :self.y_dim]
        x_ori = data[:, :, self.y_dim:(self.y_dim + self.x_dim)]
        a_ori = data[:, :, (self.y_dim + self.x_dim):(self.y_dim + self.x_dim + self.a_dim)]

        input = torch.cat((a_ori, x_ori), 2)    # a : [batch_size, seq_len, self.a_dim + self.x_dim]
        x_list = []
        r = torch.zeros((self.batch_size, 1, self.r_dim))    # Initialization
        for i in range(self.seq_len):
            mix_input = torch.cat((input[:, i, :].unsqueeze(1), r), 2).float()  # [batch_size, 1, r_dim + x_dim + a_dim]
            r = self.rnet[i](mix_input)                     # [batch_size, 1, r_dim]
            x = self.xnet[i](r) # [batch_size, 1, x_dim]
            x_list.append(x.squeeze(1))                     # append [batch_size, x_dim]
        
        y_hat = self.ynet(r) # [batch_size, 1, y_dim]
        x_hat = torch.stack(x_list, 1)                      # [batch_size, seq_len, x_dim]

        return torch.sum(self.MSELoss(x_hat, x_ori)) + self.MSELoss(y_hat, y_ori[:, -1 ,:].unsqueeze(1))

