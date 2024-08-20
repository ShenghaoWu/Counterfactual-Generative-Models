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
from scipy.spatial import cKDTree as KDTree

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def exp_decay(base,decay_coeff,d):
        return(list(np.exp(-decay_coeff*np.arange(d)[::-1])*base))
    
def interleave(arrys):
    narry = len(arrys)
    if isinstance(arrys[0], list) or len(arrys[0].shape)==1:
        total_size = np.sum([a.size for a in arrys])
        res = np.zeros(total_size)
    else:
        total_size = np.sum([a.shape[0] for a in arrys])
        res = np.zeros([total_size,arrys[0].shape[1]]) 
    for i in range(narry):
        res[i::narry] = arrys[i]
    return(res)

def savename_from_config(config):
    #generate a savename string from a config dic
    names = []
    for item in config:
        names.append(item) 
        names.append(str(config[item]))

    return(''.join(names))


    

def hist_1d2d(ax_to_plot,data,weights,Y_dim,color,label):
        
        if Y_dim == 1:
            hists = np.histogram(data[:,0],density=True, weights = weights,bins=100)
            (dens, cutoffs) = hists[0],hists[1]
            ax_to_plot.plot((cutoffs[1:]+cutoffs[:-1])/2,dens,color=color,linewidth=2,alpha=0.8,label=label)
            
        else:
            
            h,xedge,yedge = np.histogram2d(data[:,0],data[:,1], weights=weights ,bins=50)
            xcoord, ycoord = np.meshgrid((xedge[1:]+xedge[:-1])/2, (yedge[1:]+yedge[:-1])/2)
            ax_to_plot.contour(xcoord, ycoord, h, colors=color,levels=1)
            

            #ax_to_plot.hist2d(data[:,0],data[:,1],density=True,weights = weights,bins=100, color=color,linewidth=2,alpha=0.8,label=label)
            #inds = np.random.choice(range(len(data)),1000,replace = False)
            #ax_to_plot.scatter(data[inds,0],data[inds,1], color=color,alpha=0.8,label=label)

def plot_outcome_dist(Y,A,iptw=None,Y_as=[]):
    
    '''
    Plot the observed and counterfactual distributions
    
    Input:
    A: [#samples, d]
    Y: [#samples, Y_dim] 
    iptw:[#samples, 1]: IPTW weights
    '''
            
    d = A.shape[1]
    Y_dim = Y.shape[1]
    A_unique = list(itertools.product([0, 1], repeat=d))
    n_treatments = len(A_unique)
    nrow = int(np.ceil(n_treatments/4))
    ncol = min(n_treatments,4)
    fig, axs = plt.subplots(nrow,ncol,figsize=(ncol*3,nrow*3))
    
    cnt = 0
    m = 0
    m_obs = 0
    for i in range(nrow):
        for j in range(ncol):
            a = A_unique[cnt]
                
            if d == 1:
                inds  = np.where(A==a)[0]#for d=1
            else:
                inds  = np.where(np.all(A==a,axis=1))#for d>1
                
                        
            # if no treatment is observed, continue
            if len(inds)==0:
                continue
                
            if nrow==1:
                ax_to_plot = axs[j]
            else:
                ax_to_plot = axs[i,j]  
                
            hist_1d2d(ax_to_plot,Y[inds],weights = None,Y_dim=Y_dim,color = npg_color_palettes[0],label='obs')
            m_obs+=compute_metric(Y[inds],Y_as[cnt],'mean')
            if isinstance(iptw, list):
                # if iptw is a list, it is a list of samples from an algo e.g. MSCVAE
                hist_1d2d(ax_to_plot,iptw[cnt],weights = None,Y_dim=Y_dim,color = npg_color_palettes[1],label='model')
                m += compute_metric(iptw[cnt],Y_as[cnt],'mean')
            else:
                # if iptw is a np array, it's actual iptw array
                hist_1d2d(ax_to_plot,Y[inds],weights = iptw[inds],Y_dim=Y_dim,color = npg_color_palettes[1],label='iptw')
            if len(Y_as)>0:
                hist_1d2d(ax_to_plot,Y_as[cnt],weights = None,Y_dim=Y_dim,color = npg_color_palettes[2],label='causal')
            # compute metrics
            
            
            
            
            ax_to_plot.set_title(r'$\overline{a}=$('+','.join([str(int(s)) for s in str(A_unique[cnt]) if s.isdigit()])+')'
                                +'\nobs mean: %.3f \ncausal mean: %.3f' %(np.mean(Y[inds]),np.mean(Y_as[cnt])))
            
            '''
            hists = np.histogram(Y_a[0],density=True,bins=100)
            (dens, cutoffs) = hists[0],hists[1]
            plt.plot((cutoffs[1:]+cutoffs[:-1])/2,dens,color=npg_color_palettes[2],linewidth=2,alpha=0.8,label='causal')
            '''
            ax_to_plot.spines['right'].set_visible(False)
            ax_to_plot.spines['top'].set_visible(False)
            
            cnt += 1
            if cnt == n_treatments:
                break

    plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(hspace=0.5)
    if nrow==1:
        axs[0].legend()
    else:
        axs[0,0].legend()
        
    print( 'mean diff between obs and causal: %.3f'%(m_obs/np.power(2,d)))
    print( 'mean diff between iptw and causal: %.3f'%(m/np.power(2,d)))
    plt.show()



def compute_metric(x,y,metric):
    if len(x.shape)!= len(y.shape):
        raise ValueError('two distributions need to have same number of dims!')  
    nsamples = min(len(x),len(y))
    nsamples = min(nsamples, 5000 * x.shape[1]) # number of samples scale with number of dims
    indx = np.random.choice(range(len(x)),nsamples,replace=False)
    indy = np.random.choice(range(len(x)),nsamples,replace=False)
    if metric == 'mean':
        return(mean_metric(x[indx],y[indy]))
    elif metric == 'kl':
        return(kl_metric(x[indx],y[indy]))
    
    if metric == 'mmd':
        return(mmd_metric(x[indx],y[indy]))
        
def mean_metric(x,y):
    # input: [#data #dim]

    if len(x.shape)==1:
        return(np.abs(np.mean(x)-np.mean(y)))
    else:
        return(np.linalg.norm(np.mean(x,axis=0)-np.mean(y,axis=0)))


def kl_metric(x, y):
    """
    Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.

    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    https://mail.python.org/pipermail/scipy-user/2011-May/029521.html

    """
        

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

def mmd_metric(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    
    xx, yy, zz = (torch.mm(torch.Tensor(x), torch.Tensor(x).t()),
                  torch.mm(torch.Tensor(y), torch.Tensor(y).t()),
                  torch.mm(torch.Tensor(x), torch.Tensor(y).t()))
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)


        




def clip_w(w,a=1,b=99,mean_norm=True):
    #clip the IPTW weights to to percentiles within [a,b] and normalize them by means.
    if a>0 or b<100:
        a,b = np.percentile(w.flatten(),[a,b])
        w = np.clip(w,a,b)
    if mean_norm:
        w=w/w.mean()
    return(w)



def prepare_data_outcome(simu_data,params,propensity_network=None):
    '''
    Return a training dataset for the conditional generative model
    Input
        [X: [S,T,X_dim], F: [S,T], Y: [S,T]]: raw time series 
        parmas: simulation parameters
    Return: 
        A_bar: [#samples d]
        Y: [#samples Y_dim] 
        wt:[#samples]: IPTW weights
    '''
    d = params['d']
    S = params['S']
    T = params['T']
    X_dim = params['X_dim']
    Y_dim = params['Y_dim']
    
    X,F,A,Y = simu_data['X'],simu_data['F'],simu_data['A'],simu_data['Y']

    iptw = []
    
    if d == 1:
        if propensity_network:
            print('------------')
            print('Using a propensity network to compute iptw')
            F_pred = [eval_model(propensity_network,X[t:t+1]) for t in range(len(X))]
            F_pred=np.array(F_pred)
            iptw = 1/(np.multiply(F_pred,A)+np.multiply(1-F_pred,1-A))
        else:
            iptw = 1/(np.multiply(F,A)+np.multiply(1-F,1-A))
        return({'Y':Y.reshape(-1,Y_dim), 'A_bar':A.reshape(-1)[:,None], 'iptw':iptw.reshape(-1)})
    
    
    if propensity_network:
        F_pred = []
        for t in range(Y.shape[1]-d+1):
            F_pred.append(eval_model(propensity_network,
                                     np.concatenate((X[:,t:t+d].reshape(-1,d*X_dim),A[:,t:t+d-1]),axis=1)))
        F_pred=np.array(F_pred).swapaxes(0,1)
        A_bar = []
        F_pred_bar = []
        for t in range(Y.shape[1]-2*d+2):
            A_bar.append(A[:,t+d-1:t+2*d-1])  # [S,d]
            F_pred_bar.append(F_pred[:,t:t+d])  # [S,d]

        A_bar = np.swapaxes(np.array(A_bar),0,1).reshape(-1,d) #A_bar: [T-d+1, S, d] -> [S * T-d+1, d]
        F_pred_bar = np.swapaxes(np.array(F_pred_bar),0,1).reshape(-1,d) #A_bar: [T-d+1, S, d] -> [S * T-d+1, d]

        # if a propensity network is not provided, use the groundtruth probability to compute IPTW
        iptw = np.prod(1/(np.multiply(F_pred_bar,A_bar)+np.multiply(1-F_pred_bar,1-A_bar)),axis=1)

        return({'Y':Y[:,2*d-2:].reshape(-1,Y_dim), 'A_bar':A_bar,'iptw':iptw})
        
        
    
    else:
        A_bar = []
        F_bar = []
        for t in range(Y.shape[1]-d+1):
            A_bar.append(A[:,t:t+d])  # [S,d]
            F_bar.append(F[:,t:t+d])  # [S,d]

        A_bar = np.swapaxes(np.array(A_bar),0,1).reshape(-1,d) #A_bar: [T-d+1, S, d] -> [S * T-d+1, d]
        F_bar = np.swapaxes(np.array(F_bar),0,1).reshape(-1,d) #A_bar: [T-d+1, S, d] -> [S * T-d+1, d]

        # if a propensity network is not provided, use the groundtruth probability to compute IPTW
        iptw = np.prod(1/(np.multiply(F_bar,A_bar)+np.multiply(1-F_bar,1-A_bar)),axis=1)

    
    
        #Y: [S, T,Y_dim] -> [S * T-d+1,Y_dim] 
        return({'Y':Y[:,d-1:].reshape(-1,Y_dim), 'A_bar':A_bar,'iptw':iptw})
    
    
def prepare_data_propensity(simu_data,params,x_collapse=True):
    '''
    Return a training dataset for the propensity network
    Input
        [X: [S,T,X_dim], F: [S,T], A: [S,T]]: raw time series 
        parmas: simulation parameters
    Return: 
        A: [#samples]
        F: [#samples] # Not involved in training, just for sanity check
        A_bar:[#samples, d-1]
        X_bar:[#samples, d*X_dim]
    
    '''
    d = params['d']
    S = params['S']
    T = params['T']
    X_dim = params['X_dim']
    
    X,F,A,Y = simu_data['X'],simu_data['F'],simu_data['A'],simu_data['Y']
    A_bar, X_bar = [],[]
    if d == 1:
        return({'A':A.reshape(-1), 'F':F.reshape(-1), 'A_bar':A.reshape(-1)[:,None], 'X_bar':X.reshape(-1,X_dim)})
        
    for t in range(X.shape[1]-d+1):
        A_bar.append(A[:,t:t+d-1])  # [ S,d-1]
        X_bar.append(X[:,t:t+d])    # [ S,d,X_dim]
    
    A_bar = np.swapaxes(np.array(A_bar),0,1).reshape(-1,d-1) #A_bar: [T-d+1, S, d-1] -> [S * T-d+1, d-1]
    if x_collapse:
        X_bar = np.swapaxes(np.array(X_bar),0,1).reshape(A_bar.shape[0],d*X_dim) #X_bar: [T-d+1, S, d, X_dim] -> [S * T-d+1, d*X_dim]
    else:
        X_bar = np.swapaxes(np.array(X_bar),0,1).reshape(A_bar.shape[0],d,X_dim) #X_bar: [T-d+1, S, d, X_dim] -> [S * T-d+1, d*X_dim]

    #A: [S, T] -> [S * T-d+1] 
    return({'A':A[:,d-1:].reshape(-1), 'F':F[:,d-1:].reshape(-1), 'A_bar':A_bar, 'X_bar':X_bar})
    
   

  
# evaluating model

def eval_model(net,test_data):
    if torch.is_tensor(test_data):  
        return(net(test_data).detach().numpy())
    else:
        return(net(torch.Tensor(test_data)).detach().numpy())
