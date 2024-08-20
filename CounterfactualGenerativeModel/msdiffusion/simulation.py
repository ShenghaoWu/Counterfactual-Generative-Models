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
from utils import *




class time_varying_mnist_generator_phi:
    '''
    Generate the time-varying simulated data
    Params
    *S: number of individuals
    *T: number of time points per individual
    *mu, sigma: mean/std of the observational noise for Y
    *seed: random seed
    *is_plot: if plot simulated data 
    '''
    def __init__(self, params):
        self.params = params
        self.params['buffer'] = self.params['d']*3
        h = HCMNIST()
        self.imgs = h.x
        self.phis = h.phi
    def simulate_obs(self):
        
        print('------------')
        print('Start simulating observations')
        print('------------')
        
        S = self.params['S']
        T = self.params['T']
        d = self.params['d']
        decay_coeff = self.params['decay_coeff']
        seed = self.params['seed']
        sigma=self.params['noise_sigma']
        is_plot = self.params['is_plot']    
        X_dim = self.params['X_dim'] 
        Y_dim = self.params['Y_dim'] 
        
        if seed >=0:
            np.random.seed(seed)
            
        self.X = np.zeros([S,T,X_dim])
        self.F = np.zeros([S,T])
        self.A = np.zeros([S,T])
        self.Y = np.zeros([S,T,Y_dim])
        
        
        # X_filter: bias[1], X_X [d-1]*, X_A [d-1, X_dim]
        # F_filter: bias[1], F_X [d*X_dim], F_A [d-1]
        # Y_filter: bias[1], Y_X [d*X_dim, Y_dim], F_A [d, Y_dim]
        X_filter, F_filter, Y_filter = self._generate_simulation_filter(d,decay_coeff,X_dim,Y_dim)
        self.X_filter = X_filter
        self.F_filter = F_filter
        self.Y_filter = Y_filter

        #initial X from a uniform dist
        self.X[:,0,:] = np.random.uniform(size=[S,X_dim])    
        self.F[:,0] = sigmoid(self.F_filter['bias']+self.X[:,0]@self.F_filter['X'][:X_dim])
        self.A[:,0] = np.random.binomial(1,self.F[:,0])  
        if d == 1:
            self._simulate_observation_1d()
            if is_plot:
                self.plot_simu(self.datasets)
            return
    
        for t in range(1,d):
            # X(t) depends on X(t-d+1,...,t-1), A(t-d+1,...,t-1)
            # X_X: [S, t, X_dim]@[t] -> [S,X_dim,t]@[t] -> [S,X_dim]
            # X_A: [S,t]@[t, X_dim]-> [S,X_dim]
            self.X[:,t] = (self.X_filter['bias']
                           +(self.X[:,:t].swapaxes(1,2)@self.X_filter['X'][:t])
                           +(self.A[:,:t]@self.X_filter['A'][:t]))
                            
            # A(t) depends on  X(t-d+1,...,t), A(t-d+1,...,t-1)
            # F_X: [S, t+1, X_dim]@[(t+1)*X_dim] -> [S]
            # F_A: [S,t]@[t]-> [S,1,t]@[t,1] -> [S]
          
            self.F[:,t] = sigmoid(self.F_filter['bias']
                                      +self.X[:,:t+1].reshape(S,-1)@self.F_filter['X'][:(t+1)*X_dim]
                                      +self.A[:,:t]@self.F_filter['A'][:t])
            
            
            self.A[:,t] = np.random.binomial(1,self.F[:,t])

        for t in range(d,T):
            # X(t) dependency is diff from F(t)! here we simply linearly combine then instead of matmul
            self.X[:,t] = (self.X_filter['bias']
                           +(self.X[:,t-d+1:t].swapaxes(1,2)@self.X_filter['X'])
                           +(self.A[:,t-d+1:t]@self.X_filter['A']))
            
            
            # A(t) depends on  X(t-d+1,...,t), A(t-d+1,...,t-1)
            self.F[:,t] = sigmoid(self.F_filter['bias']
                                  +self.X[:,t-d+1:t+1].reshape(S,-1)@self.F_filter['X']
                                  +self.A[:,t-d+1:t]@self.F_filter['A'])

            self.A[:,t] = np.random.binomial(1,self.F[:,t])
            
            # Y(t) depends on X(t-d+1,...,t), A(t-d+1,...,t)
            self.Y[:,t] = self._generate_Y(self.X[:,t-d+1:t+1].reshape(S,-1),
                                           self.A[:,t-d+1:t+1])
            '''
            self.Y[:,t] = (self.Y_filter['bias']
                           +self.X[:,t-d+1:t+1].reshape(S,-1)@self.Y_filter['X']
                           +self.A[:,t-d+1:t+1]@self.Y_filter['A'])  
            '''
        if sigma>0:
            self.Y=self.Y+np.random.normal(0,sigma,self.Y.shape) 
        if is_plot:
            self.plot_simu(self.datasets)
            
    def _simulate_observation_1d(self):
        
         # X_filter: bias[1], X_X [0]*, X_A [0, X_dim]
        # F_filter: bias[1], F_X [X_dim], F_A [0]
        # Y_filter: bias[1], Y_X [X_dim, Y_dim], F_A [1, Y_dim]

        seed = self.params['seed']
        d = self.params['d']
        S = self.params['S']
        T = self.params['T']
        X_dim = self.params['X_dim'] 
        Y_dim = self.params['Y_dim'] 
        sigma=self.params['noise_sigma']
        for t in range(1,T):
            self.X[:,t] = np.random.uniform(size=[S,X_dim])
            self.F[:,t] = sigmoid(self.F_filter['bias']+self.X[:,t]@self.F_filter['X'])
            self.A[:,t] = np.random.binomial(1,self.F[:,t])  
            
            self.Y[:,t] = self._generate_Y(self.X[:,t],
                               self.A[:,t:t+1])
            '''
            self.Y[:,t] = (self.Y_filter['bias'] + self.X[:,t]@self.Y_filter['X']
                           +self.A[:,t:t+1]@self.Y_filter['A'])  
            '''
        self.Y=self.Y+np.random.normal(0,sigma,self.Y.shape) 

    def simulate_counterfactual_all(self):
        print('------------')
        print('Start simulating counterfactuals')
        print('------------')
        A_unique = list(itertools.product([0, 1], repeat=self.params['d']))
        Y_as = []
        for a in A_unique:
            Y_as.append(self.simulate_counterfactual(a))
        return(Y_as)
    
    
    def _simulate_counterfactual_1d(self, a):
        d = self.params['d']
        S = self.params['S']
        T = int(self.params['T']/3) #don't need that many samples!!
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']
        buffer = self.params['buffer']
        Y_a = []
        for t in range(d,T):
            Y_a.append(self._generate_Y(self.X[:,t],
                                        np.repeat(a,S)[:,None]))
            '''
            Y_a.append(self.Y_filter['bias'] + self.X[:,t]@self.Y_filter['X']
                           +np.repeat(a,S)[:,None]@self.Y_filter['A'])
            '''
        return(np.array(Y_a[max(0,buffer-d):]).reshape(-1,Y_dim))
            
    
    def simulate_counterfactual(self, a):
        # a: assigned treatment sequence at (A(t,...,t+d-1).
        # modify: U(t+1,...,t+d-1), X(t+1,...,t+d-1), Y(t+d-1)
        d = self.params['d']
        S = self.params['S']
        T = int(self.params['T']/3) #don't need that many samples!!
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']
        buffer = self.params['buffer']
        #If d=1 (static case), the DAG will be different
        if d == 1:
            Y_a = self._simulate_counterfactual_1d(a)
            return Y_a
        
        Y_a = []
        #Starting from time d, modify A(t,...,t+d-1), then  U(t+1,...,t+d-1), X(t+1,...,t+d-1), Y(t+d-1)
        #Make copies from the original arrays
        for t in range(d,T-d+1):
            #for t:t+d th segment, starting from the t+1 th X depends on t-(d-2) th covariates
            X_copy = self.X[:,(t-d+2):t+d].copy()
            A_copy = self.A[:,(t-d+2):t+d].copy()
            A_copy[:,-d:] = a
            for k in range(d-1):
                
                # X(t) depends on X(t-d+1,...,t-1), A(t-d+1,...,t-1)
                X_copy[:,k+d-1]=(self.X_filter['bias']
                                 +X_copy[:,k:k+d-1].swapaxes(1,2)@self.X_filter['X']
                                 +A_copy[:,k:k+d-1]@self.X_filter['A'])
    
            # Y(t) depends on X(t-d+1,...,t), A(t-d+1,...,t)
        
            Y_a.append(self._generate_Y(X_copy[:,-d:].reshape(S,-1),
                                        A_copy[:,-d:])) 
            '''
            Y_a.append(self.Y_filter['bias'] 
                       +X_copy[:,-d:].reshape(S,-1)@self.Y_filter['X']
                       +A_copy[:,-d:]@self.Y_filter['A']) 
            '''
        return(np.array(Y_a[max(0,buffer-d):]).reshape(-1,Y_dim))

            
            
    @property
    def datasets(self):
        # return the simulated datasets.
        T = self.params['T']
        buffer = self.params['buffer']
        assert T > buffer, 'T should be longer than buffer'

        return({'X':self.X[:,buffer:],'F':self.F[:,buffer:],'A':self.A[:,buffer:],'Y': self.Y[:,buffer:]})
    
    @staticmethod
    def plot_simu(simu_data):
        '''
        Plot simulated data
        '''
        X,F,A,Y = simu_data['X'],simu_data['F'],simu_data['A'],simu_data['Y']
        fig,ax = plt.subplots(1,4, figsize=(22,2.5))
        tt = min(Y.shape[1],150)
        data_to_plot = [Y,X,A,F]
        labels = ['Y','X','A','F']
        for i in range(4):

            if len(data_to_plot[i].shape)>2:
                ax[i].plot((data_to_plot[i][:2,:tt,0]).T)
            else:
                ax[i].plot((data_to_plot[i][:2,:tt]).T)
            ax[i].set_ylabel(labels[i])
            ax[i].set_xlabel('Time')
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        plt.subplots_adjust(wspace=0.5)
        plt.show()

        fig,ax = plt.subplots(1,4, figsize=(22,2.5))
        for i in range(4):
            if len(data_to_plot[i].shape)>2 and data_to_plot[i].shape[-1]>1:
                #ax[i].hist2d(data_to_plot[i][:,:,0].flatten(),data_to_plot[i][:,:,1].flatten(),bins=100)
                ax[i].scatter(data_to_plot[i][:,:,0].flatten(),data_to_plot[i][:,:,1].flatten(),s=1)
                ax[i].set_xlabel(labels[i]+'1')
                ax[i].set_ylabel(labels[i]+'2')
            else:
                ax[i].hist(data_to_plot[i].flatten(),bins=100)
                ax[i].set_xlabel(labels[i])
                ax[i].set_ylabel('count')
            
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        plt.subplots_adjust(wspace=0.5)
        plt.show()
    
    def _generate_simulation_filter(self, d, decay_coeff, X_dim, Y_dim):
    
        # X_filter: bias[1], X_X [(d-1)]*, X_A [d-1, X_dim]
        # F_filter: bias[1], F_X [d*X_dim], F_A [d-1]
        # Y_filter: bias[1], Y_X [d*X_dim, Y_dim], Y_A [d, Y_dim]

        seed = self.params['seed']
        if seed >=0:
                    np.random.seed(seed)
        d = self.params['d']
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']                

        X_filter = {'bias': 0,
                    'X': np.array(exp_decay(-1,decay_coeff,d-1)),#X-X coupling, simplified65
                    'A': np.tile(np.array(exp_decay(1,decay_coeff,d-1))[:,None],reps = (1,X_dim)) } #X-A coupling

        F_filter = {'bias': -1 ,
                    'X': interleave([np.array(np.power(-1,np.arange(d)[::-1])) for _ in range(X_dim)]), #F-X coupling
                    'A': np.array(0.5*np.power(-1,np.arange(d-1)[::-1])) } #F-A coupling
        
        Y_filter = {'bias': -3,
                    'X':np.tile(np.array(exp_decay(-1,decay_coeff,d))[:,None],(X_dim,1))  , #Y-X coupling
                    'A': np.tile(np.array(exp_decay(2,decay_coeff,d))[:,None],reps = (1,1)) } #Y-A coupling


        return(X_filter, F_filter, Y_filter)
    
    def _generate_Y(self, X,A):
        #print(X.shape,Y_filter['Y_xlayer1'].shape)
        #print(sigmoid(A@Y_filter['Y_alayer1']).shape)
        phi = (2/0.3)*(sigmoid(self.Y_filter['bias'] 
                       +X@self.Y_filter['X']
                       +A@self.Y_filter['A'])-0.3)
        return(self._phi2mnist(np.squeeze(phi)))


    def _phi2mnist(self,phi):
        return(self.imgs[np.argmin(np.abs(self.phis - phi),axis=0)])
    
    
    
    
    
class time_varying_policy_generator:
    '''
    Generate the time-varying simulated data
    Params
    *S: number of individuals
    *T: number of time points per individual
    *mu, sigma: mean/std of the observational noise for Y
    *seed: random seed
    *is_plot: if plot simulated data 
    '''
    def __init__(self, params):
        self.params = params
        self.params['buffer'] = self.params['d']*5
                
    def simulate_obs(self):
        
        print('------------')
        print('Start simulating observations')
        print('------------')
        
        S = self.params['S']
        T = self.params['T']
        d = self.params['d']
        decay_coeff = self.params['decay_coeff']
        seed = self.params['seed']
        sigma=self.params['noise_sigma']
        is_plot = self.params['is_plot']    
        X_dim = self.params['X_dim'] 
        Y_dim = self.params['Y_dim'] 
        
        if seed >=0:
            np.random.seed(seed)
            
        self.X = np.zeros([S,T,X_dim])
        self.F = np.zeros([S,T])
        self.A = np.zeros([S,T])
        self.Y = np.zeros([S,T,Y_dim])
        
        
        # X_filter: bias[1], X_X [d-1]*, X_A [d-1, X_dim]
        # F_filter: bias[1], F_X [d*X_dim], F_A [d-1]
        # Y_filter: bias[1], Y_X [d*X_dim, Y_dim], F_A [d, Y_dim]
        X_filter, F_filter, Y_filter = self._generate_simulation_filter(d,decay_coeff,X_dim,Y_dim)
        self.X_filter = X_filter
        self.F_filter = F_filter
        self.Y_filter = Y_filter

        #initial X from a uniform dist
        self.X[:,0,:] = np.random.uniform(size=[S,X_dim])    
        self.F[:,0] = sigmoid(self.F_filter['bias']+self.X[:,0]@self.F_filter['X'][:X_dim])
        self.A[:,0] = np.random.binomial(1,self.F[:,0])  
        if d == 1:
            self._simulate_observation_1d()
            if is_plot:
                self.plot_simu(self.datasets)
            return
    
        for t in range(1,d):
            # X(t) depends on X(t-d+1,...,t-1), A(t-d+1,...,t-1)
            # X_X: [S, t, X_dim]@[t] -> [S,X_dim,t]@[t] -> [S,X_dim]
            # X_A: [S,t]@[t, X_dim]-> [S,X_dim]
            self.X[:,t] = (self.X_filter['bias']
                           +(self.X[:,:t].swapaxes(1,2)@self.X_filter['X'][:t])
                           +(self.A[:,:t]@self.X_filter['A'][:t]))
                            
            # A(t) depends on  X(t-d+1,...,t), A(t-d+1,...,t-1)
            # F_X: [S, t+1, X_dim]@[(t+1)*X_dim] -> [S]
            # F_A: [S,t]@[t]-> [S,1,t]@[t,1] -> [S]
          
            self.F[:,t] = sigmoid(self.F_filter['bias']
                                      +self.X[:,:t+1].reshape(S,-1)@self.F_filter['X'][:(t+1)*X_dim]
                                      +self.A[:,:t]@self.F_filter['A'][:t])
            
            
            self.A[:,t] = np.random.binomial(1,self.F[:,t])

        for t in range(d,T):
            # X(t) dependency is diff from F(t)! here we simply linearly combine then instead of matmul
            self.X[:,t] = (self.X_filter['bias']
                           +(self.X[:,t-d+1:t].swapaxes(1,2)@self.X_filter['X'])
                           +(self.A[:,t-d+1:t]@self.X_filter['A']))
            
            
            # A(t) depends on  X(t-d+1,...,t), A(t-d+1,...,t-1)
            self.F[:,t] = sigmoid(self.F_filter['bias']
                                  +self.X[:,t-d+1:t+1].reshape(S,-1)@self.F_filter['X']
                                  +self.A[:,t-d+1:t]@self.F_filter['A'])

            self.A[:,t] = np.random.binomial(1,self.F[:,t])
            
            # Y(t) depends on X(t-d+1,...,t), A(t-d+1,...,t)
            self.Y[:,t] = self._generate_Y(self.X[:,t-d+1:t+1].reshape(S,-1),
                                           self.A[:,t-d+1:t+1])
            '''
            self.Y[:,t] = (self.Y_filter['bias']
                           +self.X[:,t-d+1:t+1].reshape(S,-1)@self.Y_filter['X']
                           +self.A[:,t-d+1:t+1]@self.Y_filter['A'])  
            '''
        if sigma>0:
            self.Y=self.Y+np.random.normal(0,sigma,self.Y.shape) 
        if is_plot:
            self.plot_simu(self.datasets)
            
    def _simulate_observation_1d(self):
        
         # X_filter: bias[1], X_X [0]*, X_A [0, X_dim]
        # F_filter: bias[1], F_X [X_dim], F_A [0]
        # Y_filter: bias[1], Y_X [X_dim, Y_dim], F_A [1, Y_dim]

        seed = self.params['seed']
        d = self.params['d']
        S = self.params['S']
        T = self.params['T']
        X_dim = self.params['X_dim'] 
        Y_dim = self.params['Y_dim'] 
        sigma=self.params['noise_sigma']
        for t in range(1,T):
            self.X[:,t] = np.random.uniform(size=[S,X_dim])
            self.F[:,t] = sigmoid(self.F_filter['bias']+self.X[:,t]@self.F_filter['X'])
            self.A[:,t] = np.random.binomial(1,self.F[:,t])  
            
            self.Y[:,t] = self._generate_Y(self.X[:,t],
                               self.A[:,t:t+1])
            '''
            self.Y[:,t] = (self.Y_filter['bias'] + self.X[:,t]@self.Y_filter['X']
                           +self.A[:,t:t+1]@self.Y_filter['A'])  
            '''
        self.Y=self.Y+np.random.normal(0,sigma,self.Y.shape) 

    def simulate_counterfactual_all(self):
        print('------------')
        print('Start simulating counterfactuals')
        print('------------')
        A_unique = list(itertools.product([0, 1], repeat=self.params['d']))
        Y_as = []
        for a in A_unique:
            Y_as.append(self.simulate_counterfactual(a))
        return(Y_as)
    
    
    def _simulate_counterfactual_1d(self, a):
        d = self.params['d']
        S = self.params['S']
        T = self.params['T']
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']
        buffer = self.params['buffer']
        Y_a = []
        for t in range(d,T):
            Y_a.append(self._generate_Y(self.X[:,t],
                                        np.repeat(a,S)[:,None]))
            '''
            Y_a.append(self.Y_filter['bias'] + self.X[:,t]@self.Y_filter['X']
                           +np.repeat(a,S)[:,None]@self.Y_filter['A'])
            '''
        return(np.array(Y_a[max(0,buffer-d):]).reshape(-1,Y_dim))
            
    
    def simulate_counterfactual(self, a):
        # a: assigned treatment sequence at (A(t,...,t+d-1).
        # modify: U(t+1,...,t+d-1), X(t+1,...,t+d-1), Y(t+d-1)
        d = self.params['d']
        S = self.params['S']
        T = self.params['T']
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']
        buffer = self.params['buffer']
        #If d=1 (static case), the DAG will be different
        if d == 1:
            Y_a = self._simulate_counterfactual_1d(a)
            return Y_a
        
        Y_a = []
        #Starting from time d, modify A(t,...,t+d-1), then  U(t+1,...,t+d-1), X(t+1,...,t+d-1), Y(t+d-1)
        #Make copies from the original arrays
        for t in range(d,T-d+1):
            #for t:t+d th segment, starting from the t+1 th X depends on t-(d-2) th covariates
            X_copy = self.X[:,(t-d+2):t+d].copy()
            A_copy = self.A[:,(t-d+2):t+d].copy()
            A_copy[:,-d:] = a
            for k in range(d-1):
                
                # X(t) depends on X(t-d+1,...,t-1), A(t-d+1,...,t-1)
                X_copy[:,k+d-1]=(self.X_filter['bias']
                                 +X_copy[:,k:k+d-1].swapaxes(1,2)@self.X_filter['X']
                                 +A_copy[:,k:k+d-1]@self.X_filter['A'])
    
            # Y(t) depends on X(t-d+1,...,t), A(t-d+1,...,t)
        
            Y_a.append(self._generate_Y(X_copy[:,-d:].reshape(S,-1),
                                        A_copy[:,-d:])) 
            '''
            Y_a.append(self.Y_filter['bias'] 
                       +X_copy[:,-d:].reshape(S,-1)@self.Y_filter['X']
                       +A_copy[:,-d:]@self.Y_filter['A']) 
            '''
        return(np.array(Y_a[max(0,buffer-d):]).reshape(-1,Y_dim))

            
            
    @property
    def datasets(self):
        # return the simulated datasets.
        T = self.params['T']
        buffer = self.params['buffer']
        assert T > buffer, 'T should be longer than buffer'

        return({'X':self.X[:,buffer:],'F':self.F[:,buffer:],'A':self.A[:,buffer:],'Y': self.Y[:,buffer:]})
    
    @staticmethod
    def plot_simu(simu_data):
        '''
        Plot simulated data
        '''
        X,F,A,Y = simu_data['X'],simu_data['F'],simu_data['A'],simu_data['Y']
        fig,ax = plt.subplots(1,4, figsize=(22,2.5))
        tt = min(Y.shape[1],150)
        data_to_plot = [Y,X,A,F]
        labels = ['Y','X','A','F']
        for i in range(4):

            if len(data_to_plot[i].shape)>2:
                ax[i].plot((data_to_plot[i][:2,:tt,0]).T)
            else:
                ax[i].plot((data_to_plot[i][:2,:tt]).T)
            ax[i].set_ylabel(labels[i])
            ax[i].set_xlabel('Time')
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        plt.subplots_adjust(wspace=0.5)
        plt.show()

        fig,ax = plt.subplots(1,4, figsize=(22,2.5))
        for i in range(4):
            if len(data_to_plot[i].shape)>2 and data_to_plot[i].shape[-1]>1:
                #ax[i].hist2d(data_to_plot[i][:,:,0].flatten(),data_to_plot[i][:,:,1].flatten(),bins=100)
                ax[i].scatter(data_to_plot[i][:,:,0].flatten(),data_to_plot[i][:,:,1].flatten(),s=1)
                ax[i].set_xlabel(labels[i]+'1')
                ax[i].set_ylabel(labels[i]+'2')
            else:
                ax[i].hist(data_to_plot[i].flatten(),bins=100)
                ax[i].set_xlabel(labels[i])
                ax[i].set_ylabel('count')
            
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        plt.subplots_adjust(wspace=0.5)
        plt.show()
    
    def _generate_simulation_filter(self, d, decay_coeff, X_dim, Y_dim):
    
        # X_filter: bias[1], X_X [(d-1)]*, X_A [d-1, X_dim]
        # F_filter: bias[1], F_X [d*X_dim], F_A [d-1]
        # Y_filter: bias[1], Y_X [d*X_dim, Y_dim], Y_A [d, Y_dim]

        seed = self.params['seed']
        if seed >=0:
                    np.random.seed(seed)
        d = self.params['d']
        X_dim = self.params['X_dim']
        Y_dim = self.params['Y_dim']                

        X_filter = {'bias': 0,
                    'X': np.array(exp_decay(-1,decay_coeff,d-1)),#X-X coupling, simplified65
                    'A': np.tile(np.array(exp_decay(1,decay_coeff,d-1))[:,None],reps = (1,X_dim)) } #X-A coupling

        F_filter = {'bias': -0.5 ,
                    'X': interleave([np.array(0.5*np.power(-1,np.arange(d)[::-1])) for _ in range(X_dim)]), #F-X coupling
                    'A': np.array(0.5*np.power(-1,np.arange(d-1)[::-1])) } #F-A coupling
        '''
        Y_filter = {'bias': -3,
                    'X':np.tile(np.array(exp_decay(-1,decay_coeff,d))[:,None],(X_dim,Y_dim))  , #Y-X coupling
                    'A': np.tile(np.array(exp_decay(2,decay_coeff,d))[:,None],reps = (1,Y_dim)) } #Y-A coupling
        '''
        Y_filter = {'bias':0,
            'Y_xlayer1':np.random.uniform(-1,1,size=(d*X_dim,d*X_dim)),
            'Y_xlayer2':np.random.uniform(-1,1,size=(d*X_dim,Y_dim)),
            'Y_alayer1':np.random.uniform(-2,2,size=(d,d)),
            'Y_alayer2':np.random.uniform(-2,2,size=(d,Y_dim))}
            # larger y_alayer sets outcomes for different treatments apart

        return(X_filter, F_filter, Y_filter)
    
    def _generate_Y(self, X,A):
        #print(X.shape,Y_filter['Y_xlayer1'].shape)
        #print(sigmoid(A@Y_filter['Y_alayer1']).shape)
        return(10*sigmoid(self.Y_filter['bias'] + sigmoid(X@self.Y_filter['Y_xlayer1'])@self.Y_filter['Y_xlayer2'] 
               +sigmoid(A@self.Y_filter['Y_alayer1'])@self.Y_filter['Y_alayer2']))
