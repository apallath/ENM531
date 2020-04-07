#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:16:50 2019

@author: paris
"""

import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm

np.random.seed(1234)


class RWMetropolis:
    def __init__(self, log_p, sigma):
        self.log_p = log_p
        self.sigma = sigma
        self.samples = []
    def sample(self, x0, nIter = 100):
        accepted = 0
        x = x0
        dim = x.shape[-1]
        logp = self.log_p(x)
        for i in tqdm(range(nIter)):
            x_prop = x + self.sigma * np.random.randn(dim)
            logp_prop = self.log_p(x_prop)
            alpha = min(1, np.exp(logp_prop - logp))
            if np.random.rand() < alpha:
                x = x_prop
                logp = logp_prop
                accepted += 1
            self.samples.append(x)
        print('Acceptance rate: %.2f%%' % (100.0*accepted/nIter))
        return np.asarray(self.samples)
            
    
class ODEfit:
    def __init__(self, t, X, x0, dxdt):      
        self.t = t
        self.x0 = x0
        self.X = X
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.dxdt = dxdt      
        self.sampler = RWMetropolis(self.log_posterior, 0.1)
        
    def log_prior(self, theta):
        sigma_sq = 1e2
        log_prior = -0.5*np.log(sigma_sq) - \
                     0.5*np.trace(np.matmul(theta, np.transpose(theta))/sigma_sq) - \
                     0.5*theta.shape[1]*np.log(2.0*np.pi)
        return log_prior
    
    def log_likelihood(self, theta):
        sigma_sq = 1e1
        mu = odeint(self.dxdt, self.x0, self.t, args= tuple(theta.flatten()))
        log_likelihood = -0.5*self.N*np.log(sigma_sq) - \
                          0.5*np.trace(np.matmul(self.X - mu, np.transpose(self.X - mu))/sigma_sq) - \
                          0.5*self.N*self.dim*np.log(2.0*np.pi)
        return log_likelihood
    
    
    def log_posterior(self, theta):
        log_posterior = self.log_likelihood(theta) + self.log_prior(theta)
        return log_posterior

        
