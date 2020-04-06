#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:59:50 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
sns.set(style="white")

from models import ODEfit

if __name__ == '__main__':
    
    # Define dynamics
    def pend(x, t, b, c):
        theta, omega = x
        dxdt = [omega, -b*omega - c*np.sin(theta)]
        return dxdt
    
    # True parameters
    b = 0.25
    c = 5.0
    params = np.array([b,c])
    
    n = 101
    dim = 2
    noise = 0.05
    
    x0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 10, n)
    
    # Generate time-series data
    X = odeint(pend, x0, t, args = tuple(params))
    X = X + noise*X.var(0)*np.random.randn(n, dim)
    
    # Create model
    model = ODEfit(t, X, x0, pend)
    
    # Sample
    burn_in = 500
    theta_init = np.array([[1.5,4.0]])
    samples = model.sampler.sample(theta_init, nIter = 5000)[burn_in:,:]
    
    # Posterior trajectories
    X_pred = []
    num_samples = 2000
    for i in range(num_samples):
        sol = odeint(pend, x0, t, args = tuple(samples[-num_samples+i,:,:].flatten()))
        X_pred.append(sol)

    # Visualize
    df = pd.DataFrame(samples[:,0,:], columns=['b', 'c'])
    
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.kdeplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.kdeplot, lw=3)
    plt.savefig('./pairgrid.png', dpi = 300)
    
    g = sns.jointplot(df['b'], df['c'], kind="kde", height=7, space=0)
    plt.savefig('./jointplot.png', dpi = 300)
    
    plt.figure()
    for i in range(num_samples):
        plt.plot(t, X_pred[i][:, 0], 'k', linewidth = 0.05, alpha = 0.2)
        plt.plot(t, X_pred[i][:, 1], 'k', linewidth = 0.05, alpha = 0.2)
    plt.plot(t, X[:, 0], 'bo', label=r'$\theta(t)$', alpha = 0.5)
    plt.plot(t, X[:, 1], 'go', label=r'$\omega(t)$', alpha = 0.5)
    plt.legend(loc='best')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.savefig('./time-series.png', dpi = 300)
    