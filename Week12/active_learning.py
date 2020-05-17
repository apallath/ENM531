#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:22:57 2018

@author: paris
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from models_autograd import GPRegression

np.random.seed(1234)

def f(x):
    return x * np.sin(4.0*np.pi*x)


if __name__ == "__main__":    
    
    N = 15
    D = 1
    lb = -0.5*np.ones(D)
    ub = 1.0*np.ones(D)
    noise = 0.3
    tol = 1e-4
    nsteps = 200
    
    # Training data    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X)
    y = y + np.std(y)*noise*np.random.randn(N,D)
    
    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)[:,None]
    y_star = f(X_star)
    
    # Define model
    model = GPRegression(X, y)
    
    plt.figure(1, facecolor = 'w')
    
    for i in range(0,nsteps):
        # Train 
        model.train()
        
        # Predict
        y_pred, y_var = model.predict(X_star)
        y_var = np.abs(np.diag(y_var))[:,None]
        
        # Sample where posterior variance is maximized
        idx = np.argmax(y_var)
        new_X = X_star[idx:idx+1,:]   
        new_y = f(new_X)
        new_y = new_y + np.std(y)*noise*np.random.randn(1,D)
        
        # Check for convergence
        if np.max(y_var) < tol:
            print("Converged!")
            break
        
        # Plot
        plt.cla()
        plt.plot(X_star, y_star, 'b-', label = "Exact", linewidth=2)
        plt.plot(X_star, y_pred, 'r--', label = "Prediction", linewidth=2)
        lower = y_pred - 2.0*np.sqrt(y_var)
        upper = y_pred + 2.0*np.sqrt(y_var)
        plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                         facecolor='orange', alpha=0.5, label="Two std band")
        plt.plot(X,y,'bo', label = "Data")
        plt.plot(new_X*np.ones((2,1)), np.linspace(-2,2,2),'k--')
        ax = plt.gca()
        ax.set_xlim([lb[0], ub[0]])
        ax.set_ylim([-2,2])
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title("Iteration #%d" % (i+1))
        plt.pause(0.5)
#        plt.savefig("./figures/AL_it_%d.png" % (i+1), format='png', dpi=300)
        
        # Add new point to the training set
        model.update(new_X, new_y)
        X = np.concatenate([X, new_X], axis = 0)
        y = np.concatenate([y, new_y], axis = 0)
      
    



   
   