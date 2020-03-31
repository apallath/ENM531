#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:55:11 2018

@author: paris
"""

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    
    def p(x):
        return st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)
    
    
    def q(x):
        return st.norm.pdf(x, loc=50, scale=30)
    
    
    x = np.arange(-50, 151)
    k = max(p(x) / q(x))
    
    
    def rejection_sampling(iter=1000):
        samples = []
    
        for i in range(iter):
            z = np.random.normal(50, 30)
            u = np.random.uniform(0, k*q(z))
    
            if u <= p(z):
                samples.append(z)
    
        return np.array(samples)

    s = rejection_sampling(iter=20000)
    
    plt.legend()
    plt.plot(x, p(x), label = 'p_tilde(x)$')
    plt.plot(x, k*q(x), label = '$k*q(x)$')
    plt.legend()
    plt.show()
    sns.distplot(s)
