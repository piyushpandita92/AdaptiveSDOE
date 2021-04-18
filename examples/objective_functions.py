import os
import sys
import numpy as np
from scipy.stats import norm
import math

__all__ = ['Ex1Func', 'Ex2Func', 'Ex3Func', 'Ex4Func', 'Ex6Func', 'Ex7Func', 'Ex8Func']
	
class Ex1Func(object):
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
    	x = 6. * x
        return (4 * (1. - np.sin(x[0] + 8 * np.exp(x[0] - 7.))) - 10.) / 5. 

class Ex2Func(object):
    def __init__(self, sigma_noise=lambda x: 0.5, mu1=0, sigma1=1, mu2=0.5, sigma2=1):
        self.sigma_noise = sigma_noise
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def __call__(self, x):
    	return norm.pdf(x[0], loc=self.mu1, scale=self.sigma1) + norm.pdf(x[0], loc=self.mu2, scale=self.sigma2) 

class Ex3Func(object):

    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        """
        Dette et. al. function.
        Dette, Holger, and Andrey Pepelyshev. "Generalized Latin hypercube design for computer experiments." Technometrics 52, no. 4 (2010): 421-429.

        """
        y = 4 * ((x[0] - 2 + 8 * x[1] - 8 * (x[1] ** 2)) ** 2) + (3 - 4 * x[1]) ** 2 + 16 * np.sqrt(x[2] + 1) * ((2 * x[2] - 1)**2)
        return (y - 50) / 50.

class Ex4Func(object):
    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        """
        Taken from Knowles et al. ref 16 of the paper.
        """
        y = 10 * math.sin(np.pi * x[0] * x[1]) + (20 * ((x[2] - 5) ** 2)) + 10 * x[3] + 5 * x[4] 
        return (y - 400) / 50.

class Ex6Func(object):

    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        
        if x[0] < 0.5:
            return x[0]**2. +  self.sigma(x[0]) * np.random.randn()
        if x[0] == 0.5:
            return 1.
        if x[0] > 0.5:
            return 2. - (x[0] - 0.5) ** 2.

class Ex7Func(object):

    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        
        if x[0] < 0.5:
            return -1.
        if x[0] == 0.5:
            return -1.
        if x[0] > 0.5:
            return 1.

class Ex8Func(object):

    def __init__(self, sigma=lambda x: 0.5):
        self.sigma = sigma

    def __call__(self, x):
        y = math.sin(30. * ((x[0] - 0.9) ** 4.)) * math.cos(2. * (x[0] - 0.9)) + (x[0] - 0.9) / 2.
        return y