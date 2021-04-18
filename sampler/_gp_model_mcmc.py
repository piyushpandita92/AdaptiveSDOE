"""
Returns a NSGP model object.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.set_random_seed(1224)
import gpflow
from _core import *
import numpy as np
import sys
from GPHetero import hetero_kernels, hetero_likelihoods, hetero_gpmc
from pyDOE import *

__all__ = ['NSGPMCMCModel']

class NSGPMCMCModel(object):
	"""
	NSGP MCMC model object.
	"""
	def __init__(self, X, Y, 
		ell_kern=gpflow.kernels.RBF,
		ss_kern=gpflow.kernels.RBF,
		noise_kern=gpflow.kernels.RBF,
		mcmc_samples=20,
		hmc_epsilon=0.00005,
		hmc_burn=100,
		hmc_thin=2,
		hmc_lmax=160,
		map_max_iter=500,
		nugget=1e-3,
		noisy=False,
		mean_const=-4, 
		**kwargs):
		assert X.ndim == 2
		assert Y.ndim == 2
		self.X = X
		self.Y = Y
		self.dim = self.X.shape[1]
		self.ell_kern = ell_kern
		self.ss_kern = ss_kern
		self.noise_kern  = noise_kern
		self.mcmc_samples = mcmc_samples
		self.map_max_iter = map_max_iter
		self.hmc_burn = hmc_burn
		self.hmc_thin = hmc_thin
		self.hmc_lmax = hmc_lmax
		self.mean_const = mean_const
		self.hmc_epsilon = hmc_epsilon
		self.nugget = nugget
		self.noisy = noisy
		if "ell_kern_variance_prior_list" in kwargs:
			self.ell_kern_variance_prior = kwargs["ell_kern_variance_prior_list"]
		else:
			self.ell_kern_variance_prior = gpflow.priors.Gamma(1., 1.)
		if "ell_kern_lengthscale_prior_list" in kwargs:
			self.ell_kern_lengthscale_prior = kwargs["ell_kern_lengthscale_prior_list"]
		else:
			self.ell_kern_lengthscale_prior = gpflow.priors.Gamma(1., 1.)
		if "ss_kern_variance_prior_list" in kwargs:
			self.ss_kern_variance_prior = kwargs["ss_kern_variance_prior_list"]
		else:
			self.ss_kern_variance_prior = gpflow.priors.Gamma(1., 1.)
		if "ss_kern_lengthscale_prior_list" in kwargs:
			self.ss_kern_lengthscale_prior = kwargs["ss_kern_lengthscale_prior_list"]
		else:
			self.ss_kern_lengthscale_prior = gpflow.priors.Gamma(1., 1.)
		if "mean_func_ell_prior_list" in kwargs:
			self.mean_func_ell_prior = kwargs["mean_func_ell_prior_list"]
		else:
			self.mean_func_ell_prior = None
		if "mean_func_ss_prior_list" in kwargs:
			self.mean_func_ss_prior = kwargs["mean_func_ss_prior_list"]
		else:
			self.mean_func_ss_prior = None
		if "mean_func_ell_const_list" in kwargs:
			self.mean_func_ell_const = kwargs["mean_func_ell_const_list"]
		else:
			self.mean_func_ell_const = None
		if "mean_func_ss_const_list" in kwargs:
			self.mean_func_ss_const = kwargs["mean_func_ss_const_list"]
		else:
			self.mean_func_ss_const = None

	def make_model(self):
		"""
		Perform fully Bayesian inference on the latent physical response.
		"""
		kerns_ell_list = []
		kerns_ss_list = []
		mean_ell_func = []
		mean_ss_func = []
		for i in xrange(self.dim):
			kerns_ell_list.append(self.ell_kern(input_dim=1))
			kerns_ss_list.append(self.ss_kern(input_dim=1))
			mean_ell_func.append(gpflow.mean_functions.Constant(1))
			mean_ss_func.append(gpflow.mean_functions.Constant(1)) # initialization remains the same for all problems
		nonstat = hetero_kernels.NonStationaryRBF()
		m = hetero_gpmc.GPMCAdaptiveLengthscaleMultDimEllSSDev(self.X, self.Y, kerns_ell_list, kerns_ss_list, mean_ell_func, mean_ss_func, nonstat)
		for i in xrange(self.dim):
			m.kerns_ell_list[i].lengthscales.prior = self.ell_kern_lengthscale_prior[i]
			m.kerns_ell_list[i].variance.prior = self.ell_kern_variance_prior[i]
			if self.mean_func_ell_const is not None:
				m.mean_funcs_ell_list[i].c = self.mean_func_ell_const[i]
				m.mean_funcs_ell_list[i].c.fixed = True 
			else:
				m.mean_funcs_ell_list[i].c.prior = self.mean_func_ell_prior[i]
			m.kerns_ss_list[i].lengthscales.prior = self.ss_kern_lengthscale_prior[i]
			m.kerns_ss_list[i].variance.prior = self.ss_kern_variance_prior[i]
			if self.mean_func_ss_const is not None:
				m.mean_funcs_ss_list[i].c = self.mean_func_ss_const[i]
				m.mean_funcs_ss_list[i].c.fixed = True 
			else:
				m.mean_funcs_ss_list[i].c.prior = self.mean_func_ss_prior[i]
		m.likelihood.variance = self.nugget ** 2
		m.likelihood.variance.fixed = True
		try: 
			m.optimize(maxiter=self.map_max_iter) 												 	# start near MAP
		except:
			print '>... optimization fail!! could not find the MAP'
			pass
		samples, acceptance_ratio = m.sample(self.mcmc_samples, 
			verbose=True, epsilon=self.hmc_epsilon, 
			thin=self.hmc_thin, burn=self.hmc_burn, 
			Lmax=self.hmc_lmax, return_acc_ratio=True)
		sample_df = m.get_samples_df(samples)
		return m, sample_df, acceptance_ratio
