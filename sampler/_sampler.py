"""
Information acquisition for Bayesian optimal design of experiments.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gpflow
import scipy
# from scipy.optimize import minimize
import math
import GPy
from pyDOE import *
from _core import *
# from _gp_model import *
from _saving_log import *
from _gp_model_mcmc import *
import time
import sys
from copy import copy
from scipy.stats import multivariate_normal
from scipy.stats import norm
start_time = time.time()

__all__ = ['KLSampler']

class KLSampler(object):
	"""
	This class computes the sensitivity of a set of inputs
	by taking the posterior expectation of the var of the
	corresponding effect functions.
	"""

	def __init__(self, X, Y, x_hyp, noisy, bounds, qoi_func, 
				qoi_idx=1,
				obj_func=None,
				true_func=None,
				ego_kern=GPy.kern.RBF,
				ell_kern=gpflow.kernels.RBF,
				noise_kern=gpflow.kernels.RBF,
				jitter=1e-4,
				mcmc_from_scratch=False,
				mcmc_samples=1500,
				map_max_iter=5000,
				start_from_map=True,
				hmc_burn=1500,
				hmc_thin=2,
				hmc_epsilon=0.00005,
				hmc_lmax=160,
				num_quad_points=100,
				energy=0.95,
				nugget=1e-3,
				kld_tol=1e-2,
				func_name='ex1',
				quad_points=None,
				quad_points_weight=None,
				opt_each_sample=False,
				max_it=50,
				lat_points='train',
				_acc_ratio_factor=0.65,
				**kwargs):
		"""
		:param X:		the inputs of the training data as an array.
		:param Y:		the outputs of the training data as an array.
		:param idx: 	set of indicies for which the
						effect function is needed.
		:param all:		if all lower level indicies are needed as well.
		"""
		assert X.ndim == 2
		self.X = X
		assert Y.ndim == 2
		self.Y = Y
		assert self.X.shape[0] == self.Y.shape[0]
		self.qoi_func = qoi_func
		self.dim = self.X.shape[1]
		self.num_obj = self.Y.shape[1]
		if obj_func is not None:
			self.obj_func = obj_func
		else:
			print "WARNING! sequential acquisition not possible without obj. function ..."
		self.true_func = true_func
		self.qoi_idx = qoi_idx
		self.nugget = nugget
		self.jitter = jitter
		self.noisy = noisy
		self.ego_kern = ego_kern
		self.ell_kern = ell_kern
		self.noise_kern  = noise_kern
		self.mcmc_from_scratch = mcmc_from_scratch
		self.mcmc_samples = mcmc_samples
		self.map_max_iter = map_max_iter
		self.start_from_map = start_from_map
		self.hmc_burn = hmc_burn
		self.hmc_thin = hmc_thin
		self.hmc_lmax = hmc_lmax
		self.hmc_epsilon = hmc_epsilon
		self.num_quad_points = num_quad_points
		self.energy = energy
		self.x_hyp = x_hyp
		self.mcmc_params = kwargs
		self._acc_ratio_factor = _acc_ratio_factor
		self.model, self.samples_df, self.acceptance_ratio = self.make_model()
		if quad_points is None:
			self.quad_points = np.linspace(0, 1, self.num_quad_points)
			self.quad_points_weight = np.eye(self.num_quad_points)
		else:
			self.quad_points = quad_points
			self.quad_points_weight = quad_points_weight
		self.bounds = bounds
		self.opt_each_sample  = opt_each_sample
		self.kld_tol = kld_tol
		self.func_name = func_name
		self.max_it = max_it

	def make_model(self, mcmc_from_scratch=True):
		"""
		Currently, supports only NSGP.
		"""
		if mcmc_from_scratch:
			print '>... making model from scratch'
			model = NSGPMCMCModel(X=self.X,
				Y=self.Y,
				ell_kern=self.ell_kern,
				noise_kern=self.noise_kern,
				mcmc_samples=self.mcmc_samples,
				hmc_epsilon=self.hmc_epsilon,
				hmc_burn=self.hmc_burn,
				hmc_thin=self.hmc_thin,
				hmc_lmax=self.hmc_lmax,
				map_max_iter=self.map_max_iter,
				nugget=self.nugget,
				noisy=self.noisy,
				**self.mcmc_params)
			model, sample_df, acceptance_ratio = model.make_model()
			return model, sample_df, acceptance_ratio
		else:
			print '>... using traces from the posterior from the previous iteration'
			self.model.X = self.X 			# Update inputs
			self.model.Y = self.Y 			# Update outputs
			if self.start_from_map:
				try:
					self.model.optimize(maxiter=self.map_max_iter)
				except:
					print '>... optimization fail!! could not find the MAP'
					pass
			else:
				pass
			try:
				samples, acceptance_ratio = self.model.sample(self.mcmc_samples, 
					verbose=True, epsilon=self.hmc_epsilon, 
					thin=self.hmc_thin, burn=self.hmc_burn, 
					Lmax=self.hmc_lmax, return_acc_ratio=True)
				sample_df = self.model.get_samples_df(samples)
			except:
				print '>... mcmc fail!! could not perform MCMC'
				acceptance_ratio = np.zeros(1)
				sample_df = self.samples_df
			return self.model, sample_df, acceptance_ratio
		
	def make_ego_gp(self, X, Y, num_restarts=40):
		"""
		Makes the GP model for the internal EKLD optimization.
		:param x:
		:param y:
		:return:
		"""
		model = GPy.models.GPRegression(X, Y, self.ego_kern(input_dim=X.shape[1], ARD=True))
		model.likelihood.variance.constrain_fixed(self.jitter ** 2)
		try:
			model.optimize_restarts(num_restarts=num_restarts, verbose=False)
		except:
			print '>... failed to optimize EKLD GP!'
			model = GPy.models.GPRegression(X, Y, self.ego_kern(input_dim=X.shape[1], ARD=True))
			model.likelihood.variance.constrain_fixed(self.jitter ** 2)
			return model
		return model

	def eig_func(self, x, w_j, x_d, val_trunc, vec_trunc, model):
		"""
		Constructing the eigenfunctions for the given eigenvalues at ```x```.
		"""
		k_x_d_x = model.pred_cov(x_grid, np.atleast_2d(x_hyp))
		eig_func = (1. / val_trunc) * np.sum(np.multiply(np.multiply(w_j, vec_trunc), k_x_d_x))
		return eig_func

	def eig_val_vec(self, model):
		"""
		Eigendecomposition of the ```B``` matrix in equation 15.88 of Handbook of UQ, chapter 15.
		"""
		x_d = self.quad_points
		p_x_d = self.quad_points_weight
		K_x_d = model.predict(x_d)[1]
		w_j = np.sqrt(((1. / (np.sum(self.quad_points_weight))) * np.diag(p_x_d)))
		B = np.matmul(np.matmul(w_j, K_x_d), w_j)
		val, vec = np.linalg.eigh(B)
		val[val<0] = 0																# taking care of the negative eigenvalues
		idx_sort = np.argsort(-val)
		val_sort = val[idx_sort]
		vec_sort = vec[:, idx_sort]
		tot_val = 1. * (np.cumsum(val_sort)) / np.sum(val_sort)
		try:
			idx_dim = min(np.where(tot_val >= self.energy)[0])
		except:
			energy_redu = self.energy / 2.
			try:
				print '>... trying with reduced energy'
				idx_dim = min(np.where(tot_val >= energy_redu)[0])
			except:
				print '>... eigendecomposition not possible'
				sys.exit()
		val_trunc = val_sort[:idx_dim + 1, ]
		vec_trunc = vec_sort[:, :idx_dim + 1]
		phi_x_dx = np.array([np.mean(np.sum(np.multiply(np.multiply(vec_trunc[:, j][:, None], (np.sqrt(((p_x_d / np.sum(self.quad_points_weight)))))[:, None]), K_x_d), axis=0), axis=0) for j in xrange(vec_trunc.shape[1])]) / val_trunc
		return val_trunc, vec_trunc, w_j, x_d, phi_x_dx

	def sample_xi_hyp(self, dim, val_trunc, eig_funcs, m_x_hyp, x_hyp, y_hyp, num_samp=1, model=None):
		"""
		Samples a multivariate random variable conditioned on the data and a 
		hypothetical observation.
		:param m_x:			keep in mind this is the posterior mean conditional 
							on data and a hypothetical observation.
		:param dim:			number of reduced dimensions of the eigenvalues.
		:param val_trunc:	eigenvalues after truncation.
		:param eig_funcs:	eigenvectors after truncation.
		:param y:			hypothetical sampled observation.	
		"""
		if x_hyp is None:
			x_hyp = self.x_hyp
		sigma_inv = np.multiply(np.matmul(np.sqrt(val_trunc)[:, None], np.sqrt(val_trunc)[None, :]), np.matmul(eig_funcs[:, None], eig_funcs[None, :]))
		if self.noisy:
			noise = np.exp(model.predict_n(np.atleast_2d(x_hyp))[0][0]) 
		else:
			noise = self.nugget
		sigma_inv_2 = sigma_inv / (noise ** 2)
		sigma_inv_1 = np.eye(dim)
		sigma_3 = np.linalg.inv(sigma_inv_1 + sigma_inv_2) 
		try:
			sigma_3_inv = np.linalg.inv(sigma_3)
			mu_3 = ((y_hyp - m_x_hyp) / (noise ** 2)) * np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs[:, None]))
		except:
			sigma_3 = np.linalg.inv(sigma_inv_1 + sigma_inv_2) + (self.nugget ** 2) * np.eye(dim)
			mu_3 = ((y_hyp - m_x_hyp) / (noise ** 2)) * np.matmul(sigma_3, np.multiply(np.sqrt(val_trunc)[:, None], eig_funcs[:, None]))
		try:
			xi = np.random.multivariate_normal(mu_3[:, 0], sigma_3, num_samp).T
		except:
			print mu_3, sigma_3
			print '>... could not sample from MVN for posterior of xi!'
			sys.stdout.flush()
			xi = -1.
		return xi

	def sample_xi(self, dim, num_samp=1):
		"""
		Samples a multivariate standard random variable.
		"""
		mu = np.zeros(dim, )
		sigma = np.eye(dim)
		xi = np.random.multivariate_normal(mu, sigma, num_samp).T
		return xi

	def obj_est(self, x_grid, num_samp=1, m_x=None, model=None):
		"""
		Samples a value of the QOI at a given design point.
		"""
		assert x_grid.shape[0] == self.quad_points.shape[0]
		assert x_grid.shape[1] == self.quad_points.shape[1]
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		sample_xi  = self.sample_xi(val_trunc.shape[0], num_samp=num_samp)
		eig_funcs = np.multiply(vec_trunc, 1. / (np.diag(w_j))[:, None]) 
		if m_x is None:
			samp = model.predict(np.atleast_2d(x_grid))[0] + np.matmul(eig_funcs, np.multiply(sample_xi, (np.sqrt(val_trunc))[:, None]))
		else:	
			samp = m_x + np.matmul(eig_funcs, np.multiply(sample_xi, (np.sqrt(val_trunc))[:, None]))
		if num_samp == 1:
			return samp.flatten(), val_trunc, eig_funcs
		else:
			return samp, val_trunc, eig_funcs

	def obj_est_hyp(self, x_grid, x_hyp, y_hyp=None, num_samp=1, model=None, k_x_d_x=None, m_x=None):
		# Repeating the process after adding the hypothetical observation to the data set
		assert x_grid.shape[0] == self.quad_points.shape[0]
		assert x_grid.shape[1] == self.quad_points.shape[1]
		if y_hyp is None:
			y_hyp = self.y_hyp
		m_x_hyp = model.predict(np.atleast_2d(x_hyp))[0][0]
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		if k_x_d_x is None:
			k_x_d_x = model.pred_cov(x_grid, np.atleast_2d(x_hyp))
		eig_funcs_f_hyp = np.multiply(vec_trunc, 1. / (np.diag(w_j))[:, None])
		eig_funcs_hyp = (1. / val_trunc) * np.sum(np.multiply(np.multiply((w_j[w_j > 0])[:, None], vec_trunc), k_x_d_x), axis=0)
		sample_xi_hyp = self.sample_xi_hyp(val_trunc.shape[0], val_trunc, eig_funcs_hyp, m_x_hyp, x_hyp, y_hyp, num_samp, model=model)
		if isinstance(sample_xi_hyp, np.ndarray):
			if m_x is None:
				samp_hyp = model.predict(np.atleast_2d(x_grid))[0] + np.matmul(eig_funcs_f_hyp, np.multiply(sample_xi_hyp, (np.sqrt(val_trunc))[:, None]))
			else:
				samp_hyp = m_x + np.matmul(eig_funcs_f_hyp, np.multiply(sample_xi_hyp, (np.sqrt(val_trunc))[:, None]))
			if num_samp == 1:
				return samp_hyp.flatten(), val_trunc, eig_funcs_f_hyp, m_x_hyp
			else:
				return samp_hyp, val_trunc, eig_funcs_f_hyp, m_x_hyp
		elif sample_xi_hyp == -1:
			return [-1]

	def get_eig_funcs_hyp(self, x, w_j, x_d, val_trunc, vec_trunc, model):
		"""
		Computes the values of the eigenfunctions at a point ```x```.
		:returns:			1-d array with the eigenfunctions evaluated at ```x```.
		"""
		eig_funcs_hyp = np.zeros(len(val_trunc))
		k_x_d_x = model.pred_cov(x_d, np.atleast_2d(x))
		eig_funcs_hyp = np.sum(np.multiply(vec_trunc, np.multiply((w_j[w_j>0])[:, None], k_x_d_x)), axis=0) / val_trunc
		return eig_funcs_hyp

	def qoi_qd(self, num_samp, m_x, model):
		"""
		Sampling the QoI ```num_samp``` number of times.
		:param num_samp:	Number of samples of the QoI taken for a given sample of hyperparameters theta^b. 
							This is the ```M``` of the paper.
		:param model:		This is the theta^b of the paper. Basically one of the ```B``` thetas sampled from the posterior.
		:returns:			Samples of the QoI ```Q|D_n``` obtained using the operator object ```qoi_func``` 
							on the samples of the underlying function obtained using KLE.
		"""
		# st =  time.time()
		qoi_qd_samp = self.obj_est(x_grid=self.quad_points, num_samp=num_samp, m_x=m_x,  model=model)[0]
		# print 'end', time.time() - st
		return self.qoi_func(qoi_qd_samp, qoi=self.qoi_idx)

	def qoi_qd_hyp(self, x_hyp, y_hyp, k_x_d_x, m_x, num_samp, model):
		"""
		:param x_hyp:		The hypothetical design or experiment.
		:param y_hyp:		A sampled hypothetical observation at ```x_hyp```.
		:param num_samp:	This is the number of samples of the samples of the QoI. Thisis the ```M``` of the paper.
		:param k_x_d_x:		The covariance between the quad_points and the hypothetical point.
		:param model:		This is the theta^b of the paper. Basically one of the ```B``` thetas sampled from the posterior.
		:returns:			Samples of the QoI ```Q|D_n, x_hyp, y_hyp``` obtained using the operator object ```qoi_func``` 
							on the samples of the underlying function obtained using KLE.
		"""
		# st_hyp =  time.time()
		qoi_qd_hyp_samp = self.obj_est_hyp(x_grid=self.quad_points, x_hyp=x_hyp, 
			y_hyp=y_hyp, num_samp=num_samp, k_x_d_x=k_x_d_x, m_x=m_x, model=model)[0]
		if isinstance(qoi_qd_hyp_samp, np.ndarray):
			return self.qoi_func(qoi_qd_hyp_samp, qoi=self.qoi_idx)
		else:
			return -1

	def optimize_for_qd_hyp(self, x, sample_xi_hyp, model):
		"""
		:param 	x:				Input at which a sample of the function has to be returned for a given sample_xi_hyp
		:param sample_xi_hyp:	Sampled xi from its posterior.
		:param model:			This is the theta^b of the paper. One of 
								the ```B``` thetas sampled from the posterior.
		:returns:				a scalar, a sample of the function at ```x```.
		"""
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		eig_funcs_f_hyp = self.get_eig_funcs_hyp(x, w_j, x_d, val_trunc, vec_trunc, model)
		fx = model.predict(np.atleast_2d(x))[0] + np.matmul(eig_funcs_f_hyp, np.multiply(sample_xi_hyp, (np.sqrt(val_trunc))[:, None]))
		if self.qoi_idx == 3:
			return np.ndarray.item(-fx)
		else:
			return np.ndarray.item(fx)

	def get_qoi_qd_hyp_for_opt(self, x_hyp, y_hyp, k_x_d_x, num_samp, model):
		"""
		Return an array of samples of the QoI when the QoI is the maximum or the minimum of the black-box code.
		:param x_hyp:		The hypothetical design or experiment.
		:param y_hyp:		A sampled hypothetical observation at ```x_hyp```.
		:param num_samp:	This is the number of samples of the samples of the QoI. This is the ```M``` of the paper.
		:param k_x_d_x:		The covariance between the quad_points and the hypothetical point.
		:param model:		This is the theta^b of the paper. One of 
							the ```B``` thetas sampled from the posterior.
		returns:			array of the samples of the QoI
		"""
		q_d_samp = np.ndarray(num_samp)
		m_x_hyp = model.predict(np.atleast_2d(x_hyp))[0][0]
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		eig_funcs = (1. / val_trunc) * np.sum(np.multiply(np.multiply((w_j[w_j > 0])[:, None], vec_trunc), k_x_d_x), axis=0)
		sample_xi_hyp  = self.sample_xi_hyp(dim=val_trunc.shape[0], val_trunc=val_trunc, eig_funcs=eig_funcs, m_x_hyp=m_x_hyp, x_hyp=x_hyp, y_hyp=y_hyp, num_samp=num_samp, model=model)
		if isinstance(sample_xi_hyp, np.ndarray):
			sample_xi_hyp = sample_xi_hyp.T
			for i in xrange(num_samp):
				if self.qoi_idx == 3:
					opt_res = scipy.optimize.minimize(fun=self.optimize_for_qd_hyp, x0=0.5 * np.ones(self.dim), method='L-BFGS-B', bounds=([[0, 1]] * self.dim), args=(sample_xi_hyp[i, ][:, None], model), options={'maxiter':500})
					q_d_samp[i] = -opt_res.fun
				elif self.qoi_idx == 4:
					opt_res = scipy.optimize.minimize(fun=self.optimize_for_qd_hyp, x0=0.5 * np.ones(self.dim), method='L-BFGS-B', bounds=([[0, 1]] * self.dim), args=(sample_xi_hyp[i, ][:, None], model), options={'maxiter':500})
					q_d_samp[i] = opt_res.fun
			return q_d_samp
		elif sample_xi_hyp == -1:
			return -1
		
	def get_params_qd(self, x_hyp, y_hyp, k_x_d_x, m_x, num_samp, model):
		"""
		Returns the mean and variance of the Q|D_n. The mu_1 and sigma_1^2 of the paper.
		:param x_hyp:		Hypothetical design (array-like).
		:param y_hyp:		Hypothetical value of the underlying objective, sampled from the posterior gaussian process.
		:param num_samp:	number of samples of the
		:returns:			mean and variance of the ```q_d``` and ```q_d_hyp``` in that order, respectively.
		"""
		if self.opt_each_sample:
			if self.qoi_idx == 3 or self.qoi_idx == 4:
				qoi_qd_hyp_samp = self.get_qoi_qd_hyp_for_opt(x_hyp, y_hyp, k_x_d_x, num_samp=num_samp, model=model)
			else:
				qoi_qd_hyp_samp = self.qoi_qd_hyp(x_hyp, y_hyp, k_x_d_x, m_x, num_samp=num_samp, model=model)
		else:
				qoi_qd_hyp_samp = self.qoi_qd_hyp(x_hyp, y_hyp, k_x_d_x, m_x, num_samp=num_samp, model=model)
		if isinstance(qoi_qd_hyp_samp, np.ndarray):
			return (np.mean(qoi_qd_hyp_samp), np.var(qoi_qd_hyp_samp)) 
		else:
			return -1

	def optimize_for_qd(self, x, sample_xi=None, model=None):
		"""
		Sample the black-box function using KLE around the posterior mean of the NSGP at a design ```x``` to optimize f(x; xi, theta).
		"""
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		eig_funcs = self.get_eig_funcs_hyp(x, w_j, x_d, val_trunc, vec_trunc, model)
		fx = model.predict(np.atleast_2d(x))[0] + np.matmul(eig_funcs, np.multiply(sample_xi, (np.sqrt(val_trunc))[:, None]))
		if self.qoi_idx == 3:
			return np.ndarray.item(-fx)
		else:
			return np.ndarray.item(fx)

	def get_qoi_qd_for_opt(self, num_samp, model):
		"""
		Return an array of samples of the QoI when it is the maximum or the minimum of the black-box code.
		:param num_samp:	This is the number of samples of the samples of the QoI. Thisis the ```M``` of the paper.
		:param model:		This is the theta^b of the paper. Basically one of the ```B``` thetas sampled from the posterior.
		:returns:				array containing ```num_samp``` samples of the QoI.
		"""
		q_d_samp = np.ndarray(num_samp)
		val_trunc, vec_trunc, w_j, x_d, phi_x_dx = self.get_val_vec
		sample_xi  = self.sample_xi(dim=val_trunc.shape[0], num_samp=num_samp).T
		for i in xrange(num_samp):
			# clock = time.time()
			if self.qoi_idx == 3:
				opt_res = scipy.optimize.minimize(fun=self.optimize_for_qd, x0=0.5 * np.ones(self.dim), method='L-BFGS-B', bounds=([[0, 1]] * self.dim), args=(sample_xi[i, :][:, None], model), options={'maxiter':500})
				q_d_samp[i] = -opt_res.fun
			elif self.qoi_idx == 4:
				opt_res = scipy.optimize.minimize(fun=self.optimize_for_qd, x0=0.5 * np.ones(self.dim), method='L-BFGS-B', bounds=([[0, 1]] * self.dim), args=(sample_xi[i, :][:, None], model), options={'maxiter':500})
				q_d_samp[i] = opt_res.fun
			# print opt_res
			# print 'computed the optimum in ', time.time() - clock
		return q_d_samp

	def qoi_qd_mcmc(self, val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc, num_samp=50, num_samp_gp=20):
		"""
		Computes the mean and variance of Q|D for each of the ```num_samp_gp``` GPs or ```B``` samples from the posterior of the
		hyperparameters.
		:param val_trunc_mcmc: 	eigenvalues from the eigendecomposition of the covariance matrix at ```quad_points```.
		:param vec_trunc_mcmc:		eigenvectors from the eigendecomposition of the covariance matrix at ```quad_points```.
		:param W_h_mcmc:			weights of the ```quad_points```. chosen according to a simple LHS quadrature rule to be  1 / num_quad_points.
		:param x_d_mcmc:			quadrature points.
		:param phi_x_dx_mcmc:		this is the integral of the eigenfunctions at the quadrature points.
		:num_samp:					number of samples of the QoI to be taken.
		:num_samp_gp:				number of samples of the hyperparameters taken from their posterior distribution.
		"""
		samp_qd = []
		x_d = self.quad_points
		sample_df = self.samples_df
		m = self.model
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			m_x = self._get_m_x(x_d, model=m)
			idx = i - (self.mcmc_samples - num_samp_gp)
			self.get_val_vec = self._get_val_vec_gp(idx, val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc)
			if self.opt_each_sample:														# Only for higher dimensional functions
				if self.qoi_idx == 3 or self.qoi_idx == 4:
					samp_qd.append(self.get_qoi_qd_for_opt(num_samp=num_samp, model=m))
				else:
					samp_qd.append(self.qoi_qd(num_samp=num_samp, m_x=m_x, model=m))
			else:
				samp_qd.append(self.qoi_qd(num_samp=num_samp, m_x=m_x, model=m))
		return samp_qd

	def get_qd_mcmc(self, samp_qd):
		"""
		Computes the mean and variance for samples of the QoI conditioned on a given theta^b from the paper.
		:param 	samp_qd:	a list of samples of the QoI over all the retained ```B``` samples from the posterior of the hyperparameters.
		:returns:			scalar or 1-d array-like ```mu_qd``` (mean) and ```sigma_qd``` variance of the QoI.
		"""
		samp_qd = np.hstack(samp_qd) 
		mu_qd = np.mean(samp_qd)
		sigma_qd = np.var(samp_qd)
		return mu_qd, sigma_qd

	def avg_kld(self, x_hyp, val_trunc, vec_trunc, W_h, x_d, phi_x_dx, num_samp=100, num_samp_yhyp=100, mu_qd=None, sigma_qd=None, m_x=None, model=None):
		"""
		:param x_hyp: 		hypothetical design at which the EKLD is to be computed.
		:param val_trunc: 	truncated eigenvalues of the covariance matrix.
		:param vec_trunc: 	truncated eigenvectors of the covariance matrix.
		:param W_h: 		weights across the fine grid of the input space.
		:param x_d: 		the discretized grid in the input space.
		:param phi_x_dx:	integrated eigenfunction across the input space.
		:param model:		gaussian process surrogate model of the physical process.
		:param m_x:			predicted mean at the quad points.
		:returns:			a scalar, the sample averaged EKLD for ```x_hyp```.
		"""
		kl_hyp = 0
		k_x_d_x = model.pred_cov(x_d, np.atleast_2d(x_hyp))
		y_hyp = model.posterior_samples(np.atleast_2d(x_hyp), num_samp_yhyp)
		for i in xrange(num_samp_yhyp):
			params_qd_hyp = self.get_params_qd(x_hyp, y_hyp[i, 0], k_x_d_x, m_x, num_samp=num_samp, model=model)
			if params_qd_hyp == -1.:
				kl_hyp += 0
			else:
				mu_qd_hyp = params_qd_hyp[0]
				sigma_qd_hyp = params_qd_hyp[1]
				kl_hyp += 0.5 * np.log(sigma_qd / sigma_qd_hyp) + ((sigma_qd_hyp + ((mu_qd - mu_qd_hyp) ** 2)) / (2. * sigma_qd)) - 0.5
		ekld_hyp = kl_hyp / num_samp_yhyp
		return ekld_hyp

	def update_XY(self, x_best, y_obs):
		"""
		Augment the observed set with the newly added design and the
		corresponding function value.
		:param x_best:		the array-like new design selected by BODE.
		:param y_obs:		the simulation output at ```x_best```.
		"""
		self.X = np.vstack([self.X, np.atleast_2d([x_best])])
		self.Y = np.vstack([self.Y, np.atleast_2d([y_obs])])

	def _get_m_x(self, x_d=None, model=None):
		"""
		Predicts the posterior mean at a point(s) ```x_d```.
		:param 	x_d:		array of points at which the posterior mean is to be computed.
		:param  model:		GP model object with a sample of hyperparameters from their posterior distribution.
		:returns m_x:		scalar or array-like posterior mean of the function for a given theta^b.
		"""
		m_x = model.predict(x_d)[0]
		return m_x

	def mcmc_ekld(self, x_hyp, val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc, num_samp=50, num_samp_yhyp=50, num_samp_gp=20, samp_qd=None):
		"""
		:param x_hyp:			The hypothetical input/design for which the Information Gain (EKLD) is being approximated.
		:param num_samp_gp:		The number of samples from the posterior of hyperparameters. This is the ```B``` of the paper.
		:param num_samp_yhyp:	The number of samples of y hypothetical at a hyopthetical experiment. This is the ```B``` of the paper.
		:param num_samp:		The number of samples of the QoI. This is the ```M``` of the paper.
		:param samp_qd:			The samples of Q|D_n evaluated for each of the ```S``` (num_samp_gp) hyperparameters.
		:returns:				The sample averaged EKLD over all retained hyperparameters at a hypothetical design ```x_hyp```. 
								This is the G(x) of the paper.
		"""
		sample_df = self.samples_df
		m = self.model
		ekld_x_hyp_mcmc = np.zeros(num_samp_gp)
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):	# Looping over all ```S``` \theta s from the posterior.
			m.set_parameter_dict(sample_df.iloc[i])
			idx = i - (self.mcmc_samples - num_samp_gp)
			m_x = self._get_m_x(np.atleast_2d(x_d_mcmc[idx]), model=m)
			mu_qd, sigma_qd = self.get_qd_mcmc(samp_qd[idx])				# This gets the Q_d|D_n conditioned on \theta^s.
			self.get_val_vec = self._get_val_vec_gp(idx, val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc)
			ekld_x_hyp_mcmc[idx, ] = self.avg_kld(x_hyp, val_trunc_mcmc[idx], vec_trunc_mcmc[idx], W_h_mcmc[idx], x_d_mcmc[idx], phi_x_dx_mcmc[idx],
								num_samp=num_samp, num_samp_yhyp=num_samp_yhyp, mu_qd=mu_qd, sigma_qd=sigma_qd, m_x=m_x, model=m)
		ekld_hyp = np.mean(ekld_x_hyp_mcmc)
		print '>... ekld computed for x = ', x_hyp, '>... ekld = ', ekld_hyp
		return ekld_hyp

	def get_val_vec_mcmc(self, num_samp_gp):
		"""
		Get the KLE decomposition for each GP sampled using MCMC.
		:param num_samp_gp:		Number of hyperparameters retained from the posterior. This is the ```B``` of the paper.
		:returns:				A list containing all the eigenvalues, eigenvectors, weights, quadrature points and
								integral of the eigenfunctions at the quadrature points for each theta^b.
		"""
		val_trunc_mcmc = []
		vec_trunc_mcmc = []
		W_h_mcmc = []
		x_d_mcmc = []
		phi_x_dx_mcmc = []
		m = self.model
		sample_df = self.samples_df
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			val_trunc, vec_trunc, W_h, x_d, phi_x_dx = self.eig_val_vec(m)
			val_trunc_mcmc.append(val_trunc)
			vec_trunc_mcmc.append(vec_trunc)
			W_h_mcmc.append(W_h)
			x_d_mcmc.append(x_d)
			phi_x_dx_mcmc.append(phi_x_dx)
		return val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc

	def _get_val_vec_gp(self, idx, val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc):
		"""
		:returns:			 the KLE components for the ```theta^b``` model.
		"""
		return val_trunc_mcmc[idx], vec_trunc_mcmc[idx], W_h_mcmc[idx], x_d_mcmc[idx], phi_x_dx_mcmc[idx]

	def optimize(self, X_design=None, num_designs=1000, verbose=0, plots=0, num_designs_ego=50, num_samp=100, num_samp_yhyp=100, num_samp_gp=20, ekld_lhs_fac=0.2, num_post_samp=1000):
		"""
		:param num_designs:		A discretized set of hypothetical designs
		:param plots:			To plot the lengthscales from the posterior plots should be greater than 1. 
								If the EKLD is to be plotted along with the state of the algorithm plots should be greater than 2.
		:param ekld_lhs_fac:	Fraction of ekld iterations to be used for initial design
		:returns:				Final set of inputs, and outputs, ...
		"""
		rel_kld = np.zeros(self.max_it)
		kld_all = np.ndarray((self.max_it, num_designs))
		mu_qoi = []
		sigma_qoi = []
		models = []
		samples = []
		for it in xrange(self.max_it):
			print 'iteration no. ', it + 1, 'of ', self.max_it
			kld = np.zeros(num_designs)
			val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc = self.get_val_vec_mcmc(num_samp_gp=num_samp_gp)
			samp_qd = self.qoi_qd_mcmc(val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc, num_samp=num_samp, num_samp_gp=num_samp_gp)
			mu_qd, sigma_qd = self.get_qd_mcmc(samp_qd)
			mu_qoi.append(mu_qd)
			sigma_qoi.append(sigma_qd)
			models.append(self.model)
			samples.append(self.samples_df)
			print '>... current mean of the QoI', mu_qd
			print '>... current variance of the QoI', sigma_qd
			num_lhs_ego = int(num_designs_ego * ekld_lhs_fac)
			if num_lhs_ego <= 1:
				raise ValueErrorr("number of ego designs should be greater than 10 !")
			num_seq_ego = int(num_designs_ego * (1 - ekld_lhs_fac))
			ekld_lhs_ego = np.ndarray((num_lhs_ego, 1))
			if X_design is not None:
				ego_lhs = X_design[np.random.randint(0, num_designs, num_lhs_ego), ]
			else:
				ego_lhs = lhs(self.X.shape[1], num_lhs_ego, criterion='center')
			print '>... computing the EKLD for the EGO initial designs.'
			for i in xrange(num_lhs_ego):
				ekld_lhs_ego[i, ] = -1. * self.mcmc_ekld(ego_lhs[i, :], num_samp=num_samp, 
					num_samp_yhyp=num_samp_yhyp, num_samp_gp=num_samp_gp, 
					samp_qd=samp_qd, 
					val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, 
					W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc)
			mu_ekld = np.mean(ekld_lhs_ego, axis=0)
			sigma_ekld = np.sqrt(np.var(ekld_lhs_ego, axis=0))
			ekld_lhs_ego = (ekld_lhs_ego - mu_ekld) / sigma_ekld
			ego_model = self.make_ego_gp(ego_lhs, ekld_lhs_ego)
			print '>... done.'
			for i in xrange(num_seq_ego):
				if X_design is None:
					X_design = lhs(self.X.shape[1], num_designs, criterion='center') 
				else:
					pass
				ego_min = min(ego_model.predict(ego_lhs, full_cov=False, include_likelihood=False)[0])
				mu, sigma = ego_model.predict(X_design, full_cov=False, include_likelihood=False)
				ei_ekld = ei(mu, sigma, ego_min)
				x_best_ego = X_design[np.argmax(ei_ekld), :]
				print '>... design selected for EKLD computation: ', x_best_ego
				y_obs_ego = (-1. * self.mcmc_ekld(x_best_ego, 
					num_samp=num_samp, num_samp_yhyp=num_samp_yhyp, 
					num_samp_gp=num_samp_gp, samp_qd=samp_qd, 
					val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, 
					W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc) - mu_ekld) / sigma_ekld
				ego_lhs = np.vstack([ego_lhs, np.atleast_2d([x_best_ego])])
				ekld_lhs_ego = np.vstack([ekld_lhs_ego, np.atleast_2d([y_obs_ego])])
				print '>... reconstructing EKLD EGO surrogate model.'
				ego_model = self.make_ego_gp(ego_lhs, ekld_lhs_ego)
				print '>... done.'
			idx_best = np.argmin(ekld_lhs_ego)
			print '>... maximum EKLD:', max(-ekld_lhs_ego)
			x_best = ego_lhs[idx_best, ]
			if verbose > 0:
				print '>... run the next experiment at design: ', x_best
			y_obs = self.obj_func(x_best)
			kld = -1. * (mu[:, 0] * sigma_ekld + mu_ekld)
			kld_all[it, :] = -1. * ((mu[:, 0] * sigma_ekld) + mu_ekld)
			rel_kld[it] = max(-1. * ((mu[:, 0] * sigma_ekld) + mu_ekld))
			if verbose > 0:
				print '>... simulated the output at the selected design', y_obs
			if plots > 0:
				ekld_norm = {'mu_ekld':mu_ekld, 'sigma_ekld':sigma_ekld}
				self.make_plots(it, -ekld_lhs_ego, X_design, x_best, y_obs, 
					ekld_model=ego_model, ekld_norm=ekld_norm, plots=plots, 
					num_post_samp=num_post_samp, num_samp_gp=num_samp_gp)
			self.update_XY(x_best, y_obs)
			if self.dim > 1:
				self.quad_points = lhs(self.dim, self.num_quad_points) 					# Refresh the quad points
			if verbose > 0:
				print '>... reconstructing surrogate model(s)'
			if self.acceptance_ratio[-1] < self._acc_ratio_factor:
				self.model, self.samples_df, self.acceptance_ratio = self.make_model(mcmc_from_scratch=True)
			else:
				self.model, self.samples_df, self.acceptance_ratio = self.make_model(mcmc_from_scratch=self.mcmc_from_scratch)
			tol_ratio = (max(kld) / max(rel_kld))
			if tol_ratio < self.kld_tol:
				print '>... relative ekld below specified tolerance ... stopping optimization now.'
				break
			if it == self.max_it-1:
				val_trunc_mcmc, vec_trunc_mcmc, W_h_mcmc, x_d_mcmc, phi_x_dx_mcmc = self.get_val_vec_mcmc(num_samp_gp=num_samp_gp)
				samp_qd = self.qoi_qd_mcmc(val_trunc_mcmc=val_trunc_mcmc, vec_trunc_mcmc=vec_trunc_mcmc, W_h_mcmc=W_h_mcmc, x_d_mcmc=x_d_mcmc, phi_x_dx_mcmc=phi_x_dx_mcmc, num_samp=num_samp, num_samp_gp=num_samp_gp)
				mu_qd, sigma_qd = self.get_qd_mcmc(samp_qd)
				mu_qoi.append(mu_qd)
				sigma_qoi.append(sigma_qd)
				models.append(self.model)
				samples.append(self.samples_df)
		return self.X, self.Y, kld_all, X_design, mu_qoi, sigma_qoi, models, samples

	def make_plots(self, it, kld, X_design, x_best, y_obs, ekld_model=None, ekld_norm=None, plots=1, num_post_samp=1000, num_samp_gp=20):
		# matplotlib.use('PS')
		# matplotlib.use('Qt4Agg')
		# import seaborn as sns
		sns.set_style("white")
		sns.set_context("paper")
		n = self.X.shape[0]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		x_grid = self.quad_points
		if self.true_func:
			y_grid = np.array([self.true_func(x_grid[i]) for i in xrange(x_grid.shape[0])])
			ax1.plot(x_grid, y_grid, c=sns.color_palette()[0], linewidth=4.0, label='true function')
		sample_df = self.samples_df
		m = self.model
		if self.noisy:
			y_pos_n = []
		if plots > 1:
			y_m_ell = []
			y_m_ss = []
		y_pos = []
		for i in range(self.mcmc_samples - num_samp_gp, self.mcmc_samples):
			m.set_parameter_dict(sample_df.iloc[i])
			y_pos.append(m.posterior_samples(x_grid, num_post_samp))
			if self.noisy:
				y_pos_n.append(m.posterior_samples_n(x_grid, num_post_samp))
			if plots > 1:
				y_m_ell.append(m.predict_l(x_grid)[0])	
				y_m_ss.append(m.predict_s(x_grid)[0])						# Note: makes sense for a 1D function only.
		y_pos = np.vstack(y_pos)
		y_m = np.percentile(y_pos, 50, axis=0)
		y_l = np.percentile(y_pos, 2.5, axis=0)
		y_u = np.percentile(y_pos, 97.5, axis=0)
		if self.noisy:	
			y_pos_n = np.vstack(y_pos_n)
			y_m_n = np.percentile(y_pos_n, 50, axis=0)
			y_l_n = np.percentile(y_pos_n, 2.5, axis=0)
			y_u_n = np.percentile(y_pos_n, 97.5, axis=0)
			ax1.fill_between(x_grid[:, 0], y_l_n, y_u_n, color=sns.color_palette()[1], alpha=0.25, zorder=3)
		ax1.plot(x_grid, y_m, '--', c=sns.color_palette()[1], linewidth=3.0, label='NSGP', zorder=3)
		ax1.fill_between(x_grid[:, 0], y_l, y_u, color=sns.color_palette()[1], alpha=0.5, zorder=3)
		if it == self.max_it-1:
			ax1.scatter(x_best, y_obs, marker='X', s=80, c='black', zorder=10)
			if self.noisy:
				ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
			else:
				ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		else:
			ax1.scatter(x_best, y_obs, marker='D', s=80, c=sns.color_palette()[3], label='latest experiment', zorder=10)
			if self.noisy:
				ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
			else:
				ax1.scatter(self.X[:, 0], self.Y[:, 0], marker='X', s=80, c='black', label='observed data', zorder=10)
		if plots > 2:
			idx = np.argsort(X_design[:, ], axis=0)[:, 0]
			mu_ekld = ekld_norm['mu_ekld']
			sigma_ekld = ekld_norm['sigma_ekld']
			try:
				y_ekld_pos = -1 * (ekld_model.posterior_samples_f(X_design, 1000) * sigma_ekld + mu_ekld)
				y_ekld_m = np.percentile(y_ekld_pos, 50, axis=1) 
				y_ekld_l = np.percentile(y_ekld_pos, 2.5, axis=1) 
				y_ekld_u = np.percentile(y_ekld_pos, 97.5, axis=1) 
				ekld = ax2.plot(X_design[idx[:]], y_ekld_m[idx[:]], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], label='EKLD GP', zorder=5)
				# ax2.fill_between(X_design[idx[:], 0], y_ekld_l[idx[:]], y_ekld_u[idx[:]], color=sns.color_palette()[2], alpha=0.25, zorder=5)
			except:
				print ">... plotting error! sampling from EKLD posterior (GP) not possible. moving on without plotting the EKLD."
				pass
		ax1.set_xlabel('$x$', fontsize=16)
		ax2.set_ylabel('$G(x)$', fontsize=16)
		ax2.set_ylim(0, 1)
		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc=9, fontsize=12)
		plt.xticks(fontsize=16)
		ax1.tick_params(axis='both', which='both', labelsize=16)
		ax2.tick_params(axis='both', which='both', labelsize=16)
		ax2.spines['right'].set_color(sns.color_palette()[2])
		ax2.yaxis.label.set_color(sns.color_palette()[2])
		ax2.tick_params(axis='y', colors=sns.color_palette()[2])
		ax1.set_ylabel('$f(x)$', fontsize=16)
		ax1.set_xlim(self.bounds["a"], self.bounds["b"])
		plt.savefig(self.func_name + '_kld_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
		plt.clf()
		if plots > 1:
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			[ax1.plot(x_grid, np.exp(y_m_ell[i][0]), linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5) for i in xrange(num_samp_gp)]
			ax1.plot([-1], [-1], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5, label='ell GP')
			ax1.set_xlabel('$x$', fontsize=16)
			ax1.set_ylabel('$ell(x)$', fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.legend(fontsize=12)
			ax1.set_xlim(self.bounds["a"], self.bounds["b"])
			ax1.set_ylim(min([min(np.exp(y_m_ell[i][0])) for i in xrange(num_samp_gp)]), max([max(np.exp(y_m_ell[i][0])) for i in xrange(num_samp_gp)]))
			plt.savefig(self.func_name + '_ell_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
			plt.clf()
			fig = plt.figure()
			ax1 = fig.add_subplot(111)
			[ax1.plot(x_grid, np.exp(y_m_ss[i][0]), linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5) for i in xrange(num_samp_gp)]
			ax1.plot([-1], [-1], linestyle='-.', linewidth=3.0, c=sns.color_palette()[2], zorder=5, label='ss GP')
			ax1.set_xlabel('$x$', fontsize=16)
			ax1.set_ylabel('$ss(x)$', fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.legend(fontsize=12)
			ax1.set_xlim(self.bounds["a"], self.bounds["b"])
			ax1.set_ylim(min([min(np.exp(y_m_ss[i][0])) for i in xrange(num_samp_gp)]), max([max(np.exp(y_m_ss[i][0])) for i in xrange(num_samp_gp)]))
			plt.savefig(self.func_name + '_ss_' + str(it + 1).zfill(len(str(self.max_it))) + '.png', dpi=(300), figsize=(3.25, 3.25))
			plt.clf()
